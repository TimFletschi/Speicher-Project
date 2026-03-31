from dataclasses import dataclass, replace
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    RangeSet,
    UnitInterval,
    Var,
    maximize,
    value,
)

from Storage_Test import (
    Inputs as BaseInputs,
    evaluate_results,
    export_results_and_timeseries,
    load_timeseries,
    opportunity_price_eur_per_kwh,
    print_results,
    solve_model,
)


@dataclass
class InputsNPV(BaseInputs):
    # Diskontsatz (Hurdle Rate), z. B. 6%
    discount_rate: float = 0.06
    # Baukosten Speicher (CAPEX) explizit im NPV-Tool setzbar
    capex_batt_E_eur_per_kwh: float = 120.0
    # Standardverbrauch für dieses NPV-Tool
    annual_consumption_kwh: float = 400000.0

    # PV-Größen-Sweep
    run_pv_size_sweep: bool = True
    pv_size_sweep_min_kwp: float = 100.0
    pv_size_sweep_max_kwp: float = 2000.0
    pv_size_sweep_step_kwp: float = 400.0


def _yearly_degradation_factors(inp: InputsNPV) -> np.ndarray:
    if inp.horizon_years <= 0:
        raise ValueError("horizon_years muss > 0 sein.")
    if not (0.0 < inp.batt_capacity_retention_after_horizon <= 1.0):
        raise ValueError("batt_capacity_retention_after_horizon muss im Intervall (0, 1] liegen.")
    if inp.horizon_years == 1:
        return np.array([inp.batt_capacity_retention_after_horizon], dtype=float)
    return np.linspace(1.0, inp.batt_capacity_retention_after_horizon, inp.horizon_years, dtype=float)


def npv_multiplier(inp: InputsNPV) -> float:
    """Konstanter Multiplikator für den jährlichen Mehrwert inkl. Degradation und Diskontierung."""
    if inp.discount_rate <= -1.0:
        raise ValueError("discount_rate muss > -1.0 sein.")

    factors = _yearly_degradation_factors(inp)
    years = np.arange(1, inp.horizon_years + 1, dtype=float)
    disc = (1.0 + inp.discount_rate) ** years
    return float(np.sum(factors / disc))


def build_model_npv(prices: np.ndarray, pv_per_kwp: np.ndarray, load: np.ndarray, inp: InputsNPV) -> ConcreteModel:
    T = len(prices)
    m = ConcreteModel()
    m.T = RangeSet(0, T - 1)
    neg_price = (prices < 0.0).astype(float)
    nonneg_price = (prices >= 0.0).astype(float)
    opp_price = opportunity_price_eur_per_kwh(prices, inp.pv_size_kwp)

    pv_gen = pv_per_kwp * inp.pv_size_kwp
    pv_direct_to_load_baseline = np.minimum(pv_gen, load)
    pv_surplus_baseline = np.maximum(0.0, pv_gen - pv_direct_to_load_baseline)
    pv_direct_to_grid_baseline = np.where(prices >= 0.0, pv_surplus_baseline, 0.0)
    annual_value_only_pv_baseline = float(
        np.sum(opp_price * pv_direct_to_grid_baseline)
        + inp.pv_to_load_remuneration_eur_per_kwh * np.sum(pv_direct_to_load_baseline)
    )

    m.E_max = Var(domain=NonNegativeReals, bounds=inp.batt_E_bounds)
    m.P_max = Var(domain=NonNegativeReals)

    if (inp.fixed_batt_E_kwh is None) ^ (inp.fixed_batt_P_kw is None):
        raise ValueError("Bitte entweder fixed_batt_E_kwh und fixed_batt_P_kw beide setzen oder beide auf None lassen.")

    optimize_battery_size = inp.fixed_batt_E_kwh is None and inp.fixed_batt_P_kw is None
    if not optimize_battery_size:
        m.E_max.fix(inp.fixed_batt_E_kwh)
        m.P_max.fix(inp.fixed_batt_P_kw)

    if inp.fixed_batt_P_kw is not None:
        p_big_m = float(inp.fixed_batt_P_kw)
    else:
        p_big_m = float(inp.batt_E_bounds[1]) / 2.0
    if p_big_m <= 0:
        raise ValueError("Ungültiges Big-M für Leistung. Bitte batt_E_bounds oder fixed_batt_P_kw prüfen.")

    m.sell_pv = Var(m.T, domain=NonNegativeReals)
    m.sell_batt = Var(m.T, domain=NonNegativeReals)
    m.pv_to_load = Var(m.T, domain=NonNegativeReals)
    m.batt_to_load = Var(m.T, domain=NonNegativeReals)
    m.grid_import = Var(m.T, domain=NonNegativeReals)
    m.ch = Var(m.T, domain=NonNegativeReals)
    m.dis = Var(m.T, domain=NonNegativeReals)
    m.soc = Var(m.T, domain=NonNegativeReals)
    m.pv_spill = Var(m.T, domain=NonNegativeReals)

    if inp.use_binary_charge_switch:
        m.is_charging = Var(m.T, domain=Binary)
    else:
        m.is_charging = Var(m.T, domain=UnitInterval)

    m.soc[0].fix(0.0)
    m.soc_cycle = Constraint(expr=m.soc[T - 1] == m.soc[0])

    def soc_balance(mm, t):
        if t == 0:
            return mm.soc[t] == inp.eta_ch * mm.ch[t] - mm.dis[t] / inp.eta_dis
        return mm.soc[t] == mm.soc[t - 1] + inp.eta_ch * mm.ch[t] - mm.dis[t] / inp.eta_dis

    m.soc_balance = Constraint(m.T, rule=soc_balance)
    m.pv_balance = Constraint(
        m.T,
        rule=lambda mm, t: mm.sell_pv[t] + mm.ch[t] + mm.pv_to_load[t] + mm.pv_spill[t] == pv_per_kwp[t] * inp.pv_size_kwp,
    )
    m.pv_spill_only_if_negative_price = Constraint(
        m.T,
        rule=lambda mm, t: mm.pv_spill[t] <= (pv_per_kwp[t] * inp.pv_size_kwp) * neg_price[t],
    )
    m.no_pv_feed_in_if_negative_price = Constraint(
        m.T,
        rule=lambda mm, t: mm.sell_pv[t] <= (pv_per_kwp[t] * inp.pv_size_kwp) * nonneg_price[t],
    )
    m.load_balance = Constraint(
        m.T,
        rule=lambda mm, t: mm.pv_to_load[t] + mm.batt_to_load[t] + mm.grid_import[t] == load[t],
    )
    m.dis_split = Constraint(m.T, rule=lambda mm, t: mm.dis[t] == mm.sell_batt[t] + mm.batt_to_load[t])

    m.soc_cap = Constraint(m.T, rule=lambda mm, t: mm.soc[t] <= mm.E_max)
    m.charge_cap = Constraint(m.T, rule=lambda mm, t: mm.ch[t] <= mm.P_max)
    m.discharge_cap = Constraint(m.T, rule=lambda mm, t: mm.dis[t] <= mm.P_max)
    m.charge_or_discharge_1 = Constraint(m.T, rule=lambda mm, t: mm.ch[t] <= p_big_m * mm.is_charging[t])
    m.charge_or_discharge_2 = Constraint(m.T, rule=lambda mm, t: mm.dis[t] <= p_big_m * (1 - mm.is_charging[t]))

    if optimize_battery_size:
        m.storage_2h = Constraint(expr=m.E_max == 2.0 * m.P_max)

    npv_mult = npv_multiplier(inp)

    def objective_rule(mm):
        pv_feed_in_value = sum(opp_price[t] * mm.sell_pv[t] for t in mm.T)
        batt_feed_in_value = sum(prices[t] * mm.sell_batt[t] for t in mm.T)
        battery_charging_opportunity_cost = sum(opp_price[t] * mm.ch[t] for t in mm.T)
        pv_self_consumption_value = sum(inp.pv_to_load_remuneration_eur_per_kwh * mm.pv_to_load[t] for t in mm.T)
        batt_self_consumption_value = sum(inp.batt_to_load_remuneration_eur_per_kwh * mm.batt_to_load[t] for t in mm.T)
        annual_battery_revenue = batt_feed_in_value + batt_self_consumption_value - battery_charging_opportunity_cost

        annual_gross_value = (
            pv_feed_in_value
            + batt_feed_in_value
            + pv_self_consumption_value
            + batt_self_consumption_value
            - battery_charging_opportunity_cost
        )
        storage_invest = inp.capex_batt_E_eur_per_kwh * mm.E_max
        annual_storage_opex = inp.storage_opex_pct_of_invest_per_year * storage_invest
        annual_meter_cost = inp.meter_cost_eur_per_year
        annual_pv_direct_marketing_cost = 0.0
        annual_battery_marketer_cost = inp.marketer_share_of_battery_revenue * annual_battery_revenue

        annual_net_value = (
            annual_gross_value
            - annual_storage_opex
            - annual_meter_cost
            - annual_pv_direct_marketing_cost
            - annual_battery_marketer_cost
        )
        annual_pv_only_net_baseline = annual_value_only_pv_baseline - annual_pv_direct_marketing_cost
        annual_incremental_storage_value = annual_net_value - annual_pv_only_net_baseline
        storage_cost = inp.capex_batt_E_eur_per_kwh * mm.E_max

        # NPV-Optimierung: abgezinster Mehrwert über den Horizont minus CAPEX heute
        return npv_mult * annual_incremental_storage_value - storage_cost

    m.obj = Objective(rule=objective_rule, sense=maximize)
    return m


def discounted_total_delta(annual_delta: float, inp: InputsNPV) -> float:
    return npv_multiplier(inp) * float(annual_delta)


def run_pv_size_sweep(inp: InputsNPV):
    t0_total = time.perf_counter()

    if inp.pv_size_sweep_step_kwp <= 0:
        raise ValueError("pv_size_sweep_step_kwp muss > 0 sein.")
    if inp.pv_size_sweep_min_kwp <= 0 or inp.pv_size_sweep_max_kwp <= 0:
        raise ValueError("pv_size_sweep_min_kwp und pv_size_sweep_max_kwp müssen > 0 sein.")
    if inp.pv_size_sweep_max_kwp <= inp.pv_size_sweep_min_kwp:
        raise ValueError("pv_size_sweep_max_kwp muss größer als pv_size_sweep_min_kwp sein.")

    pv_sizes = np.arange(
        inp.pv_size_sweep_min_kwp,
        inp.pv_size_sweep_max_kwp + 0.5 * inp.pv_size_sweep_step_kwp,
        inp.pv_size_sweep_step_kwp,
        dtype=float,
    )
    prices, pv_per_kwp, load = load_timeseries(inp)

    rows: list[dict] = []
    for i, pv_size in enumerate(pv_sizes, start=1):
        t0_run = time.perf_counter()
        inp_i = replace(inp, pv_size_kwp=float(pv_size))

        model = build_model_npv(prices, pv_per_kwp, load, inp_i)
        solve_model(model, inp_i)
        results = evaluate_results(model, prices, pv_per_kwp, load, inp_i)

        obj_npv = float(value(model.obj))
        rows.append(
            {
                "pv_size_kwp": float(pv_size),
                "E_opt_kwh": float(results["E_opt"]),
                "P_opt_kw": float(results["P_opt"]),
                "annual_delta_eur": float(results["annual_delta"]),
                "discounted_total_delta_npv_eur": float(discounted_total_delta(results["annual_delta"], inp_i)),
                "storage_cost_eur": float(results["storage_cost"]),
                "objective_npv_eur": obj_npv,
            }
        )
        dt_run = time.perf_counter() - t0_run
        dt_total = time.perf_counter() - t0_total
        avg_per_run = dt_total / i
        eta = avg_per_run * (len(pv_sizes) - i)
        print(
            f"[{i:03d}/{len(pv_sizes)}] PV={pv_size:,.1f} kWp | "
            f"E_opt={results['E_opt']:,.1f} kWh | P_opt={results['P_opt']:,.1f} kW | "
            f"NPV={obj_npv:,.2f} € "
            f"| Laufzeit Run={dt_run:,.1f}s | Gesamt={dt_total:,.1f}s | ETA={eta:,.1f}s"
        )

    df = pd.DataFrame(rows)
    out_dir = Path(inp.export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "npv_pv_size_sweep.csv"
    df.to_csv(csv_path, sep=";", decimal=",", index=False)

    fig = plt.figure(figsize=(12, 5.5))
    plt.plot(df["pv_size_kwp"], df["objective_npv_eur"], linewidth=2.0, marker="o", markersize=3)
    plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    i_best = int(df["objective_npv_eur"].idxmax())
    pv_best = float(df.loc[i_best, "pv_size_kwp"])
    npv_best = float(df.loc[i_best, "objective_npv_eur"])
    plt.scatter([pv_best], [npv_best], s=70, color="red", zorder=5, label=f"Maximum: {pv_best:.1f} kWp")

    # Speichergrößen je Ergebnis direkt im Diagramm anzeigen
    for _, r in df.iterrows():
        plt.annotate(
            f"E={r['E_opt_kwh']:.0f} kWh\nP={r['P_opt_kw']:.0f} kW",
            (r["pv_size_kwp"], r["objective_npv_eur"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=6,
            alpha=0.85,
        )

    plt.title("NPV über PV-Anlagengröße (E/P optimiert)")
    plt.xlabel("PV-Anlagengröße [kWp]")
    plt.ylabel("NPV [EUR]")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")

    param_text = (
        f"Parameter: CAPEX Speicher = {inp.capex_batt_E_eur_per_kwh:,.1f} EUR/kWh | "
        f"Verbrauch = {inp.annual_consumption_kwh:,.0f} kWh/a"
    )
    plt.gcf().text(0.5, 0.01, param_text, ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    plot_path = out_dir / "npv_pv_size_sweep.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print("\nSweep abgeschlossen:")
    print(f"- CSV:     {csv_path}")
    print(f"- Diagramm:{plot_path}")
    print(f"- Bestes Ergebnis: PV={pv_best:,.1f} kWp | NPV={npv_best:,.2f} €")
    print(f"- Gesamtlaufzeit: {time.perf_counter() - t0_total:,.1f} s")


def main():
    inp = InputsNPV()
    t0_main = time.perf_counter()

    if inp.run_pv_size_sweep:
        run_pv_size_sweep(inp)
        print(f"Laufzeit gesamt (Programm): {time.perf_counter() - t0_main:,.1f} s")
        return

    prices, pv_per_kwp, load = load_timeseries(inp)
    model = build_model_npv(prices, pv_per_kwp, load, inp)
    solve_model(model, inp)

    results = evaluate_results(model, prices, pv_per_kwp, load, inp)
    results["npv_discount_rate"] = inp.discount_rate
    results["npv_multiplier"] = npv_multiplier(inp)
    results["discounted_total_delta_npv"] = discounted_total_delta(results["annual_delta"], inp)
    results["objective_npv"] = value(model.obj)

    print_results(results, inp)
    print(f"NPV-Diskontsatz:                     {100.0*inp.discount_rate:,.2f} %")
    print(f"NPV-Multiplikator (degr+diskont):    {results['npv_multiplier']:,.3f}")
    print(f"Abgezinster Mehrwert (ohne CAPEX):   {results['discounted_total_delta_npv']:,.2f} €")
    print(f"NPV-Zielfunktionswert (mit CAPEX):   {results['objective_npv']:,.2f} €")

    exports = export_results_and_timeseries(results, prices, inp)
    print("\nExportierte Dateien:")
    print(f"- Zeitreihen: {exports['timeseries']}")
    print(f"- Summary:    {exports['summary']}")
    print(f"- Inputs:     {exports['inputs']}")

    from plotting_machine import run_all_plots

    run_all_plots(
        inp.export_dir,
        start_date=inp.plot_start_date,
        n_days=inp.plot_n_days,
        template_docx=inp.template_word_path,
    )
    print(f"Laufzeit gesamt (Programm): {time.perf_counter() - t0_main:,.1f} s")


if __name__ == "__main__":
    main()
