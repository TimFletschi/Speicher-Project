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
    solve_model,
)


@dataclass
class InputsNPV(BaseInputs):
    # Diskontsatz (Hurdle Rate), z. B. 6%
    discount_rate: float = 0.06
    # Baukosten Speicher (CAPEX) explizit im NPV-Tool setzbar
    capex_batt_E_eur_per_kwh: float = 245
    # Standardverbrauch für dieses NPV-Tool
    annual_consumption_kwh: float = 230000

    # PV-Größen-Sweep
    run_pv_size_sweep: bool = False
    pv_size_sweep_min_kwp: float = 100.0
    pv_size_sweep_max_kwp: float = 2000.0
    pv_size_sweep_step_kwp: float = 400.0

    # Optional: feste Speichergröße (kWh) vorgeben, damit NPV/IRR für diese Größe berechnet wird.
    # Bei Vorgabe wird P automatisch über storage_duration_hours abgeleitet.
    input_batt_E_kwh: float | None = 220


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

    fixed_e = inp.fixed_batt_E_kwh
    fixed_p = inp.fixed_batt_P_kw
    if fixed_e is not None and fixed_p is None:
        fixed_p = float(fixed_e) / float(inp.storage_duration_hours)
    elif fixed_p is not None and fixed_e is None:
        fixed_e = float(fixed_p) * float(inp.storage_duration_hours)

    optimize_battery_size = fixed_e is None and fixed_p is None
    if not optimize_battery_size:
        m.E_max.fix(float(fixed_e))
        m.P_max.fix(float(fixed_p))

    if inp.storage_duration_hours not in (2.0, 4.0):
        raise ValueError("storage_duration_hours muss 2.0 oder 4.0 sein.")

    if inp.fixed_batt_P_kw is not None:
        p_big_m = float(inp.fixed_batt_P_kw)
    else:
        p_big_m = float(inp.batt_E_bounds[1]) / float(inp.storage_duration_hours)
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
        m.storage_duration_coupling = Constraint(expr=m.E_max == float(inp.storage_duration_hours) * m.P_max)

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


def _baseline_pv_curtailment_without_storage_kwh(
    prices: np.ndarray,
    pv_per_kwp: np.ndarray,
    load: np.ndarray,
    pv_size_kwp: float,
) -> float:
    pv_gen = pv_per_kwp * pv_size_kwp
    pv_to_load = np.minimum(pv_gen, load)
    pv_surplus = np.maximum(0.0, pv_gen - pv_to_load)
    pv_curtailment = np.where(prices < 0.0, pv_surplus, 0.0)
    return float(np.sum(pv_curtailment))


def _irr_unlevered(cashflows: list[float], tol: float = 1e-9, max_iter: int = 200) -> float | None:
    """Robuste IRR-Berechnung via Bisektion."""

    def npv(rate: float) -> float:
        return float(sum(cf / ((1.0 + rate) ** t) for t, cf in enumerate(cashflows)))

    low, high = -0.999, 10.0
    f_low = npv(low)
    f_high = npv(high)

    if np.isnan(f_low) or np.isnan(f_high):
        return None
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if f_low * f_high > 0:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)


def build_requested_outputs(
    results: dict,
    model: ConcreteModel,
    prices: np.ndarray,
    pv_per_kwp: np.ndarray,
    load: np.ndarray,
    inp: InputsNPV,
) -> tuple[dict, pd.DataFrame]:
    e_opt = float(results["E_opt"])
    p_opt = float(results["P_opt"])

    batt_to_grid_kwh = float(results["annual_feed_in_batt_kwh"])
    batt_to_load_kwh = float(results["annual_batt_to_load_kwh"])
    batt_output_total_kwh = batt_to_grid_kwh + batt_to_load_kwh

    full_cycles = batt_output_total_kwh / e_opt if e_opt > 1e-9 else 0.0

    annual_load_kwh = float(np.sum(load))
    autarky_with_storage_pct = 100.0 * float(results["annual_self_consumption_kwh_model"]) / annual_load_kwh if annual_load_kwh > 0 else 0.0

    pv_curtailment_without_storage_kwh = _baseline_pv_curtailment_without_storage_kwh(prices, pv_per_kwp, load, inp.pv_size_kwp)
    pv_curtailment_with_storage_kwh = float(results["annual_curtailment_kwh"])

    yearly_deg = _yearly_degradation_factors(inp)
    storage_cost = float(results["storage_cost"])
    annual_bess_work_price = float(results["annual_battery_revenue_eur"]) * yearly_deg
    opp_price = opportunity_price_eur_per_kwh(prices, inp.pv_size_kwp)
    annual_opportunity_cost_base = float(np.sum(opp_price * np.asarray(results["ch"], dtype=float)))
    yearly_opportunity_cost = annual_opportunity_cost_base * yearly_deg
    annual_meter_cost = float(results["annual_meter_cost_eur"])
    annual_storage_opex = float(results["annual_storage_opex_eur"])

    # Excel-Logik:
    # Jahres-CF = Arbeitspreis BESS - Zähler - OPEX
    # Im ersten Jahr zusätzlich: - Investition
    yearly_surplus = annual_bess_work_price - annual_meter_cost - annual_storage_opex
    yearly_surplus = yearly_surplus.astype(float)
    yearly_surplus[0] -= storage_cost

    cashflows = [float(v) for v in yearly_surplus]
    irr = _irr_unlevered(cashflows)
    irr_pct = 100.0 * irr if irr is not None else np.nan

    years = np.arange(1, inp.horizon_years + 1, dtype=float)
    npv_eur = float(np.sum(yearly_surplus / ((1.0 + inp.discount_rate) ** years)))

    yearly_table = pd.DataFrame(
        {
            "year": np.arange(1, inp.horizon_years + 1, dtype=int),
            "opportunity_cost_eur": yearly_opportunity_cost,
            "cashflow_eur": cashflows,
        }
    )

    requested = {
        # --- BESS Output ---
        "bess_full_cycles_per_year": full_cycles,
        "bess_capacity_kwh": e_opt,
        "bess_max_power_kw": p_opt,
        "bess_energy_to_grid_kwh": batt_to_grid_kwh,
        "bess_energy_to_household_kwh": batt_to_load_kwh,
        "autarky_with_storage_pct": autarky_with_storage_pct,
        "investment_total_eur": float(results["storage_cost"]),
        "pv_curtailment_without_storage_kwh": pv_curtailment_without_storage_kwh,
        "pv_curtailment_with_storage_kwh": pv_curtailment_with_storage_kwh,
        # --- Wirtschaftlichkeit ---
        "npv_eur": npv_eur,
        "annual_surplus_over_horizon_eur": float(np.sum(yearly_surplus)),
        "annual_opportunity_cost_year1_eur": float(yearly_opportunity_cost[0]),
        "total_opportunity_cost_over_horizon_eur": float(np.sum(yearly_opportunity_cost)),
        "irr_unlevered_pct": float(irr_pct),
    }

    return requested, yearly_table


def plot_yearly_opportunity_costs(yearly_table: pd.DataFrame, out_dir: Path, show: bool = False) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4.8))
    plt.bar(yearly_table["year"], yearly_table["opportunity_cost_eur"], color="#4c78a8")
    plt.title("Jährliche Opportunitätskosten (PV-Ladung Batterie)")
    plt.xlabel("Jahr")
    plt.ylabel("EUR/Jahr")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = out_dir / "yearly_opportunity_costs.png"
    plt.savefig(plot_path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return plot_path


def _round_numeric_scalars_for_export(results: dict, digits: int = 2) -> dict:
    out: dict = {}
    for k, v in results.items():
        if np.isscalar(v) and isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = round(float(v), digits)
        else:
            out[k] = v
    return out


def print_requested_outputs(outputs: dict, inp: InputsNPV):
    print("\n-----------------------------------")
    print("BESS OUTPUT")
    print("-----------------------------------")
    print(f"Vollladezyklen pro Jahr:                    {outputs['bess_full_cycles_per_year']:,.2f}")
    print(f"Speicherkapazität:                          {outputs['bess_capacity_kwh']:,.2f} kWh")
    print(f"Max. Leistung:                              {outputs['bess_max_power_kw']:,.2f} kW")
    print(f"Strommenge BATT -> Netz:                    {outputs['bess_energy_to_grid_kwh']:,.2f} kWh/a")
    print(f"Strommenge BATT -> Haushalt:                {outputs['bess_energy_to_household_kwh']:,.2f} kWh/a")
    print(f"Autarkie Vor-Ort-Verbrauch (mit Speicher):  {outputs['autarky_with_storage_pct']:,.2f} %")
    print(f"Investitionskosten gesamt:                  {outputs['investment_total_eur']:,.2f} €")
    print(f"PV-Abregelung ohne Speicher:                {outputs['pv_curtailment_without_storage_kwh']:,.2f} kWh/a")
    print(f"PV-Abregelung mit Speicher:                 {outputs['pv_curtailment_with_storage_kwh']:,.2f} kWh/a")

    print("\n-----------------------------------")
    print("WIRTSCHAFTLICHKEIT")
    print("-----------------------------------")
    print(f"NPV:                                        {outputs['npv_eur']:,.2f} €")
    print(f"Jahresüberschuss über {inp.horizon_years} Jahre:        {outputs['annual_surplus_over_horizon_eur']:,.2f} €")
    irr_val = outputs["irr_unlevered_pct"]
    if np.isnan(irr_val):
        print("IRR unlevered:                              n/a (keine eindeutige IRR-Nullstelle)")
    else:
        print(f"IRR unlevered:                              {irr_val:,.2f} %")


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
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(2)
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

    if inp.input_batt_E_kwh is not None:
        if inp.input_batt_E_kwh <= 0:
            raise ValueError("input_batt_E_kwh muss > 0 sein.")
        inp = replace(
            inp,
            fixed_batt_E_kwh=float(inp.input_batt_E_kwh),
            fixed_batt_P_kw=float(inp.input_batt_E_kwh) / float(inp.storage_duration_hours),
        )

    t0_main = time.perf_counter()

    if inp.run_pv_size_sweep:
        run_pv_size_sweep(inp)
        print(f"Laufzeit gesamt (Programm): {time.perf_counter() - t0_main:,.1f} s")
        return

    prices, pv_per_kwp, load = load_timeseries(inp)
    model = build_model_npv(prices, pv_per_kwp, load, inp)
    solve_model(model, inp)

    results = evaluate_results(model, prices, pv_per_kwp, load, inp)
    requested_outputs, yearly_table = build_requested_outputs(results, model, prices, pv_per_kwp, load, inp)

    results["npv_discount_rate"] = inp.discount_rate
    results["npv_multiplier"] = npv_multiplier(inp)
    results["discounted_total_delta_npv"] = discounted_total_delta(results["annual_delta"], inp)
    results["objective_npv"] = value(model.obj)
    results.update(requested_outputs)

    print_requested_outputs(requested_outputs, inp)

    results_export = _round_numeric_scalars_for_export(results, digits=2)
    exports = export_results_and_timeseries(results_export, prices, inp)

    out_dir = Path(inp.export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cashflow_path = out_dir / "yearly_unlevered_cashflow.csv"
    yearly_table_export = yearly_table.copy()
    num_cols = yearly_table_export.select_dtypes(include=[np.number]).columns
    yearly_table_export[num_cols] = yearly_table_export[num_cols].round(2)
    yearly_table_export.to_csv(cashflow_path, sep=";", decimal=",", index=False)
    opp_plot_path = plot_yearly_opportunity_costs(yearly_table, out_dir, show=inp.plot_show_interactive)

    print("\nExportierte Dateien:")
    print(f"- Summary:    {exports['summary']}")
    print(f"- Inputs:     {exports['inputs']}")
    print(f"- Cashflows:  {cashflow_path}")
    print(f"- Plot Opp.-Kosten: {opp_plot_path}")
    print("- Zeitreihen: exportiert (ohne Terminal-Ausgabe)")

    from plotting_machine import run_all_plots

    run_all_plots(
        inp.export_dir,
        start_date=inp.plot_start_date,
        n_days=inp.plot_n_days,
        template_docx=inp.template_word_path,
        print_tables=False,
        include_project_data_table=False,
    )
    print(f"Laufzeit gesamt (Programm): {time.perf_counter() - t0_main:,.1f} s")


if __name__ == "__main__":
    main()
