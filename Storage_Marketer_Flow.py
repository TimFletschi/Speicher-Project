from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Storage_Test as st


@dataclass
class InputsMarketerFlow:
    _BASE_DIR: Path = Path(__file__).resolve().parent

    # Daten- und Anlagenparameter
    pv_size_kwp: float = 312.0
    annual_consumption_kwh: float = 230000.0
    pv_specific_yield_kwh_per_kwp_per_year: float | None = 919.0

    # AC- und Speicherdimensionierung
    pv_ac_share_of_kwp: float = 0.90
    bess_share_of_pv_ac: float = 0.80
    storage_duration_hours: float = 2.0

    # Vermarkter-Parameter (als Eingabe)
    marketer_remuneration_eur_per_kw_year: float = 95.0
    marketer_share_of_revenue: float = 0.08

    # Randbedingungen
    self_consumption_tolerance_pct: float = 2.0
    service_level_pct: float = 90.0

    # Plot/Export
    plot_start_date: str = "2024-07-01"
    plot_n_days: int = 14
    export_dir: Path = _BASE_DIR / "exports"


def _build_base_inputs(inp: InputsMarketerFlow) -> st.Inputs:
    base = st.Inputs()
    base.pv_size_kwp = inp.pv_size_kwp
    base.annual_consumption_kwh = inp.annual_consumption_kwh
    base.pv_specific_yield_kwh_per_kwp_per_year = inp.pv_specific_yield_kwh_per_kwp_per_year
    return base


def _simulate_priority_flows(
    pv_gen: np.ndarray,
    load: np.ndarray,
    contract_power_kw: float,
) -> dict[str, np.ndarray | float]:
    contract = max(0.0, float(contract_power_kw))
    batt_charge = np.minimum(contract, pv_gen)  # Batterie hat Vorrang (reservierte PV-Leistung)

    pv_after_batt = np.maximum(0.0, pv_gen - batt_charge)
    pv_to_load = np.minimum(pv_after_batt, load)
    pv_to_grid = np.maximum(0.0, pv_after_batt - pv_to_load)

    baseline_pv_to_load = np.minimum(pv_gen, load)
    pv_active_mask = pv_gen > 1e-9
    if np.any(pv_active_mask):
        availability_pct = 100.0 * float(np.mean(pv_gen[pv_active_mask] >= contract - 1e-12))
    else:
        availability_pct = 0.0

    return {
        "contract_power_kw": contract,
        "batt_charge": batt_charge,
        "pv_to_load": pv_to_load,
        "pv_to_grid": pv_to_grid,
        "baseline_pv_to_load": baseline_pv_to_load,
        "pv_active_hours": float(np.sum(pv_active_mask)),
        "availability_pct": availability_pct,
        "annual_pv_to_load_kwh": float(np.sum(pv_to_load)),
        "annual_baseline_pv_to_load_kwh": float(np.sum(baseline_pv_to_load)),
        "annual_batt_charge_kwh": float(np.sum(batt_charge)),
        "max_batt_charge_kw": float(np.max(batt_charge)) if len(batt_charge) else 0.0,
    }


def _find_max_scale_for_tolerance(
    pv_gen: np.ndarray,
    load: np.ndarray,
    contract_power_cap_kw: float,
    tol_pct: float,
    service_level_pct: float,
) -> tuple[float, dict[str, np.ndarray | float]]:
    baseline = float(np.sum(np.minimum(pv_gen, load)))
    tol = max(0.0, tol_pct) / 100.0
    min_allowed = baseline * (1.0 - tol)
    target_service = float(np.clip(service_level_pct, 0.0, 100.0))

    def ok(contract_kw: float) -> tuple[bool, dict[str, np.ndarray | float]]:
        res = _simulate_priority_flows(pv_gen, load, contract_kw)
        pv2load = float(res["annual_pv_to_load_kwh"])
        sl_ok = float(res["availability_pct"]) >= target_service
        return (pv2load >= min_allowed) and sl_ok, res

    lo, hi = 0.0, max(0.0, float(contract_power_cap_kw))
    best = _simulate_priority_flows(pv_gen, load, 0.0)

    for _ in range(35):
        mid = 0.5 * (lo + hi)
        cond, res = ok(mid)
        if cond:
            lo = mid
            best = res
        else:
            hi = mid

    return lo, best


def _plot_2_weeks(ts: pd.DataFrame, start_date: str, n_days: int, target_path: Path):
    start = pd.Timestamp(start_date)
    end = start + pd.Timedelta(days=n_days)
    w = ts[(ts["timestamp"] >= start) & (ts["timestamp"] < end)].copy()
    if w.empty:
        w = ts.copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(w["timestamp"], w["pv_gen_kwh"], label="PV-Erzeugung [kWh/h]", linewidth=1.5)
    ax.plot(w["timestamp"], w["load_kwh"], label="Last [kWh/h]", linewidth=1.2)
    ax.plot(w["timestamp"], w["baseline_pv_to_load_kwh"], label="PV->Last Basis [kWh/h]", linewidth=1.2)
    ax.plot(w["timestamp"], w["pv_to_load_kwh"], label="PV->Last mit Vermarkter [kWh/h]", linewidth=1.4)
    ax.plot(w["timestamp"], w["batt_charge_kwh"], label="PV->Batterie (reserviert) [kWh/h]", linewidth=1.4)
    ax.plot(w["timestamp"], w["contract_power_kw"], label="Vorgehaltene PV-Leistung [kW]", linewidth=1.1, linestyle="--")

    ax.set_title("Energieflüsse (2 Wochen) mit fester reservierter PV-Leistung")
    ax.set_xlabel("Zeit")
    ax.set_ylabel("kWh/h")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    fig.tight_layout()
    fig.savefig(target_path, dpi=160)
    plt.close(fig)


def main():
    inp = InputsMarketerFlow()
    out_dir = Path(inp.export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_inputs = _build_base_inputs(inp)
    prices, pv_per_kwp, load = st.load_timeseries(base_inputs)

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=len(prices), freq="1h")
    pv_gen = pv_per_kwp * inp.pv_size_kwp

    pv_ac_kw = inp.pv_size_kwp * inp.pv_ac_share_of_kwp
    bess_power_kw_max = pv_ac_kw * inp.bess_share_of_pv_ac
    bess_energy_kwh_max = bess_power_kw_max * inp.storage_duration_hours

    contract_star_kw, sim = _find_max_scale_for_tolerance(
        pv_gen=pv_gen,
        load=load,
        contract_power_cap_kw=bess_power_kw_max,
        tol_pct=inp.self_consumption_tolerance_pct,
        service_level_pct=inp.service_level_pct,
    )

    annual_baseline_pv_to_load = float(sim["annual_baseline_pv_to_load_kwh"])
    annual_pv_to_load_with_marketer = float(sim["annual_pv_to_load_kwh"])
    delta_pct = (
        100.0 * (annual_pv_to_load_with_marketer - annual_baseline_pv_to_load) / annual_baseline_pv_to_load
        if annual_baseline_pv_to_load > 0
        else 0.0
    )

    effective_bess_power_kw = contract_star_kw
    effective_bess_energy_kwh = effective_bess_power_kw * inp.storage_duration_hours

    gross_marketer_revenue = inp.marketer_remuneration_eur_per_kw_year * effective_bess_power_kw
    marketer_fee = inp.marketer_share_of_revenue * gross_marketer_revenue
    net_marketer_revenue = gross_marketer_revenue - marketer_fee

    ts = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pv_gen_kwh": pv_gen,
            "load_kwh": load,
            "baseline_pv_to_load_kwh": sim["baseline_pv_to_load"],
            "pv_to_load_kwh": sim["pv_to_load"],
            "pv_to_grid_kwh": sim["pv_to_grid"],
            "batt_charge_kwh": sim["batt_charge"],
            "contract_power_kw": np.full(len(timestamps), contract_star_kw),
        }
    )

    summary = pd.DataFrame(
        [
            ("pv_size_kwp", inp.pv_size_kwp),
            ("pv_ac_share_of_kwp", inp.pv_ac_share_of_kwp),
            ("pv_ac_kw", pv_ac_kw),
            ("bess_share_of_pv_ac", inp.bess_share_of_pv_ac),
            ("bess_power_kw_max", bess_power_kw_max),
            ("bess_energy_kwh_max", bess_energy_kwh_max),
            ("self_consumption_tolerance_pct", inp.self_consumption_tolerance_pct),
            ("service_level_pct", inp.service_level_pct),
            ("contract_power_kw", contract_star_kw),
            ("effective_bess_power_kw", effective_bess_power_kw),
            ("effective_bess_energy_kwh", effective_bess_energy_kwh),
            ("annual_baseline_pv_to_load_kwh", annual_baseline_pv_to_load),
            ("annual_pv_to_load_with_marketer_kwh", annual_pv_to_load_with_marketer),
            ("delta_pv_to_load_pct", delta_pct),
            ("pv_active_hours", sim["pv_active_hours"]),
            ("availability_pct_pv_active_hours", sim["availability_pct"]),
            ("annual_batt_charge_kwh", sim["annual_batt_charge_kwh"]),
            ("max_batt_charge_kw", sim["max_batt_charge_kw"]),
            ("marketer_remuneration_eur_per_kw_year", inp.marketer_remuneration_eur_per_kw_year),
            ("marketer_share_of_revenue", inp.marketer_share_of_revenue),
            ("gross_marketer_revenue_eur_per_year", gross_marketer_revenue),
            ("marketer_fee_eur_per_year", marketer_fee),
            ("net_marketer_revenue_eur_per_year", net_marketer_revenue),
        ],
        columns=["metric", "value"],
    )

    ts_path = out_dir / "marketer_timeseries_export.csv"
    summary_path = out_dir / "marketer_summary_export.csv"
    plot_path = out_dir / "marketer_energy_flows_2weeks.png"

    ts.to_csv(ts_path, sep=";", decimal=",", index=False)
    summary.to_csv(summary_path, sep=";", decimal=",", index=False)
    _plot_2_weeks(ts, inp.plot_start_date, inp.plot_n_days, plot_path)

    print("\n-----------------------------------")
    print("VERMARKTER-ENERGIEFLUSS MODELL")
    print("-----------------------------------")
    print(f"PV kWp:                               {inp.pv_size_kwp:,.2f}")
    print(f"PV AC Leistung:                        {pv_ac_kw:,.2f} kW")
    print(f"Max. BESS-Leistung (80% von AC):       {bess_power_kw_max:,.2f} kW")
    print(f"Max. BESS-Energie ({inp.storage_duration_hours:,.1f}h):       {bess_energy_kwh_max:,.2f} kWh")
    print(f"Toleranz Eigenverbrauch:               ±{inp.self_consumption_tolerance_pct:,.2f} %")
    print(f"Service-Level Vorgabe (PV-aktiv):      {inp.service_level_pct:,.2f} %")
    print(f"Reservierte PV-Leistung (max.):        {contract_star_kw:,.2f} kW")
    print(f"PV-aktive Stunden:                     {float(sim['pv_active_hours']):,.0f} h/a")
    print(f"Erreichte Verfügbarkeit (PV-aktiv):    {float(sim['availability_pct']):,.2f} %")
    print(f"Effektive BESS-Leistung:               {effective_bess_power_kw:,.2f} kW")
    print(f"Effektive BESS-Energie:                {effective_bess_energy_kwh:,.2f} kWh")
    print(f"PV->Last Basis:                        {annual_baseline_pv_to_load:,.2f} kWh/a")
    print(f"PV->Last mit Vermarkter:               {annual_pv_to_load_with_marketer:,.2f} kWh/a")
    print(f"Abweichung PV->Last:                   {delta_pct:,.2f} %")
    print(f"Batterieladung aus PV:                 {float(sim['annual_batt_charge_kwh']):,.2f} kWh/a")
    print("-----------------------------------")
    print(f"Vermarkter-Vergütung:                  {inp.marketer_remuneration_eur_per_kw_year:,.2f} €/kW*a")
    print(f"Vermarkter-Anteil:                     {100*inp.marketer_share_of_revenue:,.2f} %")
    print(f"Brutto-Erlös:                          {gross_marketer_revenue:,.2f} €/a")
    print(f"Vermarkter-Gebühr:                     {marketer_fee:,.2f} €/a")
    print(f"Netto-Erlös:                           {net_marketer_revenue:,.2f} €/a")
    print("-----------------------------------")
    print(f"Export Zeitreihe:                      {ts_path}")
    print(f"Export Summary:                        {summary_path}")
    print(f"Plot (2 Wochen):                       {plot_path}")


if __name__ == "__main__":
    main()
