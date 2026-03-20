from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    RangeSet,
    Var,
    maximize,
    value,
)
from pyomo.opt import SolverFactory


# =============================================================
# 1) EINLESEN
# =============================================================
def read_15min_series_values(path_csv: Path) -> np.ndarray:
    """Liest robust eine Zeitreihe aus CSV (erste zwei Spalten),
    ignoriert Meta-Zeilen und nutzt nur Zeilen mit Uhrzeit.
    """
    if not path_csv.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {path_csv}")

    df = pd.read_csv(
        path_csv,
        sep=";",
        decimal=",",
        engine="python",
        header=None,
        usecols=[0, 1],
        on_bad_lines="skip",
    )

    col_time = df.iloc[:, 0].astype(str).str.strip().str.replace('"', "", regex=False)
    col_val = pd.to_numeric(df.iloc[:, 1], errors="coerce")

    # Datenzeilen enthalten in der Zeitspalte normalerweise ':'
    mask = col_time.str.contains(":", na=False) & col_val.notna()
    values = col_val[mask].to_numpy(dtype=float)

    if len(values) == 0:
        raise ValueError(f"Keine gültigen Zeitreihenwerte in {path_csv} gefunden.")

    return values


def load_bdew_g0_profile_15min(path_bdew: Path, n_steps: int, target_annual_kwh: float, reference_annual_kwh: float) -> np.ndarray:
    values_arr = read_15min_series_values(path_bdew)
    if len(values_arr) < n_steps:
        raise ValueError(
            f"BDEW CSV hat zu wenige 15-min Werte: {len(values_arr)} statt mindestens {n_steps}."
        )
    if len(values_arr) > n_steps:
        values_arr = values_arr[:n_steps]

    if reference_annual_kwh <= 0:
        raise ValueError("g0_reference_annual_kwh muss > 0 sein.")
    if target_annual_kwh < 0:
        raise ValueError("annual_consumption_kwh muss >= 0 sein.")

    # Gewünschte Logik:
    # 1) Referenzprofil (z.B. 400.000 kWh/a) auf 1 kWh normieren
    # 2) mit Eingabeverbrauch hochskalieren
    load_norm = values_arr / reference_annual_kwh
    return load_norm * target_annual_kwh


def load_pv_per_kwp_hourly(
    path_pv: Path,
    n_hours: int,
    pv_reference_kwp: float,
    pv_specific_yield_kwh_per_kwp_per_year: float | None = None,
) -> np.ndarray:
    if pv_reference_kwp <= 0:
        raise ValueError("pv_reference_kwp muss > 0 sein.")

    pv_values = read_15min_series_values(path_pv)

    # Unterstützt stündlich ODER 15-min direkt (bei 15-min wird auf Stunden aufsummiert)
    if len(pv_values) == n_hours:
        pv_per_kwp_hour = pv_values / pv_reference_kwp
    elif len(pv_values) >= 4 * n_hours:
        pv_15 = pv_values[: 4 * n_hours]
        pv_per_kwp_hour = pv_15.reshape(n_hours, 4).sum(axis=1) / pv_reference_kwp
    else:
        raise ValueError(
            f"PV CSV hat {len(pv_values)} Werte. Erlaubt sind {n_hours} (stündlich) oder mindestens {4*n_hours} (15-min)."
        )

    # Optional: standortspezifischen Jahresertrag (kWh/kWp) vorgeben.
    # Das Profil wird nur skaliert, die zeitliche Form bleibt gleich.
    if pv_specific_yield_kwh_per_kwp_per_year is not None:
        if pv_specific_yield_kwh_per_kwp_per_year <= 0:
            raise ValueError("pv_specific_yield_kwh_per_kwp_per_year muss > 0 sein.")
        current_specific_yield = float(np.sum(pv_per_kwp_hour))
        if current_specific_yield <= 0:
            raise ValueError("PV-Profil hat keinen positiven Jahresertrag und kann nicht skaliert werden.")
        scale = pv_specific_yield_kwh_per_kwp_per_year / current_specific_yield
        pv_per_kwp_hour = pv_per_kwp_hour * scale

    return pv_per_kwp_hour


def load_timeseries(inp: "Inputs") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_price = pd.read_csv(inp.path_prices, sep=";", decimal=",")
    prices_hour = pd.to_numeric(df_price["price"], errors="coerce").fillna(0.0).values
    if inp.price_in_ct_per_kwh:
        prices_hour = prices_hour / 100.0

    pv_per_kwp_hour = load_pv_per_kwp_hourly(
        inp.path_pv,
        len(prices_hour),
        inp.pv_reference_kwp,
        inp.pv_specific_yield_kwh_per_kwp_per_year,
    )
    load_hour = load_bdew_g0_profile_15min(
        inp.path_bdew_g0,
        len(prices_hour),
        inp.annual_consumption_kwh,
        inp.g0_reference_annual_kwh,
    )
    return prices_hour, pv_per_kwp_hour, load_hour


# =============================================================
# 2) EINGABEN
# =============================================================
@dataclass
class Inputs:
    path_prices: Path = Path(r"C:\Users\TimFletschinger\OneDrive - empact GmbH\Desktop\04 Simulation\Spotmarktpreis.csv")
    path_pv: Path = Path(r"C:\Users\TimFletschinger\OneDrive - empact GmbH\Desktop\04 Simulation\PV-Daten_400kwp_stuendlich.csv")
    path_bdew_g0: Path = Path(r"C:\Users\TimFletschinger\OneDrive - empact GmbH\Desktop\04 Simulation\G0Verbrauch_400.000kwh_stuendlich.csv")

    # PV-Größe in kWp eingeben
    pv_size_kwp: float = 312
    # Größe der Anlage des Beispiel Erzeugungsprofils (z.B. 400 kWp), um die PV-Erzeugung pro kWp zu normieren.
    pv_reference_kwp: float = 400.0
    # Optional: standortspezifischer Jahresertrag zur Skalierung des PV-Profils.
    # Beispiel: 950.0 oder 1100.0 kWh/kWp*a. None = Wert aus CSV unverändert nutzen.
    pv_specific_yield_kwh_per_kwp_per_year: float | None = None

    # Verbrauch eingeben
    annual_consumption_kwh: float = 232000
    #Größe des Jahresverbrauchs des Beispiel Lastprofils (z.B. 400.000 kWh/a), um das Lastprofil zu normieren und auf den gewünschten Verbrauch hochzuskalieren.
    g0_reference_annual_kwh: float = 400000.0

    # Speicherparameter
    # Effizienzen (0-1) für Ladung und Entladung
    eta_ch: float = 0.95
    eta_dis: float = 0.95

    horizon_years: int = 15
    price_in_ct_per_kwh: bool = True
    # Getrennte Vergütung für Eigenverbrauch, in €/kWh
    pv_to_load_remuneration_eur_per_kwh: float = 0.14
    batt_to_load_remuneration_eur_per_kwh: float = 0.14
    capex_batt_E_eur_per_kwh: float = 230.0
    # Laufende Kosten (veränderbar)
    storage_opex_pct_of_invest_per_year: float = 0.01
    meter_cost_eur_per_year: float = 300.0
    marketer_share_of_battery_revenue: float = 0.10
    pv_direct_marketing_cost_eur_per_month: float = 60.0

    batt_E_bounds: tuple[float, float] = (0.0, 120000.0)
    fixed_batt_E_kwh: float | None = 450
    fixed_batt_P_kw: float | None = 220

    cbc_executable: str = r"C:\Users\TimFletschinger\Downloads\cbc\bin\cbc.exe"


# =============================================================
# 3) MODELLAUFBAU
# =============================================================
def build_model(prices: np.ndarray, pv_per_kwp: np.ndarray, load: np.ndarray, inp: Inputs) -> ConcreteModel:
    T = len(prices)
    m = ConcreteModel()
    m.T = RangeSet(0, T - 1)
    neg_price = (prices < 0.0).astype(float)
    nonneg_price = (prices >= 0.0).astype(float)

    pv_gen = pv_per_kwp * inp.pv_size_kwp
    pv_direct_to_load_baseline = np.minimum(pv_gen, load)
    pv_surplus_baseline = np.maximum(0.0, pv_gen - pv_direct_to_load_baseline)
    pv_direct_to_grid_baseline = np.where(prices >= 0.0, pv_surplus_baseline, 0.0)
    annual_value_only_pv_baseline = float(
        np.sum(prices * pv_direct_to_grid_baseline)
        + inp.pv_to_load_remuneration_eur_per_kwh * np.sum(pv_direct_to_load_baseline)
    )

    m.E_max = Var(domain=NonNegativeReals, bounds=inp.batt_E_bounds)
    m.P_max = Var(domain=NonNegativeReals)

    # Optional: vorhandenes Speichersystem vorgeben
    if (inp.fixed_batt_E_kwh is None) ^ (inp.fixed_batt_P_kw is None):
        raise ValueError("Bitte entweder fixed_batt_E_kwh und fixed_batt_P_kw beide setzen oder beide auf None lassen.")

    optimize_battery_size = inp.fixed_batt_E_kwh is None and inp.fixed_batt_P_kw is None
    if not optimize_battery_size:
        m.E_max.fix(inp.fixed_batt_E_kwh)
        m.P_max.fix(inp.fixed_batt_P_kw)

    m.sell_pv = Var(m.T, domain=NonNegativeReals)
    m.sell_batt = Var(m.T, domain=NonNegativeReals)
    m.pv_to_load = Var(m.T, domain=NonNegativeReals)
    m.batt_to_load = Var(m.T, domain=NonNegativeReals)
    m.grid_import = Var(m.T, domain=NonNegativeReals)
    m.ch = Var(m.T, domain=NonNegativeReals)
    m.dis = Var(m.T, domain=NonNegativeReals)
    m.soc = Var(m.T, domain=NonNegativeReals)
    m.pv_spill = Var(m.T, domain=NonNegativeReals)
    m.is_charging = Var(m.T, domain=Binary)

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
    m.charge_or_discharge_1 = Constraint(m.T, rule=lambda mm, t: mm.ch[t] <= mm.P_max * mm.is_charging[t])
    m.charge_or_discharge_2 = Constraint(m.T, rule=lambda mm, t: mm.dis[t] <= mm.P_max * (1 - mm.is_charging[t]))

    # 2h Speicher nur im Optimierungsmodus
    if optimize_battery_size:
        m.storage_2h = Constraint(expr=m.E_max == 2.0 * m.P_max)

    def objective_rule(mm):
        pv_feed_in_value = sum(prices[t] * mm.sell_pv[t] for t in mm.T)
        batt_feed_in_value = sum(prices[t] * mm.sell_batt[t] for t in mm.T)
        battery_charging_opportunity_cost = sum(prices[t] * mm.ch[t] for t in mm.T)
        pv_self_consumption_value = sum(
            inp.pv_to_load_remuneration_eur_per_kwh * mm.pv_to_load[t]
            for t in mm.T
        )
        batt_self_consumption_value = sum(
            inp.batt_to_load_remuneration_eur_per_kwh * mm.batt_to_load[t]
            for t in mm.T
        )
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
        annual_pv_direct_marketing_cost = 12.0 * inp.pv_direct_marketing_cost_eur_per_month
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
        total_value = inp.horizon_years * annual_incremental_storage_value
        storage_cost = inp.capex_batt_E_eur_per_kwh * mm.E_max
        return total_value - storage_cost

    m.obj = Objective(rule=objective_rule, sense=maximize)
    return m


# =============================================================
# 4) AUSGABE DER ERGEBNISSE
# =============================================================
def solve_model(model: ConcreteModel, inp: Inputs):
    solver = SolverFactory("cbc", executable=inp.cbc_executable)
    return solver.solve(model, tee=True)


def evaluate_results(model: ConcreteModel, prices: np.ndarray, pv_per_kwp: np.ndarray, load: np.ndarray, inp: Inputs) -> dict:
    T = len(prices)
    E_opt = value(model.E_max)
    P_opt = value(model.P_max)

    pv_gen = pv_per_kwp * inp.pv_size_kwp
    pv_direct_to_load = np.minimum(pv_gen, load)
    pv_surplus = np.maximum(0.0, pv_gen - pv_direct_to_load)
    pv_direct_to_grid = np.where(prices >= 0.0, pv_surplus, 0.0)

    annual_value_only_pv = float(
            np.sum(prices * pv_direct_to_grid)
            + inp.pv_to_load_remuneration_eur_per_kwh * np.sum(pv_direct_to_load)
    )

    annual_value_with_storage = float(
        sum(
            prices[t] * (value(model.sell_pv[t]) + value(model.sell_batt[t]) - value(model.ch[t]))
            + inp.pv_to_load_remuneration_eur_per_kwh * value(model.pv_to_load[t])
            + inp.batt_to_load_remuneration_eur_per_kwh * value(model.batt_to_load[t])
            for t in range(T)
        )
    )

    annual_battery_revenue_eur = float(
        sum(
            prices[t] * (value(model.sell_batt[t]) - value(model.ch[t]))
            + inp.batt_to_load_remuneration_eur_per_kwh * value(model.batt_to_load[t])
            for t in range(T)
        )
    )

    storage_cost = inp.capex_batt_E_eur_per_kwh * E_opt
    annual_storage_opex_eur = inp.storage_opex_pct_of_invest_per_year * storage_cost
    annual_meter_cost_eur = inp.meter_cost_eur_per_year
    annual_pv_direct_marketing_cost_eur = 12.0 * inp.pv_direct_marketing_cost_eur_per_month
    annual_battery_marketer_cost_eur = inp.marketer_share_of_battery_revenue * annual_battery_revenue_eur
    annual_fixed_costs_with_storage_eur = (
        annual_storage_opex_eur
        + annual_meter_cost_eur
        + annual_pv_direct_marketing_cost_eur
    )
    annual_value_only_pv_net = annual_value_only_pv - annual_pv_direct_marketing_cost_eur
    annual_value_with_storage_after_fixed_costs = annual_value_with_storage - annual_fixed_costs_with_storage_eur
    annual_delta_before_fixed_costs = annual_value_with_storage - annual_value_only_pv
    annual_delta_after_fixed_costs = annual_value_with_storage_after_fixed_costs - annual_value_only_pv_net

    annual_total_recurring_costs_eur = (
        annual_fixed_costs_with_storage_eur
        + annual_battery_marketer_cost_eur
    )

    annual_net_value_with_storage = annual_value_with_storage - annual_total_recurring_costs_eur
    annual_delta = annual_net_value_with_storage - annual_value_only_pv_net
    total_delta = inp.horizon_years * annual_delta

    sell_pv = np.array([value(model.sell_pv[t]) for t in range(T)])
    sell_batt = np.array([value(model.sell_batt[t]) for t in range(T)])
    pv_to_load = np.array([value(model.pv_to_load[t]) for t in range(T)])
    batt_to_load = np.array([value(model.batt_to_load[t]) for t in range(T)])
    grid_import = np.array([value(model.grid_import[t]) for t in range(T)])
    ch = np.array([value(model.ch[t]) for t in range(T)])
    dis = np.array([value(model.dis[t]) for t in range(T)])
    soc = np.array([value(model.soc[t]) for t in range(T)])
    pv_spill = np.array([value(model.pv_spill[t]) for t in range(T)])
    curtailment = pv_spill

    annual_self_consumption_kwh_model = float(np.sum(pv_to_load + batt_to_load))
    annual_self_consumption_kwh_no_storage = float(np.sum(pv_direct_to_load))
    annual_feed_in_pv_kwh = float(np.sum(sell_pv))
    annual_feed_in_batt_kwh = float(np.sum(sell_batt))
    annual_feed_in_total_kwh = annual_feed_in_pv_kwh + annual_feed_in_batt_kwh
    annual_pv_to_load_kwh = float(np.sum(pv_to_load))
    annual_batt_to_load_kwh = float(np.sum(batt_to_load))
    annual_curtailment_kwh = float(np.sum(curtailment))
    annual_pv_to_load_remuneration_eur = inp.pv_to_load_remuneration_eur_per_kwh * annual_pv_to_load_kwh
    annual_batt_to_load_remuneration_eur = inp.batt_to_load_remuneration_eur_per_kwh * annual_batt_to_load_kwh

    max_soc_kwh = float(np.max(soc))
    max_charge_kwh_per_hour = float(np.max(ch))
    max_discharge_kwh_per_hour = float(np.max(dis))
    # 1h Energiemenge -> kW
    max_charge_kw = max_charge_kwh_per_hour
    max_discharge_kw = max_discharge_kwh_per_hour

    return {
        "E_opt": E_opt,
        "P_opt": P_opt,
        "annual_value_only_pv": annual_value_only_pv,
        "annual_value_only_pv_net": annual_value_only_pv_net,
        "annual_value_with_storage": annual_value_with_storage,
        "annual_net_value_with_storage": annual_net_value_with_storage,
        "annual_delta": annual_delta,
        "total_delta": total_delta,
        "storage_cost": storage_cost,
        "annual_storage_opex_eur": annual_storage_opex_eur,
        "annual_meter_cost_eur": annual_meter_cost_eur,
        "annual_pv_direct_marketing_cost_eur": annual_pv_direct_marketing_cost_eur,
        "annual_battery_revenue_eur": annual_battery_revenue_eur,
        "annual_battery_marketer_cost_eur": annual_battery_marketer_cost_eur,
        "annual_fixed_costs_with_storage_eur": annual_fixed_costs_with_storage_eur,
        "annual_value_with_storage_after_fixed_costs": annual_value_with_storage_after_fixed_costs,
        "annual_delta_before_fixed_costs": annual_delta_before_fixed_costs,
        "annual_delta_after_fixed_costs": annual_delta_after_fixed_costs,
        "annual_total_recurring_costs_eur": annual_total_recurring_costs_eur,
        "pays": total_delta > storage_cost,
        "objective": value(model.obj),
        "annual_self_consumption_kwh_model": annual_self_consumption_kwh_model,
        "annual_self_consumption_kwh_no_storage": annual_self_consumption_kwh_no_storage,
        "annual_feed_in_pv_kwh": annual_feed_in_pv_kwh,
        "annual_feed_in_batt_kwh": annual_feed_in_batt_kwh,
        "annual_feed_in_total_kwh": annual_feed_in_total_kwh,
        "annual_pv_to_load_kwh": annual_pv_to_load_kwh,
        "annual_batt_to_load_kwh": annual_batt_to_load_kwh,
        "annual_curtailment_kwh": annual_curtailment_kwh,
        "annual_pv_to_load_remuneration_eur": annual_pv_to_load_remuneration_eur,
        "annual_batt_to_load_remuneration_eur": annual_batt_to_load_remuneration_eur,
        "max_soc_kwh": max_soc_kwh,
        "max_charge_kwh_per_hour": max_charge_kwh_per_hour,
        "max_discharge_kwh_per_hour": max_discharge_kwh_per_hour,
        "max_charge_kw": max_charge_kw,
        "max_discharge_kw": max_discharge_kw,
        "pv_gen": pv_gen,
        "load": load,
        "sell_pv": sell_pv,
        "sell_batt": sell_batt,
        "pv_to_load": pv_to_load,
        "batt_to_load": batt_to_load,
        "grid_import": grid_import,
        "ch": ch,
        "dis": dis,
        "curtailment": curtailment,
    }


def print_results(results: dict, inp: Inputs):
    diff_cost_vs_benefit = results["storage_cost"] - results["total_delta"]

    print("\n-----------------------------------")
    print("ERGEBNISSE")
    if inp.fixed_batt_E_kwh is None and inp.fixed_batt_P_kw is None:
        print("Speichermodus:                        Optimierung (E und P)")
    else:
        print("Speichermodus:                        Vorgabe (E und P fix)")
    print(f"PV-Größe (Vorgabe):                   {inp.pv_size_kwp:,.2f} kWp")
    print(f"Jahresverbrauch (Vorgabe):            {inp.annual_consumption_kwh:,.0f} kWh")
    print(f"Optimale Speicherkapazität E:         {results['E_opt']:,.2f} kWh")
    print(f"Optimale Speicherleistung P:          {results['P_opt']:,.2f} kW")
    print("-----------------------------------")
    print(f"Jahreswert nur PV (vor Fixkosten):    {results['annual_value_only_pv']:,.2f} €")
    print(f"Jahreswert nur PV (nach Fixkosten):   {results['annual_value_only_pv_net']:,.2f} €")
    print(f"Jahreswert mit Speicher (vor Fixk.):  {results['annual_value_with_storage']:,.2f} €")
    print(f"Jahreswert mit Speicher (nach Fixk.): {results['annual_value_with_storage_after_fixed_costs']:,.2f} €")
    print(f"Betriebskosten Speicher (1% Invest):  {results['annual_storage_opex_eur']:,.2f} € /a")
    print(f"Zählerkosten:                         {results['annual_meter_cost_eur']:,.2f} € /a")
    print(f"Direktvermarktung PV (fix):           {results['annual_pv_direct_marketing_cost_eur']:,.2f} € /a")
    print(f"Fixkosten mit Speicher gesamt:        {results['annual_fixed_costs_with_storage_eur']:,.2f} € /a")
    print(f"Batterieerlös (Differenzlogik):       {results['annual_battery_revenue_eur']:,.2f} € /a")
    print(f"Vermarkterkosten Batterie:            {results['annual_battery_marketer_cost_eur']:,.2f} € /a")
    print(f"Laufende Kosten gesamt:               {results['annual_total_recurring_costs_eur']:,.2f} € /a")
    print(f"Jahreswert mit Speicher (netto):      {results['annual_net_value_with_storage']:,.2f} €")
    print(f"Eigenverbrauch (ohne Speicher):       {results['annual_self_consumption_kwh_no_storage']:,.2f} kWh/a")
    print(f"Eigenverbrauch (mit Speicher):        {results['annual_self_consumption_kwh_model']:,.2f} kWh/a")
    print(f"Netzeinspeisung PV:                   {results['annual_feed_in_pv_kwh']:,.2f} kWh/a")
    print(f"Netzeinspeisung Batterie:             {results['annual_feed_in_batt_kwh']:,.2f} kWh/a")
    print(f"Netzeinspeisung gesamt (PV+BATT):     {results['annual_feed_in_total_kwh']:,.2f} kWh/a")
    print(f"PV -> Last:                           {results['annual_pv_to_load_kwh']:,.2f} kWh/a")
    print(f"Batterie -> Last:                     {results['annual_batt_to_load_kwh']:,.2f} kWh/a")
    print(f"Abregelung gesamt:                    {results['annual_curtailment_kwh']:,.2f} kWh/a")
    print(f"Vergütung PV -> Last:                 {results['annual_pv_to_load_remuneration_eur']:,.2f} € /a")
    print(f"Vergütung Batterie -> Last:           {results['annual_batt_to_load_remuneration_eur']:,.2f} € /a")
    print(f"Max. SOC (tatsächlich genutzt):       {results['max_soc_kwh']:,.2f} kWh")
    print(f"Max. Ladeenergie (1h):                {results['max_charge_kwh_per_hour']:,.2f} kWh")
    print(f"Max. Entladeenergie (1h):             {results['max_discharge_kwh_per_hour']:,.2f} kWh")
    print(f"Max. Ladeleistung (abgeleitet):       {results['max_charge_kw']:,.2f} kW")
    print(f"Max. Entladeleistung (abgeleitet):    {results['max_discharge_kw']:,.2f} kW")
    print(f"Mehrerlös vor Fixkosten:              {results['annual_delta_before_fixed_costs']:,.2f} €")
    print(f"Mehrerlös nach Fixkosten:             {results['annual_delta_after_fixed_costs']:,.2f} €")
    print(f"Mehrerlös Speicher pro Jahr (netto):  {results['annual_delta']:,.2f} €")
    print(f"Mehrwert über {inp.horizon_years} Jahre:             {results['total_delta']:,.2f} €")
    print(f"Speicherkosten (E-basiert):           {results['storage_cost']:,.2f} €")
    print(f"Speicherkosten - Mehrwert:            {diff_cost_vs_benefit:,.2f} €")
    print(f"Mehrwert > Kosten:                    {results['pays']}")
    print(f"Zielfunktionswert gesamt:             {results['objective']:,.2f} €")


# =============================================================
# 5) PLOTS
# =============================================================
def plot_soc(model: ConcreteModel, T: int):
    soc = np.array([value(model.soc[t]) for t in range(T)])
    plt.figure(figsize=(14, 4))
    plt.plot(soc, linewidth=1.2)
    plt.title("SOC über das Jahr")
    plt.xlabel("Stunden-Intervall")
    plt.ylabel("SOC [kWh]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_avg_charge_discharge_by_hour(ch: np.ndarray, dis: np.ndarray):
    steps_per_day = 24
    n_days = len(ch) // steps_per_day
    if n_days == 0:
        return

    ch_day = ch[: n_days * steps_per_day].reshape(n_days, steps_per_day)
    dis_day = dis[: n_days * steps_per_day].reshape(n_days, steps_per_day)
    ch_hour = ch_day
    dis_hour = dis_day

    hours = np.arange(24)
    plt.figure(figsize=(12, 5))
    plt.plot(hours, ch_hour.mean(axis=0), marker="o", linewidth=2, label="Laden")
    plt.plot(hours, dis_hour.mean(axis=0), marker="o", linewidth=2, label="Entladen")
    plt.xticks(hours)
    plt.xlabel("Stunde des Tages")
    plt.ylabel("Energie [kWh pro Stunde]")
    plt.title("Gemittelte Lade-/Entladezeiten pro Tag")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _date_window_indices(n_steps: int, start_date: str, n_days: int) -> tuple[pd.DatetimeIndex, int, int]:
    idx = pd.date_range("2024-01-01 00:00:00", periods=n_steps, freq="1h")
    start_ts = pd.Timestamp(start_date)
    end_ts = start_ts + pd.Timedelta(days=n_days)
    s = int(idx.searchsorted(start_ts, side="left"))
    e = int(idx.searchsorted(end_ts, side="left"))
    s = max(0, min(s, n_steps))
    e = max(0, min(e, n_steps))
    return idx, s, e


def plot_energy_balance(results: dict, start_date: str = "2024-07-01", n_days: int = 14):
    T = len(results["pv_gen"])
    time_idx, s, e = _date_window_indices(T, start_date, n_days)
    if s >= e:
        return

    x = time_idx[s:e]
    pv_gen = results["pv_gen"][s:e]
    load = results["load"][s:e]
    sell_pv = results["sell_pv"][s:e]
    sell_batt = results["sell_batt"][s:e]
    pv_to_load = results["pv_to_load"][s:e]
    batt_to_load = results["batt_to_load"][s:e]
    grid_import = results["grid_import"][s:e]
    ch = results["ch"][s:e]
    non_feedin_neg_price = results["curtailment"][s:e]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(x, pv_gen, color="black", linewidth=1.2, label="PV-Erzeugung")
    axes[0].stackplot(
        x,
        pv_to_load,
        sell_pv,
        ch,
        non_feedin_neg_price,
        labels=["PV -> Last", "PV -> Netz", "PV -> Speicher", "Nicht-Einspeisung (Preis negativ)"],
        alpha=0.8,
    )
    axes[0].set_title("Aufteilung der PV-Erzeugung")
    axes[0].set_ylabel("Energie [kWh/h]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    axes[1].stackplot(
        x,
        pv_to_load,
        batt_to_load,
        grid_import,
        sell_pv,
        sell_batt,
        labels=["PV -> Last", "Speicher -> Last", "Netzbezug", "PV -> Netz", "Speicher -> Netz"],
        alpha=0.85,
    )
    axes[1].plot(x, load, color="black", linewidth=1.0, label="Last")
    axes[1].set_title("Lastdeckung und Einspeisung")
    axes[1].set_xlabel("Datum")
    axes[1].set_ylabel("Energie [kWh/h]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    plt.tight_layout()
    plt.show()


def plot_load_selfconsumption_feedin_bess_charge(results: dict, start_date: str = "2024-07-01", n_days: int = 14):
    T = len(results["load"])
    time_idx, s, e = _date_window_indices(T, start_date, n_days)
    if s >= e:
        return

    x = time_idx[s:e]
    load = results["load"][s:e]
    self_consumption = (results["pv_to_load"] + results["batt_to_load"])[s:e]
    feed_in = (results["sell_pv"] + results["sell_batt"])[s:e]
    bess_charge = results["ch"][s:e]

    plt.figure(figsize=(16, 5))
    plt.plot(x, load, linewidth=1.8, label="Last")
    plt.plot(x, self_consumption, linewidth=1.8, label="Eigenverbrauch")
    plt.plot(x, feed_in, linewidth=1.8, label="Einspeisung")
    plt.plot(x, bess_charge, linewidth=1.8, label="Ladung BESS")
    plt.title("Last, Eigenverbrauch, Einspeisung und BESS-Ladung")
    plt.xlabel("Datum")
    plt.ylabel("Energie [kWh/h]")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_battery_discharge_split(results: dict, start_date: str = "2024-07-01", n_days: int = 14):
    T = len(results["sell_batt"])
    time_idx, s, e = _date_window_indices(T, start_date, n_days)
    if s >= e:
        return

    x = time_idx[s:e]
    batt_to_grid = results["sell_batt"][s:e]
    batt_to_load = results["batt_to_load"][s:e]

    total_to_grid = float(np.sum(results["sell_batt"]))
    total_to_load = float(np.sum(results["batt_to_load"]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(x, batt_to_grid, linewidth=1.8, label="Speicher -> Netz")
    axes[0].plot(x, batt_to_load, linewidth=1.8, label="Speicher -> Last")
    axes[0].set_title("Aufteilung der Speicherentladung (Zeitverlauf)")
    axes[0].set_xlabel("Datum")
    axes[0].set_ylabel("Energie [kWh/h]")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    axes[1].bar(["Speicher -> Netz", "Speicher -> Last"], [total_to_grid, total_to_load], alpha=0.85)
    axes[1].set_title("Aufteilung der Speicherentladung (Jahressumme)")
    axes[1].set_ylabel("Energie [kWh/a]")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    inp = Inputs()

    prices, pv_per_kwp, load = load_timeseries(inp)
    model = build_model(prices, pv_per_kwp, load, inp)
    solve_model(model, inp)

    results = evaluate_results(model, prices, pv_per_kwp, load, inp)
    print_results(results, inp)

    plot_soc(model, len(prices))
    plot_avg_charge_discharge_by_hour(results["ch"], results["dis"])
    plot_energy_balance(results, start_date="2024-07-01", n_days=14)
    plot_load_selfconsumption_feedin_bess_charge(results, start_date="2024-07-01", n_days=14)
    plot_battery_discharge_split(results, start_date="2024-07-01", n_days=14)


if __name__ == "__main__":
    main()
