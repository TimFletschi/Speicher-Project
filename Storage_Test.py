from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    UnitInterval,
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
    base_dir = Path(__file__).resolve().parent
    path_prices = inp.path_prices if inp.path_prices.exists() else base_dir / inp.path_prices.name
    path_pv = inp.path_pv if inp.path_pv.exists() else base_dir / inp.path_pv.name
    path_bdew = inp.path_bdew_g0 if inp.path_bdew_g0.exists() else base_dir / inp.path_bdew_g0.name

    df_price = pd.read_csv(path_prices, sep=";", decimal=",")
    prices_hour = pd.to_numeric(df_price["price"], errors="coerce").fillna(0.0).values
    if inp.price_in_ct_per_kwh:
        prices_hour = prices_hour / 100.0

    pv_per_kwp_hour = load_pv_per_kwp_hourly(
        path_pv,
        len(prices_hour),
        inp.pv_reference_kwp,
        inp.pv_specific_yield_kwh_per_kwp_per_year,
    )
    load_hour = load_bdew_g0_profile_15min(
        path_bdew,
        len(prices_hour),
        inp.annual_consumption_kwh,
        inp.g0_reference_annual_kwh,
    )
    return prices_hour, pv_per_kwp_hour, load_hour


def anzulegender_mischpreis_eur_per_kwh(pv_size_kwp: float) -> float:
    """Gewichteter anzulegender Wert (EUR/kWh) nach Leistungsstufen.

    Stufen (ct/kWh):
    - bis 10 kW:   8.18
    - bis 40 kW:   7.13
    - bis 1000 kW: 5.90

    Für Leistungen >1000 kW wird die letzte Stufe (5.90 ct/kWh) fortgeführt.
    """
    if pv_size_kwp <= 0:
        raise ValueError("pv_size_kwp muss > 0 sein.")

    # kW-Anteile je Stufe
    s1 = min(pv_size_kwp, 10.0)
    s2 = min(max(pv_size_kwp - 10.0, 0.0), 30.0)
    s3 = max(pv_size_kwp - 40.0, 0.0)

    # ct/kWh
    w1 = 8.18
    w2 = 7.13
    w3 = 5.90

    weighted_ct = (s1 * w1 + s2 * w2 + s3 * w3) / pv_size_kwp
    return weighted_ct / 100.0


def opportunity_price_eur_per_kwh(prices: np.ndarray, pv_size_kwp: float) -> np.ndarray:
    """Opportunitätspreis für PV->Speicher-Ladung.

    Logik:
    - Spot <= 0: Opportunitätskosten = 0
    - Anlagen >= 1000 kW und Spot > 0: fixer Opportunitätspreis 9.6 ct/kWh
    - Sonst (Spot > 0): Opportunitätswert = max(Spot, anzulegender Mischpreis)
    """
    prices = np.asarray(prices, dtype=float)

    if pv_size_kwp >= 1000.0:
        # Ab 1000 kW: dauerhafter Preis 9.6 ct/kWh, außer wenn Spot <= 0 dann 0
        return np.where(prices <= 0.0, 0.0, 0.096).astype(float)

    mixed = anzulegender_mischpreis_eur_per_kwh(pv_size_kwp)
    out = np.where(prices <= 0.0, 0.0, np.maximum(prices, mixed))
    return out.astype(float)


# =============================================================
# 2) EINGABEN
# =============================================================
@dataclass
class Inputs:
    _BASE_DIR: Path = Path(__file__).resolve().parent
    path_prices: Path = _BASE_DIR / "Data" / "Spotmarktpreis.csv"
    path_pv: Path = _BASE_DIR / "Data" / "PV-Daten_400kwp_stuendlich.csv"
    path_bdew_g0: Path = _BASE_DIR / "Data" / "G0Verbrauch_400.000kwh_stuendlich.csv"

    # PV-Größe in kWp eingeben
    pv_size_kwp: float = 400
    # Größe der Anlage des Beispiel Erzeugungsprofils (z.B. 400 kWp), um die PV-Erzeugung pro kWp zu normieren.
    pv_reference_kwp: float = 400.0
    # Optional: standortspezifischer Jahresertrag zur Skalierung des PV-Profils.
    # Beispiel: 950.0 oder 1100.0 kWh/kWp*a. None = Wert aus CSV unverändert nutzen.
    pv_specific_yield_kwh_per_kwp_per_year: float | None = None

    # Verbrauch eingeben
    annual_consumption_kwh: float = 230000
    #Größe des Jahresverbrauchs des Beispiel Lastprofils (z.B. 400.000 kWh/a), um das Lastprofil zu normieren und auf den gewünschten Verbrauch hochzuskalieren.
    g0_reference_annual_kwh: float = 400000.0

    # Speicherparameter
    # Effizienzen (0-1) für Ladung und Entladung
    eta_ch: float = 0.95
    eta_dis: float = 0.95

    horizon_years: int = 15
    # Degradation: Restkapazität am Ende des Horizonts (z.B. 0.80 nach 15 Jahren)
    batt_capacity_retention_after_horizon: float = 0.80
    price_in_ct_per_kwh: bool = True
    # Getrennte Vergütung für Eigenverbrauch, in €/kWh
    pv_to_load_remuneration_eur_per_kwh: float = 0.14
    batt_to_load_remuneration_eur_per_kwh: float = 0.14
    capex_batt_E_eur_per_kwh: float = 190.0
    # Laufende Kosten (veränderbar)
    storage_opex_pct_of_invest_per_year: float = 0.01
    meter_cost_eur_per_year: float = 300.0
    marketer_share_of_battery_revenue: float = 0.10
    pv_direct_marketing_cost_eur_per_month: float = 60.0

    batt_E_bounds: tuple[float, float] = (0.0, 120000.0)
    fixed_batt_E_kwh: float | None = None
    fixed_batt_P_kw: float | None = None

    cbc_executable: str = r"C:\Users\TimFletschinger\Downloads\cbc\bin\cbc.exe"
    # Laufzeit-Optimierung
    use_binary_charge_switch: bool = True
    solver_tee: bool = False
    cbc_time_limit_sec: int = 300
    cbc_mip_gap: float = 0.01
    cbc_threads: int = 0
    export_dir: Path = _BASE_DIR / "exports"
    template_word_path: Path | None = _BASE_DIR / "PDF" / "VORLAGE_Ausführungsbeschreibung_PVA_Kunde1.docx"
    plot_start_date: str = "2024-07-01"
    plot_n_days: int = 14


# =============================================================
# 3) MODELLAUFBAU
# =============================================================
def capacity_degradation_multiplier(inp: Inputs) -> float:
    """Linearer Degradationsfaktor über den Horizont.

    Beispiel: Start 100%, Ende 80% nach 15 Jahren -> mittlere Verfügbarkeit 90%.
    Der jährliche Mehrwert wird mit diesem mittleren Faktor über den Horizont skaliert.
    """
    if inp.horizon_years <= 0:
        raise ValueError("horizon_years muss > 0 sein.")
    if not (0.0 < inp.batt_capacity_retention_after_horizon <= 1.0):
        raise ValueError("batt_capacity_retention_after_horizon muss im Intervall (0, 1] liegen.")
    return 0.5 * (1.0 + inp.batt_capacity_retention_after_horizon)


def build_model(prices: np.ndarray, pv_per_kwp: np.ndarray, load: np.ndarray, inp: Inputs) -> ConcreteModel:
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

    # Optional: vorhandenes Speichersystem vorgeben
    if (inp.fixed_batt_E_kwh is None) ^ (inp.fixed_batt_P_kw is None):
        raise ValueError("Bitte entweder fixed_batt_E_kwh und fixed_batt_P_kw beide setzen oder beide auf None lassen.")

    optimize_battery_size = inp.fixed_batt_E_kwh is None and inp.fixed_batt_P_kw is None
    if not optimize_battery_size:
        m.E_max.fix(inp.fixed_batt_E_kwh)
        m.P_max.fix(inp.fixed_batt_P_kw)

    # Lineares Big-M für Lade-/Entlade-Logik (verhindert Nichtlinearität mm.P_max * binary)
    if inp.fixed_batt_P_kw is not None:
        p_big_m = float(inp.fixed_batt_P_kw)
    else:
        # Bei 2h-Beziehung gilt P_max = E_max / 2
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
        # LP-Modus (schneller): Relaxierung statt binärer Variable
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

    # 2h Speicher nur im Optimierungsmodus
    if optimize_battery_size:
        m.storage_2h = Constraint(expr=m.E_max == 2.0 * m.P_max)

    def objective_rule(mm):
        degr_mult = capacity_degradation_multiplier(inp)
        pv_feed_in_value = sum(opp_price[t] * mm.sell_pv[t] for t in mm.T)
        batt_feed_in_value = sum(prices[t] * mm.sell_batt[t] for t in mm.T)
        battery_charging_opportunity_cost = sum(opp_price[t] * mm.ch[t] for t in mm.T)
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
        # DV PV wird nicht der Batterie zugerechnet
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
        total_value = inp.horizon_years * degr_mult * annual_incremental_storage_value
        storage_cost = inp.capex_batt_E_eur_per_kwh * mm.E_max
        return total_value - storage_cost

    m.obj = Objective(rule=objective_rule, sense=maximize)
    return m


# =============================================================
# 4) AUSGABE DER ERGEBNISSE
# =============================================================
def solve_model(model: ConcreteModel, inp: Inputs):
    solver = SolverFactory("cbc", executable=inp.cbc_executable)
    if inp.cbc_time_limit_sec > 0:
        solver.options["seconds"] = int(inp.cbc_time_limit_sec)
    if inp.cbc_mip_gap is not None and inp.cbc_mip_gap > 0:
        solver.options["ratioGap"] = float(inp.cbc_mip_gap)
    if inp.cbc_threads is not None and inp.cbc_threads > 0:
        solver.options["threads"] = int(inp.cbc_threads)
    return solver.solve(model, tee=inp.solver_tee)


def evaluate_results(model: ConcreteModel, prices: np.ndarray, pv_per_kwp: np.ndarray, load: np.ndarray, inp: Inputs) -> dict:
    degr_mult = capacity_degradation_multiplier(inp)
    effective_years_with_degradation = inp.horizon_years * degr_mult

    T = len(prices)
    opp_price = opportunity_price_eur_per_kwh(prices, inp.pv_size_kwp)
    E_opt = value(model.E_max)
    P_opt = value(model.P_max)

    pv_gen = pv_per_kwp * inp.pv_size_kwp
    pv_direct_to_load = np.minimum(pv_gen, load)
    pv_surplus = np.maximum(0.0, pv_gen - pv_direct_to_load)
    pv_direct_to_grid = np.where(prices >= 0.0, pv_surplus, 0.0)

    annual_value_only_pv = float(
        np.sum(opp_price * pv_direct_to_grid)
        + inp.pv_to_load_remuneration_eur_per_kwh * np.sum(pv_direct_to_load)
    )

    annual_value_with_storage = float(
        sum(
            opp_price[t] * value(model.sell_pv[t])
            + prices[t] * value(model.sell_batt[t])
            - opp_price[t] * value(model.ch[t])
            + inp.pv_to_load_remuneration_eur_per_kwh * value(model.pv_to_load[t])
            + inp.batt_to_load_remuneration_eur_per_kwh * value(model.batt_to_load[t])
            for t in range(T)
        )
    )

    annual_battery_revenue_eur = float(
        sum(
            prices[t] * value(model.sell_batt[t])
            - opp_price[t] * value(model.ch[t])
            + inp.batt_to_load_remuneration_eur_per_kwh * value(model.batt_to_load[t])
            for t in range(T)
        )
    )

    storage_cost = inp.capex_batt_E_eur_per_kwh * E_opt
    annual_storage_opex_eur = inp.storage_opex_pct_of_invest_per_year * storage_cost
    annual_meter_cost_eur = inp.meter_cost_eur_per_year
    # DV PV wird nicht der Batterie zugerechnet
    annual_pv_direct_marketing_cost_eur = 0.0
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
    total_delta = effective_years_with_degradation * annual_delta

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
        "batt_capacity_retention_after_horizon": inp.batt_capacity_retention_after_horizon,
        "capacity_degradation_multiplier": degr_mult,
        "effective_years_with_degradation": effective_years_with_degradation,
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
        "soc": soc,
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
    print(f"Restkapazität nach {inp.horizon_years} Jahren:      {100*results['batt_capacity_retention_after_horizon']:,.1f} %")
    print(f"Degradationsfaktor (linear gemittelt): {results['capacity_degradation_multiplier']:,.3f}")
    print(f"Effektive Jahre (mit Degradation):     {results['effective_years_with_degradation']:,.2f}")
    print(f"Mehrwert über {inp.horizon_years} Jahre:             {results['total_delta']:,.2f} €")
    print(f"Speicherkosten (E-basiert):           {results['storage_cost']:,.2f} €")
    print(f"Speicherkosten - Mehrwert:            {diff_cost_vs_benefit:,.2f} €")
    print(f"Mehrwert > Kosten:                    {results['pays']}")
    print(f"Zielfunktionswert gesamt:             {results['objective']:,.2f} €")


# =============================================================
# 5) EXPORT
# =============================================================
def export_results_and_timeseries(results: dict, prices: np.ndarray, inp: Inputs) -> dict[str, Path]:
    out_dir = inp.export_dir
    if out_dir.parent.exists() and not out_dir.parent.is_dir():
        raise NotADirectoryError(
            f"Exportpfad ungültig: Der Parent-Pfad ist eine Datei und kein Ordner: {out_dir.parent}. "
            "Bitte inp.export_dir auf einen echten Ordner setzen (z. B. ...\\04 Simulation\\exports)."
        )
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(
            f"Exportpfad ungültig: {out_dir} ist eine Datei, aber es wird ein Ordner benötigt."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    T = len(prices)
    opp_price = opportunity_price_eur_per_kwh(prices, inp.pv_size_kwp)
    time_idx = pd.date_range("2024-01-01 00:00:00", periods=T, freq="1h")

    sell_pv = results["sell_pv"]
    sell_batt = results["sell_batt"]
    pv_to_load = results["pv_to_load"]
    batt_to_load = results["batt_to_load"]
    ch = results["ch"]

    rev_pv_feed_in = prices * sell_pv
    rev_batt_feed_in = prices * sell_batt
    rev_pv_to_load = inp.pv_to_load_remuneration_eur_per_kwh * pv_to_load
    rev_batt_to_load = inp.batt_to_load_remuneration_eur_per_kwh * batt_to_load
    cost_charge_opportunity = opp_price * ch

    cost_storage_opex = np.full(T, results["annual_storage_opex_eur"] / T)
    cost_meter = np.full(T, results["annual_meter_cost_eur"] / T)
    cost_pv_direct_marketing = np.full(T, results["annual_pv_direct_marketing_cost_eur"] / T)
    cost_battery_marketer = inp.marketer_share_of_battery_revenue * (
        rev_batt_feed_in + rev_batt_to_load - cost_charge_opportunity
    )

    ts = pd.DataFrame(
        {
            "timestamp": time_idx,
            "price_eur_per_kwh": prices,
            "opportunity_price_eur_per_kwh": opp_price,
            "pv_gen_kwh": results["pv_gen"],
            "load_kwh": results["load"],
            "sell_pv_kwh": sell_pv,
            "sell_batt_kwh": sell_batt,
            "pv_to_load_kwh": pv_to_load,
            "batt_to_load_kwh": batt_to_load,
            "grid_import_kwh": results["grid_import"],
            "ch_kwh": ch,
            "dis_kwh": results["dis"],
            "soc_kwh": results["soc"],
            "curtailment_kwh": results["curtailment"],
            "rev_pv_feed_in_eur": rev_pv_feed_in,
            "rev_batt_feed_in_eur": rev_batt_feed_in,
            "rev_pv_to_load_eur": rev_pv_to_load,
            "rev_batt_to_load_eur": rev_batt_to_load,
            "cost_charge_opportunity_eur": cost_charge_opportunity,
            "cost_storage_opex_eur": cost_storage_opex,
            "cost_meter_eur": cost_meter,
            "cost_pv_direct_marketing_eur": cost_pv_direct_marketing,
            "cost_battery_marketer_eur": cost_battery_marketer,
        }
    )

    ts["cum_price_sum"] = ts["price_eur_per_kwh"].cumsum()
    ts["cum_price_positive_sum"] = ts["price_eur_per_kwh"].clip(lower=0.0).cumsum()
    ts["cum_price_negative_sum"] = ts["price_eur_per_kwh"].clip(upper=0.0).cumsum()

    ts["cum_rev_pv_feed_in_eur"] = ts["rev_pv_feed_in_eur"].cumsum()
    ts["cum_rev_batt_feed_in_eur"] = ts["rev_batt_feed_in_eur"].cumsum()
    ts["cum_rev_pv_to_load_eur"] = ts["rev_pv_to_load_eur"].cumsum()
    ts["cum_rev_batt_to_load_eur"] = ts["rev_batt_to_load_eur"].cumsum()

    ts["cum_cost_charge_opportunity_eur"] = ts["cost_charge_opportunity_eur"].cumsum()
    ts["cum_cost_storage_opex_eur"] = ts["cost_storage_opex_eur"].cumsum()
    ts["cum_cost_meter_eur"] = ts["cost_meter_eur"].cumsum()
    ts["cum_cost_pv_direct_marketing_eur"] = ts["cost_pv_direct_marketing_eur"].cumsum()
    ts["cum_cost_battery_marketer_eur"] = ts["cost_battery_marketer_eur"].cumsum()

    ts["cum_revenue_total_eur"] = (
        ts["cum_rev_pv_feed_in_eur"]
        + ts["cum_rev_batt_feed_in_eur"]
        + ts["cum_rev_pv_to_load_eur"]
        + ts["cum_rev_batt_to_load_eur"]
    )
    ts["cum_cost_total_eur"] = (
        ts["cum_cost_charge_opportunity_eur"]
        + ts["cum_cost_storage_opex_eur"]
        + ts["cum_cost_meter_eur"]
        + ts["cum_cost_pv_direct_marketing_eur"]
        + ts["cum_cost_battery_marketer_eur"]
    )
    ts["cum_net_eur"] = ts["cum_revenue_total_eur"] - ts["cum_cost_total_eur"]

    timeseries_path = out_dir / "timeseries_export.csv"
    ts.to_csv(timeseries_path, sep=";", decimal=",", index=False)

    summary_rows = [
        {"metric": k, "value": v}
        for k, v in results.items()
        if np.isscalar(v)
    ]
    summary_path = out_dir / "summary_export.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep=";", decimal=",", index=False)

    inputs_path = out_dir / "inputs_export.csv"
    inp_dict = {k: str(v) if isinstance(v, Path) else v for k, v in inp.__dict__.items()}
    pd.DataFrame(list(inp_dict.items()), columns=["parameter", "value"]).to_csv(
        inputs_path, sep=";", decimal=",", index=False
    )

    return {
        "timeseries": timeseries_path,
        "summary": summary_path,
        "inputs": inputs_path,
    }


def main():
    inp = Inputs()

    prices, pv_per_kwp, load = load_timeseries(inp)
    model = build_model(prices, pv_per_kwp, load, inp)
    solve_model(model, inp)

    results = evaluate_results(model, prices, pv_per_kwp, load, inp)
    print_results(results, inp)

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


if __name__ == "__main__":
    main()
