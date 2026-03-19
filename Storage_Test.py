from dataclasses import dataclass, field
from pathlib import Path

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
    Var,
    maximize,
    value,
)
from pyomo.opt import SolverFactory


@dataclass
class Config:
    # -----------------------------
    # Dateien
    # -----------------------------
    path_prices: Path = Path(r"C:\Users\TimFletschinger\OneDrive - empact GmbH\Desktop\04 Simulation\Spotmarktpreis.csv")
    path_pv: Path = Path(r"C:\Users\TimFletschinger\OneDrive - empact GmbH\Desktop\04 Simulation\PV-Daten.csv")

    # -----------------------------
    # Technische Parameter
    # -----------------------------
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    horizon_years: int = 15
    price_in_ct_per_kwh: bool = True

    # WICHTIG: Wenn die PV-Datei bereits für eine konkrete Referenzanlage gilt,
    # hier die Referenzleistung in kWp eintragen (z.B. 500 oder 1000).
    # Dann wird PV im Modell korrekt in kWp optimiert.
    pv_reference_kwp: float = 1000.0

    # -----------------------------
    # Kostenannahmen (einmalige CAPEX)
    # PV-Kosten werden hier nicht berücksichtigt (PV steht bereits)
    # -----------------------------
    capex_batt_E: float = 230.0       # €/kWh

    # -----------------------------
    # Entscheidungsgrenzen
    # -----------------------------
    pv_kwp_bounds: tuple[float, float] = (0.0, 10000.0)
    batt_E_bounds: tuple[float, float] = (0.0, 120000.0)
    allowed_power_values_kw: list[float] = field(
        default_factory=lambda: [0.0, 50.0, 100.0, 150.0, 170.0, 200.0, 220.0, 250.0, 300.0, 500.0]
        + [float(v) for v in range(1000, 50001, 500)]
    )

    # Optional: feste Eingaben für Speicher (None => wird optimiert)
    fixed_batt_E_kwh: float | None = None
    fixed_batt_P_kw: float | None = None
    fixed_pv_kwp: float | None = None

    # Solver
    cbc_executable: str = r"C:\Users\TimFletschinger\Downloads\cbc\bin\cbc.exe"


def load_data(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    df_price = pd.read_csv(cfg.path_prices, sep=";", decimal=",")
    df_pv = pd.read_csv(cfg.path_pv, sep=";", decimal=",")

    prices_ct = pd.to_numeric(df_price["price"], errors="coerce").fillna(0.0).values
    prices_eur = prices_ct / 100.0 if cfg.price_in_ct_per_kwh else prices_ct

    pv_raw = pd.to_numeric(df_pv["pv"], errors="coerce").fillna(0.0).values

    if len(prices_eur) != len(pv_raw):
        raise ValueError(f"Längen passen nicht! Preise={len(prices_eur)}, PV={len(pv_raw)}")
    if cfg.pv_reference_kwp <= 0:
        raise ValueError("pv_reference_kwp muss > 0 sein.")

    # kWh je Stunde und je kWp installierter PV-Leistung
    pv_per_kwp = pv_raw / cfg.pv_reference_kwp
    return prices_eur, pv_per_kwp


def build_model(prices: np.ndarray, pv_per_kwp: np.ndarray, cfg: Config) -> ConcreteModel:
    T = len(prices)

    m = ConcreteModel()
    m.T = RangeSet(0, T - 1)

    # Variablen
    m.pv_kwp = Var(domain=NonNegativeReals, bounds=cfg.pv_kwp_bounds)
    m.E_max = Var(domain=NonNegativeReals, bounds=cfg.batt_E_bounds)
    m.P_max = Var(domain=NonNegativeReals)

    m.sell_pv = Var(m.T, domain=NonNegativeReals)
    m.ch = Var(m.T, domain=NonNegativeReals)
    m.dis = Var(m.T, domain=NonNegativeReals)
    m.soc = Var(m.T, domain=NonNegativeReals)

    # Falls feste Eingaben für Speicher gewünscht sind
    if cfg.fixed_pv_kwp is not None:
        m.pv_kwp.fix(cfg.fixed_pv_kwp)
    if cfg.fixed_batt_E_kwh is not None:
        m.E_max.fix(cfg.fixed_batt_E_kwh)
    if cfg.fixed_batt_P_kw is not None:
        m.P_max.fix(cfg.fixed_batt_P_kw)
    else:
        # Diskrete Leistungsstufen (inkl. 0 kW als "kein Speicher")
        power_options = sorted(set(cfg.allowed_power_values_kw))
        if len(power_options) == 0:
            raise ValueError("allowed_power_values_kw darf nicht leer sein.")
        if power_options[0] < 0:
            raise ValueError("allowed_power_values_kw darf keine negativen Werte enthalten.")

        m.J = RangeSet(0, len(power_options) - 1)
        m.y = Var(m.J, domain=Binary)
        m.power_choice = Constraint(expr=sum(m.y[j] for j in m.J) == 1)
        m.power_match = Constraint(expr=m.P_max == sum(power_options[j] * m.y[j] for j in m.J))

    # Start-SOC
    m.soc[0].fix(0.0)

    # Zyklusbedingung (Jahresende = Jahresanfang)
    m.soc_cycle = Constraint(expr=m.soc[T - 1] == m.soc[0])

    # SOC-Dynamik
    def soc_balance(mm, t):
        if t == 0:
            return mm.soc[t] == cfg.eta_ch * mm.ch[t] - mm.dis[t] / cfg.eta_dis
        return mm.soc[t] == mm.soc[t - 1] + cfg.eta_ch * mm.ch[t] - mm.dis[t] / cfg.eta_dis

    m.soc_balance = Constraint(m.T, rule=soc_balance)

    # PV-Aufteilung: direkte Einspeisung + Speicherladung <= PV-Erzeugung
    def pv_balance(mm, t):
        return mm.sell_pv[t] + mm.ch[t] <= pv_per_kwp[t] * mm.pv_kwp

    m.pv_balance = Constraint(m.T, rule=pv_balance)

    # Speichergrenzen
    m.soc_cap = Constraint(m.T, rule=lambda mm, t: mm.soc[t] <= mm.E_max)
    m.charge_cap = Constraint(m.T, rule=lambda mm, t: mm.ch[t] <= mm.P_max)
    m.discharge_cap = Constraint(m.T, rule=lambda mm, t: mm.dis[t] <= mm.P_max)
    m.power_energy_ratio = Constraint(expr=m.P_max <= 0.5 * m.E_max)

    # Ziel: Gesamtgewinn über Horizont = Erlöse - CAPEX
    def objective_rule(mm):
        annual_revenue = sum(prices[t] * (mm.sell_pv[t] + mm.dis[t]) for t in mm.T)
        total_revenue = cfg.horizon_years * annual_revenue
        capex = cfg.capex_batt_E * mm.E_max
        return total_revenue - capex

    m.obj = Objective(rule=objective_rule, sense=maximize)
    return m


def solve_model(model: ConcreteModel, cfg: Config):
    solver = SolverFactory("cbc", executable=cfg.cbc_executable)
    return solver.solve(model, tee=True)


def calculate_results(model: ConcreteModel, prices: np.ndarray, pv_per_kwp: np.ndarray, cfg: Config) -> dict:
    T = len(prices)
    pv_kwp_opt = value(model.pv_kwp)
    E_opt = value(model.E_max)
    P_opt = value(model.P_max)

    annual_rev_direct = float(np.sum(prices * (pv_per_kwp * pv_kwp_opt)))
    annual_rev_model = float(sum(prices[t] * (value(model.sell_pv[t]) + value(model.dis[t])) for t in range(T)))

    annual_extra_revenue_storage = annual_rev_model - annual_rev_direct
    total_extra_revenue_storage = cfg.horizon_years * annual_extra_revenue_storage

    storage_cost = cfg.capex_batt_E * E_opt

    capex_total = cfg.capex_batt_E * E_opt
    profit_total = value(model.obj)

    ch = np.array([value(model.ch[t]) for t in range(T)])
    dis = np.array([value(model.dis[t]) for t in range(T)])
    sell_pv = np.array([value(model.sell_pv[t]) for t in range(T)])
    pv_gen = pv_per_kwp * pv_kwp_opt
    curtailment = np.maximum(0.0, pv_gen - sell_pv - ch)

    return {
        "pv_kwp_opt": pv_kwp_opt,
        "E_opt": E_opt,
        "P_opt": P_opt,
        "annual_rev_direct": annual_rev_direct,
        "annual_rev_model": annual_rev_model,
        "annual_extra_revenue_storage": annual_extra_revenue_storage,
        "total_extra_revenue_storage": total_extra_revenue_storage,
        "storage_cost": storage_cost,
        "capex_total": capex_total,
        "profit_total": profit_total,
        "pv_gen": pv_gen,
        "sell_pv": sell_pv,
        "ch": ch,
        "dis": dis,
        "curtailment": curtailment,
    }


def print_results(results: dict, cfg: Config):
    pv_kwp_opt = results["pv_kwp_opt"]
    E_opt = results["E_opt"]
    P_opt = results["P_opt"]
    annual_rev_direct = results["annual_rev_direct"]
    annual_rev_model = results["annual_rev_model"]
    annual_extra_revenue_storage = results["annual_extra_revenue_storage"]
    total_extra_revenue_storage = results["total_extra_revenue_storage"]
    storage_cost = results["storage_cost"]
    capex_total = results["capex_total"]
    profit_total = results["profit_total"]

    economic_gap = storage_cost - total_extra_revenue_storage
    storage_pays = total_extra_revenue_storage > storage_cost

    print("\n-----------------------------------")
    print("ERGEBNISSE")
    print(f"Optimale PV-Leistung: {pv_kwp_opt:,.2f} kWp")
    print(f"Optimale Speicherkapazität: {E_opt:,.2f} kWh")
    print(f"Optimale Speicherleistung: {P_opt:,.2f} kW")
    print("-----------------------------------")
    print(f"Jahreserlös nur PV-Einspeisung:         {annual_rev_direct:,.2f} €")
    print(f"Jahreserlös mit Speicher:               {annual_rev_model:,.2f} €")
    print(f"Mehrerlös pro Jahr durch Speicher:      {annual_extra_revenue_storage:,.2f} €")
    print(f"Mehrerlös über {cfg.horizon_years} Jahre:          {total_extra_revenue_storage:,.2f} €")
    print(f"Speicherkosten (nur Batterie, E-basiert): {storage_cost:,.2f} €")
    print(f"Speicherkosten - Mehrerlös:             {economic_gap:,.2f} €")
    print(f"Speicher wirtschaftlich (Mehrerlös > Kosten): {storage_pays}")
    print(f"Gesamt-CAPEX (nur Speicher):            {capex_total:,.2f} €")
    print(f"Gesamtgewinn über {cfg.horizon_years} Jahre:       {profit_total:,.2f} €")


def print_quick_diagnostics(prices: np.ndarray, pv_per_kwp: np.ndarray, cfg: Config):
    annual_rev_per_kwp = float(np.sum(prices * pv_per_kwp))

    print("\nDIAGNOSE")
    print(f"Jahreserlös je 1 kWp (ohne Speicher): {annual_rev_per_kwp:,.2f} €/kWp*a")
    print("PV-CAPEX wird nicht berücksichtigt (Bestandsanlage).")
    if cfg.price_in_ct_per_kwh:
        print("Preis-Umrechnung aktiv: Eingangsdatei wird als ct/kWh interpretiert.")
    else:
        print("Preis-Umrechnung AUS: Eingangsdatei wird als €/kWh interpretiert.")


def plot_soc(model: ConcreteModel, T: int):
    soc = np.array([value(model.soc[t]) for t in range(T)])
    plt.figure(figsize=(14, 4))
    plt.plot(soc, linewidth=1.2)
    plt.title("SOC über das Jahr")
    plt.xlabel("Stunde")
    plt.ylabel("SOC [kWh]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_avg_charge_discharge_by_hour(ch: np.ndarray, dis: np.ndarray):
    daily_hours = 24
    n_days = len(ch) // daily_hours
    if n_days == 0:
        return

    ch_matrix = ch[: n_days * daily_hours].reshape(n_days, daily_hours)
    dis_matrix = dis[: n_days * daily_hours].reshape(n_days, daily_hours)

    avg_ch_hour = ch_matrix.mean(axis=0)
    avg_dis_hour = dis_matrix.mean(axis=0)

    hours = np.arange(daily_hours)
    plt.figure(figsize=(12, 5))
    plt.plot(hours, avg_ch_hour, marker="o", linewidth=2, label="Laden (Durchschnitt)")
    plt.plot(hours, avg_dis_hour, marker="o", linewidth=2, label="Entladen (Durchschnitt)")
    plt.xticks(hours)
    plt.xlabel("Stunde des Tages")
    plt.ylabel("Leistung / Energiefluss [kWh pro Stunde]")
    plt.title("Gemittelte Lade- und Entladezeiten pro Tag")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_energy_balance_timeseries(results: dict, start_hour: int = 0, hours: int = 168):
    pv_gen = results["pv_gen"]
    sell_pv = results["sell_pv"]
    ch = results["ch"]
    dis = results["dis"]
    curtailment = results["curtailment"]

    T = len(pv_gen)
    if T == 0:
        return

    start = max(0, start_hour)
    end = min(T, start + hours)
    if start >= end:
        return

    x = np.arange(start, end)
    pv_h = pv_gen[start:end]
    sell_h = sell_pv[start:end]
    ch_h = ch[start:end]
    dis_h = dis[start:end]
    curt_h = curtailment[start:end]
    export_total_h = sell_h + dis_h

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # 1) Aufteilung der PV-Erzeugung in jedem Zeitpunkt
    axes[0].plot(x, pv_h, color="black", linewidth=1.5, label="PV-Erzeugung")
    axes[0].stackplot(
        x,
        sell_h,
        ch_h,
        curt_h,
        labels=["Aus PV ins Netz", "In Speicher geladen", "Abregelung/Rest"],
        alpha=0.8,
    )
    axes[0].set_title("Energiebilanz je Zeitpunkt: Aufteilung der PV-Erzeugung")
    axes[0].set_ylabel("Energie [kWh pro Stunde]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    # 2) Netzeinspeisung nach Quelle
    axes[1].stackplot(
        x,
        sell_h,
        dis_h,
        labels=["PV -> Netz", "Speicher -> Netz"],
        alpha=0.85,
    )
    axes[1].plot(x, export_total_h, color="black", linewidth=1.2, label="Gesamt-Einspeisung")
    axes[1].set_title("Netzeinspeisung je Zeitpunkt nach Quelle")
    axes[1].set_xlabel("Stunde im Jahr")
    axes[1].set_ylabel("Energie [kWh pro Stunde]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def main():
    # -----------------------------
    # EINGABEFELDER
    # -----------------------------
    # None  => wird optimiert
    # Zahl  => wird fix vorgegeben
    batt_E_input_kwh: float | None = None
    batt_P_input_kw: float | None = None
    pv_input_kwp: float | None = 400

    cfg = Config(
        fixed_batt_E_kwh=batt_E_input_kwh,
        fixed_batt_P_kw=batt_P_input_kw,
        fixed_pv_kwp=pv_input_kwp,
    )

    prices, pv_per_kwp = load_data(cfg)
    print_quick_diagnostics(prices, pv_per_kwp, cfg)
    model = build_model(prices, pv_per_kwp, cfg)
    solve_model(model, cfg)
    results = calculate_results(model, prices, pv_per_kwp, cfg)
    print_results(results, cfg)
    plot_soc(model, len(prices))
    plot_avg_charge_discharge_by_hour(results["ch"], results["dis"])
    plot_energy_balance_timeseries(results, start_hour=0, hours=168)


if __name__ == "__main__":
    main()

