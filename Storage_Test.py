import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory
import os

# -------------------------------------------------------------
# DATEN LADEN
# -------------------------------------------------------------

# Preisdatei
path_prices = r"C:\Users\TimFletschinger\OneDrive - empact GmbH\Desktop\04 Simulation\Spotmarktpreis.csv"
df_price = pd.read_csv(path_prices, sep=";", decimal=",")
prices_ct = df_price["price"].astype(float).values # ct/kWh
prices = prices_ct/100

# PV-Datei
path_pv = r"C:\Users\TimFletschinger\OneDrive - empact GmbH\Desktop\04 Simulation\PV-Daten.csv"
df_pv = pd.read_csv(path_pv, sep=";", decimal=",")
pv = df_pv["pv"].astype(float).values # kWh

# Prüfen gleiche Länge
if len(prices) != len(pv):
    raise ValueError(f"Längen passen nicht! Preise={len(prices)}, PV={len(pv)}")

T = len(prices)

# -------------------------------------------------------------
# Variablen
# -------------------------------------------------------------
n_years = 15 # Jahre Laufzeit
eta_dis = 0.95  # Wirkungsgrad Entladung
eta_ch  = 0.95 # Wirkungsgrad Beladung
cost_E = 210   # €/kWh für Speicher-Kapazität
annual_cost_E = cost_E / n_years

# -------------------------------------------------------------
# ERLÖS OHNE SPEICHER
# -------------------------------------------------------------
revenue_no_battery = (pv * prices).sum()
print("\nErlös ohne Speicher:", revenue_no_battery, "€")

#Modell Bildung
model = ConcreteModel()
model.T = RangeSet(0, T - 1)

# Variablen im Modell
model.dis = Var(model.T, domain=NonNegativeReals)    # Entladung
model.ch = Var(model.T, domain=NonNegativeReals)     # Ladung
model.soc = Var(model.T, domain=NonNegativeReals)    # Speicherfüllstand
model.E_max = Var(domain=NonNegativeReals, bounds=(0, 5000 ))           # Kapazität
model.P_max = Var(domain=NonNegativeReals, bounds=(0, 2000))           # Leistung
model.sell_pv = Var(model.T, domain=NonNegativeReals)

# -------------------------------------------------------------
# SOC STARTWERT
# -------------------------------------------------------------
model.soc[0].fix(0)

# -------------------------------------------------------------
# ZYKLUSBEDINGUNG
# -------------------------------------------------------------
def soc_cycle(m):
    return m.soc[T - 1] == m.soc[0]
model.soc_cycle = Constraint(rule=soc_cycle)

# -------------------------------------------------------------
# SOC-DYNAMIK
# -------------------------------------------------------------

def soc_balance(m, t):
    if t == 0:
        return m.soc[t] == eta_ch*m.ch[t] - m.dis[t]/eta_dis
    return m.soc[t] == m.soc[t-1] + eta_ch*m.ch[t] - m.dis[t]/eta_dis

model.soc_balance = Constraint(model.T, rule=soc_balance)

# -------------------------------------------------------------
# KAPAZITÄTSGRENZEN (notwendig)
# -------------------------------------------------------------
def pv_balance(m, t):
    return m.sell_pv[t] + m.ch[t] <= pv[t]

model.pv_balance = Constraint(model.T, rule=pv_balance)   # PV Aufteilung

def soc_cap(m, t):
    return m.soc[t] <= m.E_max
model.soc_cap = Constraint(model.T, rule=soc_cap)

def charge_cap(m, t):
    return m.ch[t] <= m.P_max
model.charge_cap = Constraint(model.T, rule=charge_cap)

def discharge_cap(m, t):
    return m.dis[t] <= m.P_max
model.discharge_cap = Constraint(model.T, rule=discharge_cap)

def charge_from_pv(m, t):                                          # Ladung muss aus PV-Anlage stammen
    return m.ch[t] <= pv[t]
model.charge_from_pv = Constraint(model.T, rule=charge_from_pv)


# -------------------------------------------------------------
# ZIELFUNKTION
# -------------------------------------------------------------

def objective_rule(m):
    revenue = sum(prices[t] * (m.sell_pv[t] + m.dis[t]) for t in m.T)
    cost_storage = annual_cost_E * m.E_max
    return n_years * revenue - cost_storage

model.obj = Objective(rule=objective_rule, sense=maximize)


# -------------------------------------------------------------
# CBC SOLVER EINBINDEN (KORREKTE COIN-OR VERSION)
# -------------------------------------------------------------

solver = SolverFactory("cbc", executable=r"C:\Users\TimFletschinger\Downloads\cbc\bin\cbc.exe")

# WICHTIG: Pfad explizit übergeben!
result = solver.solve(model, tee=True)

# -------------------------------------------------------------
# ERGEBNISSE
# -------------------------------------------------------------
revenue_with_battery = value(model.obj)
E_opt = value(model.E_max)
P_opt = value(model.P_max)

print("\n-----------------------------------")
print("ERGEBNISSE:")
print("Erlös mit Speicher:", revenue_with_battery, "€")
print("Optimaler Speicher (kWh):", E_opt)
print("Optimale Leistung (kW):", P_opt)
print("-----------------------------------")

print("\nMehrerlös durch Speicher:", revenue_with_battery - revenue_no_battery, "€")

import numpy as np
import matplotlib.pyplot as plt

def plot_soc_year():
    year_len = T  # z.B. 8760 Stunden
    hrs = np.arange(year_len)
    
    soc_year = np.array([value(model.soc[t]) for t in hrs])

    # Optional: tägliche Mittelwerte für Übersicht
    daily_hours = 24
    soc_daily = soc_year.reshape(-1, daily_hours).mean(axis=1)

    plt.figure(figsize=(16, 4))
    plt.plot(soc_daily, linewidth=2)
    plt.title("Durchschnittlicher Speicherfüllstand pro Tag über ein Jahr")
    plt.xlabel("Tag")
    plt.ylabel("SOC [kWh]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_soc_year()

def plot_horizon_year(week_len=168):
    weeks = T // week_len
    cum_rev_no = []
    cum_rev_with = []

    plt.figure(figsize=(16, 9))
    
    for w in range(weeks):
        start = w * week_len
        end = start + week_len
        hrs = np.arange(start, end)

        pv_h = pv[hrs]
        price_h = prices[hrs]
        dis_h = np.array([value(model.dis[t]) for t in hrs])

        revenue_no = pv_h * price_h
        revenue_with = dis_h * price_h

        cum_rev_no.extend(np.cumsum(revenue_no) + (cum_rev_no[-1] if cum_rev_no else 0))
        cum_rev_with.extend(np.cumsum(revenue_with) + (cum_rev_with[-1] if cum_rev_with else 0))

    rel = np.arange(len(cum_rev_no))

    plt.plot(rel, cum_rev_no, linestyle="--", label="kumuliert ohne Speicher")
    plt.plot(rel, cum_rev_with, linewidth=2, label="kumuliert mit Speicher")
    plt.title("Kumulierte Erlöse über ein Jahr")
    plt.xlabel("Stunde")
    plt.ylabel("Erlös [€]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_horizon_year()

def plot_revenue_split_year(week_len=168):
    weeks = T // week_len
    rev_pv_weekly = []
    rev_batt_weekly = []

    for w in range(weeks):
        start = w * week_len
        end = start + week_len
        hrs = np.arange(start, end)

        sell_pv = np.array([value(model.sell_pv[t]) for t in hrs])
        dis = np.array([value(model.dis[t]) for t in hrs])
        price_h = prices[hrs]

        rev_pv_weekly.append(np.sum(sell_pv * price_h))
        rev_batt_weekly.append(np.sum(dis * price_h))

    weeks_idx = np.arange(weeks)

    plt.figure(figsize=(16, 5))
    plt.bar(weeks_idx, rev_pv_weekly, label="Erlös PV-Direkt", alpha=0.7)
    plt.bar(weeks_idx, rev_batt_weekly, bottom=rev_pv_weekly, label="Erlös Speicher", alpha=0.9)

    plt.title("Wöchentliche Erlösaufteilung PV-Direkt vs Speicher über ein Jahr")
    plt.xlabel("Woche")
    plt.ylabel("Erlös [€]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_revenue_split_year()

def plot_average_discharge_by_hour():
    daily_hours = 24
    n_days = T // daily_hours

    # Entladung pro Stunde
    dis_h = np.array([value(model.dis[t]) for t in range(T)])

    # Reshape: Zeilen = Tage, Spalten = Stunden des Tages
    dis_matrix = dis_h[:n_days*daily_hours].reshape(n_days, daily_hours)

    # Mittelwert über alle Tage pro Stunde
    dis_hourly_avg = dis_matrix.mean(axis=0)

    plt.figure(figsize=(12,5))
    plt.plot(range(daily_hours), dis_hourly_avg, marker='o', color='orange', linewidth=2)
    plt.xticks(range(daily_hours))
    plt.xlabel("Stunde des Tages")
    plt.ylabel("Durchschnittliche Entladung [kWh]")
    plt.title("Durchschnittliche Speicherentladung pro Stunde des Tages")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_average_discharge_by_hour()

def plot_average_price_by_hour():
    daily_hours = 24
    n_days = T // daily_hours  # Anzahl ganzer Tage

    # Preise für alle Stunden
    prices_h = prices[:n_days*daily_hours]

    # Reshape: Zeilen = Tage, Spalten = Stunden des Tages
    price_matrix = prices_h.reshape(n_days, daily_hours)

    # Mittelwert über alle Tage pro Stunde
    price_hourly_avg = price_matrix.mean(axis=0)

    plt.figure(figsize=(12,5))
    plt.plot(range(daily_hours), price_hourly_avg, marker='o', color='blue', linewidth=2)
    plt.xticks(range(daily_hours))
    plt.xlabel("Stunde des Tages")
    plt.ylabel("Durchschnittlicher Spotpreis [€/kWh]")
    plt.title("Durchschnittlicher Spotpreis pro Stunde des Tages")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_average_price_by_hour()

