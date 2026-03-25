from dataclasses import dataclass, replace
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

import Storage_Test as st


@dataclass
class SweepConfig:
    # Vorgaben laut Anforderung
    pv_sizes_kwp: list[float] | None = None
    annual_consumptions_kwh: list[float] | None = None
    storage_capex_eur_per_kwh: list[float] | None = None

    # Laufzeitsteuerung
    use_binary_charge_switch: bool = True
    cbc_time_limit_sec: int = 120
    cbc_mip_gap: float = 0.02
    cbc_threads: int = 0
    solver_tee: bool = False

    # Ausgabe
    output_dir: Path = Path(r"C:\Users\TimFletschinger\OneDrive - empact GmbH\Desktop\04 Simulation\exports\sweep")


def default_pv_grid() -> list[float]:
    # 100 kWp bis 2 MW in 200er Schritten; 2000 explizit ergänzen
    vals = list(range(100, 2000, 200))
    if 2000 not in vals:
        vals.append(2000)
    return [float(v) for v in vals]


def default_load_grid() -> list[float]:
    return [float(v) for v in range(200_000, 800_001, 100_000)]


def default_capex_grid() -> list[float]:
    return [float(v) for v in range(190, 241, 10)]


def run_sweep(cfg: SweepConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SweepConfig()
    pv_grid = cfg.pv_sizes_kwp or default_pv_grid()
    load_grid = cfg.annual_consumptions_kwh or default_load_grid()
    capex_grid = cfg.storage_capex_eur_per_kwh or default_capex_grid()

    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = st.Inputs()
    rows: list[dict[str, Any]] = []

    run_id = 0
    total_runs = len(pv_grid) * len(load_grid) * len(capex_grid)

    # Preise/PV-spezifisch nur einmal laden (Last wird je Verbrauch neu skaliert)
    prices, pv_per_kwp, _ = st.load_timeseries(base)

    for load_kwh in load_grid:
        inp_load = replace(base, annual_consumption_kwh=load_kwh)
        _, _, load = st.load_timeseries(inp_load)

        for pv_kwp in pv_grid:
            for capex in capex_grid:
                run_id += 1
                t0 = time.perf_counter()

                inp = replace(
                    base,
                    annual_consumption_kwh=load_kwh,
                    pv_size_kwp=pv_kwp,
                    capex_batt_E_eur_per_kwh=capex,
                    use_binary_charge_switch=cfg.use_binary_charge_switch,
                    cbc_time_limit_sec=cfg.cbc_time_limit_sec,
                    cbc_mip_gap=cfg.cbc_mip_gap,
                    cbc_threads=cfg.cbc_threads,
                    solver_tee=cfg.solver_tee,
                    # Für Sweep soll jedes Mal neu optimiert werden
                    fixed_batt_E_kwh=None,
                    fixed_batt_P_kw=None,
                )

                print(f"[{run_id:04d}/{total_runs}] PV={pv_kwp:.0f} kWp | Last={load_kwh:,.0f} kWh | CAPEX={capex:.0f} €/kWh")

                try:
                    model = st.build_model(prices, pv_per_kwp, load, inp)
                    solve_res = st.solve_model(model, inp)
                    res = st.evaluate_results(model, prices, pv_per_kwp, load, inp)

                    solver_status = str(solve_res.solver.status)
                    termination = str(solve_res.solver.termination_condition)
                    ok = True
                    err = ""
                except Exception as ex:
                    res = {}
                    solver_status = "error"
                    termination = "error"
                    ok = False
                    err = str(ex)

                dt = time.perf_counter() - t0

                rows.append(
                    {
                        "run_id": run_id,
                        "pv_size_kwp": pv_kwp,
                        "annual_consumption_kwh": load_kwh,
                        "capex_batt_E_eur_per_kwh": capex,
                        "ok": ok,
                        "solver_status": solver_status,
                        "termination": termination,
                        "runtime_sec": dt,
                        "annual_delta_eur": res.get("annual_delta"),
                        "total_delta_eur": res.get("total_delta"),
                        "E_opt_kwh": res.get("E_opt"),
                        "P_opt_kw": res.get("P_opt"),
                        "objective_eur": res.get("objective"),
                        "error": err,
                    }
                )

    df = pd.DataFrame(rows)
    csv_path = out_dir / "sweep_results.csv"
    df.to_csv(csv_path, sep=";", decimal=",", index=False)
    print(f"\nErgebnisse gespeichert: {csv_path}")

    make_plots(df, out_dir)
    return df


def make_plots(df: pd.DataFrame, out_dir: Path):
    ok_df = df[df["ok"]].copy()

    # 1) Übersicht Inputs je Run
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].step(df["run_id"], df["pv_size_kwp"], where="mid")
    axes[0].set_ylabel("PV [kWp]")
    axes[0].grid(alpha=0.3)

    axes[1].step(df["run_id"], df["annual_consumption_kwh"], where="mid")
    axes[1].set_ylabel("Verbrauch [kWh/a]")
    axes[1].grid(alpha=0.3)

    axes[2].step(df["run_id"], df["capex_batt_E_eur_per_kwh"], where="mid")
    axes[2].set_ylabel("CAPEX [€/kWh]")
    axes[2].set_xlabel("Simulationslauf")
    axes[2].grid(alpha=0.3)

    fig.suptitle("Sweep-Eingaben je Simulationslauf")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_inputs_over_runs.png", dpi=160)
    plt.close(fig)

    # 2) Flexibel durchgelaufen / Status
    status_counts = df["termination"].fillna("unknown").value_counts()
    fig = plt.figure(figsize=(10, 4))
    status_counts.plot(kind="bar")
    plt.title("Solver-Termination je Simulationslauf")
    plt.ylabel("Anzahl")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_solver_status.png", dpi=160)
    plt.close(fig)

    if ok_df.empty:
        print("Keine erfolgreichen Läufe für Ergebnisdiagramme.")
        return

    # 3) Mehrerlös über alle Runs
    fig = plt.figure(figsize=(14, 5))
    plt.plot(ok_df["run_id"], ok_df["annual_delta_eur"], marker="o", linewidth=1)
    plt.title("Mehrerlös Speicher pro Jahr über Simulationsläufe")
    plt.xlabel("Simulationslauf")
    plt.ylabel("EUR/a")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_annual_delta_over_runs.png", dpi=160)
    plt.close(fig)

    # 4) Optimale Speichergröße über alle Runs
    fig = plt.figure(figsize=(14, 5))
    plt.plot(ok_df["run_id"], ok_df["E_opt_kwh"], marker="o", linewidth=1)
    plt.title("Optimale Speichergröße E über Simulationsläufe")
    plt.xlabel("Simulationslauf")
    plt.ylabel("E_opt [kWh]")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "plot_eopt_over_runs.png", dpi=160)
    plt.close(fig)

    # 5) Gewünschte Sicht: je (PV, Verbrauch) die CAPEX-Variation
    # Ein Diagramm pro (PV, Verbrauch), zeigt Mehrerlös + E_opt über CAPEX
    combo_dir = out_dir / "plots_by_combo"
    combo_dir.mkdir(parents=True, exist_ok=True)

    grouped = ok_df.groupby(["pv_size_kwp", "annual_consumption_kwh"], dropna=True)
    for (pv, load), g in grouped:
        g = g.sort_values("capex_batt_E_eur_per_kwh")

        fig, ax1 = plt.subplots(figsize=(10, 4.8))
        ax1.plot(g["capex_batt_E_eur_per_kwh"], g["annual_delta_eur"], marker="o", label="Mehrerlös [EUR/a]")
        ax1.set_xlabel("CAPEX Speicher [€/kWh]")
        ax1.set_ylabel("Mehrerlös [EUR/a]")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(g["capex_batt_E_eur_per_kwh"], g["E_opt_kwh"], marker="s", linestyle="--", label="E_opt [kWh]")
        ax2.set_ylabel("E_opt [kWh]")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(f"PV={pv:.0f} kWp | Verbrauch={load:,.0f} kWh/a")
        plt.tight_layout()
        fname = f"pv_{int(pv)}_load_{int(load)}.png"
        plt.savefig(combo_dir / fname, dpi=160)
        plt.close(fig)

    print(f"Diagramme gespeichert in: {out_dir}")


if __name__ == "__main__":
    # Standardlauf mit den geforderten Rastern
    run_sweep()
