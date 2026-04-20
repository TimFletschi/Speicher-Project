"""Plotting- und Reporting-Modul für die Speichersimulation.

- Liest exportierte Zeitreihen/Ergebnisse aus CSV.
- Erstellt interaktive Diagramme.
- Erzeugt einen PDF-Bericht mit Tabellen und allen Diagrammen.
- Nutzt optional eine Word-Vorlage (.docx) für Basis-Schriftstil und Logo.
"""

import argparse
import io
from datetime import datetime
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg


GREEN = (0.0, 1.0, 0.0)
# klare Abstufung von Grün -> Grau -> fast Schwarz
GRAY_1 = (0.12, 0.60, 0.12)
GRAY_2 = (0.35, 0.45, 0.35)
GRAY_3 = (0.48, 0.48, 0.48)
GRAY_4 = (0.30, 0.30, 0.30)
GRAY_5 = (0.08, 0.08, 0.08)

plt.rcParams["axes.prop_cycle"] = cycler(color=[GREEN, GRAY_1, GRAY_2, GRAY_3, GRAY_4, GRAY_5])


def load_exported_data(export_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lädt die drei Standard-Exportdateien aus dem Exportordner."""
    export_dir = Path(export_dir)
    ts_path = export_dir / "timeseries_export.csv"
    summary_path = export_dir / "summary_export.csv"
    inputs_path = export_dir / "inputs_export.csv"

    if not ts_path.exists() or not summary_path.exists() or not inputs_path.exists():
        raise FileNotFoundError(
            "Exportdateien fehlen. Bitte zuerst Storage_Test.py vollständig laufen lassen, "
            f"damit die CSVs unter {export_dir} erzeugt werden. "
            f"Erwartet: {ts_path.name}, {summary_path.name}, {inputs_path.name}."
        )

    ts = pd.read_csv(ts_path, sep=";", decimal=",", parse_dates=["timestamp"])
    summary = pd.read_csv(summary_path, sep=";", decimal=",")
    inputs = pd.read_csv(inputs_path, sep=";", decimal=",")
    return ts, summary, inputs


def _window(ts: pd.DataFrame, start_date: str, n_days: int) -> pd.DataFrame:
    """Schneidet ein Datumfenster aus der Zeitreihe; fallback = komplette Reihe."""
    start_ts = pd.Timestamp(start_date)
    end_ts = start_ts + pd.Timedelta(days=n_days)
    w = ts[(ts["timestamp"] >= start_ts) & (ts["timestamp"] < end_ts)].copy()
    if len(w) == 0:
        return ts.copy()
    return w


def _get_float_from_table(df: pd.DataFrame, key_col: str, key: str, value_col: str) -> float:
    """Liest einen numerischen Wert aus einer Key-Value-Tabelle."""
    row = df.loc[df[key_col].astype(str) == key, value_col]
    if row.empty:
        raise KeyError(f"'{key}' nicht in Tabelle gefunden.")
    return float(row.iloc[0])


def resolve_template_path(export_dir: Path, template_path: Path | None = None) -> Path | None:
    """Sucht ein Word-Template (bevorzugt .docx) im PDF-Ordner."""
    if template_path is not None:
        tp = Path(template_path)
        return tp if tp.exists() else None

    base = Path(export_dir).parent
    pdf_dir = base / "PDF"
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        return None

    candidates = sorted(list(pdf_dir.glob("*.docx")) + list(pdf_dir.glob("*.doc")))
    return candidates[0] if candidates else None


def _extract_docx_base_style(template_path: Path) -> dict[str, float | str] | None:
    """Extrahiert Basis-Schriftinformationen aus styles.xml einer DOCX-Datei."""
    if template_path.suffix.lower() != ".docx" or not template_path.exists():
        return None
    try:
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        with zipfile.ZipFile(template_path, "r") as zf:
            with zf.open("word/styles.xml") as f:
                tree = ET.parse(f)
        root = tree.getroot()
        rpr = root.find(".//w:docDefaults/w:rPrDefault/w:rPr", ns)
        if rpr is None:
            return None
        font_name = None
        font_size = None
        rfonts = rpr.find("w:rFonts", ns)
        if rfonts is not None:
            font_name = rfonts.get(f"{{{ns['w']}}}ascii") or rfonts.get(f"{{{ns['w']}}}hAnsi")
        sz = rpr.find("w:sz", ns)
        if sz is not None:
            sz_val = sz.get(f"{{{ns['w']}}}val")
            if sz_val is not None:
                font_size = float(sz_val) / 2.0
        out: dict[str, float | str] = {}
        if font_name:
            out["font_name"] = font_name
        if font_size:
            out["font_size"] = font_size
        return out if out else None
    except Exception:
        return None


def _apply_template_style_if_available(template_path: Path | None):
    """Wendet erkannte Template-Schrift auf Matplotlib-Defaults an."""
    if template_path is None:
        return
    style = _extract_docx_base_style(template_path)
    if not style:
        return
    if "font_name" in style:
        plt.rcParams["font.family"] = str(style["font_name"])
    if "font_size" in style:
        plt.rcParams["font.size"] = float(style["font_size"])


def resolve_report_path(export_dir: Path, template_path: Path | None) -> Path:
    """Definiert den Zielpfad für den PDF-Report im PDF-Unterordner."""
    base = Path(export_dir).parent
    target_dir = base / "PDF"
    target_dir.mkdir(parents=True, exist_ok=True)
    if template_path is not None:
        return target_dir / f"Simulation_Report_{template_path.stem}.pdf"
    return target_dir / "Simulation_Report.pdf"


def build_result_tables(ts: pd.DataFrame, summary: pd.DataFrame, inputs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Erzeugt deutsche Berichtstabellen für Kennzahlen sowie Erlöse/Kosten."""
    storage_cost = _get_float_from_table(summary, "metric", "storage_cost", "value")
    horizon_years = _get_float_from_table(inputs, "parameter", "horizon_years", "value")
    annualized_storage_capex = storage_cost / horizon_years if horizon_years > 0 else storage_cost

    revenue_cost = pd.DataFrame(
        {
            "Kategorie": [
                "Erlös PV->Netz",
                "Erlös BATT->Netz",
                "Erlös PV->Last",
                "Erlös BATT->Last",
                "Kosten Opp. Laden",
                "Kosten OPEX",
                "Kosten Zähler",
                "Kosten DV PV",
                "Kosten Vermarkter BATT",
                "Kosten CAPEX/Jahr",
            ],
            "EUR_pro_Jahr": [
                float(ts["rev_pv_feed_in_eur"].sum()),
                float(ts["rev_batt_feed_in_eur"].sum()),
                float(ts["rev_pv_to_load_eur"].sum()),
                float(ts["rev_batt_to_load_eur"].sum()),
                -float(ts["cost_charge_opportunity_eur"].sum()),
                -float(ts["cost_storage_opex_eur"].sum()),
                -float(ts["cost_meter_eur"].sum()),
                -float(ts["cost_pv_direct_marketing_eur"].sum()),
                -float(ts["cost_battery_marketer_eur"].sum()),
                -float(annualized_storage_capex),
            ],
        }
    )

    key_metrics = summary[summary["metric"].isin([
        "E_opt",
        "P_opt",
        "annual_delta",
        "total_delta",
        "storage_cost",
        "objective",
    ])].copy()
    key_metrics.rename(columns={"metric": "Kennzahl", "value": "Wert"}, inplace=True)

    anlagengroesse_kwp = _get_float_from_table(inputs, "parameter", "pv_size_kwp", "value")
    projektdaten = pd.DataFrame(
        {
            "Feld": [
                "Projektstandort",
                "Adresse",
                "",
                "Anlagenparameter",
                "Anlagengröße in kWp",
                "Wechselrichter",
                "Module",
                "Unterkonstruktion",
                "Statische Prüfung Dachfläche",
                "Zusage Netzanschluss",
                "Wärmeentwicklung (WR)",
                "Geräuschentwicklung (WR) in dB",
                "",
                "Ansprechpartner",
                "empact",
                "Objektbetreuer (Kunde)",
            ],
            "Wert": [
                "",
                "",
                "",
                "Überschrift",
                f"{anlagengroesse_kwp:.2f}",
                "",
                "",
                "Aerocompact SN2plus",
                "Noch ausstehend",
                "offen",
                "",
                "",
                "",
                "",
                "Ansprechpartner: \nTelefon: \nMail:",
                "Ansprechpartner: \nTelefon: \nMail:",
            ],
        }
    )

    cashflow_jaehrlich = build_yearly_cashflow_table(summary, inputs)

    return {
        "tabelle_projektdaten": projektdaten,
        "tabelle_erloese_kosten": revenue_cost,
        "tabelle_kennzahlen": key_metrics,
        "tabelle_cashflow_jaehrlich": cashflow_jaehrlich,
    }


def show_and_export_tables(export_dir: Path, tables: dict[str, pd.DataFrame], print_to_console: bool = True):
    """Zeigt Tabellen im Terminal und schreibt sie als CSV in report_tables/."""
    table_dir = Path(export_dir) / "report_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df_export = df.copy()
        num_cols = df_export.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df_export[num_cols] = df_export[num_cols].round(2)
        if print_to_console:
            print(f"\n=== {name} ===")
            print(df_export.to_string(index=False))
        df_export.to_csv(table_dir / f"{name}.csv", sep=";", decimal=",", index=False)


def _add_df_table_page(pdf: PdfPages, title: str, df: pd.DataFrame):
    """Fügt eine linkbündige Tabelle als PDF-Seite hinzu."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="upper left", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)

    for (_, _), cell in tbl.get_celld().items():
        cell.set_text_props(ha="left")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _extract_logo_from_docx(template_path: Path | None):
    """Extrahiert möglichst ein Logo-Bild aus der DOCX-Vorlage."""
    if template_path is None or template_path.suffix.lower() != ".docx" or not template_path.exists():
        return None
    try:
        with zipfile.ZipFile(template_path, "r") as zf:
            media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
            if not media_files:
                return None
            media_files.sort(key=lambda n: ("logo" not in n.lower(), len(n)))
            blob = zf.read(media_files[0])
        return mpimg.imread(io.BytesIO(blob))
    except Exception:
        return None


def _extract_logo_image(export_dir: Path, template_path: Path | None):
    """Lädt ein Logo-Bild aus PDF/ (bevorzugt Dateinamen mit 'logo'), fallback: DOCX-Logo."""
    pdf_dir = Path(export_dir).parent / "PDF"
    if pdf_dir.exists() and pdf_dir.is_dir():
        img_candidates: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.gif"):
            img_candidates.extend(pdf_dir.glob(ext))
        if img_candidates:
            img_candidates.sort(key=lambda p: ("logo" not in p.name.lower(), len(p.name)))
            try:
                return mpimg.imread(img_candidates[0])
            except Exception:
                pass

    return _extract_logo_from_docx(template_path)


def _safe_get_float(df: pd.DataFrame, key_col: str, key: str, value_col: str, default: float = 0.0) -> float:
    try:
        return _get_float_from_table(df, key_col, key, value_col)
    except Exception:
        return default


def build_yearly_cashflow_table(summary: pd.DataFrame, inputs: pd.DataFrame) -> pd.DataFrame:
    """Erstellt eine Batterie-Jahres-Cashflow-Tabelle (Jahr 1..N) inkl. Degradation.

    Wichtig: Nur Batterie-Perspektive.
    - Keine PV-Erlöse (PV->Netz, PV->Last)
    - Keine Kosten der PV-Direktvermarktung
    """
    horizon_years = int(_safe_get_float(inputs, "parameter", "horizon_years", "value", 15.0))
    horizon_years = max(1, horizon_years)

    annual_battery_revenue = _safe_get_float(summary, "metric", "annual_battery_revenue_eur", "value", 0.0)
    annual_storage_opex = _safe_get_float(summary, "metric", "annual_storage_opex_eur", "value", 0.0)
    annual_meter_cost = _safe_get_float(summary, "metric", "annual_meter_cost_eur", "value", 0.0)
    annual_battery_marketer_cost = _safe_get_float(summary, "metric", "annual_battery_marketer_cost_eur", "value", 0.0)
    storage_cost = _safe_get_float(summary, "metric", "storage_cost", "value", 0.0)
    retention = _safe_get_float(summary, "metric", "batt_capacity_retention_after_horizon", "value", 1.0)

    retention = min(1.0, max(0.0, retention))
    if horizon_years == 1:
        yearly_factor = np.array([retention])
    else:
        yearly_factor = np.linspace(1.0, retention, horizon_years)

    years = np.arange(1, horizon_years + 1)
    # Batterie-Erlöse skalieren mit Degradation
    erlöse_batterie_jaehrlich = annual_battery_revenue * yearly_factor
    # Kosten Batterie: OPEX + Zähler (fix) + Vermarkter Batterie (degradationsabhängig)
    kosten_batterie_jaehrlich = (
        annual_storage_opex
        + annual_meter_cost
        + annual_battery_marketer_cost * yearly_factor
    )
    netto_batterie_jaehrlich = erlöse_batterie_jaehrlich - kosten_batterie_jaehrlich
    kum_netto_batterie = np.cumsum(netto_batterie_jaehrlich)
    kum_netto_nach_capex = kum_netto_batterie - storage_cost

    return pd.DataFrame(
        {
            "Jahr": years,
            "Faktor_Degradation": yearly_factor,
            "Batterie_Erloese_EUR": erlöse_batterie_jaehrlich,
            "Batterie_Kosten_EUR": kosten_batterie_jaehrlich,
            "Batterie_Netto_Jahr_EUR": netto_batterie_jaehrlich,
            "Kumulierter_Batterie_Netto_EUR": kum_netto_batterie,
            "Kumulierter_Netto_nach_CAPEX_EUR": kum_netto_nach_capex,
        }
    )


def _add_report_cover(
    pdf: PdfPages,
    template_path: Path | None,
    export_dir: Path,
    inputs: pd.DataFrame,
    summary: pd.DataFrame,
):
    """Erstellt Titelseite mit Template-Info und optionalem Logo."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.text(0.02, 0.92, "Simulationsbericht", fontsize=22, fontweight="bold", color=GREEN)

    anlagengroesse_kwp = _safe_get_float(inputs, "parameter", "pv_size_kwp", "value", 0.0)
    bess_kwh = _safe_get_float(summary, "metric", "E_opt", "value", 0.0)

    if template_path is None:
        ax.text(0.02, 0.82, "Designvorlage: keine Word-Vorlage gefunden", fontsize=11)
    else:
        ax.text(0.02, 0.82, f"Designvorlage: {template_path.name}", fontsize=11)
        if template_path.suffix.lower() == ".doc":
            ax.text(
                0.02,
                0.76,
                "Hinweis: .doc erkannt. Für automatische Stilübernahme bitte als .docx speichern.",
                fontsize=10,
            )
        else:
            ax.text(0.02, 0.76, "Word-Vorlage erkannt (.docx).", fontsize=10)

    # Gewünschte Projektinfos auf Titelseite
    cover_text = (
        "Projektstandort: \n"
        "Adresse: \n\n"
        "Anlagenparameter\n"
        f"Anlagengröße in kWp: {anlagengroesse_kwp:,.2f}\n"
        f"Größe BESS in kWh: {bess_kwh:,.2f}\n"
        "Wärmeentwicklung (WR): \n"
        "Geräuschentwicklung (WR) in dB: \n\n"
        "Ansprechpartner\n"
        "empact Ansprechpartner:\n"
        "Telefon:\n"
        "Mail:\n\n"
        "Objektbetreuer (Kunde) Ansprechpartner:\n"
        "Telefon:\n"
        "Mail:"
    )
    ax.text(0.02, 0.68, cover_text, fontsize=11, va="top")

    logo_img = _extract_logo_image(export_dir, template_path)
    if logo_img is not None:
        logo_ax = fig.add_axes([0.68, 0.80, 0.28, 0.16])
        logo_ax.imshow(logo_img)
        logo_ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf_report(
    export_dir: Path,
    start_date: str,
    n_days: int,
    template_docx: Path | None = None,
    include_project_data_table: bool = True,
):
    """Erzeugt vollständigen PDF-Bericht mit Tabellen und sämtlichen Diagrammen."""
    ts, summary, inputs = load_exported_data(Path(export_dir))
    tables = build_result_tables(ts, summary, inputs)
    if not include_project_data_table:
        tables.pop("tabelle_projektdaten", None)
    resolved_template = resolve_template_path(Path(export_dir), template_docx)
    _apply_template_style_if_available(resolved_template)

    def _write_report_pages(pdf: PdfPages):
        _add_report_cover(pdf, resolved_template, Path(export_dir), inputs, summary)
        if "tabelle_projektdaten" in tables:
            _add_df_table_page(pdf, "Projekt- und Anlagenparameter", tables["tabelle_projektdaten"])
        _add_df_table_page(pdf, "Kennzahlen", tables["tabelle_kennzahlen"])
        _add_df_table_page(pdf, "Cashflow Jahr 1 bis Jahr N", tables["tabelle_cashflow_jaehrlich"])
        _add_df_table_page(pdf, "Erlöse und Kosten", tables["tabelle_erloese_kosten"])

        # Diagrammseiten (alle relevanten Diagramme explizit anhängen)
        fig = plot_soc(ts, start_date=start_date, n_days=n_days, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_avg_charge_discharge_by_hour(ts, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_avg_spot_price_by_hour(ts, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_batt_feed_in_price_summer_2weeks(ts, start_date=start_date, n_days=n_days, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_batt_feed_in_price_summer_2weeks(ts, start_date="2024-02-01", n_days=14, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_average_battery_prices_over_horizon(ts, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_energy_balance(ts, start_date=start_date, n_days=n_days, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_load_selfconsumption_feedin_bess_charge(ts, start_date=start_date, n_days=n_days, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_battery_discharge_split(ts, start_date=start_date, n_days=n_days, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_bess_revenue_costs_2weeks(ts, start_date=start_date, n_days=n_days, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_revenue_cost_comparison_bars(ts, summary, inputs, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_objective_cashflow_over_horizon(summary, inputs, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig = plot_discounted_cashflow_over_horizon(summary, inputs, discount_rate=0.06, show=False)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


    report_path = resolve_report_path(Path(export_dir), resolved_template)
    try:
        with PdfPages(report_path) as pdf:
            _write_report_pages(pdf)
    except PermissionError:
        ts_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_path.with_name(f"{report_path.stem}_{ts_suffix}{report_path.suffix}")
        with PdfPages(report_path) as pdf:
            _write_report_pages(pdf)
        print("Hinweis: Ursprüngliche PDF war gesperrt (evtl. in Acrobat geöffnet).")

    if resolved_template is not None:
        print(f"Word-Vorlage erkannt: {resolved_template}")
        if resolved_template.suffix.lower() == ".doc":
            print("Hinweis: .doc kann nicht automatisch gestylt werden. Für exakte Stilübernahme bitte .docx verwenden.")

    print(f"\nPDF-Bericht erstellt: {report_path}")


def plot_soc(ts: pd.DataFrame, start_date: str = "2024-07-01", n_days: int = 14, show: bool = True):
    w = _window(ts, start_date, n_days)
    fig = plt.figure(figsize=(14, 4))
    plt.fill_between(w["timestamp"], w["soc_kwh"], alpha=0.35, label="SOC", color=GREEN)
    plt.plot(w["timestamp"], w["soc_kwh"], linewidth=1.2, color=GRAY_1)
    plt.title(f"SOC über {n_days} Tage ab {start_date}")
    plt.xlabel("Datum")
    plt.ylabel("SOC [kWh]")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_avg_charge_discharge_by_hour(ts: pd.DataFrame, show: bool = True):
    tmp = ts.copy()
    tmp["hour"] = tmp["timestamp"].dt.hour
    grp = tmp.groupby("hour", as_index=False)[["ch_kwh", "dis_kwh"]].mean()

    # Gemittelte Preise je Tagesstunde (energiemengengewichtet)
    # Hinweis: Verkaufspreis nur für BATT -> Netz (nicht BATT -> Last)
    avg_sell_price = []
    avg_charge_price = []
    for h in range(24):
        hh = tmp[tmp["hour"] == h]

        sell_grid_weight = float(hh["sell_batt_kwh"].sum())
        if sell_grid_weight > 0:
            # Mengengewichteter Auspeicherpreis ins Netz:
            # (BATT->Netz Erlös) / (BATT->Netz Energiemenge)
            p_sell = float(hh["rev_batt_feed_in_eur"].sum() / sell_grid_weight)
        else:
            p_sell = np.nan

        ch_weight = float(hh["ch_kwh"].sum())
        if ch_weight > 0:
            # Mengengewichteter Einspeicherpreis:
            # Opportunitätskosten Laden / Ladeenergie
            p_charge = float(hh["cost_charge_opportunity_eur"].sum() / ch_weight)
        else:
            p_charge = np.nan

        avg_sell_price.append(p_sell)
        avg_charge_price.append(p_charge)

    grp["avg_sell_price_eur_per_kwh"] = avg_sell_price
    grp["avg_charge_price_eur_per_kwh"] = avg_charge_price

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    ax1.plot(grp["hour"], grp["ch_kwh"], marker="o", linewidth=2, label="Laden")
    ax1.plot(grp["hour"], grp["dis_kwh"], marker="o", linewidth=2, label="Entladen")
    ax1.set_xticks(np.arange(24))
    ax1.set_xlabel("Stunde des Tages")
    ax1.set_ylabel("Energie [kWh pro Stunde]")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        grp["hour"],
        grp["avg_sell_price_eur_per_kwh"],
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label="Ø Auspeicherpreis ins Netz",
    )
    ax2.plot(
        grp["hour"],
        grp["avg_charge_price_eur_per_kwh"],
        marker="^",
        linewidth=1.8,
        linestyle=":",
        label="Ø Einspeicherpreis (bei Ladung)",
    )
    ax2.set_ylabel("Preis [EUR/kWh]")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.title("Gemittelte Lade-/Entladezeiten inkl. gemittelter Netz-Auspeicher- und Einspeicherpreise")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_avg_spot_price_by_hour(ts: pd.DataFrame, show: bool = True):
    """Plottet den über alle Tage gemittelten Spotmarktpreis je Tagesstunde."""
    tmp = ts.copy()
    tmp["hour"] = tmp["timestamp"].dt.hour
    grp = tmp.groupby("hour", as_index=False)["price_eur_per_kwh"].mean()

    fig = plt.figure(figsize=(12, 4.8))
    plt.plot(grp["hour"], grp["price_eur_per_kwh"], marker="o", linewidth=2.0, label="Ø Spotmarktpreis")
    plt.axhline(0.0, color=GRAY_5, linewidth=1.0, linestyle="--")
    plt.xticks(np.arange(24))
    plt.xlabel("Stunde des Tages")
    plt.ylabel("Preis [EUR/kWh]")
    plt.title("Über den Tag gemittelter Spotmarktpreis je Stunde")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_batt_feed_in_price_summer_2weeks(
    ts: pd.DataFrame,
    start_date: str = "2024-07-01",
    n_days: int = 14,
    show: bool = True,
):
    """Plottet Spotpreis und BATT->Netz-Auspeicherpreis für ein Sommer-2-Wochen-Fenster."""
    w = _window(ts, start_date, n_days)

    # Nur Stunden mit Batterieeinspeisung ins Netz
    w_sell = w[w["sell_batt_kwh"] > 0].copy()

    # Mengengewichteter Auspeicherpreis im Fenster (nur BATT -> Netz)
    sell_kwh_window = float(w_sell["sell_batt_kwh"].sum())
    avg_out_price_window = float(w_sell["rev_batt_feed_in_eur"].sum() / sell_kwh_window) if sell_kwh_window > 0 else np.nan

    # Jahreswert zum Vergleich
    sell_kwh_year = float(ts["sell_batt_kwh"].sum())
    avg_out_price_year = float(ts["rev_batt_feed_in_eur"].sum() / sell_kwh_year) if sell_kwh_year > 0 else np.nan

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.plot(w["timestamp"], w["price_eur_per_kwh"], linewidth=1.2, alpha=0.7, label="Spotmarktpreis (alle Stunden)")

    if len(w_sell) > 0:
        sc = ax.scatter(
            w_sell["timestamp"],
            w_sell["price_eur_per_kwh"],
            c=w_sell["sell_batt_kwh"],
            cmap="Greens",
            s=30,
            alpha=0.9,
            label="Stunden mit BATT -> Netz (Farbe = kWh)",
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("BATT -> Netz [kWh]")

    if np.isfinite(avg_out_price_window):
        ax.axhline(
            avg_out_price_window,
            color=GRAY_5,
            linestyle="--",
            linewidth=1.5,
            label=f"Ø Auspeicherpreis Fenster: {avg_out_price_window:.3f} EUR/kWh",
        )
    if np.isfinite(avg_out_price_year):
        ax.axhline(
            avg_out_price_year,
            color=GRAY_3,
            linestyle=":",
            linewidth=1.5,
            label=f"Ø Auspeicherpreis Jahr: {avg_out_price_year:.3f} EUR/kWh",
        )

    ax.set_title(f"Ausspeicherpreise (BATT -> Netz): {n_days} Tage ab {start_date}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Preis [EUR/kWh]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_average_battery_prices_over_horizon(ts: pd.DataFrame, show: bool = True):
    """Mittlere Batterie-Preise über die gesamte Laufzeit."""
    w_dis = float(ts["dis_kwh"].sum())
    avg_price_batt_sell_total = float(
        (ts["rev_batt_feed_in_eur"] + ts["rev_batt_to_load_eur"]).sum() / w_dis
    ) if w_dis > 0 else np.nan

    w_grid = float(ts["sell_batt_kwh"].sum())
    avg_price_batt_to_grid = float(ts["rev_batt_feed_in_eur"].sum() / w_grid) if w_grid > 0 else np.nan

    w_load = float(ts["batt_to_load_kwh"].sum())
    avg_price_batt_to_load = float(ts["rev_batt_to_load_eur"].sum() / w_load) if w_load > 0 else np.nan

    w_charge = float(ts["ch_kwh"].sum())
    avg_charge_price = float(ts["cost_charge_opportunity_eur"].sum() / w_charge) if w_charge > 0 else np.nan

    labels = [
        "Ø Verkaufspreis gesamt\n(BATT-Entladung)",
        "Ø Verkaufspreis\nBatterie -> Netz",
        "Ø Verkaufspreis\nBatterie -> Last",
        "Ø Einspeicherpreis\n(Batterie-Ladung)",
    ]
    values = [avg_price_batt_sell_total, avg_price_batt_to_grid, avg_price_batt_to_load, avg_charge_price]

    fig = plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, alpha=0.85)
    plt.ylabel("Preis [EUR/kWh]")
    plt.title("Gemittelte Batterie-Preise über die Laufzeit")
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, values):
        if np.isfinite(v):
            plt.text(b.get_x() + b.get_width() / 2.0, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_energy_balance(ts: pd.DataFrame, start_date: str = "2024-07-01", n_days: int = 14, show: bool = True):
    w = _window(ts, start_date, n_days)
    x = w["timestamp"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(x, w["pv_gen_kwh"], color=GRAY_1, linewidth=1.2, label="PV-Erzeugung")
    axes[0].stackplot(
        x,
        w["pv_to_load_kwh"],
        w["sell_pv_kwh"],
        w["ch_kwh"],
        w["curtailment_kwh"],
        colors=[GREEN, GRAY_2, GRAY_3, GRAY_5],
        labels=["PV -> Last", "PV -> Netz", "PV -> Speicher", "Nicht-Einspeisung (Preis negativ)"],
        alpha=0.8,
    )
    axes[0].set_title("Aufteilung der PV-Erzeugung")
    axes[0].set_ylabel("Energie [kWh/h]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].stackplot(
        x,
        w["pv_to_load_kwh"],
        w["batt_to_load_kwh"],
        w["grid_import_kwh"],
        w["sell_pv_kwh"],
        w["sell_batt_kwh"],
        colors=[GREEN, GRAY_2, GRAY_5, GRAY_3, GRAY_4],
        labels=["PV -> Last", "Speicher -> Last", "Netzbezug", "PV -> Netz", "Speicher -> Netz"],
        alpha=0.85,
    )
    axes[1].plot(x, w["load_kwh"], color=GRAY_1, linewidth=1.0, label="Last")
    axes[1].set_title("Lastdeckung und Einspeisung")
    axes[1].set_xlabel("Datum")
    axes[1].set_ylabel("Energie [kWh/h]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    locator = mdates.DayLocator(interval=2)
    formatter = mdates.DateFormatter("%d.%m")
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(formatter)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_load_selfconsumption_feedin_bess_charge(ts: pd.DataFrame, start_date: str = "2024-07-01", n_days: int = 14, show: bool = True):
    w = _window(ts, start_date, n_days)
    x = w["timestamp"]
    self_consumption = w["pv_to_load_kwh"] + w["batt_to_load_kwh"]
    feed_in = w["sell_pv_kwh"] + w["sell_batt_kwh"]

    fig = plt.figure(figsize=(16, 5))
    plt.plot(x, w["load_kwh"], linewidth=1.8, label="Last")
    plt.plot(x, self_consumption, linewidth=1.8, label="Eigenverbrauch")
    plt.plot(x, feed_in, linewidth=1.8, label="Einspeisung")
    plt.plot(x, w["ch_kwh"], linewidth=1.8, label="Ladung BESS")
    plt.title("Last, Eigenverbrauch, Einspeisung und BESS-Ladung")
    plt.xlabel("Datum")
    plt.ylabel("Energie [kWh/h]")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_battery_discharge_split(ts: pd.DataFrame, start_date: str = "2024-07-01", n_days: int = 14, show: bool = True):
    w = _window(ts, start_date, n_days)
    x = w["timestamp"]

    total_to_grid = float(ts["sell_batt_kwh"].sum())
    total_to_load = float(ts["batt_to_load_kwh"].sum())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    axes[0].plot(x, w["sell_batt_kwh"], linewidth=1.8, label="Speicher -> Netz")
    axes[0].plot(x, w["batt_to_load_kwh"], linewidth=1.8, label="Speicher -> Last")
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
    if show:
        plt.show()
    return fig


def plot_cumulative_prices(ts: pd.DataFrame):
    plt.figure(figsize=(16, 5))
    x = ts["timestamp"]
    plt.fill_between(x, ts["cum_price_sum"], alpha=0.25, label="Kumulierter Preis (gesamt)")
    plt.fill_between(x, ts["cum_price_positive_sum"], alpha=0.20, label="Kumulierte positive Preise")
    plt.fill_between(x, ts["cum_price_negative_sum"], alpha=0.20, label="Kumulierte negative Preise")
    plt.plot(x, ts["cum_price_sum"], linewidth=1.5)
    plt.plot(x, ts["cum_price_positive_sum"], linewidth=1.2)
    plt.plot(x, ts["cum_price_negative_sum"], linewidth=1.2)
    plt.title("Kumulierte Spotmarktpreise über das Jahr")
    plt.xlabel("Datum")
    plt.ylabel("Σ Preis [€/kWh]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_revenues_and_costs(ts: pd.DataFrame, summary: pd.DataFrame, inputs: pd.DataFrame, show: bool = True):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    x = ts["timestamp"]

    storage_cost = _get_float_from_table(summary, "metric", "storage_cost", "value")
    horizon_years = _get_float_from_table(inputs, "parameter", "horizon_years", "value")
    annualized_storage_capex = storage_cost / horizon_years if horizon_years > 0 else storage_cost
    capex_per_step = np.full(len(ts), annualized_storage_capex / len(ts))
    cum_capex = np.cumsum(capex_per_step)

    axes[0].fill_between(x, ts["cum_rev_pv_feed_in_eur"], alpha=0.25, label="Erlös PV -> Netz")
    axes[0].fill_between(x, ts["cum_rev_batt_feed_in_eur"], alpha=0.25, label="Erlös Batterie -> Netz")
    axes[0].fill_between(x, ts["cum_rev_pv_to_load_eur"], alpha=0.25, label="Erlös PV -> Last")
    axes[0].fill_between(x, ts["cum_rev_batt_to_load_eur"], alpha=0.25, label="Erlös Batterie -> Last")
    axes[0].plot(x, ts["cum_revenue_total_eur"], label="Erlöse gesamt", linewidth=2.4, color=GREEN)
    axes[0].set_title("Kumulierte Erlöse nach Erlösart")
    axes[0].set_ylabel("EUR")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].fill_between(x, ts["cum_cost_charge_opportunity_eur"], alpha=0.25, label="Kosten Opportunität Laden")
    axes[1].fill_between(x, ts["cum_cost_storage_opex_eur"], alpha=0.25, label="Kosten OPEX Speicher")
    axes[1].fill_between(x, ts["cum_cost_meter_eur"], alpha=0.25, label="Kosten Zähler")
    axes[1].fill_between(x, ts["cum_cost_pv_direct_marketing_eur"], alpha=0.25, label="Kosten Direktvermarktung PV")
    axes[1].fill_between(x, ts["cum_cost_battery_marketer_eur"], alpha=0.25, label="Kosten Vermarkter Batterie")
    axes[1].fill_between(x, cum_capex, alpha=0.25, label="Kosten CAPEX/Jahr")

    cum_cost_total_with_capex = ts["cum_cost_total_eur"] + cum_capex
    cum_net_with_capex = ts["cum_revenue_total_eur"] - cum_cost_total_with_capex

    axes[1].plot(x, cum_cost_total_with_capex, label="Kosten gesamt (inkl. CAPEX/Jahr)", linewidth=2.4, color=GRAY_1)
    axes[1].plot(x, cum_net_with_capex, label="Netto (inkl. CAPEX/Jahr)", linewidth=2.0, color=GREEN)
    axes[1].set_title("Kumulierte Kosten nach Kostenart")
    axes[1].set_xlabel("Datum")
    axes[1].set_ylabel("EUR")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper left")

    locator = mdates.MonthLocator(interval=1)
    formatter = mdates.DateFormatter("%b")
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_bess_revenue_costs_2weeks(ts: pd.DataFrame, start_date: str = "2024-07-01", n_days: int = 14, show: bool = True):
    w = _window(ts, start_date, n_days)
    x = w["timestamp"]

    bess_rev_load = w["rev_batt_to_load_eur"]
    bess_rev_grid = w["rev_batt_feed_in_eur"]
    bess_costs = w["cost_charge_opportunity_eur"] + w["cost_battery_marketer_eur"] + w["cost_storage_opex_eur"]

    fig = plt.figure(figsize=(16, 6))
    plt.stackplot(
        x,
        bess_rev_load,
        bess_rev_grid,
        colors=[GREEN, GRAY_3],
        labels=["BESS Erlös -> Last", "BESS Erlös -> Netz"],
        alpha=0.65,
    )
    plt.fill_between(x, -bess_costs, 0, alpha=0.35, label="BESS Kosten (2 Wochen)", color=GRAY_1)
    plt.plot(x, bess_rev_load + bess_rev_grid - bess_costs, linewidth=1.8, color=GRAY_2, label="BESS Netto")
    plt.title(f"BESS Erlöse und Kosten als Fläche ({n_days} Tage ab {start_date})")
    plt.xlabel("Datum")
    plt.ylabel("EUR pro Stunde")
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_revenue_cost_comparison_bars(ts: pd.DataFrame, summary: pd.DataFrame, inputs: pd.DataFrame, show: bool = True):
    storage_cost = _get_float_from_table(summary, "metric", "storage_cost", "value")
    horizon_years = _get_float_from_table(inputs, "parameter", "horizon_years", "value")
    annualized_storage_capex = storage_cost / horizon_years if horizon_years > 0 else storage_cost

    revenue_labels = [
        "PV->Netz",
        "BATT->Netz",
        "PV->Last",
        "BATT->Last",
    ]
    revenue_values = [
        float(ts["rev_pv_feed_in_eur"].sum()),
        float(ts["rev_batt_feed_in_eur"].sum()),
        float(ts["rev_pv_to_load_eur"].sum()),
        float(ts["rev_batt_to_load_eur"].sum()),
    ]

    cost_labels = [
        "Opp. Laden",
        "OPEX",
        "Zähler",
        "DV PV",
        "Vermarkter BATT",
        "CAPEX/Jahr",
    ]
    cost_values = [
        float(ts["cost_charge_opportunity_eur"].sum()),
        float(ts["cost_storage_opex_eur"].sum()),
        float(ts["cost_meter_eur"].sum()),
        float(ts["cost_pv_direct_marketing_eur"].sum()),
        float(ts["cost_battery_marketer_eur"].sum()),
        float(annualized_storage_capex),
    ]

    labels = revenue_labels + cost_labels
    values = revenue_values + [-v for v in cost_values]
    colors = [GREEN, GRAY_1, GRAY_2, GRAY_3] + [GRAY_2, GRAY_3, GRAY_4, GRAY_5, GRAY_1, GRAY_4]

    fig = plt.figure(figsize=(14, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.85)
    plt.axhline(0, color=GRAY_1, linewidth=1.0)
    plt.title("Gegenüberstellung aller Erlös- und Kostenarten (Jahr)")
    plt.ylabel("EUR")
    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2.0, v, f"{v:,.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_objective_cashflow_over_horizon(summary: pd.DataFrame, inputs: pd.DataFrame, show: bool = True):
    """Zeigt Batterie-Jahres-Cashflows und Netto-Entwicklung über Jahr 1..N."""
    cf = build_yearly_cashflow_table(summary, inputs)

    years = cf["Jahr"].to_numpy()
    yearly_revenue = cf["Batterie_Erloese_EUR"].to_numpy()
    yearly_cost = cf["Batterie_Kosten_EUR"].to_numpy()
    yearly_delta = cf["Batterie_Netto_Jahr_EUR"].to_numpy()
    cum_net = cf["Kumulierter_Netto_nach_CAPEX_EUR"].to_numpy()

    fig, ax1 = plt.subplots(figsize=(14, 6))
    width = 0.28
    bars_revenue = ax1.bar(years - width, yearly_revenue, width=width, label="Batterie-Erlöse/Jahr")
    bars_cost = ax1.bar(years, -yearly_cost, width=width, label="Batterie-Kosten/Jahr")
    bars_net = ax1.bar(years + width, yearly_delta, width=width, label="Batterie-Netto/Jahr")

    for bars in (bars_revenue, bars_cost, bars_net):
        for b in bars:
            v = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                v,
                f"{v:,.0f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
            )
    ax1.axhline(0.0, color=GRAY_5, linewidth=1.0)
    ax1.set_xlabel("Jahr")
    ax1.set_ylabel("EUR pro Jahr")
    ax1.set_xticks(years)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(years, cum_net, linewidth=2.4, marker="o", label="Kumuliert: Batterie-Netto nach CAPEX")
    ax2.set_ylabel("Kumuliert EUR")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.title("Jährliche Batterie-Cashflows Jahr 1 bis Jahr N")
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_discounted_cashflow_over_horizon(
    summary: pd.DataFrame,
    inputs: pd.DataFrame,
    discount_rate: float = 0.06,
    show: bool = True,
):
    """Zeigt abgezinste Batterie-Cashflows (NPV) über Jahr 1..N bei gegebener Diskontierung."""
    cf = build_yearly_cashflow_table(summary, inputs)
    storage_cost = _safe_get_float(summary, "metric", "storage_cost", "value", 0.0)

    years = cf["Jahr"].to_numpy(dtype=float)
    yearly_net = cf["Batterie_Netto_Jahr_EUR"].to_numpy(dtype=float)
    discount_factors = (1.0 + float(discount_rate)) ** years
    discounted_yearly_net = yearly_net / discount_factors
    cumulative_npv = -storage_cost + np.cumsum(discounted_yearly_net)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    bars = ax1.bar(years, discounted_yearly_net, width=0.65, alpha=0.85, label="Abgezinster Netto-Cashflow/Jahr")
    ax1.axhline(0.0, color=GRAY_5, linewidth=1.0)
    ax1.set_xlabel("Jahr")
    ax1.set_ylabel("Abgezinster Cashflow [EUR]")
    ax1.set_xticks(years)
    ax1.grid(axis="y", alpha=0.3)

    for b in bars:
        v = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            v,
            f"{v:,.0f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
        )

    ax2 = ax1.twinx()
    ax2.plot(years, cumulative_npv, linewidth=2.4, marker="o", label="Kumulierter NPV (inkl. CAPEX t0)")
    ax2.set_ylabel("NPV [EUR]")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.title(f"Batterie-Cashflow mit {discount_rate*100:.1f}% Diskontierung (IRR-Hürde)")
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def run_all_plots(
    export_dir: Path,
    start_date: str = "2024-07-01",
    n_days: int = 14,
    template_docx: Path | None = None,
    show_plots: bool = False,
    print_tables: bool = True,
    include_project_data_table: bool = True,
):
    ts, summary, inputs = load_exported_data(Path(export_dir))
    tables = build_result_tables(ts, summary, inputs)
    if not include_project_data_table:
        tables.pop("tabelle_projektdaten", None)
    show_and_export_tables(export_dir, tables, print_to_console=print_tables)

    fig = plot_soc(ts, start_date=start_date, n_days=n_days, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_avg_charge_discharge_by_hour(ts, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_avg_spot_price_by_hour(ts, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_batt_feed_in_price_summer_2weeks(ts, start_date=start_date, n_days=n_days, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_batt_feed_in_price_summer_2weeks(ts, start_date="2024-02-01", n_days=14, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_average_battery_prices_over_horizon(ts, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_energy_balance(ts, start_date=start_date, n_days=n_days, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_load_selfconsumption_feedin_bess_charge(ts, start_date=start_date, n_days=n_days, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_battery_discharge_split(ts, start_date=start_date, n_days=n_days, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_bess_revenue_costs_2weeks(ts, start_date=start_date, n_days=n_days, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_revenue_cost_comparison_bars(ts, summary, inputs, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_objective_cashflow_over_horizon(summary, inputs, show=show_plots)
    if not show_plots:
        plt.close(fig)

    fig = plot_discounted_cashflow_over_horizon(summary, inputs, discount_rate=0.06, show=show_plots)
    if not show_plots:
        plt.close(fig)

    generate_pdf_report(
        export_dir,
        start_date=start_date,
        n_days=n_days,
        template_docx=template_docx,
        include_project_data_table=include_project_data_table,
    )


def main():
    parser = argparse.ArgumentParser(description="Plots für exportierte Speichersimulation")
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "exports",
        help="Ordner mit timeseries_export.csv, summary_export.csv, inputs_export.csv",
    )
    parser.add_argument("--start-date", type=str, default="2024-07-01", help="Startdatum für 14-Tage-Fenster")
    parser.add_argument("--n-days", type=int, default=14, help="Anzahl Tage im Fenster")
    parser.add_argument("--template-docx", type=Path, default=None, help="Optionales Word-Muster (Infohinweis)")
    parser.add_argument("--show-plots", action="store_true", help="Diagramme interaktiv anzeigen (sonst nur erzeugen/speichern)")
    args = parser.parse_args()

    run_all_plots(
        args.export_dir,
        start_date=args.start_date,
        n_days=args.n_days,
        template_docx=args.template_docx,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
