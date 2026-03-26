from dataclasses import dataclass, replace

from pyomo.environ import value

from Storage_Test import evaluate_results, load_timeseries, solve_model
from Storage_Test_NPV import InputsNPV, build_model_npv, npv_multiplier


@dataclass
class InputsCapexBreakEven(InputsNPV):
    """Eingaben für Break-even-CAPEX (NPV = target_npv_eur)."""

    # Wunsch-Eingaben
    input_batt_E_kwh: float = 225.0
    input_batt_P_kw: float = 110.0
    pv_size_kwp: float = 1000.0
    annual_consumption_kwh: float = 600000.0

    # Zielwert (standard: NPV = 0)
    target_npv_eur: float = 0.0

    # Dieses Tool ist Einzelrechnung, kein PV-Sweep
    run_pv_size_sweep: bool = False


def _validate(inp: InputsCapexBreakEven):
    if inp.input_batt_E_kwh <= 0:
        raise ValueError("input_batt_E_kwh muss > 0 sein.")
    if inp.input_batt_P_kw <= 0:
        raise ValueError("input_batt_P_kw muss > 0 sein.")
    if inp.pv_size_kwp <= 0:
        raise ValueError("pv_size_kwp muss > 0 sein.")
    if inp.annual_consumption_kwh <= 0:
        raise ValueError("annual_consumption_kwh muss > 0 sein.")


def _simulate(inp: InputsCapexBreakEven):
    prices, pv_per_kwp, load = load_timeseries(inp)
    model = build_model_npv(prices, pv_per_kwp, load, inp)
    solve_model(model, inp)
    results = evaluate_results(model, prices, pv_per_kwp, load, inp)
    objective_npv = float(value(model.obj))
    return results, objective_npv


def main():
    inp = InputsCapexBreakEven()
    _validate(inp)

    # 1) Referenzlauf mit CAPEX=0 zur Ermittlung des maximal möglichen jährlichen Mehrwerts
    inp_zero_capex = replace(
        inp,
        fixed_batt_E_kwh=inp.input_batt_E_kwh,
        fixed_batt_P_kw=inp.input_batt_P_kw,
        capex_batt_E_eur_per_kwh=0.0,
    )
    res0, obj0 = _simulate(inp_zero_capex)

    E = inp.input_batt_E_kwh
    npv_mult = npv_multiplier(inp_zero_capex)
    opex_pct = inp.storage_opex_pct_of_invest_per_year

    # annual_delta(capex) = annual_delta(0) - opex_pct * capex * E
    # objective_npv(capex) = npv_mult * annual_delta(capex) - capex * E
    # => objective_npv(capex) = npv_mult*annual_delta(0) - capex*E*(1 + npv_mult*opex_pct)
    denom = E * (1.0 + npv_mult * opex_pct)
    capex_break_even = (npv_mult * float(res0["annual_delta"]) - inp.target_npv_eur) / denom

    print("\n-----------------------------------")
    print("BREAK-EVEN CAPEX TOOL (NPV-Ziel)")
    print("-----------------------------------")
    print(f"PV-Größe:                          {inp.pv_size_kwp:,.2f} kWp")
    print(f"Verbrauch:                         {inp.annual_consumption_kwh:,.0f} kWh/a")
    print(f"Vorgegebene Batteriegröße E:       {inp.input_batt_E_kwh:,.2f} kWh")
    print(f"Vorgegebene Batterieleistung P:    {inp.input_batt_P_kw:,.2f} kW")
    print(f"Diskontsatz:                       {100*inp.discount_rate:,.2f} %")
    print(f"NPV-Ziel:                          {inp.target_npv_eur:,.2f} €")
    print(f"NPV-Multiplikator:                 {npv_mult:,.4f}")
    print(f"Jährlicher Mehrwert bei CAPEX=0:   {res0['annual_delta']:,.2f} € /a")
    print(f"NPV bei CAPEX=0:                   {obj0:,.2f} €")
    print("-----------------------------------")
    print(f"Ermittelter Break-even CAPEX:      {capex_break_even:,.2f} € /kWh")

    if capex_break_even < 0:
        print("Hinweis: Break-even CAPEX ist negativ. Das bedeutet: selbst bei 0 €/kWh wird das NPV-Ziel nicht erreicht.")
        return

    # 2) Plausibilitätscheck mit berechnetem CAPEX
    inp_check = replace(inp_zero_capex, capex_batt_E_eur_per_kwh=float(capex_break_even))
    _, obj_check = _simulate(inp_check)
    print(f"Plausibilitätscheck NPV bei Break-even CAPEX: {obj_check:,.4f} €")


if __name__ == "__main__":
    main()
