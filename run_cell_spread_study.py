from copy import deepcopy

from src.config import load_params
from src.scenarios import load_scenarios
from src.sim_runner import run_scenario
from src.metrics import compute_operational_metrics


def main():
    base_params = load_params()
    dt_s = float(base_params["simulation"]["dt_s"])

    # Spread seviyeleri
    spreads = [0.0, 5.0, 10.0]

    # Örnek senaryolar: nominal discharge ve UV fault
    scenarios = load_scenarios()
    scenarios_by_id = {s.id: s for s in scenarios}
    scenario_ids = ["S1_nominal_25C_1C", "S5_UV_fault_25C"]

    print("Cell parameter spread study (q & r0, ±pct)\n")

    for sp in spreads:
        print(f"=== Spread = ±{sp:.1f}% ===")
        params = deepcopy(base_params)
        params.setdefault("cell_variation", {})
        params["cell_variation"]["q_spread_pct"] = sp
        params["cell_variation"]["r0_spread_pct"] = sp

        for sid in scenario_ids:
            sc = scenarios_by_id[sid]
            result = run_scenario(sc, params)
            oper = compute_operational_metrics(result, dt_s)
            print(
                f"  {sid}: "
                f"energy={oper['energy_discharge_kwh']:.1f} kWh, "
                f"soc_delta={oper['soc_delta']:.3f}"
            )
        print()

if __name__ == "__main__":
    main()
