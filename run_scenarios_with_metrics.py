from src.config import load_params
from src.scenarios import load_scenarios
from src.sim_runner import run_scenario
from src.metrics import (
    compute_safety_metrics,
    compute_operational_metrics,
    compute_estimation_metrics,
)


def fmt(x, nd=3):
    if x is None:
        return "None"
    return f"{x:.{nd}f}"


def main():
    params = load_params()
    dt_s = float(params["simulation"]["dt_s"])

    scenarios = load_scenarios()

    for sc in scenarios:
        print("\n" + "=" * 60)
        print(f"Scenario {sc.id}: {sc.description}")

        result = run_scenario(sc, params)

        safety = compute_safety_metrics(result, dt_s)
        oper = compute_operational_metrics(result, dt_s)
        est = compute_estimation_metrics(result)

        print("\nSafety metrics:")
        print(f"  t_fault_any_s  : {fmt(safety['t_fault_any_s'])}")
        print(f"  t_emergency_s  : {fmt(safety['t_emergency_s'])}")
        print(f"  t_oc_s         : {fmt(safety['t_oc_s'])}")
        print(f"  t_ov_s         : {fmt(safety['t_ov_s'])}")
        print(f"  t_uv_s         : {fmt(safety['t_uv_s'])}")
        print(f"  t_ot_s         : {fmt(safety['t_ot_s'])}")
        print(f"  t_fire_s       : {fmt(safety['t_fire_s'])}")

        print("\nOperational / thermal metrics:")
        print(f"  energy_discharge_kwh : {fmt(oper['energy_discharge_kwh'])}")
        print(f"  soc_delta            : {fmt(oper['soc_delta'])}")
        print(f"  t_total_s            : {fmt(oper['t_total_s'])}")
        print(f"  t_discharge_s        : {fmt(oper['t_discharge_s'])}")
        print(f"  t_max_c              : {fmt(oper['t_max_c'])}")
        print(f"  delta_t_c            : {fmt(oper['delta_t_c'])}")
        print(f"  energy_charge_kwh    : {fmt(oper.get('energy_charge_kwh'))}")
        print(f"  energy_net_kwh       : {fmt(oper.get('energy_net_kwh'))}")
        print(f"  energy_abs_kwh       : {fmt(oper.get('energy_abs_kwh'))}")
        print(f"  t_charge_s           : {fmt(oper.get('t_charge_s'))}")

        print("\nEstimation (SoC) metrics:")
        print(f"  rmse_soc             : {fmt(est['rmse_soc'], nd=5)}")
        print(f"  max_abs_error_soc    : {fmt(est['max_abs_error_soc'], nd=5)}")
        print(f"  p95_abs_error_soc    : {fmt(est['p95_abs_error_soc'], nd=5)}")


if __name__ == "__main__":
    main()
