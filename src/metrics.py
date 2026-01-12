from typing import Dict, List, Any, Optional
import math


def _first_true_time(time_s: List[float], flags: List[bool]) -> Optional[float]:
    """Return first time where flag is True, or None if never."""
    for t, f in zip(time_s, flags):
        if f:
            return float(t)
    return None


def compute_safety_metrics(
    result: Dict[str, List[Any]],
    dt_s: float,
) -> Dict[str, Optional[float]]:
    """
    Safety-related metrics:
      - first fault time (any of OC/OV/UV/OT/UT/FIRE)
      - first emergency shutdown time (state_code == 3)
      - individual fault detection times
    """
    time_s = result.get("time_s", [])
    state_code = result.get("state_code", [])

    # Fault flags (default to all-False if key missing)
    n = len(time_s) if time_s else 0
    oc = result.get("oc", [False] * n)
    ov = result.get("ov", [False] * n)
    uv = result.get("uv", [False] * n)
    ot = result.get("ot", [False] * n)
    ut = result.get("ut", [False] * n)
    fire = result.get("fire", [False] * n)

    t_oc = _first_true_time(time_s, oc)
    t_ov = _first_true_time(time_s, ov)
    t_uv = _first_true_time(time_s, uv)
    t_ot = _first_true_time(time_s, ot)
    t_ut = _first_true_time(time_s, ut)
    t_fire = _first_true_time(time_s, fire)

    any_fault_flags = [
        (o or v or u or tt or uu or ff)
        for o, v, u, tt, uu, ff in zip(oc, ov, uv, ot, ut, fire)
    ]
    t_fault_any = _first_true_time(time_s, any_fault_flags)

    emergency_flags = [code == 3 for code in state_code]
    t_emergency = _first_true_time(time_s, emergency_flags)

    return {
        "t_fault_any_s": t_fault_any,
        "t_emergency_s": t_emergency,
        "t_oc_s": t_oc,
        "t_ov_s": t_ov,
        "t_uv_s": t_uv,
        "t_ot_s": t_ot,
        "t_ut_s": t_ut,
        "t_fire_s": t_fire,
    }


def compute_operational_metrics(
    result: Dict[str, List[float]],
    dt_s: float,
) -> Dict[str, Optional[float]]:
    time_s = result.get("time_s", [])
    i_act = result.get("i_act_a", [])
    v_rack = result.get("v_rack_v", [])
    soc_true = result.get("soc_true", [])
    t_rack = result.get("t_rack_c", [])

    if (not time_s) or (not i_act) or (not v_rack) or (not soc_true) or (not t_rack):
        return {
            "energy_discharge_kwh": None,
            "energy_charge_kwh": None,
            "energy_net_kwh": None,
            "energy_abs_kwh": None,
            "soc_delta": None,
            "t_total_s": None,
            "t_discharge_s": None,
            "t_charge_s": None,
            "t_max_c": None,
            "t_rack_max_c": None,
            "delta_t_c": None,
            "v_rack_min_v": None,
            "v_rack_max_v": None,
            "i_rack_min_a": None,
            "i_rack_max_a": None,
        }

    t_total_s = (time_s[-1] - time_s[0]) + float(dt_s)

    energy_discharge_j = 0.0
    energy_charge_j = 0.0
    energy_net_j = 0.0
    energy_abs_j = 0.0

    discharge_time_s = 0.0
    charge_time_s = 0.0

    for i, v in zip(i_act, v_rack):
        p = float(v) * float(i)  # W (signed)
        energy_net_j += p * dt_s
        energy_abs_j += abs(p) * dt_s

        if i > 0.0:
            energy_discharge_j += p * dt_s
            discharge_time_s += dt_s
        elif i < 0.0:
            energy_charge_j += (-p) * dt_s  # positive magnitude
            charge_time_s += dt_s

    energy_discharge_kwh = energy_discharge_j / 3.6e6
    energy_charge_kwh = energy_charge_j / 3.6e6
    energy_net_kwh = energy_net_j / 3.6e6
    energy_abs_kwh = energy_abs_j / 3.6e6

    soc_delta = float(soc_true[0]) - float(soc_true[-1])

    t_rack_max_c = max(float(x) for x in t_rack)
    delta_t_c = t_rack_max_c - float(t_rack[0])

    v_rack_min_v = min(float(x) for x in v_rack)
    v_rack_max_v = max(float(x) for x in v_rack)

    i_rack_min_a = min(float(x) for x in i_act)
    i_rack_max_a = max(float(x) for x in i_act)

    return {
        "energy_discharge_kwh": energy_discharge_kwh,
        "energy_charge_kwh": energy_charge_kwh,
        "energy_net_kwh": energy_net_kwh,
        "energy_abs_kwh": energy_abs_kwh,
        "soc_delta": soc_delta,
        "t_total_s": t_total_s,
        "t_discharge_s": discharge_time_s,
        "t_charge_s": charge_time_s,
        "t_max_c": t_rack_max_c,
        "t_rack_max_c": t_rack_max_c,
        "delta_t_c": delta_t_c,
        "v_rack_min_v": v_rack_min_v,
        "v_rack_max_v": v_rack_max_v,
        "i_rack_min_a": i_rack_min_a,
        "i_rack_max_a": i_rack_max_a,
    }


def compute_estimation_metrics(
    result: Dict[str, List[float]],
) -> Dict[str, Optional[float]]:
    soc_true = result.get("soc_true", [])
    soc_hat = result.get("soc_hat", [])

    n = len(soc_true)
    if n == 0 or n != len(soc_hat):
        return {"rmse_soc": None, "max_abs_error_soc": None, "p95_abs_error_soc": None}

    errors = [hat - true for hat, true in zip(soc_hat, soc_true)]
    mse = sum(e * e for e in errors) / n
    rmse = math.sqrt(mse)

    abs_errors = [abs(e) for e in errors]
    max_abs_err = max(abs_errors)

    abs_errors_sorted = sorted(abs_errors)
    idx_95 = int(0.95 * (n - 1))
    p95_err = abs_errors_sorted[idx_95]

    return {"rmse_soc": rmse, "max_abs_error_soc": max_abs_err, "p95_abs_error_soc": p95_err}
