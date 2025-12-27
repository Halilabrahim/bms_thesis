# run_manual_bms_scenario_cl06.py
"""
Manual BMS scenario runner (offline / standalone)

This version adds CL06 demonstration scenarios with two options:

CL06-A (recommended / deterministic):
- Keep a discharge request active while UV is injected, so you can clearly see:
  I_req stays >0, but I_act drops to 0 after FAULT.
- Then inject FIRE later to verify FAULT -> EMERGENCY_SHUTDOWN escalation.

CL06-B (more "realistic"):
- Try to provoke a *natural* UV by starting at low SoC and bypassing BMS limits,
  while also injecting FIRE later. Depending on your parameter set, natural UV
  may or may not occur before FIRE; the script prints a warning if UV never trips.

Usage (Windows / PyCharm terminal):
    python run_manual_bms_scenario_cl06.py --case A
    python run_manual_bms_scenario_cl06.py --case B
    python run_manual_bms_scenario_cl06.py --case both

Then plot with your dashboard:
    python run_bms_dashboard_v2.py results/manual/<file>.npz
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import os
import argparse

import numpy as np

from src.config import load_params
from src.sim_runner import build_models
from src.fsm import BMSState
from src.bms_logic import compute_current_limits
from src.metrics import (
    compute_safety_metrics,
    compute_operational_metrics,
    compute_estimation_metrics,
)


# ---------------------------------------------------------------------------
# Scenario description structures
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """
    A simple segment of constant current and ambient temperature.
    current_a > 0  -> discharge
    current_a < 0  -> charge
    """
    duration_s: float
    current_a: float
    t_amb_c: float


@dataclass
class FaultConfig:
    """
    Optional fault injections.
    All times are absolute simulation time (seconds from start).
    If a time is None, that fault is disabled.
    """
    uv_time_s: Optional[float] = None   # inject undervoltage at this time
    uv_v_fault: float = 2.0             # forced min cell voltage [V]

    ot_time_s: Optional[float] = None   # inject over-temp at this time
    ot_temp_c: float = 100.0            # forced rack temperature [°C]

    fire_time_s: Optional[float] = None  # gas / fire sensor alarm time [s]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _build_segment_bounds(segments: List[Segment]) -> Tuple[List[Tuple[float, float, Segment]], float]:
    seg_bounds: List[Tuple[float, float, Segment]] = []
    t_acc = 0.0
    for seg in segments:
        t_start = t_acc
        t_acc += float(seg.duration_s)
        t_end = t_acc
        seg_bounds.append((t_start, t_end, seg))
    return seg_bounds, t_acc


def _active_segment(seg_bounds: List[Tuple[float, float, Segment]], t_s: float) -> Segment:
    # default to last segment
    seg_active = seg_bounds[-1][2]
    for t_start, t_end, seg in seg_bounds:
        if t_start <= t_s < t_end:
            return seg
    return seg_active


def fmt(x, nd=3):
    if x is None:
        return "None"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_manual_profile(
    segments: List[Segment],
    faults: Optional[FaultConfig],
    params: Dict[str, Any],
    use_bms_limits: bool = True,
    true_init_soc: float = 1.0,
    ekf_init_soc: float = 1.0,
    include_initial_point: bool = True,
    # EKF update scheduling (for clearer convergence demos)
    measurement_update_start_s: float = 0.0,
    measurement_update_every_n: int = 1,
    # Optional initial polarization mismatch (RC voltages)
    true_init_v1_v: float = 0.0,
    true_init_v2_v: float = 0.0,
    ekf_init_v1_v: float = 0.0,
    ekf_init_v2_v: float = 0.0,
) -> Dict[str, List[float]]:

    if not segments:
        raise ValueError("At least one Segment must be defined.")

    # --- Build models (same core as run_scenario) ---
    models = build_models(params)
    ecm = models["ecm"]
    thermal = models["thermal"]
    ekf = models["ekf"]
    bms_params = models["bms_params"]
    fault_det = models["fault_det"]
    fsm = models["fsm"]
    dt_s = float(models["dt_s"])

    # --- Sensor noise config ---
    sensor_cfg = params.get("sensors", {})
    sigma_v = float(sensor_cfg.get("voltage_noise_std_v", 0.0))
    sigma_i = float(sensor_cfg.get("current_noise_std_a", 0.0))
    sigma_t = float(sensor_cfg.get("temp_noise_std_c", 0.0))
    seed = int(sensor_cfg.get("seed", 0))
    rng = np.random.default_rng(seed)

    # --- Segment timeline ---
    seg_bounds, total_time_s = _build_segment_bounds(segments)
    n_steps = int(np.ceil(total_time_s / dt_s))

    # --- Initial conditions ---
    true_init_soc = _clip01(true_init_soc)
    ekf_init_soc = _clip01(ekf_init_soc)

    ecm.reset(soc=true_init_soc)
    thermal.reset(t_init_c=float(segments[0].t_amb_c))
    ekf.reset(soc_init=ekf_init_soc)
    fault_det.reset()
    fsm.reset(BMSState.RUN)

    # Optional: set initial RC polarization voltages
    ecm.v1 = float(true_init_v1_v)
    ecm.v2 = float(true_init_v2_v)
    ekf.x[1, 0] = float(ekf_init_v1_v)
    ekf.x[2, 0] = float(ekf_init_v2_v)

    soc_hat = ekf.get_soc()

    # Initialize ECM output at t=0
    res = ecm.step(0.0, 0.0)
    t_rack_c = thermal.t_c

    state_code_map = {
        BMSState.OFF: 0,
        BMSState.RUN: 1,
        BMSState.FAULT: 2,
        BMSState.EMERGENCY_SHUTDOWN: 3,
    }

    # --- Logs ---
    times: List[float] = []
    i_req_list: List[float] = []
    i_act_list: List[float] = []
    soc_true: List[float] = []
    soc_hat_list: List[float] = []
    v_rack_true_list: List[float] = []
    v_rack_meas_list: List[float] = []
    t_rack_true_list: List[float] = []
    t_rack_meas_list: List[float] = []
    i_meas_list: List[float] = []
    state_codes: List[int] = []
    oc_flags: List[bool] = []
    ov_flags: List[bool] = []
    uv_flags: List[bool] = []
    ot_flags: List[bool] = []
    fire_flags: List[bool] = []

    v_cell_min_list: List[float] = []
    v_cell_max_list: List[float] = []
    soc_cell_min_list: List[float] = []
    soc_cell_max_list: List[float] = []
    soc_cell_mean_list: List[float] = []

    soc_hat_pred_list: List[float] = []  # EKF after predict, before update

    def _log_sample(
        t_s: float,
        i_req: float,
        i_act: float,
        res_dict: Dict[str, float],
        v_meas_v: float,
        i_meas_a: float,
        t_true_c: float,
        t_meas_c: float,
        flags: Dict[str, bool],
        state_code: int,
        soc_hat_pred: Optional[float] = None
    ) -> None:
        times.append(float(t_s))
        i_req_list.append(float(i_req))
        i_act_list.append(float(i_act))
        soc_true.append(float(res_dict["soc"]))
        soc_hat_list.append(float(ekf.get_soc()))
        v_rack_true_list.append(float(res_dict["v_rack"]))
        v_rack_meas_list.append(float(v_meas_v))
        t_rack_true_list.append(float(t_true_c))
        t_rack_meas_list.append(float(t_meas_c))
        i_meas_list.append(float(i_meas_a))
        state_codes.append(int(state_code))

        oc_flags.append(bool(flags["oc"]))
        ov_flags.append(bool(flags["ov"]))
        uv_flags.append(bool(flags["uv"]))
        ot_flags.append(bool(flags["ot"]))
        fire_flags.append(bool(flags["fire"]))

        v_cell_min_list.append(float(res_dict.get("v_cell_min", np.nan)))
        v_cell_max_list.append(float(res_dict.get("v_cell_max", np.nan)))
        soc_cell_min_list.append(float(res_dict.get("soc_cell_min", res_dict["soc"])))
        soc_cell_max_list.append(float(res_dict.get("soc_cell_max", res_dict["soc"])))
        soc_cell_mean_list.append(float(res_dict.get("soc_cell_mean", res_dict["soc"])))

        soc_hat_pred_list.append(float(ekf.get_soc() if soc_hat_pred is None else soc_hat_pred))

    # Optional t=0 sample
    if include_initial_point:
        flags0 = {"oc": False, "ov": False, "uv": False, "ot": False, "fire": False}
        v_meas0 = float(res["v_rack"] + (rng.normal(0.0, sigma_v) if sigma_v > 0 else 0.0))
        i_meas0 = float(0.0 + (rng.normal(0.0, sigma_i) if sigma_i > 0 else 0.0))
        t_meas0 = float(t_rack_c + (rng.normal(0.0, sigma_t) if sigma_t > 0 else 0.0))
        _log_sample(
            t_s=0.0,
            i_req=0.0,
            i_act=0.0,
            res_dict=res,
            v_meas_v=v_meas0,
            i_meas_a=i_meas0,
            t_true_c=t_rack_c,
            t_meas_c=t_meas0,
            flags=flags0,
            state_code=state_code_map[fsm.state],
            soc_hat_pred=ekf.get_soc(),
        )

    # Main loop
    for k in range(n_steps):
        t_prev = k * dt_s
        t_s = (k + 1) * dt_s

        seg = _active_segment(seg_bounds, t_prev)
        i_req = float(seg.current_a)
        t_amb_c = float(seg.t_amb_c)

        # ---- BMS selects actual current ----
        # Note: This uses *previous* FSM state (decision made at the start of the cycle).
        if fsm.state in (BMSState.EMERGENCY_SHUTDOWN, BMSState.FAULT):
            i_act = 0.0
        else:
            if use_bms_limits:
                curr_limits = compute_current_limits(
                    soc_hat=soc_hat,
                    t_rack_c=thermal.t_c,
                    v_cell_min=res["v_cell_min"],
                    v_cell_max=res["v_cell_max"],
                    params=bms_params,
                )
                if i_req >= 0.0:
                    i_act = min(i_req, float(curr_limits["i_discharge_max_allowed"]))
                else:
                    i_act = max(i_req, -float(curr_limits["i_charge_max_allowed"]))
            else:
                i_act = i_req

        # ---- EKF predict + true model update ----
        ekf.predict(i_act, dt_s)
        soc_hat_pred = ekf.get_soc()

        res = ecm.step(i_act, dt_s)
        t_rack_c = thermal.step(res["p_loss"], t_amb_c, dt_s)

        # ---- Measurements ----
        v_meas_v = float(res["v_rack"] + (rng.normal(0.0, sigma_v) if sigma_v > 0 else 0.0))
        i_meas_a = float(i_act + (rng.normal(0.0, sigma_i) if sigma_i > 0 else 0.0))
        t_meas_c = float(t_rack_c + (rng.normal(0.0, sigma_t) if sigma_t > 0 else 0.0))

        # ---- EKF update scheduling ----
        do_update = True
        if t_s < float(measurement_update_start_s):
            do_update = False
        if measurement_update_every_n > 1 and ((k + 1) % int(measurement_update_every_n) != 0):
            do_update = False
        if do_update:
            ekf.update(v_meas_v, i_meas_a)

        soc_hat = ekf.get_soc()

        # ---- Fault injection (for detector inputs) ----
        v_cell_min_used = float(res["v_cell_min"])
        v_cell_max_used = float(res["v_cell_max"])
        t_for_fault = float(t_rack_c)
        gas_alarm = False

        if faults is not None:
            if faults.uv_time_s is not None and t_s >= float(faults.uv_time_s):
                v_cell_min_used = min(v_cell_min_used, float(faults.uv_v_fault))
            if faults.ot_time_s is not None and t_s >= float(faults.ot_time_s):
                t_for_fault = max(t_for_fault, float(faults.ot_temp_c))
            if faults.fire_time_s is not None and t_s >= float(faults.fire_time_s):
                gas_alarm = True

        # ---- Fault detection + FSM update ----
        flags = fault_det.step(
            v_cell_min=v_cell_min_used,
            v_cell_max=v_cell_max_used,
            t_rack_c=t_for_fault,
            i_rack_a=i_meas_a,
            gas_alarm=gas_alarm,
        )
        state = fsm.step(flags, enable=True)

        # ---- Log ----
        _log_sample(
            t_s=t_s,
            i_req=i_req,
            i_act=i_act,
            res_dict={**res, "v_cell_min": v_cell_min_used, "v_cell_max": v_cell_max_used},
            v_meas_v=v_meas_v,
            i_meas_a=i_meas_a,
            t_true_c=t_rack_c,
            t_meas_c=t_meas_c,
            flags=flags,
            state_code=state_code_map[state],
            soc_hat_pred=soc_hat_pred,
        )

        if t_s >= total_time_s - 1e-9:
            break

    return {
        "time_s": times,
        "i_req_a": i_req_list,
        "i_act_a": i_act_list,
        "soc_true": soc_true,
        "soc_hat": soc_hat_list,
        "v_rack_v": v_rack_true_list,
        "v_rack_meas_v": v_rack_meas_list,
        "t_rack_c": t_rack_true_list,
        "t_rack_meas_c": t_rack_meas_list,
        "i_meas_a": i_meas_list,
        "state_code": state_codes,
        "oc": oc_flags,
        "ov": ov_flags,
        "uv": uv_flags,
        "ot": ot_flags,
        "fire": fire_flags,
        "v_cell_min_v": v_cell_min_list,
        "v_cell_max_v": v_cell_max_list,
        "soc_cell_min": soc_cell_min_list,
        "soc_cell_max": soc_cell_max_list,
        "soc_cell_mean": soc_cell_mean_list,
        "soc_hat_pred": soc_hat_pred_list,
    }


# ---------------------------------------------------------------------------
# CL06 scenario builders
# ---------------------------------------------------------------------------

def build_cl06_A():
    """
    Deterministic:
    - Continuous discharge request
    - Inject UV while I_req is still >0 (so I_act cut is visible)
    - Inject FIRE later to test FAULT -> EMERGENCY escalation
    """
    segments = [
        Segment(duration_s=4000.0, current_a=+250.0, t_amb_c=25.0),
    ]
    faults = FaultConfig(
        uv_time_s=1200.0,
        uv_v_fault=2.0,
        fire_time_s=1600.0,
    )
    run_kwargs = dict(
        use_bms_limits=True,
        true_init_soc=1.00,
        ekf_init_soc=1.00,
        include_initial_point=True,
    )
    out_name = "manual_CL06_A_UV_then_FIRE.npz"
    return segments, faults, run_kwargs, out_name


def build_cl06_B():
    """
    More realistic:
    - Try to provoke natural UV by low initial SoC + high discharge
    - BMS limits bypassed so the request is actually applied
    - FIRE injected late (so UV has a chance to happen first)
    """
    segments = [
        Segment(duration_s=2500.0, current_a=+320.0, t_amb_c=25.0),
    ]
    faults = FaultConfig(
        fire_time_s=2000.0,
    )
    run_kwargs = dict(
        use_bms_limits=False,
        true_init_soc=0.25,
        ekf_init_soc=0.25,
        include_initial_point=True,
    )
    out_name = "manual_CL06_B_natural_UV_plus_FIRE.npz"
    return segments, faults, run_kwargs, out_name


# ---------------------------------------------------------------------------
# Runner / metrics print
# ---------------------------------------------------------------------------

def run_and_save(case_label: str, segments: List[Segment], faults: Optional[FaultConfig],
                 run_kwargs: Dict[str, Any], out_dir: str, params: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, run_kwargs.pop("_out_name"))
    result = run_manual_profile(segments=segments, faults=faults, params=params, **run_kwargs)
    np.savez(out_path, **result)

    dt_s = float(params["simulation"]["dt_s"])
    safety = compute_safety_metrics(result, dt_s)
    oper = compute_operational_metrics(result, dt_s)
    est = compute_estimation_metrics(result)

    print("\n" + "=" * 70)
    print(f"Manual scenario {case_label} saved: {out_path}")

    print("\nSafety metrics:")
    print(f"  t_fault_any_s  : {fmt(safety.get('t_fault_any_s'))}")
    print(f"  t_emergency_s  : {fmt(safety.get('t_emergency_s'))}")
    print(f"  t_oc_s         : {fmt(safety.get('t_oc_s'))}")
    print(f"  t_ov_s         : {fmt(safety.get('t_ov_s'))}")
    print(f"  t_uv_s         : {fmt(safety.get('t_uv_s'))}")
    print(f"  t_ot_s         : {fmt(safety.get('t_ot_s'))}")
    print(f"  t_fire_s       : {fmt(safety.get('t_fire_s'))}")

    # Quick CL06 checks (non-fatal)
    if case_label.upper().endswith("A"):
        if safety.get("t_uv_s") is None:
            print("WARNING: UV did not trip in CL06-A (unexpected). Check fault injection wiring.")
        if safety.get("t_fire_s") is None:
            print("WARNING: FIRE did not trip in CL06-A (unexpected). Check fire_time_s / detector.")
        if safety.get("t_fire_s") is not None and safety.get("t_emergency_s") is None:
            print("WARNING: FIRE tripped but EMERGENCY did not. This suggests FSM does not escalate "
                  "FAULT -> EMERGENCY on FIRE. We can patch src/fsm.py if you confirm this.")

    if case_label.upper().endswith("B"):
        if safety.get("t_uv_s") is None:
            print("NOTE: Natural UV did not trip in CL06-B. This can happen depending on ECM + thresholds.")
            print("      If you want to force UV, use CL06-A; or set true_init_soc=0.15 and/or current=350A.")

    print("\nOperational / thermal metrics:")
    print(f"  energy_discharge_kwh : {fmt(oper.get('energy_discharge_kwh'))}")
    print(f"  energy_charge_kwh    : {fmt(oper.get('energy_charge_kwh'))}")
    print(f"  energy_net_kwh       : {fmt(oper.get('energy_net_kwh'))}")
    print(f"  energy_abs_kwh       : {fmt(oper.get('energy_abs_kwh'))}")
    print(f"  soc_delta            : {fmt(oper.get('soc_delta'))}")
    print(f"  t_total_s            : {fmt(oper.get('t_total_s'))}")
    print(f"  t_discharge_s        : {fmt(oper.get('t_discharge_s'))}")
    print(f"  t_charge_s           : {fmt(oper.get('t_charge_s'))}")
    print(f"  t_max_c              : {fmt(oper.get('t_max_c'))}")
    print(f"  delta_t_c            : {fmt(oper.get('delta_t_c'))}")

    print("\nEstimation (SoC) metrics:")
    print(f"  rmse_soc             : {fmt(est.get('rmse_soc'), nd=5)}")
    print(f"  max_abs_error_soc    : {fmt(est.get('max_abs_error_soc'), nd=5)}")
    print(f"  p95_abs_error_soc    : {fmt(est.get('p95_abs_error_soc'), nd=5)}")

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["A", "B", "both"], default="A",
                    help="Which CL06 scenario to run.")
    ap.add_argument("--outdir", default=os.path.join("results", "manual"),
                    help="Output directory for .npz results.")
    args = ap.parse_args()

    params = load_params()

    if args.case in ("A", "both"):
        segments, faults, run_kwargs, out_name = build_cl06_A()
        run_kwargs = dict(run_kwargs)  # copy
        run_kwargs["_out_name"] = out_name
        run_and_save("CL06-A", segments, faults, run_kwargs, args.outdir, params)

    if args.case in ("B", "both"):
        segments, faults, run_kwargs, out_name = build_cl06_B()
        run_kwargs = dict(run_kwargs)
        run_kwargs["_out_name"] = out_name
        run_and_save("CL06-B", segments, faults, run_kwargs, args.outdir, params)


if __name__ == "__main__":
    main()
