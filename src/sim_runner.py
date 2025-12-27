from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.battery_model import RackParams, RackECMModel
from src.thermal_model import ThermalParams, RackThermalModel
from src.estimator import SoCEstimatorEKF, EKFParams
from src.bms_logic import BMSControlParams, compute_current_limits
from src.faults import FaultThresholds, FaultDetector
from src.fsm import BMSStateMachine, BMSState
from src.scenarios import Scenario


def build_models(params: Dict[str, Any], soh_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Instantiate plant + estimator + protection blocks using YAML parameters.

    Notes
    -----
    - Physical plant (ECM) can be SoH-scaled (capacity + R0 multiplier).
    - EKF stays nominal (so estimation mismatch is possible and testable).
    """
    em = params["electrical_model"]
    struct = params["structure"]
    limits = params["limits"]
    therm = params["thermal_model"]
    ekf_cfg = params["estimation"]["ekf"]
    ctrl_cfg = params["bms_control"]
    faults_cfg = params["faults"]
    sim_cfg = params["simulation"]
    cell_var = params.get("cell_variation", {}) or {}

    # --- SoH scaling (physical model) ---
    soh_cfg = soh_profile or {}
    cap_rel = float(soh_cfg.get("capacity_rel", 1.0))
    r0_mult = float(soh_cfg.get("r0_mult", soh_cfg.get("r0_multiplier", 1.0)))

    q_nom_phys_ah = float(em["q_nom_ah"]) * cap_rel
    r0_phys_ohm = float(em["r0_ohm"]) * r0_mult

    rack_params = RackParams(
        q_nom_ah=q_nom_phys_ah,
        r0_ohm=r0_phys_ohm,
        r1_ohm=float(em["r1_ohm"]),
        c1_f=float(em["c1_f"]),
        r2_ohm=float(em["r2_ohm"]),
        c2_f=float(em["c2_f"]),
        n_packs=int(struct["packs_in_series_per_rack"]),
        cells_in_series_per_pack=int(struct["cells_in_series_per_pack"]),
        v_cell_min_v=float(limits["v_cell_min_v"]),
        v_cell_max_v=float(limits["v_cell_max_v"]),
        # cell variation
        cell_param_spread_q_pct=float(cell_var.get("q_spread_pct", 0.0)),
        cell_param_spread_r0_pct=float(cell_var.get("r0_spread_pct", 0.0)),
        cell_random_seed=int(cell_var.get("seed", 0)),
    )
    ecm = RackECMModel(rack_params)

    thermal_params = ThermalParams(
        c_th_j_per_k=float(therm["c_th_j_per_k"]),
        r_th_k_per_w=float(therm["r_th_k_per_w"]),
        t_init_c=float(therm["t_init_c"]),
    )
    thermal = RackThermalModel(thermal_params)

    # --- EKF (nominal values) ---
    n_series_cells = rack_params.n_packs * rack_params.cells_in_series_per_pack
    ekf_params = EKFParams(
        q_nom_ah=float(em["q_nom_ah"]),  # nominal
        r0_ohm=float(em["r0_ohm"]),      # nominal
        r1_ohm=float(em["r1_ohm"]),
        c1_f=float(em["c1_f"]),
        r2_ohm=float(em["r2_ohm"]),
        c2_f=float(em["c2_f"]),
        n_series_cells=n_series_cells,
        q_process=ekf_cfg["q_process"],
        r_meas_v=float(ekf_cfg["r_measurement"]["v_terminal"]),
        p0=ekf_cfg["initial_covariance"],
    )
    ekf = SoCEstimatorEKF(ekf_params, soc_init=1.0)

    # --- BMS current limit params ---
    bms_params = BMSControlParams(
        i_charge_max_a=float(limits["i_charge_max_a"]),
        i_discharge_max_a=float(limits["i_discharge_max_a"]),
        soc_low_cutoff=float(ctrl_cfg["soc_low_cutoff"]),
        soc_low_derate_start=float(ctrl_cfg["soc_low_derate_start"]),
        soc_high_cutoff=float(ctrl_cfg["soc_high_cutoff"]),
        soc_high_derate_start=float(ctrl_cfg["soc_high_derate_start"]),
        t_low_cutoff_c=float(ctrl_cfg["t_low_cutoff_c"]),
        t_low_derate_start_c=float(ctrl_cfg["t_low_derate_start_c"]),
        t_high_cutoff_c=float(ctrl_cfg["t_high_cutoff_c"]),
        t_high_derate_start_c=float(ctrl_cfg["t_high_derate_start_c"]),
        v_cell_min_v=float(limits["v_cell_min_v"]),
        v_cell_max_v=float(limits["v_cell_max_v"]),
        v_margin_v=float(ctrl_cfg["v_margin_v"]),
    )

    # --- Fault thresholds ---
    # Backward/forward compatible: allow both faults.fire_detection and top-level fire_detection
    fire_cfg: Dict[str, Any] = {}
    if isinstance(faults_cfg.get("fire_detection", None), dict):
        fire_cfg.update(faults_cfg.get("fire_detection", {}) or {})
    if isinstance(params.get("fire_detection", None), dict):
        fire_cfg.update(params.get("fire_detection", {}) or {})

    debounce_steps = int(fire_cfg.get("debounce_steps", faults_cfg.get("debounce_steps", 3)))
    fire_temp_c = float(fire_cfg.get("temp_threshold_c", faults_cfg.get("fire_temp_c", 80.0)))
    fire_dTdt_c_per_s = float(fire_cfg.get("dTdt_threshold_c_per_s", faults_cfg.get("fire_dTdt_c_per_s", 0.5)))

    thr = FaultThresholds(
        ov_cell_v=float(faults_cfg["ov_cell_v"]),
        uv_cell_v=float(faults_cfg["uv_cell_v"]),
        ot_rack_c=float(faults_cfg["ot_rack_c"]),
        ut_rack_c=float(faults_cfg["ut_rack_c"]),
        oc_discharge_a=float(faults_cfg["oc_discharge_a"]),
        oc_charge_a=float(faults_cfg["oc_charge_a"]),
        fire_temp_c=float(fire_temp_c),
        fire_dTdt_c_per_s=float(fire_dTdt_c_per_s),
        debounce_steps=debounce_steps,
    )

    dt_s = float(sim_cfg["dt_s"])
    fault_det = FaultDetector(thr, dt_s)
    fsm = BMSStateMachine()

    return {
        "ecm": ecm,
        "thermal": thermal,
        "ekf": ekf,
        "bms_params": bms_params,
        "fault_det": fault_det,
        "fsm": fsm,
        "dt_s": dt_s,
    }


def _seg_get(seg: Any, key: str, default: Any = None) -> Any:
    """Safe getter for SegmentSpec / dataclass / SimpleNamespace or dict-like segments."""
    if seg is None:
        return default
    if hasattr(seg, key):
        return getattr(seg, key)
    if isinstance(seg, dict):
        return seg.get(key, default)
    return default


def _active_segment(segments: List[Any], t_s: float) -> Any:
    acc = 0.0
    for seg in segments:
        dur = float(_seg_get(seg, "duration_s", 0.0))
        if dur <= 0:
            continue
        if acc <= t_s < acc + dur:
            return seg
        acc += dur
    return segments[-1]


def _get_injection_value(scenario: Any, new_key: str, old_key: str, default: float) -> float:
    """
    Read a numeric injection value in a backward-compatible way.
    Prefer the new_key (current scenarios.py), else fallback to old_key.
    """
    v = getattr(scenario, new_key, None)
    if v is None:
        v = getattr(scenario, old_key, None)
    try:
        return float(default if v is None else v)
    except Exception:
        return float(default)


def _get_injection_duration_s(scenario: Any, key: str) -> float:
    """
    Optional duration support (seconds). Missing/None/<=0 => step (latched after start).
    """
    v = getattr(scenario, key, None)
    try:
        return float(0.0 if v is None else v)
    except Exception:
        return 0.0


def _pulse_active(t_s: float, start_s: Optional[float], duration_s: float) -> bool:
    """
    True if an injection is active at time t_s.
    - If start_s is None => False
    - If duration_s <= 0 => step/latched for t >= start
    - Else active in [start, start+duration)
    """
    if start_s is None:
        return False
    try:
        s0 = float(start_s)
    except Exception:
        return False
    if s0 <= 0:
        return False
    if duration_s <= 0:
        return t_s >= s0
    return (t_s >= s0) and (t_s < (s0 + float(duration_s)))


def run_scenario(scenario: Scenario, params: Dict[str, Any]) -> Dict[str, List[float]]:
    """
    Core offline runner used by scripts and Streamlit.

    Two layers of "inputs" exist:
    1) Plant physics (ECM + thermal) is driven by i_act and ambient temperature.
    2) BMS visibility (for limiting + fault detection) can be overridden per segment
       and/or via global time-based injections (UV/OV/OT/OC/FIRE).

    Segment overrides are sensor-level: they DO NOT force the plant state.
    """
    # ---- SoH parsing ----
    soh_cfg: Optional[Dict[str, Any]] = None
    raw_soh = getattr(scenario, "soh_profile", None)

    if isinstance(raw_soh, dict):
        soh_cfg = {
            "capacity_rel": float(raw_soh.get("capacity_rel", 1.0)),
            "r0_mult": float(raw_soh.get("r0_mult", raw_soh.get("r0_multiplier", 1.0))),
        }
    elif isinstance(raw_soh, str) and raw_soh:
        if raw_soh == "aged_0p8Q_1p5R0":
            soh_cfg = {"capacity_rel": 0.8, "r0_mult": 1.5}
        elif raw_soh in ("nominal", "none"):
            soh_cfg = None
        else:
            raise ValueError(f"Unknown soh_profile {raw_soh!r} for scenario {scenario.id}")

    # ---- Build models ----
    models = build_models(params, soh_profile=soh_cfg)
    ecm = models["ecm"]
    thermal = models["thermal"]
    ekf = models["ekf"]
    bms_params = models["bms_params"]
    fault_det = models["fault_det"]
    fsm = models["fsm"]
    dt_s = float(models["dt_s"])

    # ---- Sensor noise ----
    sensor_cfg = params.get("sensors", {}) or {}
    sigma_v = float(sensor_cfg.get("voltage_noise_std_v", 0.0))
    sigma_i = float(sensor_cfg.get("current_noise_std_a", 0.0))
    sigma_t = float(sensor_cfg.get("temp_noise_std_c", 0.0))
    rng_meas = np.random.default_rng(int(sensor_cfg.get("seed", 0)))

    # ---- Initial conditions ----
    true_init_soc = float(getattr(scenario, "true_init_soc", 1.0))
    ekf_init_soc = getattr(scenario, "ekf_init_soc", None)
    ekf_init_soc = float(true_init_soc if ekf_init_soc is None else ekf_init_soc)

    # ---- Simulation horizon ----
    max_time_s_cfg = float(getattr(scenario, "max_time_s", 3600.0))
    prof = getattr(scenario, "profile_type", "constant_current")

    if prof == "segments":
        segs = list(getattr(scenario, "segments", []) or [])
        if not segs:
            raise ValueError("profile_type='segments' but segments is empty")
        seg_total = float(sum(float(_seg_get(s, "duration_s", 0.0)) for s in segs))
        max_time_s = min(seg_total, max_time_s_cfg) if max_time_s_cfg > 0 else seg_total
    else:
        max_time_s = max_time_s_cfg

    # ---- Thermal reset temp ----
    if prof == "segments":
        first_seg = list(getattr(scenario, "segments", []) or [None])[0]
        t_init_c = float(_seg_get(first_seg, "t_amb_c", getattr(scenario, "t_amb_c", 25.0)))
    else:
        t_init_c = float(getattr(scenario, "t_amb_c", 25.0))

    ecm.reset(soc=true_init_soc)
    thermal.reset(t_init_c=t_init_c)
    ekf.reset(soc_init=ekf_init_soc)
    fault_det.reset()
    fsm.reset(BMSState.RUN)

    state_code_map = {
        BMSState.OFF: 0,
        BMSState.RUN: 1,
        BMSState.FAULT: 2,
        BMSState.EMERGENCY_SHUTDOWN: 3,
    }

    # ---- Global time-based injections ----
    uv_time = getattr(scenario, "uv_inject_time_s", None)
    ov_time = getattr(scenario, "ov_inject_time_s", None)
    oc_time = getattr(scenario, "oc_inject_time_s", None)
    ot_time = getattr(scenario, "ot_inject_time_s", None)
    fire_time = getattr(scenario, "fire_inject_time_s", None)

    # Optional durations (seconds). Missing/None/<=0 => step
    uv_dur_s = _get_injection_duration_s(scenario, "uv_inject_duration_s")
    ov_dur_s = _get_injection_duration_s(scenario, "ov_inject_duration_s")
    oc_dur_s = _get_injection_duration_s(scenario, "oc_inject_duration_s")
    ot_dur_s = _get_injection_duration_s(scenario, "ot_inject_duration_s")
    fire_dur_s = _get_injection_duration_s(scenario, "fire_inject_duration_s")

    uv_v_fault = _get_injection_value(scenario, "uv_v_cell_min_v", "uv_v_fault", 2.0)
    ov_v_fault = _get_injection_value(scenario, "ov_v_cell_max_v", "ov_v_fault", 3.65)
    oc_i_fault = _get_injection_value(scenario, "oc_i_rack_a", "oc_i_fault", 0.0)
    ot_temp_c = _get_injection_value(scenario, "ot_t_rack_c", "ot_temp_c", 100.0)

    # ---- logs ----
    hist: Dict[str, List[float]] = {k: [] for k in [
        "time_s", "i_req_a", "i_act_a",
        "soc_true", "soc_hat",
        "v_rack_v", "v_rack_meas_v",
        "t_rack_c", "t_rack_meas_c",
        "i_meas_a",
        "v_cell_min_v", "v_cell_max_v",
        "v_cell_min_true_v", "v_cell_max_true_v",
        "t_rack_used_c", "i_rack_used_a",
        "state_code",
        "oc", "ov", "uv", "ot", "fire",
        "soc_cell_min", "soc_cell_max", "soc_cell_mean",
    ]}

    n_steps = int(np.ceil(max_time_s / dt_s)) if max_time_s > 0 else 0

    # ---- random profile pre-gen ----
    i_req_random = None
    if prof == "random_current":
        seg_min_steps = max(1, int(float(getattr(scenario, "segment_min_s")) / dt_s))
        seg_max_steps = max(seg_min_steps, int(float(getattr(scenario, "segment_max_s")) / dt_s))
        rng_prof = np.random.default_rng(getattr(scenario, "random_seed", 0) or 0)

        i_req_random = np.zeros(n_steps)
        k0 = 0
        while k0 < n_steps:
            seg_len = int(rng_prof.integers(seg_min_steps, seg_max_steps + 1))
            mode = rng_prof.choice(["discharge", "charge", "idle"])
            if mode == "discharge":
                level = float(rng_prof.uniform(0.3, 1.0) * float(getattr(scenario, "i_discharge_max_a")))
            elif mode == "charge":
                level = float(-rng_prof.uniform(0.3, 1.0) * float(getattr(scenario, "i_charge_max_a")))
            else:
                level = 0.0
            k1 = min(n_steps, k0 + seg_len)
            i_req_random[k0:k1] = level
            k0 = k1

    # ---- Prime plant ----
    res = ecm.step(0.0, 0.0)
    soc_hat = float(ekf.get_soc())

    def apply_overrides(
        t_s: float,
        vmin: float, vmax: float,
        t_rack: float, i_rack: float,
        seg: Optional[Any],
    ) -> Tuple[float, float, float, float, bool]:
        """
        Returns: vmin_used, vmax_used, t_used, i_used, gas_alarm

        Priority:
          1) segment overrides (direct set)
          2) time-based injections (min/max semantics) with optional duration windows
        """
        gas_alarm = False

        # --- Segment overrides ---
        if seg is not None:
            vmin_ovr = _seg_get(seg, "v_cell_min_override_v", None)
            vmax_ovr = _seg_get(seg, "v_cell_max_override_v", None)
            t_ovr = _seg_get(seg, "t_rack_override_c", None)
            i_ovr = _seg_get(seg, "i_rack_override_a", None)
            gas_ovr = _seg_get(seg, "gas_alarm", None)

            if vmin_ovr is not None:
                vmin = float(vmin_ovr)
            if vmax_ovr is not None:
                vmax = float(vmax_ovr)
            if t_ovr is not None:
                t_rack = float(t_ovr)
            if i_ovr is not None:
                i_rack = float(i_ovr)
            if gas_ovr is not None:
                gas_alarm = bool(gas_ovr)

        # --- Global injections (time window) ---
        if _pulse_active(t_s, uv_time, uv_dur_s):
            vmin = min(vmin, float(uv_v_fault))

        if _pulse_active(t_s, ov_time, ov_dur_s):
            vmax = max(vmax, float(ov_v_fault))

        if _pulse_active(t_s, ot_time, ot_dur_s):
            t_rack = max(t_rack, float(ot_temp_c))

        if _pulse_active(t_s, oc_time, oc_dur_s):
            i_rack = float(oc_i_fault)

        if _pulse_active(t_s, fire_time, fire_dur_s):
            gas_alarm = True

        return vmin, vmax, t_rack, i_rack, gas_alarm

    # ---- Main loop ----
    for k in range(n_steps):
        t_s = float(k * dt_s)

        # 1) Requested current profile + ambient
        seg_obj: Optional[Any] = None

        if prof == "constant_current":
            direction = getattr(scenario, "direction", "discharge")
            if direction == "discharge":
                i_req = float(getattr(scenario, "current_a"))
            elif direction == "charge":
                i_req = -float(getattr(scenario, "current_a"))
            else:
                raise ValueError(f"Unknown direction: {direction!r}")
            t_amb_c = float(getattr(scenario, "t_amb_c", 25.0))

        elif prof == "random_current":
            i_req = float(i_req_random[k])
            t_amb_c = float(getattr(scenario, "t_amb_c", 25.0))

        elif prof == "segments":
            seg_obj = _active_segment(list(getattr(scenario, "segments")), t_s)
            i_req = float(_seg_get(seg_obj, "current_a", 0.0))
            t_amb_c = float(_seg_get(seg_obj, "t_amb_c", 25.0))

        else:
            raise ValueError(f"Unsupported profile type: {prof!r}")

        # 2) BMS limiting uses "visible" signals (previous plant / pre-step)
        vmin_lim = float(res.get("v_cell_min", np.nan))
        vmax_lim = float(res.get("v_cell_max", np.nan))
        t_lim = float(thermal.t_c)

        vmin_lim, vmax_lim, t_lim, _, _ = apply_overrides(
            t_s, vmin_lim, vmax_lim, t_lim, 0.0, seg_obj
        )

        if fsm.state in (BMSState.FAULT, BMSState.EMERGENCY_SHUTDOWN):
            i_act = 0.0
        else:
            if bool(getattr(scenario, "use_bms_limits", True)):
                lim = compute_current_limits(
                    soc_hat=float(soc_hat),
                    t_rack_c=float(t_lim),
                    v_cell_min=float(vmin_lim),
                    v_cell_max=float(vmax_lim),
                    params=bms_params,
                )
                if i_req >= 0:
                    i_act = min(i_req, float(lim["i_discharge_max_allowed"]))
                else:
                    i_act = max(i_req, -float(lim["i_charge_max_allowed"]))
            else:
                i_act = i_req

        # 3) EKF predict + plant update
        ekf.predict(i_act, dt_s)
        res = ecm.step(i_act, dt_s)
        t_rack_c = float(thermal.step(res["p_loss"], t_amb_c, dt_s))

        # 4) Measurements
        v_meas = (float(res["v_rack"]) + float(rng_meas.normal(0.0, sigma_v))) if sigma_v > 0 else float(res["v_rack"])
        i_meas = (float(i_act) + float(rng_meas.normal(0.0, sigma_i))) if sigma_i > 0 else float(i_act)
        t_meas = (float(t_rack_c) + float(rng_meas.normal(0.0, sigma_t))) if sigma_t > 0 else float(t_rack_c)

        # 5) EKF update
        ekf.update(v_meas, i_meas)
        soc_hat = float(ekf.get_soc())

        # 6) Fault detection inputs (post-step)
        vmin_true = float(res.get("v_cell_min", np.nan))
        vmax_true = float(res.get("v_cell_max", np.nan))

        vmin_used, vmax_used, t_used, i_used, gas_alarm = apply_overrides(
            t_s, vmin_true, vmax_true, t_meas, i_meas, seg_obj
        )

        flags = fault_det.step(
            v_cell_min=float(vmin_used),
            v_cell_max=float(vmax_used),
            t_rack_c=float(t_used),
            i_rack_a=float(i_used),
            gas_alarm=bool(gas_alarm),
        )
        state = fsm.step(flags, enable=True)

        # 7) Log
        hist["time_s"].append(t_s)
        hist["i_req_a"].append(float(i_req))
        hist["i_act_a"].append(float(i_act))
        hist["soc_true"].append(float(res["soc"]))
        hist["soc_hat"].append(float(soc_hat))
        hist["v_rack_v"].append(float(res["v_rack"]))
        hist["v_rack_meas_v"].append(float(v_meas))
        hist["t_rack_c"].append(float(t_rack_c))
        hist["t_rack_meas_c"].append(float(t_meas))
        hist["i_meas_a"].append(float(i_meas))

        hist["v_cell_min_v"].append(float(vmin_used))
        hist["v_cell_max_v"].append(float(vmax_used))
        hist["v_cell_min_true_v"].append(float(vmin_true))
        hist["v_cell_max_true_v"].append(float(vmax_true))
        hist["t_rack_used_c"].append(float(t_used))
        hist["i_rack_used_a"].append(float(i_used))

        hist["state_code"].append(int(state_code_map[state]))
        hist["oc"].append(1.0 if bool(flags["oc"]) else 0.0)
        hist["ov"].append(1.0 if bool(flags["ov"]) else 0.0)
        hist["uv"].append(1.0 if bool(flags["uv"]) else 0.0)
        hist["ot"].append(1.0 if bool(flags["ot"]) else 0.0)
        hist["fire"].append(1.0 if bool(flags["fire"]) else 0.0)

        hist["soc_cell_min"].append(float(res.get("soc_cell_min", res["soc"])))
        hist["soc_cell_max"].append(float(res.get("soc_cell_max", res["soc"])))
        hist["soc_cell_mean"].append(float(res.get("soc_cell_mean", res["soc"])))

        if bool(getattr(scenario, "stop_on_emergency", False)) and state == BMSState.EMERGENCY_SHUTDOWN:
            break

        # safety: avoid logging beyond horizon if ceil caused 1 extra sample
        if (t_s + dt_s) > (max_time_s + 1e-9):
            break

    return hist
