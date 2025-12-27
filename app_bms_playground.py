# app_bms_playground.py
import io
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.config import load_params
from src.metrics import (
    compute_safety_metrics,
    compute_operational_metrics,
    compute_estimation_metrics,
)
from run_manual_bms_scenario import Segment, FaultConfig, run_manual_profile
from scada_playground_ui import render_scada_playground
from typing import Optional

@st.cache_data
def get_params():
    return load_params()


def fmt(x, nd=3):
    if x is None:
        return "None"
    return f"{x:.{nd}f}"


def _none_to_nan(x):
    return np.nan if x is None else float(x)


def _dur_ui_to_opt(val: float) -> Optional[float]:
    # 0 -> None (step)
    try:
        v = float(val)
    except Exception:
        return None
    return None if v <= 0 else v


def make_manual_npz_bytes(
    result: dict,
    *,
    segments: list,
    fault_inj: FaultConfig,
    use_limits: bool,
    true_init_soc: float,
    ekf_init_soc: float,
    dt_s: float,
) -> bytes:
    """
    Create an in-memory .npz for the manual run.
    Includes raw simulation arrays + meta so you can reproduce the run conditions later.
    """
    export = {k: np.asarray(v) for k, v in result.items()}

    # ---- Meta (scalars as 1-element arrays) ----
    export["meta_dt_s"] = np.asarray([float(dt_s)], dtype=float)
    export["meta_true_init_soc"] = np.asarray([float(true_init_soc)], dtype=float)
    export["meta_ekf_init_soc"] = np.asarray([float(ekf_init_soc)], dtype=float)
    export["meta_use_bms_limits"] = np.asarray([1 if use_limits else 0], dtype=int)

    # ---- Segment definition (arrays) ----
    export["segments_duration_s"] = np.asarray([float(s.duration_s) for s in segments], dtype=float)
    export["segments_current_a"] = np.asarray([float(s.current_a) for s in segments], dtype=float)
    export["segments_t_amb_c"] = np.asarray([float(s.t_amb_c) for s in segments], dtype=float)

    # ---- Segment overrides (store NaN if None) ----
    export["segments_v_cell_min_override_v"] = np.asarray([_none_to_nan(getattr(s, "v_cell_min_override_v", None)) for s in segments], dtype=float)
    export["segments_v_cell_max_override_v"] = np.asarray([_none_to_nan(getattr(s, "v_cell_max_override_v", None)) for s in segments], dtype=float)
    export["segments_t_rack_override_c"] = np.asarray([_none_to_nan(getattr(s, "t_rack_override_c", None)) for s in segments], dtype=float)
    export["segments_i_rack_override_a"] = np.asarray([_none_to_nan(getattr(s, "i_rack_override_a", None)) for s in segments], dtype=float)
    export["segments_gas_alarm"] = np.asarray([1 if bool(getattr(s, "gas_alarm", False)) else 0 for s in segments], dtype=int)

    # ---- Fault injection meta ----
    export["fault_uv_time_s"] = np.asarray([_none_to_nan(fault_inj.uv_time_s)], dtype=float)
    export["fault_uv_duration_s"] = np.asarray([_none_to_nan(fault_inj.uv_duration_s)], dtype=float)
    export["fault_uv_v_fault"] = np.asarray([float(fault_inj.uv_v_fault)], dtype=float)

    export["fault_ot_time_s"] = np.asarray([_none_to_nan(fault_inj.ot_time_s)], dtype=float)
    export["fault_ot_duration_s"] = np.asarray([_none_to_nan(fault_inj.ot_duration_s)], dtype=float)
    export["fault_ot_temp_c"] = np.asarray([float(fault_inj.ot_temp_c)], dtype=float)

    export["fault_oc_time_s"] = np.asarray([_none_to_nan(fault_inj.oc_time_s)], dtype=float)
    export["fault_oc_duration_s"] = np.asarray([_none_to_nan(fault_inj.oc_duration_s)], dtype=float)
    export["fault_oc_i_fault_a"] = np.asarray([float(fault_inj.oc_i_fault_a)], dtype=float)

    export["fault_fire_time_s"] = np.asarray([_none_to_nan(fault_inj.fire_time_s)], dtype=float)
    export["fault_fire_duration_s"] = np.asarray([_none_to_nan(fault_inj.fire_duration_s)], dtype=float)

    buf = io.BytesIO()
    np.savez_compressed(buf, **export)
    return buf.getvalue()


def _build_piecewise_profile(t: np.ndarray, segments: list, field: str) -> np.ndarray:
    out = np.zeros_like(t, dtype=float)
    acc = 0.0
    for seg in segments:
        dur = float(seg.duration_s)
        t0, t1 = acc, acc + dur
        acc = t1
        mask = (t >= t0) & (t < t1)
        out[mask] = float(getattr(seg, field))
    if len(segments) > 0:
        out[t >= acc] = float(getattr(segments[-1], field))
    return out


def _reset_to_safe_defaults(max_segments: int = 6):
    for i in range(max_segments):
        st.session_state[f"dur_{i}"] = 600.0 if i == 0 else 10.0
        st.session_state[f"curr_{i}"] = 250.0 if i == 0 else 0.0
        st.session_state[f"tamb_{i}"] = 25.0

        st.session_state[f"seg_{i}_t_override_enable"] = False
        st.session_state[f"seg_{i}_t_override_val"] = 25.0

        st.session_state[f"seg_{i}_v_override_enable"] = False
        st.session_state[f"seg_{i}_vmin_override_val"] = 3.20
        st.session_state[f"seg_{i}_vmax_override_val"] = 3.20

        st.session_state[f"seg_{i}_i_override_enable"] = False
        st.session_state[f"seg_{i}_i_override_val"] = 0.0

        st.session_state[f"seg_{i}_gas_alarm"] = False


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="BMS Rack Playground", layout="wide")
st.title("BMS Rack Playground")

st.write(
    """
This dashboard visualises a **single battery rack** of the container BESS.

- Currents, SoC and rack temperature are **rack-level signals**.
- The curves “Cell SoC min/mean/max” and “V_cell min/max” are **statistics over all series-connected cells** in that rack.
- Fault flags (OC, OV, UV, OT, FIRE) and the *BMS state code* are generated by the **same BMS logic** that is used in the offline scenario simulations.
"""
)

params = get_params()

# Persist last manual run for download + review
if "manual_last_result" not in st.session_state:
    st.session_state["manual_last_result"] = None
if "manual_last_npz_bytes" not in st.session_state:
    st.session_state["manual_last_npz_bytes"] = None
if "manual_last_npz_name" not in st.session_state:
    st.session_state["manual_last_npz_name"] = "manual_result.npz"
if "manual_last_segments" not in st.session_state:
    st.session_state["manual_last_segments"] = None
if "manual_last_fault_inj" not in st.session_state:
    st.session_state["manual_last_fault_inj"] = None
if "manual_last_use_limits" not in st.session_state:
    st.session_state["manual_last_use_limits"] = True

tab_manual, tab_scada = st.tabs(["Manual scenarios (offline)", "Live SCADA (offline)"])


with tab_manual:
    st.subheader("Manual scenarios (offline)")

    dt_s = float(params["simulation"]["dt_s"])

    limits = params["limits"]
    faults_yaml = params["faults"]
    ctrl = params.get("bms_control", {})

    UV_THRESHOLD = float(faults_yaml["uv_cell_v"])
    OV_THRESHOLD = float(faults_yaml["ov_cell_v"])
    SOFT_MAX_V = float(limits["v_cell_max_v"])

    OT_THRESHOLD = float(faults_yaml.get("ot_rack_c", 50.0))
    T_DERATE_START = float(ctrl.get("t_high_derate_start_c", 45.0))
    T_CUTOFF = float(ctrl.get("t_high_cutoff_c", 55.0))

    V_CELL_MIN_OP = float(limits.get("v_cell_min_v", 2.5))
    V_CELL_MAX_OP = float(limits.get("v_cell_max_v", 3.5))

    therm = params.get("thermal_model", {})
    try:
        tau_s = float(therm.get("c_th_j_per_k", 0.0)) * float(therm.get("r_th_k_per_w", 0.0))
    except Exception:
        tau_s = 0.0

    st.sidebar.markdown(f"**Simulation time step**: {dt_s} s")
    st.sidebar.markdown(
        f"**UV threshold**: {UV_THRESHOLD:.2f} V\n\n"
        f"**OV threshold**: {OV_THRESHOLD:.2f} V\n\n"
        f"**Soft max cell voltage**: {SOFT_MAX_V:.2f} V\n\n"
        f"**OT threshold**: {OT_THRESHOLD:.1f} °C\n\n"
        f"**T derate start**: {T_DERATE_START:.1f} °C\n\n"
        f"**T cutoff**: {T_CUTOFF:.1f} °C"
    )
    if tau_s > 0:
        st.sidebar.caption(
            f"Thermal time constant estimate: τ ≈ {tau_s:.0f} s (~{tau_s/3600:.1f} h). "
            "So changing T_amb in later segments may not raise T_rack quickly. "
            "Use T_rack override or OT injection for fast logic tests."
        )

    # ---------------- EKF initial condition ----------------
    st.sidebar.header("EKF Initial Condition Test")
    true_init_soc_ui = st.sidebar.number_input(
        "True initial SoC (ECM) [0..1]",
        min_value=0.0, max_value=1.0, value=0.50, step=0.01, key="true_init_soc_ui"
    )
    ekf_init_soc_ui = st.sidebar.number_input(
        "EKF initial SoC estimate [0..1]",
        min_value=0.0, max_value=1.0, value=0.80, step=0.01, key="ekf_init_soc_ui"
    )
    st.sidebar.caption("Example: True=0.50, EKF=0.80 -> watch SoC error converge toward 0.")

    st.info(
        "Tip: **Advanced overrides** change what the BMS sees (limits + fault detector), "
        "but they do not modify the plant physics. This is ideal for operator-style BMS validation."
    )

    # ---------------- 1) Segments ----------------
    st.header("1. Current / ambient temperature profile")

    colr1, colr2 = st.columns([1, 2])
    with colr1:
        if st.button("Reset to normal operation (safe defaults)", use_container_width=True):
            _reset_to_safe_defaults(max_segments=6)
            st.rerun()
    with colr2:
        st.caption(
            "Use segments for plant inputs (I_req, T_amb). "
            "If you want to force an OT/UV/OV/OC condition immediately, use **Advanced overrides** or **Fault injections**."
        )

    n_segments = st.number_input(
        "Number of segments", min_value=1, max_value=6, value=2, step=1, key="n_segments"
    )

    segments = []
    for i in range(int(n_segments)):
        st.subheader(f"Segment {i + 1}")
        c1, c2, c3 = st.columns(3)

        with c1:
            dur = st.number_input(
                f"Duration [s] (segment {i + 1})",
                min_value=1.0,
                value=600.0 if i == 0 else 10.0,
                key=f"dur_{i}",
            )
        with c2:
            default_i = 250.0 if i == 0 else 0.0
            curr = st.number_input(
                f"Current [A] (segment {i + 1})",
                value=default_i,
                step=10.0,
                key=f"curr_{i}",
            )
        with c3:
            tamb = st.number_input(
                f"Ambient temperature [°C] (segment {i + 1})",
                value=25.0,
                key=f"tamb_{i}",
            )

        with st.expander(f"Advanced overrides (BMS inputs) — segment {i + 1}", expanded=False):
            st.caption(
                f"Operating ranges (from YAML): V_cell ≈ [{V_CELL_MIN_OP:.2f}, {V_CELL_MAX_OP:.2f}] V, "
                f"T cutoff ≈ {T_CUTOFF:.1f} °C, OT threshold ≈ {OT_THRESHOLD:.1f} °C."
            )

            cA, cB = st.columns(2)

            with cA:
                t_ov_en = st.checkbox(
                    "Override rack temperature seen by BMS",
                    value=False,
                    key=f"seg_{i}_t_override_enable",
                )
                if t_ov_en:
                    t_ov_val = st.number_input(
                        "T_rack override [°C]",
                        value=25.0,
                        step=5.0,
                        key=f"seg_{i}_t_override_val",
                    )
                else:
                    t_ov_val = None

            with cB:
                i_ov_en = st.checkbox(
                    "Override rack current measurement seen by BMS",
                    value=False,
                    key=f"seg_{i}_i_override_enable",
                )
                if i_ov_en:
                    i_ov_val = st.number_input(
                        "I_rack override [A]",
                        value=0.0,
                        step=10.0,
                        key=f"seg_{i}_i_override_val",
                    )
                else:
                    i_ov_val = None

            v_ov_en = st.checkbox(
                "Override cell voltage min/max seen by BMS",
                value=False,
                key=f"seg_{i}_v_override_enable",
            )
            if v_ov_en:
                cv1, cv2 = st.columns(2)
                with cv1:
                    vmin_ov = st.number_input(
                        "V_cell_min override [V]",
                        value=3.20,
                        step=0.01,
                        key=f"seg_{i}_vmin_override_val",
                    )
                with cv2:
                    vmax_ov = st.number_input(
                        "V_cell_max override [V]",
                        value=3.20,
                        step=0.01,
                        key=f"seg_{i}_vmax_override_val",
                    )
            else:
                vmin_ov, vmax_ov = None, None

            gas_alarm = st.checkbox(
                "Force gas alarm (fire) for this segment",
                value=False,
                key=f"seg_{i}_gas_alarm",
            )

        segments.append(
            Segment(
                duration_s=float(dur),
                current_a=float(curr),
                t_amb_c=float(tamb),
                v_cell_min_override_v=None if vmin_ov is None else float(vmin_ov),
                v_cell_max_override_v=None if vmax_ov is None else float(vmax_ov),
                t_rack_override_c=None if t_ov_val is None else float(t_ov_val),
                i_rack_override_a=None if i_ov_val is None else float(i_ov_val),
                gas_alarm=bool(gas_alarm),
            )
        )

    # ---------------- 2) Fault injections ----------------
    st.header("2. Fault injections (optional)")

    col_uv, col_ot, col_oc, col_fire = st.columns(4)

    with col_uv:
        st.markdown("**Undervoltage (UV) fault**")
        uv_enable = st.checkbox("Enable UV fault", value=False, key="uv_enable")
        if uv_enable:
            uv_time_s = st.number_input("UV injection time [s]", min_value=0.0, value=1200.0, key="uv_time")
            uv_dur_s = st.number_input("UV pulse duration [s] (0 = step)", min_value=0.0, value=0.0, key="uv_dur")
            uv_v_fault = st.number_input(
                "Forced minimum cell voltage during UV [V]",
                min_value=0.0, value=2.0, key="uv_v_fault"
            )
        else:
            uv_time_s, uv_dur_s, uv_v_fault = None, 0.0, 2.0

    with col_ot:
        st.markdown("**Over-temperature (OT) fault**")
        ot_enable = st.checkbox("Enable OT fault", value=False, key="ot_enable")
        if ot_enable:
            ot_time_s = st.number_input("OT injection time [s]", min_value=0.0, value=200.0, key="ot_time")
            ot_dur_s = st.number_input("OT pulse duration [s] (0 = step)", min_value=0.0, value=0.0, key="ot_dur")
            ot_temp_c = st.number_input(
                "Forced rack temperature during OT [°C]",
                min_value=-50.0, value=100.0, step=5.0, key="ot_temp"
            )
        else:
            ot_time_s, ot_dur_s, ot_temp_c = None, 0.0, 100.0

    with col_oc:
        st.markdown("**Over-current (OC) fault**")
        oc_enable = st.checkbox("Enable OC fault", value=False, key="oc_enable")
        if oc_enable:
            oc_time_s = st.number_input("OC injection time [s]", min_value=0.0, value=300.0, key="oc_time")
            oc_dur_s = st.number_input("OC pulse duration [s] (0 = step)", min_value=0.0, value=0.0, key="oc_dur")
            oc_i_fault_a = st.number_input(
                "Forced rack current measurement during OC [A] (sign matters)",
                value=500.0, step=10.0, key="oc_i_fault"
            )
        else:
            oc_time_s, oc_dur_s, oc_i_fault_a = None, 0.0, 500.0

    with col_fire:
        st.markdown("**Fire / gas alarm**")
        fire_enable = st.checkbox("Enable fire alarm", value=False, key="fire_enable")
        if fire_enable:
            fire_time_s = st.number_input("Fire alarm time [s]", min_value=0.0, value=350.0, key="fire_time")
            fire_dur_s = st.number_input("Fire pulse duration [s] (0 = step)", min_value=0.0, value=0.0, key="fire_dur")
        else:
            fire_time_s, fire_dur_s = None, 0.0

    fault_inj = FaultConfig(
        uv_time_s=uv_time_s,
        uv_duration_s=_dur_ui_to_opt(uv_dur_s),
        uv_v_fault=uv_v_fault,

        ot_time_s=ot_time_s,
        ot_duration_s=_dur_ui_to_opt(ot_dur_s),
        ot_temp_c=ot_temp_c,

        oc_time_s=oc_time_s,
        oc_duration_s=_dur_ui_to_opt(oc_dur_s),
        oc_i_fault_a=oc_i_fault_a,

        fire_time_s=fire_time_s,
        fire_duration_s=_dur_ui_to_opt(fire_dur_s),
    )

    use_limits = st.checkbox("Apply BMS current limits", value=True, key="use_limits")

    # ---------------- Run simulation ----------------
    if st.button("Run simulation", key="run_btn"):
        with st.spinner("Running BMS simulation..."):
            result = run_manual_profile(
                segments=segments,
                faults=fault_inj,
                params=params,
                use_bms_limits=use_limits,
                true_init_soc=true_init_soc_ui,
                ekf_init_soc=ekf_init_soc_ui,
            )

        st.session_state["manual_last_result"] = result
        st.session_state["manual_last_segments"] = segments
        st.session_state["manual_last_fault_inj"] = fault_inj
        st.session_state["manual_last_use_limits"] = bool(use_limits)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["manual_last_npz_name"] = f"manual_result_{ts}.npz"
        st.session_state["manual_last_npz_bytes"] = make_manual_npz_bytes(
            result,
            segments=segments,
            fault_inj=fault_inj,
            use_limits=use_limits,
            true_init_soc=true_init_soc_ui,
            ekf_init_soc=ekf_init_soc_ui,
            dt_s=dt_s,
        )

        st.success("Simulation finished. You can download the .npz below.")

    # ---- Export button ----
    if st.session_state["manual_last_npz_bytes"] is not None:
        st.download_button(
            label="Download manual result (.npz)",
            data=st.session_state["manual_last_npz_bytes"],
            file_name=st.session_state["manual_last_npz_name"],
            mime="application/octet-stream",
            key="dl_manual_npz",
        )

    # ---- Plot if we have a result ----
    if st.session_state["manual_last_result"] is not None:
        result = st.session_state["manual_last_result"]
        segs_used = st.session_state.get("manual_last_segments", None)
        use_limits_used = bool(st.session_state.get("manual_last_use_limits", True))

        # ---- unpack ----
        t = np.array(result["time_s"], dtype=float)
        i_req = np.array(result["i_req_a"], dtype=float)
        i_act = np.array(result["i_act_a"], dtype=float)

        soc_true = np.array(result["soc_true"], dtype=float)
        soc_hat = np.array(result["soc_hat"], dtype=float)

        soc_min = np.array(result.get("soc_cell_min", soc_true), dtype=float)
        soc_mean = np.array(result.get("soc_cell_mean", soc_true), dtype=float)
        soc_max = np.array(result.get("soc_cell_max", soc_true), dtype=float)

        v_min = np.array(result["v_cell_min_v"], dtype=float)
        v_max = np.array(result["v_cell_max_v"], dtype=float)

        v_min_true = np.array(result.get("v_cell_min_true_v", v_min), dtype=float)
        v_max_true = np.array(result.get("v_cell_max_true_v", v_max), dtype=float)

        t_rack_true = np.array(result.get("t_rack_c", np.full_like(t, np.nan)), dtype=float)
        t_rack_used = np.array(result.get("t_rack_used_c", t_rack_true), dtype=float)

        state = np.array(result["state_code"], dtype=int)
        oc = np.array(result["oc"], dtype=bool)
        ov = np.array(result["ov"], dtype=bool)
        uv = np.array(result["uv"], dtype=bool)
        ot = np.array(result["ot"], dtype=bool)
        fire = np.array(result["fire"], dtype=bool)

        # ---- sanity checks ----
        if not np.all(np.isfinite(t)) or len(t) == 0:
            st.error("Invalid time vector returned from simulation.")
            st.stop()

        if np.any(np.diff(t) <= 0):
            st.warning(
                "Time vector is NOT strictly increasing. "
                "This typically indicates duplicate timestamps (e.g., two samples at t=0). "
                "Please check the runner logging."
            )

        # ---- metrics ----
        dt = float(params["simulation"]["dt_s"])
        safety = compute_safety_metrics(result, dt)
        oper = compute_operational_metrics(result, dt)
        est = compute_estimation_metrics(result)

        st.subheader("Summary metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown("**Safety**")
            st.write(f"t_fault_any_s : {fmt(safety['t_fault_any_s'])}")
            st.write(f"t_emergency_s : {fmt(safety['t_emergency_s'])}")
            st.write(f"t_uv_s        : {fmt(safety['t_uv_s'])}")
            st.write(f"t_oc_s        : {fmt(safety.get('t_oc_s'))}")
            st.write(f"t_ot_s        : {fmt(safety['t_ot_s'])}")
            st.write(f"t_fire_s      : {fmt(safety['t_fire_s'])}")
        with mcol2:
            st.markdown("**Operational / thermal**")
            st.write(f"Energy_discharge_kWh : {fmt(oper['energy_discharge_kwh'])}")
            st.write(f"soc_delta            : {fmt(oper['soc_delta'])}")
            st.write(f"t_discharge_s        : {fmt(oper['t_discharge_s'])}")
            st.write(f"t_max_c              : {fmt(oper['t_max_c'])}")
            st.write(f"delta_t_c            : {fmt(oper['delta_t_c'])}")
        with mcol3:
            st.markdown("**SoC estimation**")
            st.write(f"RMSE SoC             : {fmt(est['rmse_soc'], nd=5)}")
            st.write(f"Max abs error SoC    : {fmt(est['max_abs_error_soc'], nd=5)}")
            st.write(f"p95 abs error SoC    : {fmt(est['p95_abs_error_soc'], nd=5)}")

        if safety["t_fault_any_s"] is None:
            st.success("No safety faults were triggered during this scenario.")
        else:
            msgs = []
            for key, name in [("t_uv_s", "UV"), ("t_ov_s", "OV"), ("t_oc_s", "OC"), ("t_ot_s", "OT"), ("t_fire_s", "FIRE")]:
                if safety.get(key) is not None:
                    msgs.append(f"{name} at t = {fmt(safety[key])} s")
            st.error("Faults triggered: " + ", ".join(msgs))

        # ---- derived signals ----
        soc_err = soc_true - soc_hat
        soc_spread = soc_max - soc_min
        v_spread = v_max - v_min
        fault_any = (oc | ov | uv | ot | fire)

        def first_time(mask: np.ndarray):
            idx = np.where(mask)[0]
            return None if len(idx) == 0 else float(t[idx[0]])

        t_uv = first_time(uv)
        t_ov = first_time(ov)
        t_oc = first_time(oc)
        t_ot = first_time(ot)
        t_fire = first_time(fire)

        derate_mask = (np.abs(i_req - i_act) > 1e-6) if use_limits_used else np.zeros_like(i_req, dtype=bool)
        derate_start_idx = np.where((~derate_mask[:-1]) & (derate_mask[1:]))[0] + 1
        derate_start_times = t[derate_start_idx] if len(derate_start_idx) else np.array([])

        state_change_idx = np.where(state[1:] != state[:-1])[0] + 1
        state_change_times = t[state_change_idx] if len(state_change_idx) else np.array([])

        if segs_used is not None:
            t_amb_series = _build_piecewise_profile(t, segs_used, "t_amb_c")
        else:
            t_amb_series = np.full_like(t, np.nan)

        # ---- plots ----
        fig, axes = plt.subplots(7, 1, figsize=(10, 15), sharex=True)

        def vline(ax, x, label, style="--"):
            if x is None:
                return
            ax.axvline(x, linestyle=style, linewidth=1.2, label=label)

        # 1) Currents
        ax = axes[0]
        ax.plot(t, i_req, label="I_req [A]")
        ax.plot(t, i_act, "--", label="I_act [A]")
        for j, x in enumerate(derate_start_times):
            ax.axvline(x, linestyle=":", linewidth=1.0, label="Derating start" if j == 0 else "_nolegend_")
        ax.set_ylabel("Current [A]")
        ax.grid(True)
        ax.legend(loc="best")

        # 2) SoC
        ax = axes[1]
        ax.plot(t, soc_true, label="SoC true (rack)")
        ax.plot(t, soc_hat, "--", label="SoC EKF")
        ax.set_ylabel("SoC [-]")
        ax.grid(True)
        ax.legend(loc="best")

        # 3) SoC error
        ax = axes[2]
        ax.plot(t, soc_err, label="SoC error (true - EKF)")
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_ylabel("SoC error [-]")
        ax.grid(True)
        ax.legend(loc="best")

        # 4) Cell voltages + thresholds
        ax = axes[3]
        ax.plot(t, v_min, label="V_cell min (used)")
        ax.plot(t, v_max, label="V_cell max (used)")
        if np.nanmax(np.abs(v_min_true - v_min)) > 1e-9 or np.nanmax(np.abs(v_max_true - v_max)) > 1e-9:
            ax.plot(t, v_min_true, ":", label="V_cell min (true)")
            ax.plot(t, v_max_true, ":", label="V_cell max (true)")
        ax.axhline(UV_THRESHOLD, linestyle="--", label="UV threshold")
        ax.axhline(OV_THRESHOLD, linestyle="--", label="OV threshold")
        ax.axhline(SOFT_MAX_V, linestyle=":", label="Soft max voltage")
        vline(ax, t_uv, "UV trigger")
        vline(ax, t_ov, "OV trigger")
        ax.set_ylabel("Cell voltage [V]")
        ax.grid(True)
        ax.legend(loc="best")

        # 5) Temperature panel
        ax = axes[4]
        ax.plot(t, t_rack_true, label="T_rack true [°C]")
        ax.plot(t, t_rack_used, "--", label="T_rack used by BMS [°C]")
        if np.any(np.isfinite(t_amb_series)):
            ax.plot(t, t_amb_series, ":", label="T_amb [°C]")
        ax.axhline(OT_THRESHOLD, linestyle="--", label="OT threshold")
        ax.axhline(T_DERATE_START, linestyle=":", label="T derate start")
        ax.axhline(T_CUTOFF, linestyle="-.", label="T cutoff")
        vline(ax, t_ot, "OT trigger")
        ax.set_ylabel("Temperature [°C]")
        ax.grid(True)
        ax.legend(loc="best")

        # 6) Spread panel
        ax = axes[5]
        ax.plot(t, soc_spread, label="SoC spread (max-min)")
        ax.set_ylabel("SoC spread [-]")
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(t, v_spread, "--", label="V spread (max-min)")
        ax2.set_ylabel("V spread [V]")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")

        # 7) State timeline
        ax = axes[6]
        ax.step(t, state, where="post", label="BMS state code")
        ax.step(t, fault_any.astype(int), where="post", label="fault_any")
        for j, x in enumerate(state_change_times):
            ax.axvline(x, linestyle=":", linewidth=0.8, label="State change" if j == 0 else "_nolegend_")
        vline(ax, t_oc, "OC trigger")
        vline(ax, t_ot, "OT trigger")
        vline(ax, t_fire, "FIRE trigger")
        ax.set_ylabel("State / fault_any")
        ax.set_xlabel("Time [s]")
        ax.grid(True)
        ax.legend(loc="best")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ---- event table ----
        events = []
        for name, tx in [("UV", t_uv), ("OV", t_ov), ("OC", t_oc), ("OT", t_ot), ("FIRE", t_fire)]:
            if tx is not None:
                events.append({"event": name, "t_s": float(tx)})

        if events:
            st.subheader("Event timeline")
            st.dataframe(events, use_container_width=True)


with tab_scada:
    render_scada_playground(params)
