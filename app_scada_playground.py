"""
app_scada_playground_v2.py

Streamlit playground for a BESS rack + BMS simulation.

What you can do
- Live SCADA (interactive): delegated to scada_playground_ui.render_scada_playground()
  so Live logic is maintained in ONE place.
- Offline simulation: run a full scenario instantly and view the results.

How to run (from project root):
    streamlit run app_scada_playground_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------#
# Make sure project root and src are importable
# -----------------------------------------------------------------------------#
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.config import load_params
from src.sim_runner import build_models
from src.fsm import BMSState
from src.bms_logic import compute_current_limits


# IMPORTANT: Live SCADA UI is centralized here
from scada_playground_ui import render_scada_playground


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
STATE_CODE_MAP = {
    BMSState.OFF: 0,
    BMSState.RUN: 1,
    BMSState.FAULT: 2,
    BMSState.EMERGENCY_SHUTDOWN: 3,
}


def _resolve_config_path(p: str) -> str:
    """
    Resolve config path robustly:
    - if absolute -> resolve
    - else prefer repo-root relative (_ROOT / p) if exists, else cwd-relative resolve
    """
    pp = Path(p)
    if pp.is_absolute():
        return str(pp.resolve())
    cand = _ROOT / p
    if cand.exists():
        return str(cand.resolve())
    return str(pp.resolve())


# -----------------------------------------------------------------------------#
# Streamlit UI
# -----------------------------------------------------------------------------#
st.set_page_config(page_title="BMS SCADA Playground", layout="wide")

# Active config path (single source of truth)
CONFIG_PATH = "data/params_rack.yaml"
CONFIG_PATH_RESOLVED = _resolve_config_path(CONFIG_PATH)

st.title("BMS SCADA Playground")
st.caption(
    "A simple, interactive simulator to understand how a Battery Management System (BMS) reacts "
    "to normal operation and fault conditions."
)
st.caption(f"Active config: {CONFIG_PATH} | Resolved: {CONFIG_PATH_RESOLVED}")

with st.expander("What am I looking at? (plain English)", expanded=True):
    st.markdown(
        """
**Battery rack basics**
- **Current (A)**: positive means discharging (power delivered), negative means charging.
- **SoC (State of Charge)**: 1.0 = 100% full, 0.0 = empty.
- **Cell voltages**: a rack has many cells; we track **minimum** and **maximum** cell voltage.
- **BMS state**:
  - RUN: normal operation
  - FAULT: a fault was detected (BMS typically limits or stops current)
  - EMERGENCY_SHUTDOWN: critical condition, current is forced to zero

**Two modes**
- **Live SCADA**: step-by-step simulation (interactive), best for "what happens if I toggle X now?"
- **Offline**: run once, deterministic plots + NPZ export for thesis/report.
        """
    )

# Load params once (cached)
@st.cache_resource
def _get_params() -> Dict[str, Any]:
    return load_params(CONFIG_PATH)

params = _get_params()

# Sidebar: mode selector
st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Choose mode",
    ["Live SCADA (interactive)", "Offline simulation (run once)"],
    index=0,
)

# -----------------------------------------------------------------------------#
# LIVE SCADA mode (delegated)
# -----------------------------------------------------------------------------#
if mode == "Live SCADA (interactive)":
    render_scada_playground(
        params,
        config_path=CONFIG_PATH,
        config_path_resolved=CONFIG_PATH_RESOLVED,
    )
    st.stop()

# -----------------------------------------------------------------------------#
# OFFLINE simulation mode (run once)
# -----------------------------------------------------------------------------#
st.sidebar.divider()
st.sidebar.header("Offline settings")

t_amb = st.sidebar.number_input("Ambient temperature [°C]", value=25.0, step=1.0)
true_soc0 = st.sidebar.slider("True initial SoC", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
ekf_soc0 = st.sidebar.slider("EKF initial SoC", min_value=0.0, max_value=1.0, value=0.90, step=0.01)
use_limits = st.sidebar.checkbox("Enable BMS current limits", value=True)

st.sidebar.divider()
st.sidebar.header("Fault injection magnitudes (detector inputs)")
uv_v_fault = st.sidebar.number_input("UV injected cell voltage [V]", value=2.0, step=0.1)
ot_c_fault = st.sidebar.number_input("OT injected rack temperature [°C]", value=100.0, step=5.0)

st.subheader("Offline simulation (run once)")
st.write(
    "In offline mode, we simulate a full scenario instantly and then plot the results. "
    "This is useful to generate reproducible examples for reports or debugging."
)

st.markdown("### Scenario inputs")
st.write("Define a simple current profile as a list of segments (duration, current, ambient temperature).")

default_rows = [
    {"duration_s": 600.0, "current_a": +250.0, "t_amb_c": float(t_amb)},
    {"duration_s": 600.0, "current_a": -120.0, "t_amb_c": float(t_amb)},
    {"duration_s": 1200.0, "current_a": +300.0, "t_amb_c": float(t_amb)},
]
df_in = st.data_editor(
    pd.DataFrame(default_rows),
    num_rows="dynamic",
    use_container_width=True,
    key="offline_segments_editor",
)

st.markdown("### Optional fault injections (absolute time, seconds)")
colf1, colf2, colf3 = st.columns(3)
with colf1:
    uv_time = st.number_input("UV start time [s] (0 = disabled)", value=0.0, step=10.0)
with colf2:
    ot_time = st.number_input("OT start time [s] (0 = disabled)", value=0.0, step=10.0)
with colf3:
    fire_time = st.number_input("FIRE start time [s] (0 = disabled)", value=0.0, step=10.0)

run_btn = st.button("Run offline simulation", type="primary", use_container_width=True)

if run_btn:
    # Build models
    models = build_models(params)
    ecm = models["ecm"]
    thermal = models["thermal"]
    ekf = models["ekf"]
    bms_params = models["bms_params"]
    fault_det = models["fault_det"]
    fsm = models["fsm"]
    dt_s = float(models["dt_s"])

    # Reset IC
    ecm.reset(soc=float(true_soc0))
    thermal.reset(t_init_c=float(t_amb))
    ekf.reset(soc_init=float(ekf_soc0))
    fault_det.reset()
    fsm.reset(BMSState.RUN)

    # Parse segments
    segs = []
    for _, r in df_in.iterrows():
        try:
            dur = float(r["duration_s"])
            cur = float(r["current_a"])
            Ta = float(r["t_amb_c"])
        except Exception:
            continue
        if dur > 0:
            segs.append((dur, cur, Ta))

    if not segs:
        st.error("Please define at least one valid segment.")
        st.stop()

    # Build time schedule
    t_end = sum(d for d, _, _ in segs)
    n_steps = int(np.ceil(t_end / dt_s))

    # Fault schedule
    uv_on = uv_time > 0
    ot_on = ot_time > 0
    fire_on = fire_time > 0

    # Logs
    hist: Dict[str, List[float]] = {k: [] for k in [
        "time_s", "i_req_a", "i_act_a", "soc_true", "soc_hat", "v_cell_min_v", "v_cell_max_v", "t_rack_c",
        "state_code", "oc", "ov", "uv", "ot", "fire"
    ]}

    # Helper to find active segment
    def active_seg(t_s: float):
        acc = 0.0
        for d, cur, Ta in segs:
            if acc <= t_s < acc + d:
                return cur, Ta
            acc += d
        return segs[-1][1], segs[-1][2]

    # Prime ECM output
    res = ecm.step(0.0, 0.0)

    for k in range(n_steps):
        t_s = (k + 1) * dt_s
        i_req, Ta = active_seg((k) * dt_s)

        # BMS action
        if fsm.state in (BMSState.FAULT, BMSState.EMERGENCY_SHUTDOWN):
            i_act = 0.0
        else:
            if use_limits:
                curr_limits = compute_current_limits(
                    soc_hat=float(ekf.get_soc()),
                    t_rack_c=float(thermal.t_c),
                    v_cell_min=float(res.get("v_cell_min", np.nan)),
                    v_cell_max=float(res.get("v_cell_max", np.nan)),
                    params=bms_params,
                )
                if i_req >= 0.0:
                    i_act = min(i_req, float(curr_limits["i_discharge_max_allowed"]))
                else:
                    i_act = max(i_req, -float(curr_limits["i_charge_max_allowed"]))
            else:
                i_act = float(i_req)

        # EKF predict + plant update
        ekf.predict(i_act, dt_s)
        res = ecm.step(i_act, dt_s)
        t_rack = float(thermal.step(res["p_loss"], Ta, dt_s))

        # EKF update (ideal measurement)
        ekf.update(float(res["v_rack"]), float(i_act))

        # Fault detector inputs (+ optional injections)
        vmin = float(res.get("v_cell_min", np.nan))
        vmax = float(res.get("v_cell_max", np.nan))
        t_for_fault = float(t_rack)
        i_for_fault = float(i_act)
        gas_alarm = False

        if uv_on and t_s >= float(uv_time):
            vmin = min(vmin, float(uv_v_fault))
        if ot_on and t_s >= float(ot_time):
            t_for_fault = max(t_for_fault, float(ot_c_fault))
        if fire_on and t_s >= float(fire_time):
            gas_alarm = True

        flags = fault_det.step(
            v_cell_min=vmin,
            v_cell_max=vmax,
            t_rack_c=t_for_fault,
            i_rack_a=i_for_fault,
            gas_alarm=gas_alarm,
        )
        state = fsm.step(flags, enable=True)

        # Log
        hist["time_s"].append(float(t_s))
        hist["i_req_a"].append(float(i_req))
        hist["i_act_a"].append(float(i_act))
        hist["soc_true"].append(float(res["soc"]))
        hist["soc_hat"].append(float(ekf.get_soc()))
        hist["v_cell_min_v"].append(float(vmin))
        hist["v_cell_max_v"].append(float(vmax))
        hist["t_rack_c"].append(float(t_rack))
        hist["state_code"].append(int(STATE_CODE_MAP[state]))
        for kk in ["oc", "ov", "uv", "ot", "fire"]:
            hist[kk].append(1.0 if bool(flags.get(kk, False)) else 0.0)

        if t_s >= t_end - 1e-9:
            break

    df = pd.DataFrame(hist).set_index("time_s")

    st.success("Offline simulation completed.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Current")
        st.line_chart(df[["i_req_a", "i_act_a"]].rename(columns={"i_req_a": "I_req [A]", "i_act_a": "I_act [A]"}))
    with c2:
        st.markdown("#### SoC")
        st.line_chart(df[["soc_true", "soc_hat"]].rename(columns={"soc_true": "SoC true", "soc_hat": "SoC EKF"}))

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Cell voltages")
        st.line_chart(df[["v_cell_min_v", "v_cell_max_v"]].rename(columns={"v_cell_min_v": "V_cell min [V]", "v_cell_max_v": "V_cell max [V]"}))
    with c4:
        st.markdown("#### BMS state")
        st.line_chart(df[["state_code"]].rename(columns={"state_code": "State code"}))

    # Download NPZ
    import io
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v, dtype=float) for k, v in hist.items()})
    st.download_button(
        "Download NPZ (offline run)",
        data=buf.getvalue(),
        file_name="offline_run.npz",
        mime="application/octet-stream",
        use_container_width=True,
    )
