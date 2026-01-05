import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_params
from scada_playground_ui import render_scada_playground

st.set_page_config(page_title="Live SCADA", layout="wide")

st.title("Live SCADA")

st.markdown(
    "This page runs the offline simulation stack (ECM + Thermal + EKF + Fault Detector + FSM) step-by-step "
    "to provide a SCADA-like monitoring."
)

# -----------------------------
# Configuration (inside the page)
# -----------------------------
st.subheader("Configuration")

if "active_config_path" not in st.session_state:
    st.session_state.active_config_path = "data/params_rack.yaml"

if "cfg_path_input" not in st.session_state:
    st.session_state.cfg_path_input = st.session_state.active_config_path

c1, c2 = st.columns([3, 1])
with c1:
    st.text_input("Active config path (YAML)", key="cfg_path_input")
with c2:
    apply_cfg = st.button("Apply & rebuild")

if apply_cfg:
    new_path = str(st.session_state.cfg_path_input).strip()
    st.session_state.active_config_path = new_path

    # Stop auto-run and rebuild sim cleanly
    st.session_state.scada_running = False
    st.session_state.scada_autorun_toggle = False
    if "scada_sim" in st.session_state:
        del st.session_state["scada_sim"]

    st.rerun()

cfg_path = str(st.session_state.active_config_path)

try:
    params = load_params(cfg_path)
except Exception as e:
    st.error("Failed to load YAML config.")
    st.code(f"{cfg_path}\n\n{e}")
    st.stop()

render_scada_playground(params, config_path=cfg_path, config_path_resolved=str((ROOT / cfg_path).resolve()))
