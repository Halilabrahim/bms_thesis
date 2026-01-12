# app.py (Home)
from __future__ import annotations

from datetime import date
from typing import Any, Dict

import streamlit as st

from src.config import load_params, PROFILES  # DEFAULT_PROFILE import ETME


st.set_page_config(page_title="BMS Rack Playground — Thesis Demo", layout="wide")


@st.cache_data
def get_params(cfg_path: str) -> Dict[str, Any]:
    return load_params(cfg_path)


# --- Profile selection (Option A) ---
profiles = list(PROFILES.keys())
DEFAULT_PROFILE = "LUNA" if "LUNA" in PROFILES else profiles[0]

if "active_profile" not in st.session_state:
    st.session_state.active_profile = DEFAULT_PROFILE

st.sidebar.selectbox("Rack profile", profiles, key="active_profile")

active = st.session_state.active_profile
st.session_state.active_config_path = PROFILES[active]["params"]
st.session_state.active_scenarios_path = PROFILES[active]["scenarios"]

cfg_path = st.session_state.active_config_path
try:
    params = get_params(cfg_path)
except Exception as e:
    st.error("Failed to load YAML config for selected profile.")
    st.code(f"{cfg_path}\n\n{e}")
    st.stop()

# ---------------- Page ----------------
st.title("BMS Rack Playground — Master Thesis Demo")
st.caption("Python-based offline simulation and validation of single-rack BMS functions.")

st.sidebar.success(f"Active profile: {active}")
st.sidebar.caption(f"Config: {cfg_path}")


# --- Thesis meta (edit here anytime) ---
AUTHOR = "Halil İbrahim AYDIN"
THESIS_SCOPE = (
    "Offline simulation and deterministic validation of single-rack BMS functions "
    "(limits, fault detection, FSM reactions, and EKF-based SoC estimation). "
    "Profiles: Huawei LUNA reference + Great Power 1P416S rack."
)

THESIS_TITLE = "Python Based Development of Battery Management System Functions"
ADVISOR = "Prof. Dr. techn. Michael Sternad"
INDUSTRY_ADVISOR = "Harun Köle — Technical Manager at Saves Enerji"
START_DATE = date(2025, 9, 24)
SUBMISSION_DEADLINE = date(2026, 3, 24)
THESIS_PERIOD_MONTHS = 6

st.subheader("Thesis information")
st.markdown(
    f"""
**Author:** {AUTHOR}  
**Title:** {THESIS_TITLE}  
**Advisor:** {ADVISOR}  
**Industry advisor:** {INDUSTRY_ADVISOR}  
**Start date:** {START_DATE.isoformat()}  
**Submission deadline:** {SUBMISSION_DEADLINE.isoformat()}  
**Period:** {THESIS_PERIOD_MONTHS} months  

**Scope:** {THESIS_SCOPE}
"""
)

# (Aşağıdaki expander/metin kısımlarını senin eski dosyandan aynen bırakabilirsin.)

with st.expander("Original thesis proposal (as submitted)", expanded=True):
    st.markdown(
        """
**Objective**  
Develop a Python-based simulation framework to implement and evaluate key Battery Management System (BMS) functions for battery energy storage systems (BESS). The focus is on safe operation, robust fault handling, and reproducible validation under both normal operating conditions and fault scenarios.

**Scope**  
BMS Functions: Development in Python of core BMS tasks:

Charging/discharging control based on battery SoC, cell voltages, and temperature limits.

Safety monitoring and emergency protocols 
(e.g., over-temperature, over-voltage, fire detection leading to system shutdown and activation of a fire suppression mechanism).

**Optional**  
SoC/SoH estimation to support decision-making.

**Evaluation**  
Simulation of these functions under varying conditions and load demands. The focus will be on robustness, safety, and operational feasibility.

**Expected contribution**  
The outcome will be a Python-based framework that demonstrates how BMS algorithms ensure safe and reliable battery operation under different operating and fault scenarios, including emergency response cases."""
    )

with st.expander("What has been implemented in this app", expanded=True):
    st.markdown(
        """
**Implemented (current app capabilities)**
- Offline runner integrating **ECM plant + thermal model + EKF SoC estimation + BMS limits + fault detector + FSM**.
- Deterministic validation via:
  - **Sensor-level injections** (UV/OT/OC/FIRE) including **pulse duration** behavior,
  - Per-segment **BMS input overrides** (“used vs true” separation).
- Preset **Quick Tests library** (UV/OT/OC/FIRE + debounce pulse test) with expected outcomes.
- Plot suite including:
  - SoC error (true − EKF),
  - Used vs true voltage and temperature visibility,
  - State timeline + fault_any + derating markers,
  - Event timeline table.
- Reproducible export as **.npz** including scenario meta and segment definitions.

**Planned / next**
- References & Validation page: parameter provenance, topology sanity checks, validation snapshots, known limitations.
"""
    )


st.subheader("What this app simulates (and what it does not)")
left, right = st.columns(2)
with left:
    st.markdown(
        """
**Simulates**
- Single rack electrical behavior (ECM-based plant)
- Rack thermal behavior (lumped model)
- EKF SoC estimation in closed loop
- BMS limiting + fault detection + FSM response
- Deterministic validation via injections/segment overrides
"""
    )
with right:
    st.markdown(
        """
**Does not simulate**
- Full BESS multi-rack dispatch / EMS optimization
- Detailed electrochemistry / aging mechanisms (optional SoH is simplified scaling)
- Detailed pack balancing hardware dynamics (only “spread” visualisation if provided)
- Real SCADA networking (offline-only; SCADA page can be added later)
"""
    )

st.divider()

# --- Navigation hints ---
st.subheader("How to use the multipage app")
st.markdown(
    """
- **Offline Scenarios**: Run preset tests (UV/OT/OC/FIRE + debounce) and build custom segment profiles.
- **Glossary**: Quick lookup of abbreviations (BMS/BESS/ECM/EKF/SoC/SoH/FSM and fault codes).
"""
)

st.info(
    "Recommended workflow: start with **Offline Scenarios → Quick Tests** to validate the protection logic, "
    "then use **Manual Builder** for custom profiles and EKF convergence demos."
)
