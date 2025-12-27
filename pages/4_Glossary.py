# pages/2_Glossary.py
from __future__ import annotations

from typing import Dict, List

import pandas as pd
import streamlit as st


st.title("Glossary / Abbreviations")
st.caption("Quick reference for terms used across the BMS Rack Playground.")

GLOSSARY: List[Dict[str, str]] = [
    # System level
    {"Term": "BESS", "Meaning": "Battery Energy Storage System", "Notes": "System-level battery storage plant."},
    {"Term": "BMS", "Meaning": "Battery Management System", "Notes": "Protection + estimation + control for safe operation."},
    {"Term": "Rack", "Meaning": "Battery rack", "Notes": "A single rack in a containerized BESS (our simulated unit)."},
    {"Term": "SCADA", "Meaning": "Supervisory Control and Data Acquisition", "Notes": "Monitoring/control layer; offline demo only."},

    # Models & estimation
    {"Term": "ECM", "Meaning": "Equivalent Circuit Model", "Notes": "Electrical plant model (R0 + RC branches)."},
    {"Term": "EKF", "Meaning": "Extended Kalman Filter", "Notes": "State estimator used here for SoC estimation."},
    {"Term": "SoC", "Meaning": "State of Charge", "Notes": "Normalized charge content [0..1]."},
    {"Term": "SoH", "Meaning": "State of Health", "Notes": "Aging indicator; optional simplified scaling in simulations."},

    # State machine & logic
    {"Term": "FSM", "Meaning": "Finite State Machine", "Notes": "RUN / FAULT / EMERGENCY_SHUTDOWN handling."},
    {"Term": "Derating", "Meaning": "Current derating / limiting", "Notes": "Reducing I_act vs I_req due to constraints."},
    {"Term": "I_req", "Meaning": "Requested current", "Notes": "Profile command (segment current)."},
    {"Term": "I_act", "Meaning": "Applied current", "Notes": "After BMS limits + FSM state (may become 0 in FAULT/EMERG)."},
    {"Term": "Used vs True", "Meaning": "BMS-visible vs plant signals", "Notes": "Overrides/injections affect 'used', plant produces 'true'."},

    # Fault codes
    {"Term": "UV", "Meaning": "Under-voltage", "Notes": "Cell min voltage below threshold."},
    {"Term": "OV", "Meaning": "Over-voltage", "Notes": "Cell max voltage above threshold."},
    {"Term": "OT", "Meaning": "Over-temperature", "Notes": "Rack temperature above threshold."},
    {"Term": "UT", "Meaning": "Under-temperature", "Notes": "Rack temperature below threshold."},
    {"Term": "OC", "Meaning": "Over-current", "Notes": "Rack current above threshold (charge/discharge)."},
    {"Term": "FIRE", "Meaning": "Fire / Gas alarm", "Notes": "Gas alarm OR temperature-based fire condition; emergency."},
    {"Term": "Debounce", "Meaning": "Debouncing", "Notes": "N consecutive samples required to raise a fault flag."},
]

df = pd.DataFrame(GLOSSARY)

st.dataframe(df, use_container_width=True)

with st.expander("Interpretation notes", expanded=False):
    st.markdown(
        """
- **Debounce**: With `debounce_steps = N` and `dt = dt_s`, a step injection at time `t0` typically triggers at:
  `t_trigger ≈ t0 + (N-1) * dt_s`
- **Used vs True**: Injections/segment overrides are designed to validate logic deterministically. They do not force the plant physics.
"""
    )
