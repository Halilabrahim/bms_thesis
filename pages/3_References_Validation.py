# pages/3_References_Validation.py  (veya sende 2_References_Validation.py ise oraya)
from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import streamlit as st

from src.config import load_params


# -----------------------------
# Helpers
# -----------------------------
def _safe_dict_from_load_params(ret: Any) -> Dict[str, Any]:
    """
    Supports:
      - load_params(...) -> dict
      - load_params(...) -> (dict, anything...)
    """
    if isinstance(ret, dict):
        return ret
    if isinstance(ret, (tuple, list)) and len(ret) >= 1 and isinstance(ret[0], dict):
        return ret[0]
    raise TypeError(f"load_params returned unsupported type: {type(ret)}")


def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def check_close(a: Optional[float], b: Optional[float], tol_abs: float = 1e-6, tol_rel: float = 0.01) -> bool:
    if a is None or b is None:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    if abs(a - b) <= tol_abs:
        return True
    denom = max(abs(b), 1e-12)
    return abs(a - b) / denom <= tol_rel


def check_leq(a: Optional[float], b: Optional[float], tol_abs: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return float(a) <= float(b) + float(tol_abs)


def _maybe(a: Any) -> Optional[float]:
    v = _as_float(a, default=float("nan"))
    return v if np.isfinite(v) else None


def _derive_from_yaml(params: Dict[str, Any]) -> Dict[str, Optional[float]]:
    chem = params.get("chemistry", {}) or {}
    struct = params.get("structure", {}) or {}
    em = params.get("electrical_model", {}) or {}
    limits = params.get("limits", {}) or {}
    pack = params.get("pack", {}) or {}
    rack = params.get("rack", {}) or {}

    v_cell_nom = _as_float(chem.get("nominal_cell_voltage_v", 3.2))
    q_ah = _as_float(em.get("q_nom_ah", np.nan))

    cells_s = _as_float(struct.get("cells_in_series_per_pack", np.nan))
    packs_s = _as_float(struct.get("packs_in_series_per_rack", np.nan))

    v_pack = cells_s * v_cell_nom
    e_pack_kwh = v_pack * q_ah / 1000.0

    v_rack = packs_s * v_pack
    e_rack_kwh = packs_s * e_pack_kwh

    # Operational limits (soft) used by BMS current limiting
    v_cell_min_op = _as_float(limits.get("v_cell_min_v", np.nan))
    v_cell_max_op = _as_float(limits.get("v_cell_max_v", np.nan))
    v_rack_min_op = v_cell_min_op * cells_s * packs_s
    v_rack_max_op = v_cell_max_op * cells_s * packs_s

    # Absolute (reference-inspired) from YAML rack section
    v_rack_min_abs_yaml = _as_float(rack.get("v_min_v", np.nan))
    v_rack_max_abs_yaml = _as_float(rack.get("v_max_v", np.nan))

    # Absolute per-cell from pack section (if present)
    v_cell_min_abs = _as_float(pack.get("cell_voltage_min_v", np.nan))
    v_cell_max_abs = _as_float(pack.get("cell_voltage_max_v", np.nan))
    v_rack_min_abs_derived = v_cell_min_abs * cells_s * packs_s
    v_rack_max_abs_derived = v_cell_max_abs * cells_s * packs_s

    return {
        "v_cell_nom_v": _maybe(v_cell_nom),
        "q_ah": _maybe(q_ah),
        "cells_s": _maybe(cells_s),
        "packs_s": _maybe(packs_s),
        "v_pack_v": _maybe(v_pack),
        "e_pack_kwh": _maybe(e_pack_kwh),
        "v_rack_v": _maybe(v_rack),
        "e_rack_kwh": _maybe(e_rack_kwh),
        "v_cell_min_op_v": _maybe(v_cell_min_op),
        "v_cell_max_op_v": _maybe(v_cell_max_op),
        "v_rack_min_op_v": _maybe(v_rack_min_op),
        "v_rack_max_op_v": _maybe(v_rack_max_op),
        "v_rack_min_abs_yaml_v": _maybe(v_rack_min_abs_yaml),
        "v_rack_max_abs_yaml_v": _maybe(v_rack_max_abs_yaml),
        "v_cell_min_abs_v": _maybe(v_cell_min_abs),
        "v_cell_max_abs_v": _maybe(v_cell_max_abs),
        "v_rack_min_abs_derived_v": _maybe(v_rack_min_abs_derived),
        "v_rack_max_abs_derived_v": _maybe(v_rack_max_abs_derived),
    }


def _detect_reference(derived: Dict[str, Optional[float]]) -> str:
    """
    Heuristic:
      - Huawei LUNA-inspired: 16S per pack + 21 packs per rack
      - Great Power: 104S per pack + 4 packs per rack (416S rack)
    """
    c = derived.get("cells_s", None)
    p = derived.get("packs_s", None)
    if c is not None and p is not None:
        if abs(c - 16.0) < 1e-6 and abs(p - 21.0) < 1e-6:
            return "HUAWEI_LUNA"
        if abs(c - 104.0) < 1e-6 and abs(p - 4.0) < 1e-6:
            return "GREATPOWER"
    return "CUSTOM"


# -----------------------------
# Streamlit cached loader
# -----------------------------
@st.cache_data(show_spinner=False)
def get_params(config_path: str) -> Dict[str, Any]:
    ret = load_params(config_path)
    return _safe_dict_from_load_params(ret)


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="References & Validation", layout="wide")
st.title("References & Validation")
st.caption("Thesis traceability: parameter provenance, topology sanity checks, and reference alignment.")

# Active config path (from app selection)
default_cfg = str(st.session_state.get("active_config_path", "data/params_rack.yaml"))
with st.sidebar:
    st.subheader("Active configuration")
    st.text_input("Config path", value=default_cfg, key="rv_cfg_path")
    st.caption("This page reads the active config path from the app state. You can override it here for validation.")

cfg_path = str(st.session_state.get("rv_cfg_path", default_cfg)).strip()

# Load + derive
params = get_params(cfg_path)
derived = _derive_from_yaml(params)

# Reference dictionaries
HUAWEI_REF = {
    "label": "Huawei LUNA-inspired rack (reference model)",
    "chemistry": "LFP",
    "nominal_cell_voltage_v": 3.2,
    "cells_in_series_per_pack": 16,
    "packs_in_series_per_rack": 21,
    "rack_nominal_voltage_v": 1075.2,
    "rack_v_min_abs_v": 840.0,
    "rack_v_max_abs_v": 1226.4,
    "rack_nominal_energy_kwh_datasheet": 346.752,  # depending on pack energy definition; ~1% tolerance accepted
    "soft_cell_window_v": (2.50, 3.50),            # typical BMS operational in your YAML
    "hard_fault_uv_v": 2.50,
    "hard_fault_ov_v": 3.65,
}

GREATPOWER_REF = {
    "label": "Great Power Max-20HC-3720-1P rack (ULTRA-280Ah-1P, reference excerpt)",
    "chemistry": "LFP",
    "nominal_cell_voltage_v": 3.2,
    "cell_capacity_ah": 280.0,
    "cells_in_series_per_pack": 104,
    "packs_in_series_per_rack": 4,   # 4 packs in series -> 416S rack
    "rack_nominal_voltage_v": 1331.2,
    "rack_nominal_energy_kwh": 372.736,
    # Operational control window (soft) used by BMS logic (as per your text)
    "soft_cell_window_v": (2.60, 3.60),
    # Hard protection thresholds
    "hard_fault_uv_v": 2.50,
    "hard_fault_ov_v": 3.65,
    # Vendor rack table excerpts you provided (for context)
    "rack_voltage_range_op_v": (1081.6, 1497.6),     # 416 * (2.6..3.6)
    "rack_voltage_range_ratedchg_v": (1164.8, 1476.8),  # 416 * (2.8..3.55)
    "rated_current_a": 280.0,
    "max_current_a_5min": 320.0,
    "charge_temp_c": (0.0, 60.0),
    "discharge_temp_c": (-30.0, 60.0),
}

auto_ref = _detect_reference(derived)

with st.sidebar:
    st.subheader("Reference selection")
    ref_sel = st.selectbox(
        "Reference dataset",
        ["Auto-detect", "Huawei LUNA-inspired", "Great Power (Max-20HC / 372.7 kWh rack)"],
        index=0,
        help="Auto-detect chooses based on topology (cells-in-series per pack, packs-in-series per rack).",
        key="rv_ref_sel",
    )

if ref_sel == "Huawei LUNA-inspired":
    REF = HUAWEI_REF
    ref_key = "HUAWEI_LUNA"
elif ref_sel == "Great Power (Max-20HC / 372.7 kWh rack)":
    REF = GREATPOWER_REF
    ref_key = "GREATPOWER"
else:
    if auto_ref == "GREATPOWER":
        REF = GREATPOWER_REF
        ref_key = "GREATPOWER"
    else:
        REF = HUAWEI_REF
        ref_key = "HUAWEI_LUNA" if auto_ref == "HUAWEI_LUNA" else "CUSTOM"

st.subheader("Active config snapshot")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Config path", cfg_path)
with m2:
    st.metric("Chemistry (YAML)", str((params.get("chemistry", {}) or {}).get("name", "n/a")))
with m3:
    st.metric("Topology (YAML)", f"{_as_int(derived.get('cells_s', 0))}S × {_as_int(derived.get('packs_s', 0))} packs")
with m4:
    st.metric("Reference used", REF["label"])

st.divider()

# -----------------------------
# Great Power reference text (requested)
# -----------------------------
with st.expander("Great Power reference excerpt (Cell + Rack specification used in thesis)", expanded=(ref_key == "GREATPOWER")):
    st.markdown(
        """
### 2. System Overview

#### 2.1 Cell Configuration
The **Max-20HC-3720-1P** energy storage system is based on **Great Power ULTRA-280Ah-1P** lithium iron phosphate (LFP) cells.

**Battery Cell Specifications (excerpt, Great Power ULTRA-280Ah-1P):**
- Model: **GSP71173204F**
- Nominal capacity: **280 Ah**
- Nominal voltage: **3.2 V**
- Nominal energy: **896 Wh**
- Voltage range:
  - **2.5–3.65 V** at **T > 0°C**
  - **2.0–3.65 V** at **T ≤ 0°C**
- Charging temperature: **0–60°C**
- Discharging temperature: **−30–60°C**
- Storage temperature: **−30–60°C** (recommended **−10–35°C**)
- Cycle life: **≥ 6000 @ 70%**

#### 2.2 Rack Configuration
Great Power container architecture indicates:
- Pack: **104S1P**
- Rack: **4 packs in series → 416S1P**

**Rack configuration used in this study (Great Power rack profile):**
- Chemistry: **LFP**
- Rack nominal voltage: **1331.2 V**
- Rack nominal energy: **372.736 kWh**
- Rated current: **280 A**
- Max current: **320 A (≤ 5 min)**

**Operational and protection voltage references (as used for BMS logic):**
- Operational (soft) cell window (normal control): **2.6–3.60 V/cell**
- Hard fault thresholds (safety): **UV = 2.5 V/cell**, **OV = 3.65 V/cell**

Scope note: The **320 A / 5 min** peak-current capability is a vendor-stated specification; dedicated endurance verification is considered out of scope for the present offline validation.
        """
    )

# -----------------------------
# Sanity checks vs selected reference
# -----------------------------
st.subheader("Topology sanity checks (YAML vs selected reference)")

chem_yaml = str((params.get("chemistry", {}) or {}).get("name", "n/a")).strip()
rows: List[Dict[str, Any]] = []

# Chemistry
chem_ok = (chem_yaml.lower() == str(REF["chemistry"]).strip().lower())
rows.append({
    "Item": "Chemistry",
    "Reference": REF["chemistry"],
    "YAML/Derived": chem_yaml,
    "Check": "PASS" if chem_ok else "INFO",
})

# Nominal cell voltage
rows.append({
    "Item": "Nominal cell voltage [V]",
    "Reference": REF["nominal_cell_voltage_v"],
    "YAML/Derived": derived.get("v_cell_nom_v", None),
    "Check": "PASS" if check_close(derived.get("v_cell_nom_v"), float(REF["nominal_cell_voltage_v"]), tol_rel=0.001) else "FAIL",
})

# Cells in series per pack
rows.append({
    "Item": "Cells in series per pack",
    "Reference": REF["cells_in_series_per_pack"],
    "YAML/Derived": derived.get("cells_s", None),
    "Check": "PASS" if check_close(derived.get("cells_s"), float(REF["cells_in_series_per_pack"]), tol_rel=0.0) else "FAIL",
})

# Packs in series per rack
rows.append({
    "Item": "Packs in series per rack",
    "Reference": REF["packs_in_series_per_rack"],
    "YAML/Derived": derived.get("packs_s", None),
    "Check": "PASS" if check_close(derived.get("packs_s"), float(REF["packs_in_series_per_rack"]), tol_rel=0.0) else "FAIL",
})

# Rack nominal voltage
rows.append({
    "Item": "Rack nominal voltage [V] (derived)",
    "Reference": REF.get("rack_nominal_voltage_v", None),
    "YAML/Derived": derived.get("v_rack_v", None),
    "Check": "PASS" if check_close(derived.get("v_rack_v"), float(REF.get("rack_nominal_voltage_v", np.nan)), tol_rel=0.002) else "FAIL",
})

# Rack energy (if reference has it)
ref_e = REF.get("rack_nominal_energy_kwh", REF.get("rack_nominal_energy_kwh_datasheet", None))
if ref_e is not None:
    rows.append({
        "Item": "Rack nominal energy [kWh] (derived)",
        "Reference": float(ref_e),
        "YAML/Derived": derived.get("e_rack_kwh", None),
        "Check": "PASS" if check_close(derived.get("e_rack_kwh"), float(ref_e), tol_rel=0.02) else "WARN",
    })

# Soft operational window checks (limits.v_cell_min/max vs reference soft window)
soft = REF.get("soft_cell_window_v", None)
if soft is not None:
    vmin_ref, vmax_ref = float(soft[0]), float(soft[1])
    rows.append({
        "Item": "Soft operational V_cell_min [V] (limits.v_cell_min_v)",
        "Reference": vmin_ref,
        "YAML/Derived": derived.get("v_cell_min_op_v", None),
        "Check": "PASS" if check_close(derived.get("v_cell_min_op_v"), vmin_ref, tol_rel=0.05) else "INFO",
    })
    rows.append({
        "Item": "Soft operational V_cell_max [V] (limits.v_cell_max_v)",
        "Reference": vmax_ref,
        "YAML/Derived": derived.get("v_cell_max_op_v", None),
        "Check": "PASS" if check_close(derived.get("v_cell_max_op_v"), vmax_ref, tol_rel=0.05) else "INFO",
    })

# Hard thresholds (faults.uv_cell_v / faults.ov_cell_v vs reference)
faults = params.get("faults", {}) or {}
uv_yaml = _as_float(faults.get("uv_cell_v", np.nan))
ov_yaml = _as_float(faults.get("ov_cell_v", np.nan))

rows.append({
    "Item": "UV fault threshold [V] (faults.uv_cell_v)",
    "Reference": float(REF.get("hard_fault_uv_v", np.nan)),
    "YAML/Derived": _maybe(uv_yaml),
    "Check": "PASS" if check_close(_maybe(uv_yaml), float(REF.get("hard_fault_uv_v", np.nan)), tol_rel=0.0) else "INFO",
})
rows.append({
    "Item": "OV fault threshold [V] (faults.ov_cell_v)",
    "Reference": float(REF.get("hard_fault_ov_v", np.nan)),
    "YAML/Derived": _maybe(ov_yaml),
    "Check": "PASS" if check_close(_maybe(ov_yaml), float(REF.get("hard_fault_ov_v", np.nan)), tol_rel=0.0) else "INFO",
})

st.dataframe(rows, use_container_width=True, hide_index=True)

st.info(
    "Interpretation guidance:\n"
    "- **PASS**: matches the selected reference.\n"
    "- **INFO**: valid but not expected to match exactly (e.g., your operational BMS window may differ intentionally).\n"
    "- **WARN**: approximate match but outside tight tolerance (often acceptable for reference-inspired modeling).\n"
    "- **FAIL**: likely a topology mismatch (wrong series counts or incorrect nominal voltage)."
)

st.divider()

# -----------------------------
# Parameter provenance + limitations
# -----------------------------
st.subheader("Parameter provenance (what comes from where)")
st.markdown(
    """
- **YAML parameters**: system topology, ECM parameters, thermal constants, soft limits (`limits.*`), and protection thresholds (`faults.*`).
- **Reference datasheets**: topology targets (series counts), nominal voltage/energy, and hard operating boundaries.
- **Derived values**: pack/rack nominal voltage and energy computed from topology + nominal cell voltage + nominal capacity.
"""
)

st.subheader("Known limitations (traceability notes)")
st.markdown(
    """
- Reference models are **reference-inspired / excerpt-based** and intentionally simplified (no proprietary controls).
- The thermal model is first-order; use **overrides/injections** for deterministic safety logic demonstrations.
- The soft operational voltage window in YAML may intentionally differ from vendor “max range”; this is acceptable if explained in the thesis.
"""
)
