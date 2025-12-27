# pages/2_References_Validation.py
from __future__ import annotations

from typing import Any, Dict, Optional, List

import numpy as np
import streamlit as st

from src.config import load_params


@st.cache_data
def get_params() -> Dict[str, Any]:
    return load_params()


def _safe_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def derive_from_yaml(params: Dict[str, Any]) -> Dict[str, Optional[float]]:
    chem = params.get("chemistry", {}) or {}
    struct = params.get("structure", {}) or {}
    em = params.get("electrical_model", {}) or {}
    limits = params.get("limits", {}) or {}
    pack = params.get("pack", {}) or {}
    rack = params.get("rack", {}) or {}

    # Nominal / derived from topology + nominal cell voltage
    v_cell_nom = float(chem.get("nominal_cell_voltage_v", 3.2))
    q_ah = float(em.get("q_nom_ah", np.nan))

    cells_s = float(struct.get("cells_in_series_per_pack", np.nan))
    packs_s = float(struct.get("packs_in_series_per_rack", np.nan))

    v_pack = cells_s * v_cell_nom
    e_pack_kwh = v_pack * q_ah / 1000.0

    v_rack = packs_s * v_pack
    e_rack_kwh = packs_s * e_pack_kwh

    # Operational limits (BMS soft limits) -> derived rack range
    v_cell_min_op = float(limits.get("v_cell_min_v", np.nan))
    v_cell_max_op = float(limits.get("v_cell_max_v", np.nan))
    v_rack_min_op = v_cell_min_op * cells_s * packs_s
    v_rack_max_op = v_cell_max_op * cells_s * packs_s

    # Absolute / reference-inspired rack range (YAML rack section)
    v_rack_min_abs_yaml = float(rack.get("v_min_v", np.nan))
    v_rack_max_abs_yaml = float(rack.get("v_max_v", np.nan))

    # (Optional) Absolute derived from chemistry max if present
    v_cell_max_abs = float(pack.get("cell_voltage_max_v", np.nan))
    v_rack_max_abs_derived = v_cell_max_abs * cells_s * packs_s

    return {
        "v_cell_nom_v": v_cell_nom,
        "q_ah": q_ah,
        "cells_s": cells_s,
        "packs_s": packs_s,
        "v_pack_v": v_pack,
        "e_pack_kwh": e_pack_kwh,
        "v_rack_v": v_rack,
        "e_rack_kwh": e_rack_kwh,
        # operational (from limits)
        "v_cell_min_op_v": v_cell_min_op,
        "v_cell_max_op_v": v_cell_max_op,
        "v_rack_min_op_v": v_rack_min_op,
        "v_rack_max_op_v": v_rack_max_op,
        # absolute (from rack yaml)
        "v_rack_min_abs_yaml_v": v_rack_min_abs_yaml,
        "v_rack_max_abs_yaml_v": v_rack_max_abs_yaml,
        # absolute (derived from chemistry max, if available)
        "v_rack_max_abs_derived_v": v_rack_max_abs_derived,
    }


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
    """
    PASS if a <= b (+tol). Useful for "operational limit must be below absolute limit".
    """
    if a is None or b is None:
        return False
    if not (np.isfinite(a) and np.isfinite(b)):
        return False
    return float(a) <= float(b) + float(tol_abs)


st.set_page_config(page_title="References & Validation", layout="wide")
st.title("References & Validation")
st.caption("Thesis traceability: parameter provenance, topology sanity checks, and glossary.")

params = get_params()
derived = derive_from_yaml(params)

# ---- Huawei LUNA-inspired reference numbers (from your provided PDF) ----
HUAWEI_REF = {
    "chemistry": "LFP",
    "nominal_cell_voltage_v": 3.2,
    "pack_topology": "16S1P",
    "cells_in_series_per_pack": 16,
    "packs_in_series_per_rack": 21,
    "rack_nominal_voltage_v": 1075.2,
    "rack_v_min_v": 840.0,
    "rack_v_max_v": 1226.4,
    "rack_nominal_energy_kwh_datasheet": 346.752,
}

st.subheader("Reference rack model (Huawei LUNA-inspired)")
st.markdown(
    """
This project uses a **Huawei LUNA2000-2.0MWH-1HX inspired** rack model as a realistic reference.
The goal is **not** to replicate proprietary internals, but to simulate a credible utility-scale rack topology for
BMS logic validation and thesis-ready results.
"""
)

st.subheader("Topology sanity checks (YAML vs reference)")

chem_yaml = (params.get("chemistry", {}) or {}).get("name", "n/a")

rows = []

# Chemistry
chem_ok = (str(chem_yaml).strip().lower() == str(HUAWEI_REF["chemistry"]).strip().lower())
rows.append({
    "item": "Chemistry",
    "reference": HUAWEI_REF["chemistry"],
    "yaml/derived": str(chem_yaml),
    "check": "PASS" if chem_ok else "INFO",
})

# Nominal cell voltage
rows.append({
    "item": "Nominal cell voltage [V]",
    "reference": HUAWEI_REF["nominal_cell_voltage_v"],
    "yaml/derived": float(derived["v_cell_nom_v"]) if derived["v_cell_nom_v"] is not None else None,
    "check": "PASS" if check_close(derived["v_cell_nom_v"], HUAWEI_REF["nominal_cell_voltage_v"], tol_rel=0.001) else "FAIL",
})

# Pack topology (cells series)
rows.append({
    "item": "Pack topology (cells series)",
    "reference": HUAWEI_REF["cells_in_series_per_pack"],
    "yaml/derived": float(derived["cells_s"]) if derived["cells_s"] is not None else None,
    "check": "PASS" if check_close(derived["cells_s"], float(HUAWEI_REF["cells_in_series_per_pack"]), tol_rel=0.0) else "FAIL",
})

# Packs in series per rack
rows.append({
    "item": "Packs in series per rack",
    "reference": HUAWEI_REF["packs_in_series_per_rack"],
    "yaml/derived": float(derived["packs_s"]) if derived["packs_s"] is not None else None,
    "check": "PASS" if check_close(derived["packs_s"], float(HUAWEI_REF["packs_in_series_per_rack"]), tol_rel=0.0) else "FAIL",
})

# Rack nominal voltage (derived)
rows.append({
    "item": "Rack nominal voltage [V]",
    "reference": HUAWEI_REF["rack_nominal_voltage_v"],
    "yaml/derived": float(derived["v_rack_v"]) if derived["v_rack_v"] is not None else None,
    "check": "PASS" if check_close(derived["v_rack_v"], HUAWEI_REF["rack_nominal_voltage_v"], tol_rel=0.001) else "FAIL",
})

# Rack absolute min/max (from YAML rack section) -> compare to reference absolute range
rows.append({
    "item": "Rack voltage min [V] (absolute, YAML rack.v_min_v)",
    "reference": HUAWEI_REF["rack_v_min_v"],
    "yaml/derived": float(derived["v_rack_min_abs_yaml_v"]) if derived["v_rack_min_abs_yaml_v"] is not None else None,
    "check": "PASS" if check_close(derived["v_rack_min_abs_yaml_v"], HUAWEI_REF["rack_v_min_v"], tol_rel=0.001) else "FAIL",
})

rows.append({
    "item": "Rack voltage max [V] (absolute, YAML rack.v_max_v)",
    "reference": HUAWEI_REF["rack_v_max_v"],
    "yaml/derived": float(derived["v_rack_max_abs_yaml_v"]) if derived["v_rack_max_abs_yaml_v"] is not None else None,
    "check": "PASS" if check_close(derived["v_rack_max_abs_yaml_v"], HUAWEI_REF["rack_v_max_v"], tol_rel=0.001) else "FAIL",
})

# Rack operational max (from BMS limits) -> should be <= absolute max (not equal)
op_max = derived.get("v_rack_max_op_v", None)
abs_max = derived.get("v_rack_max_abs_yaml_v", None)

rows.append({
    "item": "Rack voltage max [V] (operational, from limits.v_cell_max_v)",
    "reference": f"≤ {HUAWEI_REF['rack_v_max_v']:.1f} (must stay below absolute)",
    "yaml/derived": float(op_max) if op_max is not None else None,
    "check": "PASS" if check_leq(op_max, abs_max, tol_abs=1e-6) else "FAIL",
})

# Rack nominal energy (derived) vs datasheet (near-match)
rows.append({
    "item": "Rack nominal energy [kWh] (derived)",
    "reference": HUAWEI_REF["rack_nominal_energy_kwh_datasheet"],
    "yaml/derived": float(derived["e_rack_kwh"]) if derived["e_rack_kwh"] is not None else None,
    "check": "PASS" if check_close(derived["e_rack_kwh"], HUAWEI_REF["rack_nominal_energy_kwh_datasheet"], tol_rel=0.01) else "WARN",
})

st.dataframe(rows, use_container_width=True)

st.info(
    "Notes:\n"
    "- The datasheet rack voltage range (840–1226.4 V) is an **absolute/chemical** range.\n"
    "- BMS **operational** voltage limits are typically lower than the absolute max; therefore we validate:\n"
    "  operational max ≤ absolute max.\n"
    "- Energy can differ slightly depending on whether the datasheet uses 320Ah vs ~322.5Ah equivalent; "
    "we accept ~1% deviation as OK for a *reference-inspired* model."
)

st.subheader("Parameter provenance (what came from where)")
st.markdown(
    """
- **YAML parameters**: system topology, limits, ECM parameters, thermal constants.
- **Reference datasheet**: topology targets (16S1P pack, 21 packs per rack, voltage range).
- **Derived constants**: pack/rack nominal voltage and energy derived from nominal cell voltage and Ah.
"""
)

st.subheader("Known limitations / TODO")
st.markdown(
    """
- The rack model is *reference-inspired* and intentionally simplified (no proprietary control loops).
- Thermal model is first-order (use overrides/injections for deterministic safety logic validation).
- Future work: add validation snapshots (V–I at SoC points, thermal step response), and parameter citations per table row.
"""
)
