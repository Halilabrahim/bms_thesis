# src/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union
import warnings

import yaml

# -----------------------------------------------------------------------------
# Streamlit profile registry
# -----------------------------------------------------------------------------
# UI şu yapıyı bekliyor:
#   PROFILES[profile_key]["params"] -> params yaml
# Opsiyonel:
#   PROFILES[profile_key]["scenarios"] -> scenarios yaml
#   PROFILES[profile_key]["label"] -> UI’da görünen isim (istersen)
# -----------------------------------------------------------------------------

PROFILES: Dict[str, Dict[str, str]] = {
    "LUNA": {
        "label": "Huawei LUNA (baseline)",
        "params": "data/params_rack.yaml",
        "scenarios": "data/scenarios.yaml",
    },
    "GREATPOWER": {
        "label": "Great Power rack",
        "params": "data/params_rack_greatpower.yaml",
        "scenarios": "data/scenarios_greatpower.yaml",
    },
}

DEFAULT_PROFILE: str = "LUNA"


def _project_root() -> Path:
    # src/config.py -> project root
    return Path(__file__).resolve().parent.parent


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _validate_params(params: Dict[str, Any]) -> None:
    """
    Lightweight sanity checks (non-fatal).
    Warns if key constraints look inconsistent, to avoid silent misconfiguration.
    """
    limits = _get(params, "limits", default={}) or {}
    faults = _get(params, "faults", default={}) or {}
    structure = _get(params, "structure", default={}) or {}
    pack = _get(params, "pack", default={}) or {}
    rack = _get(params, "rack", default={}) or {}

    v_cell_min = float(limits.get("v_cell_min_v", float("nan")))
    v_cell_max = float(limits.get("v_cell_max_v", float("nan")))
    uv_cell_v = float(faults.get("uv_cell_v", v_cell_min))
    ov_cell_v = float(faults.get("ov_cell_v", float("nan")))

    # Basic voltage hierarchy checks (soft limits vs fault thresholds)
    if (v_cell_min == v_cell_min) and (uv_cell_v == uv_cell_v):
        if uv_cell_v > v_cell_min + 1e-9:
            warnings.warn(
                f"[params sanity-check] faults.uv_cell_v ({uv_cell_v}) is above limits.v_cell_min_v ({v_cell_min}). "
                "Usually UV threshold is <= minimum operational voltage.",
                RuntimeWarning,
            )

    if (v_cell_max == v_cell_max) and (ov_cell_v == ov_cell_v):
        if ov_cell_v < v_cell_max - 1e-9:
            warnings.warn(
                f"[params sanity-check] faults.ov_cell_v ({ov_cell_v}) is below limits.v_cell_max_v ({v_cell_max}). "
                "Usually OV threshold is >= maximum operational voltage.",
                RuntimeWarning,
            )

    # Structure-derived checks (if those fields exist)
    try:
        cells_s = int(structure.get("cells_in_series_per_pack", 0))
        packs_s = int(structure.get("packs_in_series_per_rack", 0))
    except Exception:
        cells_s, packs_s = 0, 0

    if cells_s > 0 and packs_s > 0:
        # If rack nominal voltage exists, it should be close-ish to cell_nom*cells_s*packs_s
        try:
            ch = _get(params, "chemistry", default={}) or {}
            v_cell_nom = float(ch.get("nominal_cell_voltage_v", float("nan")))
            v_rack_nom = float(rack.get("nominal_voltage_v", float("nan")))
            if (v_cell_nom == v_cell_nom) and (v_rack_nom == v_rack_nom):
                est = v_cell_nom * cells_s * packs_s
                if abs(v_rack_nom - est) / max(est, 1e-9) > 0.25:
                    warnings.warn(
                        f"[params sanity-check] rack.nominal_voltage_v ({v_rack_nom}) differs from "
                        f"nominal_cell_voltage_v*cells_in_series_per_pack*packs_in_series_per_rack (~{est:.1f}).",
                        RuntimeWarning,
                    )
        except Exception:
            pass

    # Pack/rack energy consistency (non-fatal)
    try:
        e_pack = float(pack.get("nominal_energy_kwh", float("nan")))
        e_rack = float(rack.get("nominal_energy_kwh", float("nan")))
        if (e_pack == e_pack) and (e_rack == e_rack) and packs_s > 0:
            est = e_pack * packs_s
            if abs(e_rack - est) / max(est, 1e-9) > 0.25:
                warnings.warn(
                    f"[params sanity-check] rack.nominal_energy_kwh ({e_rack}) differs from "
                    f"pack.nominal_energy_kwh*packs_in_series_per_rack (~{est:.2f}).",
                    RuntimeWarning,
                )
    except Exception:
        pass


def load_params(config: Union[str, Path], *, validate: bool = True) -> Dict[str, Any]:
    """
    Load YAML parameters.

    Supports either:
      - a YAML path (relative to project root), or
      - a profile key present in PROFILES (e.g., "LUNA", "GREATPOWER")
    """
    if isinstance(config, str) and config in PROFILES:
        config = PROFILES[config]["params"]

    p = Path(str(config))
    if not p.is_absolute():
        p = (_project_root() / p).resolve()

    if not p.exists():
        raise FileNotFoundError(f"params file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}

    if validate:
        _validate_params(params)

    # Provenance (UI’da göstermek için)
    params.setdefault("meta", {})
    if isinstance(params["meta"], dict):
        params["meta"].setdefault("loaded_from", str(p))

    return params
