# src/config.py
from __future__ import annotations

from pathlib import Path
import warnings
import yaml


def _project_root() -> Path:
    # src/config.py -> src/ -> project root
    return Path(__file__).resolve().parents[1]


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _validate_params(params: dict) -> None:
    """
    Sanity-check: compare derived electrical quantities against the reference fields.
    Does not mutate params; only emits warnings (or raises on clearly invalid inputs).
    """
    # --- Required core fields (hard errors) ---
    required = [
        ("structure", "cells_in_series_per_pack"),
        ("structure", "packs_in_series_per_rack"),
        ("electrical_model", "q_nom_ah"),
        ("limits", "v_cell_min_v"),
        ("limits", "v_cell_max_v"),
        ("simulation", "dt_s"),
    ]
    for path in required:
        if _get(params, *path, default=None) is None:
            raise ValueError(f"Missing required config key: {'.'.join(path)}")

    # --- Derivations ---
    v_cell_nom = float(_get(params, "chemistry", "nominal_cell_voltage_v", default=float("nan")))
    q_ah = float(_get(params, "chemistry", "cell_capacity_ah", default=_get(params, "electrical_model", "q_nom_ah")))
    nS_pack = int(_get(params, "structure", "cells_in_series_per_pack"))
    n_pack_series = int(_get(params, "structure", "packs_in_series_per_rack"))

    # Use nominal cell voltage if present; otherwise skip nominal derivations
    if v_cell_nom == v_cell_nom:  # not NaN
        v_pack_nom_calc = nS_pack * v_cell_nom
        e_pack_kwh_calc = (v_pack_nom_calc * q_ah) / 1000.0
        v_rack_nom_calc = v_pack_nom_calc * n_pack_series
        e_rack_kwh_calc = e_pack_kwh_calc * n_pack_series

        # Compare with reference fields if present
        v_pack_ref = _get(params, "pack", "rated_voltage_v", default=None)
        e_pack_ref = _get(params, "pack", "nominal_energy_kwh", default=None)
        v_rack_ref = _get(params, "rack", "nominal_voltage_v", default=None)
        e_rack_ref = _get(params, "rack", "nominal_energy_kwh", default=None)

        def _warn_if_mismatch(name, calc, ref, tol_rel=0.01, tol_abs=0.05):
            if ref is None:
                return
            ref = float(ref)
            err = abs(calc - ref)
            if err > max(tol_abs, tol_rel * max(1.0, abs(ref))):
                warnings.warn(
                    f"[params sanity-check] {name} mismatch: calc={calc:.4g}, ref={ref:.4g} (err={err:.4g}).",
                    RuntimeWarning,
                )

        _warn_if_mismatch("pack.rated_voltage_v", v_pack_nom_calc, v_pack_ref, tol_rel=0.005, tol_abs=0.2)
        _warn_if_mismatch("pack.nominal_energy_kwh", e_pack_kwh_calc, e_pack_ref, tol_rel=0.02, tol_abs=0.2)
        _warn_if_mismatch("rack.nominal_voltage_v", v_rack_nom_calc, v_rack_ref, tol_rel=0.005, tol_abs=1.0)
        _warn_if_mismatch("rack.nominal_energy_kwh", e_rack_kwh_calc, e_rack_ref, tol_rel=0.02, tol_abs=2.0)

    # --- Limits consistency (soft warnings) ---
    v_cell_min = float(_get(params, "limits", "v_cell_min_v"))
    v_cell_max = float(_get(params, "limits", "v_cell_max_v"))
    if v_cell_min >= v_cell_max:
        raise ValueError(f"Invalid limits: v_cell_min_v ({v_cell_min}) >= v_cell_max_v ({v_cell_max})")

    v_cell_abs_max = _get(params, "pack", "cell_voltage_max_v", default=None)
    if v_cell_abs_max is not None and float(v_cell_max) > float(v_cell_abs_max) + 1e-9:
        warnings.warn(
            f"[params sanity-check] limits.v_cell_max_v ({v_cell_max}) is above pack.cell_voltage_max_v ({v_cell_abs_max}).",
            RuntimeWarning,
        )

    # --- Hardware rated current consistency (soft warning) ---
    p_rack_kw = _get(params, "hardware", "rated_power_per_rack_kw", default=None)
    v_bus = _get(params, "hardware", "rated_operating_voltage_v", default=None)
    i_rack_ref = _get(params, "hardware", "rated_current_per_rack_a", default=None)
    if p_rack_kw is not None and v_bus is not None and i_rack_ref is not None:
        i_calc = (float(p_rack_kw) * 1000.0) / max(1e-9, float(v_bus))
        if abs(i_calc - float(i_rack_ref)) > 2.0:  # ~2A tolerance
            warnings.warn(
                f"[params sanity-check] hardware rated current mismatch: "
                f"calc={i_calc:.2f}A vs rated_current_per_rack_a={float(i_rack_ref):.2f}A "
                f"(from {p_rack_kw}kW @ {v_bus}V).",
                RuntimeWarning,
            )


def load_params(path: str = "data/params_rack.yaml", *, validate: bool = True) -> dict:
    """
    Load rack and BMS parameters from a YAML file.

    - Resolves relative paths against the project root for robustness.
    - Optionally runs a sanity-check (warnings, no mutation).
    """
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = _project_root() / path_obj

    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj.resolve()}")

    with path_obj.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    if validate:
        _validate_params(params)

    return params
