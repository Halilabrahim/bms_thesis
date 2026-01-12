# src/bms_logic.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class BMSControlParams:
    """
    Parameters for BMS current limitation logic.

    Notes
    -----
    - Backward compatible: if the *_charge_* / *_discharge_* low-temp fields are not provided,
      the legacy fields t_low_cutoff_c and t_low_derate_start_c are used for both charge/discharge.
    """
    i_charge_max_a: float
    i_discharge_max_a: float

    soc_low_cutoff: float
    soc_low_derate_start: float
    soc_high_cutoff: float
    soc_high_derate_start: float

    # Legacy (fallback for both directions if split params are None)
    t_low_cutoff_c: float
    t_low_derate_start_c: float

    t_high_cutoff_c: float
    t_high_derate_start_c: float

    # Optional split low-temp limits (preferred)
    t_low_cutoff_discharge_c: Optional[float] = None
    t_low_derate_start_discharge_c: Optional[float] = None
    t_low_cutoff_charge_c: Optional[float] = None
    t_low_derate_start_charge_c: Optional[float] = None

    v_cell_min_v: float = 0.0
    v_cell_max_v: float = 0.0
    v_margin_v: float = 0.05
    v_margin_low_v: Optional[float] = None
    v_margin_high_v: Optional[float] = None


def _clip(x: float, xmin: float, xmax: float) -> float:
    return max(xmin, min(xmax, x))


def _linear_scale(x: float, x0: float, x1: float) -> float:
    """
    Piecewise linear scale between 0 and 1.
    x <= x0 -> 0
    x >= x1 -> 1
    """
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0
    # Guard against division by zero if x0 == x1
    if abs(x1 - x0) < 1e-12:
        return 0.0
    return (x - x0) / (x1 - x0)


def compute_current_limits(
    soc_hat: float,
    t_rack_c: float,
    v_cell_min: float,
    v_cell_max: float,
    params: BMSControlParams,
) -> Dict[str, float]:
    """
    Compute allowable charge and discharge current magnitudes based on
    SoC, temperature and cell voltages.

    Notes
    -----
    - Returned limits are magnitudes (positive). Sign handling is responsibility of the caller.
    - Also returns debug scalars for UI/SCADA display.

    Dominant limiter codes (float)
    ------------------------------
    0   : NONE (no derate)
    10  : TEMP_LOW_CUTOFF
    11  : TEMP_LOW_DERATE
    20  : TEMP_HIGH_CUTOFF
    21  : TEMP_HIGH_DERATE
    30  : VMIN_LIMIT (discharge)
    31  : VMIN_MARGIN (discharge)
    40  : VMAX_LIMIT (charge)
    41  : VMAX_MARGIN (charge)
    50  : SOC_LOW_CUTOFF (discharge)
    51  : SOC_LOW_DERATE (discharge)
    60  : SOC_HIGH_CUTOFF (charge)
    61  : SOC_HIGH_DERATE (charge)
    """
    # --- 1) Start from rated limits ---
    i_chg = float(params.i_charge_max_a)
    i_dis = float(params.i_discharge_max_a)

    # ---------- SoC-based derating ----------
    soc = _clip(float(soc_hat), 0.0, 1.0)

    dis_soc_scale = _linear_scale(
        soc,
        float(params.soc_low_cutoff),
        float(params.soc_low_derate_start),
    )

    chg_soc_scale = _linear_scale(
        1.0 - soc,
        1.0 - float(params.soc_high_cutoff),
        1.0 - float(params.soc_high_derate_start),
    )

    # ---------- Temperature-based derating (split charge/discharge) ----------
    t = float(t_rack_c)

    # discharge low-temp (fallback to legacy)
    t_low_cut_dis = float(params.t_low_cutoff_discharge_c) if params.t_low_cutoff_discharge_c is not None else float(params.t_low_cutoff_c)
    t_low_der_dis = float(params.t_low_derate_start_discharge_c) if params.t_low_derate_start_discharge_c is not None else float(params.t_low_derate_start_c)

    # charge low-temp (fallback to legacy)
    t_low_cut_chg = float(params.t_low_cutoff_charge_c) if params.t_low_cutoff_charge_c is not None else float(params.t_low_cutoff_c)
    t_low_der_chg = float(params.t_low_derate_start_charge_c) if params.t_low_derate_start_charge_c is not None else float(params.t_low_derate_start_c)

    t_low_scale_dis = _linear_scale(t, t_low_cut_dis, t_low_der_dis)
    t_low_scale_chg = _linear_scale(t, t_low_cut_chg, t_low_der_chg)

    # high temperature scale (shared)
    t_high_cut = float(params.t_high_cutoff_c)
    t_high_der = float(params.t_high_derate_start_c)
    t_high_scale = _linear_scale(
        t_high_cut - t,
        0.0,
        t_high_cut - t_high_der,
    )

    temp_scale_dis = t_low_scale_dis * t_high_scale
    temp_scale_chg = t_low_scale_chg * t_high_scale

    # ---------- Voltage-based derating ----------
    vmin = float(v_cell_min)
    vmax = float(v_cell_max)

    v_margin_low = float(params.v_margin_low_v) if params.v_margin_low_v is not None else float(params.v_margin_v)
    v_margin_high = float(params.v_margin_high_v) if params.v_margin_high_v is not None else float(params.v_margin_v)

    dis_v_scale = _linear_scale(
        vmin - float(params.v_cell_min_v),
        0.0,
        v_margin_low,
    )

    chg_v_scale = _linear_scale(
        float(params.v_cell_max_v) - vmax,
        0.0,
        v_margin_high,
    )

    # ---------- Dominant limiter codes ----------
    def _dominant_discharge_code() -> float:
        if dis_soc_scale <= 0.0:
            return 50.0
        if t_low_scale_dis <= 0.0:
            return 10.0
        if t_high_scale <= 0.0:
            return 20.0
        if dis_v_scale <= 0.0:
            return 30.0

        m = min(dis_soc_scale, t_low_scale_dis, t_high_scale, dis_v_scale)
        if m >= 0.999999:
            return 0.0

        tol = 1e-9
        if dis_v_scale <= m + tol:
            return 31.0
        if dis_soc_scale <= m + tol:
            return 51.0
        if t_low_scale_dis <= m + tol:
            return 11.0
        return 21.0

    def _dominant_charge_code() -> float:
        if chg_soc_scale <= 0.0:
            return 60.0
        if t_low_scale_chg <= 0.0:
            return 10.0
        if t_high_scale <= 0.0:
            return 20.0
        if chg_v_scale <= 0.0:
            return 40.0

        m = min(chg_soc_scale, t_low_scale_chg, t_high_scale, chg_v_scale)
        if m >= 0.999999:
            return 0.0

        tol = 1e-9
        if chg_v_scale <= m + tol:
            return 41.0
        if chg_soc_scale <= m + tol:
            return 61.0
        if t_low_scale_chg <= m + tol:
            return 11.0
        return 21.0

    # ---------- Combine scales (product) ----------
    dis_scale = dis_soc_scale * temp_scale_dis * dis_v_scale
    chg_scale = chg_soc_scale * temp_scale_chg * chg_v_scale

    i_dis_allowed = i_dis * dis_scale
    i_chg_allowed = i_chg * chg_scale

    return {
        # main outputs
        "i_discharge_max_allowed": float(i_dis_allowed),
        "i_charge_max_allowed": float(i_chg_allowed),

        # debug scalars for UI
        "scale_dis_total": float(dis_scale),
        "scale_chg_total": float(chg_scale),
        "scale_soc_dis": float(dis_soc_scale),
        "scale_soc_chg": float(chg_soc_scale),

        # keep legacy key name but map to DISCHARGE low-temp scale (so older UI won't break)
        "scale_t_low": float(t_low_scale_dis),
        "scale_t_low_dis": float(t_low_scale_dis),
        "scale_t_low_chg": float(t_low_scale_chg),

        "scale_t_high": float(t_high_scale),
        "scale_v_dis": float(dis_v_scale),
        "scale_v_chg": float(chg_v_scale),
        "code_limit_dis": float(_dominant_discharge_code()),
        "code_limit_chg": float(_dominant_charge_code()),
    }
