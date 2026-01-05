from dataclasses import dataclass
from typing import Dict


@dataclass
class BMSControlParams:
    """
    Parameters for BMS current limitation logic.
    """
    i_charge_max_a: float
    i_discharge_max_a: float

    soc_low_cutoff: float
    soc_low_derate_start: float
    soc_high_cutoff: float
    soc_high_derate_start: float

    t_low_cutoff_c: float
    t_low_derate_start_c: float
    t_high_cutoff_c: float
    t_high_derate_start_c: float

    v_cell_min_v: float
    v_cell_max_v: float
    v_margin_v: float


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
    - The returned limits are **magnitudes** (positive numbers). Sign handling is
      the responsibility of the caller.
    - In addition to the limits, the function also returns debug scalars that are
      convenient for SCADA/UI display (scales and a "dominant limiter" code).

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
    i_chg = params.i_charge_max_a
    i_dis = params.i_discharge_max_a

    # ---------- SoC-based derating ----------
    soc = _clip(soc_hat, 0.0, 1.0)

    # Discharge: derate as SoC approaches lower limit
    dis_soc_scale = _linear_scale(
        soc,
        params.soc_low_cutoff,        # 0 -> below cutoff
        params.soc_low_derate_start,  # 1 -> above derate start
    )

    # Charge: derate as SoC approaches upper limit
    chg_soc_scale = _linear_scale(
        1.0 - soc,
        1.0 - params.soc_high_cutoff,       # 0 -> above cutoff
        1.0 - params.soc_high_derate_start, # 1 -> below derate start
    )

    # ---------- Temperature-based derating ----------
    t = t_rack_c

    # low temperature
    t_low_scale = _linear_scale(
        t,
        params.t_low_cutoff_c,
        params.t_low_derate_start_c,
    )

    # high temperature (1 at cool, 0 at too hot)
    t_high_scale = _linear_scale(
        params.t_high_cutoff_c - t,
        0.0,
        params.t_high_cutoff_c - params.t_high_derate_start_c,
    )

    temp_scale = t_low_scale * t_high_scale

    # ---------- Voltage-based derating ----------
    vmin = v_cell_min
    vmax = v_cell_max

    # Discharge: derate when vmin -> v_cell_min_v
    dis_v_scale = _linear_scale(
        vmin - params.v_cell_min_v,
        0.0,
        params.v_margin_v,
    )

    # Charge: derate when vmax -> v_cell_max_v
    chg_v_scale = _linear_scale(
        params.v_cell_max_v - vmax,
        0.0,
        params.v_margin_v,
    )

    def _dominant_discharge_code() -> float:
        # Cutoffs first
        if dis_soc_scale <= 0.0:
            return 50.0
        if t_low_scale <= 0.0:
            return 10.0
        if t_high_scale <= 0.0:
            return 20.0
        if dis_v_scale <= 0.0:
            return 30.0

        m = min(dis_soc_scale, t_low_scale, t_high_scale, dis_v_scale)
        if m >= 0.999999:
            return 0.0

        # Deterministic tie-break: voltage -> SoC -> temp low -> temp high
        tol = 1e-9
        if dis_v_scale <= m + tol:
            return 31.0
        if dis_soc_scale <= m + tol:
            return 51.0
        if t_low_scale <= m + tol:
            return 11.0
        return 21.0

    def _dominant_charge_code() -> float:
        # Cutoffs first
        if chg_soc_scale <= 0.0:
            return 60.0
        if t_low_scale <= 0.0:
            return 10.0
        if t_high_scale <= 0.0:
            return 20.0
        if chg_v_scale <= 0.0:
            return 40.0

        m = min(chg_soc_scale, t_low_scale, t_high_scale, chg_v_scale)
        if m >= 0.999999:
            return 0.0

        # Deterministic tie-break: voltage -> SoC -> temp low -> temp high
        tol = 1e-9
        if chg_v_scale <= m + tol:
            return 41.0
        if chg_soc_scale <= m + tol:
            return 61.0
        if t_low_scale <= m + tol:
            return 11.0
        return 21.0

    # ---------- Combine scales (product) ----------
    dis_scale = dis_soc_scale * temp_scale * dis_v_scale
    chg_scale = chg_soc_scale * temp_scale * chg_v_scale

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
        "scale_t_low": float(t_low_scale),
        "scale_t_high": float(t_high_scale),
        "scale_v_dis": float(dis_v_scale),
        "scale_v_chg": float(chg_v_scale),
        "code_limit_dis": float(_dominant_discharge_code()),
        "code_limit_chg": float(_dominant_charge_code()),
    }
