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
    Compute allowable charge and discharge currents based on
    SoC, temperature and cell voltages.

    Returns
    -------
    dict with:
      - i_charge_max_allowed (negative direction)
      - i_discharge_max_allowed (positive direction)
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

    # ---------- Combine scales ----------
    dis_scale = dis_soc_scale * temp_scale * dis_v_scale
    chg_scale = chg_soc_scale * temp_scale * chg_v_scale

    i_dis_allowed = i_dis * dis_scale
    i_chg_allowed = i_chg * chg_scale

    return {
        "i_discharge_max_allowed": i_dis_allowed,
        "i_charge_max_allowed": i_chg_allowed,
    }
