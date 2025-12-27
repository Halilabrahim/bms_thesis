from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from src.sim_runner import run_scenario


@dataclass
class Segment:
    """
    UI-defined segment for profile_type='segments'.

    Base (plant inputs):
      - duration_s
      - current_a
      - t_amb_c

    Optional "BMS inputs override" (does NOT change plant physics,
    but overrides what BMS limiting + fault detector see):
      - v_cell_min_override_v
      - v_cell_max_override_v
      - t_rack_override_c
      - i_rack_override_a
      - gas_alarm
    """
    duration_s: float
    current_a: float
    t_amb_c: float

    # --- Advanced: override BMS-visible signals (optional) ---
    v_cell_min_override_v: Optional[float] = None
    v_cell_max_override_v: Optional[float] = None
    t_rack_override_c: Optional[float] = None
    i_rack_override_a: Optional[float] = None
    gas_alarm: bool = False


@dataclass
class FaultConfig:
    """
    Time-based injections for BMS-visible signals.

    Durations:
      - duration <= 0 => step (latched after start time)
      - duration > 0  => active window [t_start, t_start + duration)
    """
    # UV: force minimum cell voltage
    uv_time_s: Optional[float] = None
    uv_v_fault: float = 2.0
    uv_duration_s: float = 0.0

    # OT: force rack temperature
    ot_time_s: Optional[float] = None
    ot_temp_c: float = 100.0
    ot_duration_s: float = 0.0

    # OC: force rack current measurement
    oc_time_s: Optional[float] = None
    oc_i_fault_a: float = 800.0
    oc_duration_s: float = 0.0

    # FIRE: gas alarm
    fire_time_s: Optional[float] = None
    fire_duration_s: float = 0.0


def run_manual_profile(
    *,
    segments: List[Segment],
    faults: Optional[FaultConfig],
    params: Dict[str, Any],
    use_bms_limits: bool = True,
    true_init_soc: float = 1.0,
    ekf_init_soc: Optional[float] = None,
    stop_on_emergency: bool = False,
    soh_profile: Optional[Any] = None,
) -> Dict[str, List[float]]:
    """
    Adapter: UI manual segments -> src.sim_runner.run_scenario()
    """
    if not segments:
        raise ValueError("run_manual_profile: segments must be non-empty.")

    if ekf_init_soc is None:
        ekf_init_soc = float(true_init_soc)

    total_time_s = sum(float(s.duration_s) for s in segments)

    if faults is None:
        faults = FaultConfig()

    scenario = SimpleNamespace(
        id="manual_profile",
        description="Manual segment profile from UI",

        profile_type="segments",
        segments=segments,
        max_time_s=float(total_time_s),
        t_amb_c=float(segments[0].t_amb_c),

        use_bms_limits=bool(use_bms_limits),
        stop_on_emergency=bool(stop_on_emergency),

        true_init_soc=float(true_init_soc),
        ekf_init_soc=float(ekf_init_soc),

        soh_profile=soh_profile,

        # injections
        uv_inject_time_s=faults.uv_time_s,
        uv_v_cell_min_v=float(faults.uv_v_fault),
        uv_inject_duration_s=float(faults.uv_duration_s),

        ov_inject_time_s=None,
        ov_v_cell_max_v=0.0,
        ov_inject_duration_s=0.0,

        ot_inject_time_s=faults.ot_time_s,
        ot_t_rack_c=float(faults.ot_temp_c),
        ot_inject_duration_s=float(faults.ot_duration_s),

        oc_inject_time_s=faults.oc_time_s,
        oc_i_rack_a=float(faults.oc_i_fault_a),
        oc_inject_duration_s=float(faults.oc_duration_s),

        fire_inject_time_s=faults.fire_time_s,
        fire_inject_duration_s=float(faults.fire_duration_s),

        # random_current placeholders
        segment_min_s=1.0,
        segment_max_s=1.0,
        random_seed=0,
        i_discharge_max_a=0.0,
        i_charge_max_a=0.0,
    )

    return run_scenario(scenario, params)
