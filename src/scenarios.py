# src/scenarios.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class SegmentSpec:
    """
    Manual/segmented profile segment.

    Sign convention:
      + current = discharge
      - current = charge

    Overrides are "what the BMS sees" (sensor-level),
    not forcing the physical plant state.
    """
    duration_s: float
    current_a: float
    t_amb_c: float

    # Optional sensor overrides (BMS-visible)
    v_cell_min_override_v: Optional[float] = None
    v_cell_max_override_v: Optional[float] = None
    t_rack_override_c: Optional[float] = None
    i_rack_override_a: Optional[float] = None
    gas_alarm: bool = False


@dataclass
class Scenario:
    """
    Scenario configuration for sim_runner.run_scenario().

    profile_type:
      - constant_current
      - random_current
      - segments
    """
    id: str
    name: str = ""
    description: str = ""

    profile_type: str = "constant_current"

    # constant_current
    direction: str = "discharge"
    current_a: float = 0.0

    # random_current
    i_discharge_max_a: float = 320.0
    i_charge_max_a: float = 160.0
    segment_min_s: float = 30.0
    segment_max_s: float = 180.0
    random_seed: int = 0

    # segments
    segments: List[SegmentSpec] = field(default_factory=list)

    # BMS
    use_bms_limits: bool = True
    stop_on_emergency: bool = False

    # environment + duration
    t_amb_c: float = 25.0
    max_time_s: float = 3600.0

    # optional global time-based injections (BMS-visible)
    uv_inject_time_s: Optional[float] = None
    ov_inject_time_s: Optional[float] = None
    oc_inject_time_s: Optional[float] = None
    ot_inject_time_s: Optional[float] = None
    fire_inject_time_s: Optional[float] = None

    uv_v_cell_min_v: float = 2.0
    ov_v_cell_max_v: float = 3.7
    oc_i_rack_a: float = 500.0
    ot_t_rack_c: float = 100.0

    # SoH + IC
    soh_profile: Optional[Any] = None
    true_init_soc: float = 1.0
    ekf_init_soc: Optional[float] = None


def _as_float(x: Any, default: Optional[float]) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _as_bool(x: Any, default: bool) -> bool:
    if x is None:
        return default
    return bool(x)


def load_scenarios(path: str = "data/scenarios.yaml") -> List[Scenario]:
    """
    Backwards-compatible loader for your scenarios.yaml.

    Expected keys per scenario (matches your current file):
      id, description, t_amb_c
      profile: {type, direction, current_a, use_bms_limits}
      stop_conditions: {max_time_s, stop_on_emergency}
      init: {true_soc, ekf_soc}
      soh_profile: dict or string

    Optional extension:
      injection: {uv_time_s, ..., fire_time_s, uv_v_cell_min_v, ..., ot_t_rack_c}
      profile.type: "segments" + profile.segments: [...]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scenarios YAML not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    items = raw.get("scenarios", [])
    if not isinstance(items, list):
        raise ValueError("scenarios.yaml must contain top-level key 'scenarios' as a list")

    out: List[Scenario] = []
    for it in items:
        if not isinstance(it, dict) or "id" not in it:
            continue

        prof = it.get("profile", {}) or {}
        stop_cfg = it.get("stop_conditions", {}) or {}
        init_cfg = it.get("init", {}) or {}

        scn = Scenario(
            id=str(it["id"]),
            description=str(it.get("description", "")),
            profile_type=str(prof.get("type", "constant_current")).strip().lower(),
            t_amb_c=float(_as_float(it.get("t_amb_c"), 25.0) or 25.0),
            max_time_s=float(_as_float(stop_cfg.get("max_time_s"), 3600.0) or 3600.0),
            stop_on_emergency=_as_bool(stop_cfg.get("stop_on_emergency"), False),
            use_bms_limits=_as_bool(prof.get("use_bms_limits"), True),
            true_init_soc=float(_as_float(init_cfg.get("true_soc"), 1.0) or 1.0),
            ekf_init_soc=_as_float(init_cfg.get("ekf_soc"), None),
            soh_profile=it.get("soh_profile", None),
        )

        if scn.profile_type == "constant_current":
            scn.direction = str(prof.get("direction", "discharge")).strip().lower()
            scn.current_a = float(_as_float(prof.get("current_a"), 0.0) or 0.0)

        elif scn.profile_type == "random_current":
            scn.i_discharge_max_a = float(_as_float(prof.get("i_discharge_max_a"), scn.i_discharge_max_a) or scn.i_discharge_max_a)
            scn.i_charge_max_a = float(_as_float(prof.get("i_charge_max_a"), scn.i_charge_max_a) or scn.i_charge_max_a)
            scn.segment_min_s = float(_as_float(prof.get("segment_min_s"), scn.segment_min_s) or scn.segment_min_s)
            scn.segment_max_s = float(_as_float(prof.get("segment_max_s"), scn.segment_max_s) or scn.segment_max_s)
            scn.random_seed = int(prof.get("random_seed", scn.random_seed) or scn.random_seed)

        elif scn.profile_type == "segments":
            seg_list = prof.get("segments", []) or []
            if isinstance(seg_list, list):
                for s in seg_list:
                    if not isinstance(s, dict):
                        continue
                    dur = _as_float(s.get("duration_s"), None)
                    cur = _as_float(s.get("current_a"), None)
                    Ta = _as_float(s.get("t_amb_c"), None)
                    if dur is None or cur is None or Ta is None or dur <= 0:
                        continue
                    scn.segments.append(
                        SegmentSpec(
                            duration_s=float(dur),
                            current_a=float(cur),
                            t_amb_c=float(Ta),
                            v_cell_min_override_v=_as_float(s.get("v_cell_min_override_v"), None),
                            v_cell_max_override_v=_as_float(s.get("v_cell_max_override_v"), None),
                            t_rack_override_c=_as_float(s.get("t_rack_override_c"), None),
                            i_rack_override_a=_as_float(s.get("i_rack_override_a"), None),
                            gas_alarm=_as_bool(s.get("gas_alarm"), False),
                        )
                    )
            if scn.segments:
                scn.max_time_s = float(sum(seg.duration_s for seg in scn.segments))

        inj = it.get("injection", {}) or {}
        if isinstance(inj, dict):
            scn.uv_inject_time_s = _as_float(inj.get("uv_time_s"), None)
            scn.ov_inject_time_s = _as_float(inj.get("ov_time_s"), None)
            scn.oc_inject_time_s = _as_float(inj.get("oc_time_s"), None)
            scn.ot_inject_time_s = _as_float(inj.get("ot_time_s"), None)
            scn.fire_inject_time_s = _as_float(inj.get("fire_time_s"), None)

            scn.uv_v_cell_min_v = float(_as_float(inj.get("uv_v_cell_min_v"), scn.uv_v_cell_min_v) or scn.uv_v_cell_min_v)
            scn.ov_v_cell_max_v = float(_as_float(inj.get("ov_v_cell_max_v"), scn.ov_v_cell_max_v) or scn.ov_v_cell_max_v)
            scn.oc_i_rack_a = float(_as_float(inj.get("oc_i_rack_a"), scn.oc_i_rack_a) or scn.oc_i_rack_a)
            scn.ot_t_rack_c = float(_as_float(inj.get("ot_t_rack_c"), scn.ot_t_rack_c) or scn.ot_t_rack_c)

        out.append(scn)

    return out
