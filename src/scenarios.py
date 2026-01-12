# src/scenarios.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

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

    # optional durations (seconds). Missing/None/<=0 => step/latched after start
    uv_inject_duration_s: Optional[float] = None
    ov_inject_duration_s: Optional[float] = None
    oc_inject_duration_s: Optional[float] = None
    ot_inject_duration_s: Optional[float] = None
    fire_inject_duration_s: Optional[float] = None

    # injection values
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
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "y", "1", "on"):
            return True
        if s in ("false", "no", "n", "0", "off", ""):
            return False
    return bool(x)


def _f(x: Any, default: float) -> float:
    v = _as_float(x, None)
    return default if v is None else float(v)


def _fo(x: Any) -> Optional[float]:
    # float-or-none helper
    return _as_float(x, None)


def _i(x: Any, default: int) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def load_scenarios(path: str = "data/scenarios.yaml") -> List[Scenario]:
    """
    Backwards-compatible loader for scenarios.yaml.

    Expected keys per scenario:
      id, description, t_amb_c
      profile: {type, direction, current_a, use_bms_limits}
      stop_conditions: {max_time_s, stop_on_emergency}
      init: {true_soc, ekf_soc}
      soh_profile: dict or string

    Optional extension:
      injection: {
        uv_time_s, ov_time_s, oc_time_s, ot_time_s, fire_time_s,
        uv_duration_s, ov_duration_s, oc_duration_s, ot_duration_s, fire_duration_s,
        uv_v_cell_min_v, ov_v_cell_max_v, oc_i_rack_a, ot_t_rack_c
      }
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

            t_amb_c=_f(it.get("t_amb_c"), 25.0),
            max_time_s=_f(stop_cfg.get("max_time_s"), 3600.0),

            stop_on_emergency=_as_bool(stop_cfg.get("stop_on_emergency"), False),
            use_bms_limits=_as_bool(prof.get("use_bms_limits"), True),

            true_init_soc=_f(init_cfg.get("true_soc"), 1.0),
            ekf_init_soc=_fo(init_cfg.get("ekf_soc")),
            soh_profile=it.get("soh_profile", None),
        )

        if scn.profile_type == "constant_current":
            scn.direction = str(prof.get("direction", "discharge")).strip().lower()
            scn.current_a = _f(prof.get("current_a"), 0.0)

        elif scn.profile_type == "random_current":
            scn.i_discharge_max_a = _f(prof.get("i_discharge_max_a"), scn.i_discharge_max_a)
            scn.i_charge_max_a = _f(prof.get("i_charge_max_a"), scn.i_charge_max_a)
            scn.segment_min_s = _f(prof.get("segment_min_s"), scn.segment_min_s)
            scn.segment_max_s = _f(prof.get("segment_max_s"), scn.segment_max_s)
            scn.random_seed = _i(prof.get("random_seed"), scn.random_seed)

        elif scn.profile_type == "segments":
            seg_list = prof.get("segments", []) or []
            if isinstance(seg_list, list):
                for s in seg_list:
                    if not isinstance(s, dict):
                        continue
                    dur = _fo(s.get("duration_s"))
                    cur = _fo(s.get("current_a"))
                    Ta = _fo(s.get("t_amb_c"))
                    if dur is None or cur is None or Ta is None or dur <= 0:
                        continue
                    scn.segments.append(
                        SegmentSpec(
                            duration_s=float(dur),
                            current_a=float(cur),
                            t_amb_c=float(Ta),
                            v_cell_min_override_v=_fo(s.get("v_cell_min_override_v")),
                            v_cell_max_override_v=_fo(s.get("v_cell_max_override_v")),
                            t_rack_override_c=_fo(s.get("t_rack_override_c")),
                            i_rack_override_a=_fo(s.get("i_rack_override_a")),
                            gas_alarm=_as_bool(s.get("gas_alarm"), False),
                        )
                    )
            if scn.segments:
                scn.max_time_s = float(sum(seg.duration_s for seg in scn.segments))

        inj = it.get("injection", {}) or {}
        if isinstance(inj, dict):
            # times
            scn.uv_inject_time_s = _fo(inj.get("uv_time_s"))
            scn.ov_inject_time_s = _fo(inj.get("ov_time_s"))
            scn.oc_inject_time_s = _fo(inj.get("oc_time_s"))
            scn.ot_inject_time_s = _fo(inj.get("ot_time_s"))
            scn.fire_inject_time_s = _fo(inj.get("fire_time_s"))

            # durations (support both *duration_s and *inject_duration_s keys)
            scn.uv_inject_duration_s = _fo(inj.get("uv_duration_s", inj.get("uv_inject_duration_s")))
            scn.ov_inject_duration_s = _fo(inj.get("ov_duration_s", inj.get("ov_inject_duration_s")))
            scn.oc_inject_duration_s = _fo(inj.get("oc_duration_s", inj.get("oc_inject_duration_s")))
            scn.ot_inject_duration_s = _fo(inj.get("ot_duration_s", inj.get("ot_inject_duration_s")))
            scn.fire_inject_duration_s = _fo(inj.get("fire_duration_s", inj.get("fire_inject_duration_s")))

            # values (NO "or" usage -> 0.0 stays 0.0)
            scn.uv_v_cell_min_v = _f(inj.get("uv_v_cell_min_v"), scn.uv_v_cell_min_v)
            scn.ov_v_cell_max_v = _f(inj.get("ov_v_cell_max_v"), scn.ov_v_cell_max_v)
            scn.oc_i_rack_a = _f(inj.get("oc_i_rack_a"), scn.oc_i_rack_a)
            scn.ot_t_rack_c = _f(inj.get("ot_t_rack_c"), scn.ot_t_rack_c)

        out.append(scn)

    return out
