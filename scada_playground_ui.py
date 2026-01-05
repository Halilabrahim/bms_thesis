# scada_playground_ui.py
from __future__ import annotations

import io
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.sim_runner import build_models
from src.fsm import BMSState
from src.bms_logic import compute_current_limits

UI_BUILD = "2026-01-05_scada_fix_used_signals_v4"


STATE_CODE_MAP = {
    BMSState.OFF: 0,
    BMSState.RUN: 1,
    BMSState.FAULT: 2,
    BMSState.EMERGENCY_SHUTDOWN: 3,
}
STATE_NAME_MAP = {v: k.name for k, v in STATE_CODE_MAP.items()}

LIMIT_CODE_TEXT = {
    0: "NONE",
    10: "TEMP_LOW_CUTOFF",
    11: "TEMP_LOW_DERATE",
    20: "TEMP_HIGH_CUTOFF",
    21: "TEMP_HIGH_DERATE",
    30: "VMIN_MARGIN",
    31: "VMIN_LIMIT",
    40: "VMAX_MARGIN",
    41: "VMAX_LIMIT",
    50: "SOC_LOW_CUTOFF",
    51: "SOC_LOW_DERATE",
    60: "SOC_HIGH_CUTOFF",
    61: "SOC_HIGH_DERATE",
}


def _lim_code_name(x: float) -> str:
    try:
        i = int(x)
    except Exception:
        return "N/A"
    return LIMIT_CODE_TEXT.get(i, f"CODE_{i}")


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _resolve_path(p: str) -> str:
    try:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        cand = (Path.cwd() / pp).resolve()
        if cand.exists():
            return str(cand)
        here = Path(__file__).resolve().parent
        cand2 = (here / pp).resolve()
        if cand2.exists():
            return str(cand2)
        return str((here.parent / pp).resolve())
    except Exception:
        return str(p)


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


def _render_active_config_sidebar(params: Dict[str, Any], *, config_path: str, dt_s: float) -> None:
    meta = _get(params, "meta", default={}) or {}
    faults = _get(params, "faults", default={}) or {}
    limits = _get(params, "limits", default={}) or {}
    bms = _get(params, "bms_control", default={}) or {}
    thermal = _get(params, "thermal_model", default={}) or {}

    debounce_steps = _as_int(faults.get("debounce_steps", 0), default=0)

    uv_thr = _as_float(faults.get("uv_cell_v", limits.get("v_cell_min_v", float("nan"))))
    ov_thr = _as_float(faults.get("ov_cell_v", float("nan")))
    soft_vmax = _as_float(limits.get("v_cell_max_v", float("nan")))

    ot_thr = _as_float(faults.get("ot_rack_c", float("nan")))
    ut_thr = _as_float(faults.get("ut_rack_c", float("nan")))

    t_derate_start = _as_float(bms.get("t_high_derate_start_c", float("nan")))
    t_cutoff = _as_float(bms.get("t_high_cutoff_c", limits.get("t_max_c", float("nan"))))

    t_low_derate = _as_float(bms.get("t_low_derate_start_c", float("nan")))
    t_low_cutoff = _as_float(bms.get("t_low_cutoff_c", limits.get("t_min_c", float("nan"))))

    c_th = _as_float(thermal.get("c_th_j_per_k", float("nan")))
    r_th = _as_float(thermal.get("r_th_k_per_w", float("nan")))
    tau_s = float(c_th * r_th) if (np.isfinite(c_th) and np.isfinite(r_th)) else float("nan")
    tau_h = float(tau_s / 3600.0) if np.isfinite(tau_s) else float("nan")

    oc_dis = _as_float(faults.get("oc_discharge_a", float("nan")))
    oc_chg = _as_float(faults.get("oc_charge_a", float("nan")))
    fire_temp = _as_float(
        faults.get("fire_temp_c", _get(params, "fire_detection", "temp_threshold_c", default=float("nan")))
    )

    rec_uv = (uv_thr - 0.05) if np.isfinite(uv_thr) else float("nan")
    rec_ov = (ov_thr + 0.02) if np.isfinite(ov_thr) else float("nan")
    rec_ot = (ot_thr + 5.0) if np.isfinite(ot_thr) else float("nan")
    rec_ut = (ut_thr - 5.0) if np.isfinite(ut_thr) else float("nan")
    rec_oc_dis = (oc_dis + 50.0) if np.isfinite(oc_dis) else float("nan")
    rec_oc_chg = (oc_chg + 50.0) if np.isfinite(oc_chg) else float("nan")
    rec_fire_t = (fire_temp + 1.0) if np.isfinite(fire_temp) else float("nan")

    chemistry = _get(params, "chemistry", default={}) or {}
    structure = _get(params, "structure", default={}) or {}
    pack = _get(params, "pack", default={}) or {}
    rack = _get(params, "rack", default={}) or {}
    hardware = _get(params, "hardware", default={}) or {}

    with st.sidebar:
        st.subheader("Active config")
        st.caption(f"Config: {config_path}")
        st.write(f"dt_s: {dt_s:.3g} s")

        if debounce_steps:
            st.write(f"debounce_steps: {debounce_steps}")

        if np.isfinite(uv_thr):
            st.write(f"UV threshold: {uv_thr:.2f} V")
        if np.isfinite(ov_thr):
            st.write(f"OV threshold: {ov_thr:.2f} V")
        if np.isfinite(soft_vmax):
            st.write(f"Soft max cell voltage: {soft_vmax:.2f} V")

        if np.isfinite(ot_thr):
            st.write(f"OT threshold: {ot_thr:.1f} °C")
        if np.isfinite(ut_thr):
            st.write(f"UT threshold: {ut_thr:.1f} °C")

        if np.isfinite(t_low_derate):
            st.write(f"T low derate start: {t_low_derate:.1f} °C")
        if np.isfinite(t_low_cutoff):
            st.write(f"T low cutoff: {t_low_cutoff:.1f} °C")

        if np.isfinite(t_derate_start):
            st.write(f"T high derate start: {t_derate_start:.1f} °C")
        if np.isfinite(t_cutoff):
            st.write(f"T high cutoff: {t_cutoff:.1f} °C")

        if np.isfinite(tau_s):
            st.write(f"Thermal time constant estimate: τ ≈ {tau_s:.0f} s (~{tau_h:.1f} h).")
            st.caption(
                "So changing T_amb in later segments may not raise T_rack quickly. "
                "Use T_rack override / injections for fast logic tests."
            )

        sys_name = str(meta.get("system_name", "")).strip()
        ver = str(meta.get("version", "")).strip()
        if sys_name or ver:
            st.divider()
            st.caption("meta")
            if sys_name:
                st.write(f'system_name: "{sys_name}"')
            if ver:
                st.write(f'version: "{ver}"')

        st.divider()
        with st.expander("Rack / pack key specs", expanded=True):
            ch_name = str(chemistry.get("name", "")).strip()
            if ch_name:
                st.write(f"chemistry: {ch_name}")
            nc = _as_float(chemistry.get("nominal_cell_voltage_v", float("nan")))
            cap = _as_float(chemistry.get("cell_capacity_ah", float("nan")))
            if np.isfinite(nc):
                st.write(f"nominal_cell_voltage: {nc:.2f} V")
            if np.isfinite(cap):
                st.write(f"cell_capacity: {cap:.1f} Ah")

            s16 = structure.get("cells_in_series_per_pack", None)
            p21 = structure.get("packs_in_series_per_rack", None)
            if s16 is not None and p21 is not None:
                st.write(f"structure: {int(s16)}s × {int(p21)} packs (rack)")

            pv = _as_float(pack.get("rated_voltage_v", float("nan")))
            pe = _as_float(pack.get("nominal_energy_kwh", float("nan")))
            rv = _as_float(rack.get("nominal_voltage_v", float("nan")))
            re = _as_float(rack.get("nominal_energy_kwh", float("nan")))
            if np.isfinite(pv):
                st.write(f"pack rated voltage: {pv:.1f} V")
            if np.isfinite(pe):
                st.write(f"pack nominal energy: {pe:.2f} kWh")
            if np.isfinite(rv):
                st.write(f"rack nominal voltage: {rv:.1f} V")
            if np.isfinite(re):
                st.write(f"rack nominal energy: {re:.1f} kWh")

            hv = _as_float(hardware.get("rated_operating_voltage_v", float("nan")))
            hp = _as_float(hardware.get("rated_power_per_rack_kw", float("nan")))
            hi = _as_float(hardware.get("rated_current_per_rack_a", float("nan")))
            if np.isfinite(hv):
                st.write(f"rated operating voltage: {hv:.0f} V")
            if np.isfinite(hp):
                st.write(f"rated power per rack: {hp:.1f} kW")
            if np.isfinite(hi):
                st.write(f"rated current per rack: {hi:.1f} A")

        st.divider()
        st.caption("Fault quick reference (for fast triggering)")
        rows = []
        if np.isfinite(uv_thr):
            rows.append(
                {
                    "Fault": "UV",
                    "Threshold": f"{uv_thr:.2f} V",
                    "Quick trigger": f"Override v_cell_min ≈ {rec_uv:.2f} V",
                    "Best knob": "Manual override v_cell_min OR UV injection",
                }
            )
        if np.isfinite(ov_thr):
            rows.append(
                {
                    "Fault": "OV",
                    "Threshold": f"{ov_thr:.2f} V",
                    "Quick trigger": f"Override v_cell_max ≈ {rec_ov:.2f} V",
                    "Best knob": "Manual override v_cell_max",
                }
            )
        if np.isfinite(ot_thr):
            rows.append(
                {
                    "Fault": "OT",
                    "Threshold": f"{ot_thr:.1f} °C",
                    "Quick trigger": f"Override t_rack_used ≈ {rec_ot:.1f} °C",
                    "Best knob": "Manual override t_rack OR OT injection",
                }
            )
        if np.isfinite(ut_thr):
            rows.append(
                {
                    "Fault": "UT",
                    "Threshold": f"{ut_thr:.1f} °C",
                    "Quick trigger": f"Override t_rack_used ≈ {rec_ut:.1f} °C",
                    "Best knob": "Manual override t_rack OR UT injection",
                }
            )
        if np.isfinite(oc_dis) or np.isfinite(oc_chg):
            thr = []
            if np.isfinite(oc_dis):
                thr.append(f"dis>{oc_dis:.0f}A")
            if np.isfinite(oc_chg):
                thr.append(f"chg>{oc_chg:.0f}A")
            rows.append(
                {
                    "Fault": "OC",
                    "Threshold": ", ".join(thr),
                    "Quick trigger": f"Override i_rack ≈ {rec_oc_dis:.0f}A (dis) / {rec_oc_chg:.0f}A (chg)",
                    "Best knob": "Manual override i_rack OR OC injection",
                }
            )
        rows.append(
            {
                "Fault": "FIRE",
                "Threshold": "gas_alarm=True",
                "Quick trigger": f"Enable FIRE injection (or gas_alarm override). Temp ref: {rec_fire_t:.1f}°C",
                "Best knob": "FIRE injection OR gas_alarm override",
            }
        )
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _ensure_event_log(maxlen: int = 200) -> None:
    if "scada_event_log" not in st.session_state:
        st.session_state.scada_event_log = deque(maxlen=maxlen)
    if "scada_prev_flags" not in st.session_state:
        st.session_state.scada_prev_flags = {"oc": 0, "ov": 0, "uv": 0, "ot": 0, "ut": 0, "fire": 0}
    if "scada_prev_state_code" not in st.session_state:
        st.session_state.scada_prev_state_code = None


def _event_add(t_s: float, severity: str, kind: str, message: str) -> None:
    _ensure_event_log()
    st.session_state.scada_event_log.appendleft({"t_s": float(t_s), "severity": severity, "type": kind, "message": message})


def _severity_for_fault(fault: str) -> str:
    if fault in ("fire", "ot", "ov", "uv"):
        return "CRITICAL"
    if fault in ("ut", "oc"):
        return "WARNING"
    return "INFO"


def _update_events_from_step(step_out: Dict[str, float]) -> None:
    _ensure_event_log()
    t_s = float(step_out.get("t_s", np.nan))

    cur_flags = {
        "oc": int(step_out.get("oc", 0.0) > 0.5),
        "ov": int(step_out.get("ov", 0.0) > 0.5),
        "uv": int(step_out.get("uv", 0.0) > 0.5),
        "ot": int(step_out.get("ot", 0.0) > 0.5),
        "ut": int(step_out.get("ut", 0.0) > 0.5),
        "fire": int(step_out.get("fire", 0.0) > 0.5),
    }
    prev_flags = dict(st.session_state.scada_prev_flags)

    for k in ("oc", "ov", "uv", "ot", "ut", "fire"):
        if prev_flags.get(k, 0) == 0 and cur_flags[k] == 1:
            _event_add(t_s=t_s, severity=_severity_for_fault(k), kind="FAULT_ASSERT", message=f"{k.upper()} asserted")
        if prev_flags.get(k, 0) == 1 and cur_flags[k] == 0:
            _event_add(t_s=t_s, severity="INFO", kind="FAULT_CLEAR", message=f"{k.upper()} cleared")

    cur_state_code = int(step_out.get("state_code", -999))
    prev_state_code = st.session_state.scada_prev_state_code
    if prev_state_code is None:
        st.session_state.scada_prev_state_code = cur_state_code
    else:
        if cur_state_code != prev_state_code:
            _event_add(
                t_s=t_s,
                severity="INFO",
                kind="STATE_CHANGE",
                message=(
                    f"State changed: {STATE_NAME_MAP.get(prev_state_code, prev_state_code)}"
                    f" → {STATE_NAME_MAP.get(cur_state_code, cur_state_code)}"
                ),
            )
            st.session_state.scada_prev_state_code = cur_state_code

    st.session_state.scada_prev_flags = cur_flags


@dataclass
class LiveInjections:
    uv: bool = False
    ot: bool = False
    ut: bool = False
    oc: bool = False
    fire: bool = False


@dataclass
class ManualOverrides:
    enabled: bool = False
    v_cell_min_override_v: Optional[float] = None
    v_cell_max_override_v: Optional[float] = None
    t_rack_override_c: Optional[float] = None
    i_rack_override_a: Optional[float] = None
    gas_alarm_override: Optional[bool] = None


@dataclass
class LiveProfile:
    mode: str = "random"  # "random" | "setpoint"
    i_setpoint_a: float = 0.0
    ramp_a_per_s: float = 0.0


class LiveBmsSim:
    """
    Live SCADA simulator.

    IMPORTANT:
    - "true" signals are from the plant models (ECM + thermal)
    - "used" signals are the ones the BMS sees after manual overrides / injections
    Derating (compute_current_limits) and fault detection both use the USED signals.

    Key visibility signals:
    - I_req: operator/profile request
    - I_act: applied current after BMS limits
    - I_used: BMS perceived current (measurement layer after overrides/injections)
    """

    def __init__(
        self,
        params: Dict[str, Any],
        *,
        t_amb_c: float = 25.0,
        true_init_soc: float = 0.7,
        ekf_init_soc: float = 0.7,
        use_bms_limits: bool = True,
        i_discharge_max_a: float = 300.0,
        i_charge_max_a: float = 300.0,
        segment_min_s: float = 5.0,
        segment_max_s: float = 20.0,
        seed_profile: int = 0,
        profile: Optional[LiveProfile] = None,
        uv_v_fault: float = 2.5,
        ot_temp_c: float = 65.0,
        ut_temp_c: float = -30.0,
        oc_i_fault_a: float = 500.0,
        history_max_points: int = 30000,
    ) -> None:
        self.params = params
        self.models = build_models(params)

        self.ecm = self.models["ecm"]
        self.thermal = self.models["thermal"]
        self.ekf = self.models["ekf"]
        self.bms_params = self.models["bms_params"]
        self.fault_det = self.models["fault_det"]
        self.fsm = self.models["fsm"]

        dt = float(self.models.get("dt_s", _get(params, "simulation", "dt_s", default=1.0)))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0
        self.dt_s = float(dt)

        sensor_cfg = _get(params, "sensors", default={}) or {}
        self.sigma_v = float(sensor_cfg.get("voltage_noise_std_v", 0.0))
        self.sigma_i = float(sensor_cfg.get("current_noise_std_a", 0.0))
        self.sigma_t = float(sensor_cfg.get("temp_noise_std_c", 0.0))
        self.rng_meas = np.random.default_rng(int(sensor_cfg.get("seed", 0)))

        self.t_amb_c = float(t_amb_c)
        self.true_init_soc = _clip01(true_init_soc)
        self.ekf_init_soc = _clip01(ekf_init_soc)
        self.use_bms_limits = bool(use_bms_limits)

        self.i_discharge_max_a = float(i_discharge_max_a)
        self.i_charge_max_a = float(i_charge_max_a)
        self.segment_min_s = float(segment_min_s)
        self.segment_max_s = float(segment_max_s)
        self.rng_prof = np.random.default_rng(int(seed_profile))

        self.profile = profile if profile is not None else LiveProfile()
        self.i_req_a = 0.0
        self.next_switch_s = 0.0

        self.inj = LiveInjections()
        self.ovr = ManualOverrides()
        self.uv_v_fault = float(uv_v_fault)
        self.ot_temp_c = float(ot_temp_c)
        self.ut_temp_c = float(ut_temp_c)
        self.oc_i_fault_a = float(oc_i_fault_a)

        self.time_s = 0.0
        self.last_res: Optional[Dict[str, Any]] = None

        self.history_max_points = int(max(1000, history_max_points))
        self.hist: Dict[str, Deque[float]] = {}
        self._init_hist()
        self.reset()

    def _init_hist(self) -> None:
        keys = [
            "time_s",
            "i_req_a",
            "i_act_a",
            "i_rack_used_a",        # BMS perceived current
            "soc_true",
            "soc_hat",
            "v_cell_min_true_v",
            "v_cell_max_true_v",
            "v_cell_min_v",          # used
            "v_cell_max_v",          # used
            "t_rack_true_c",
            "t_rack_used_c",
            "t_amb_c",
            "state_code",
            "oc",
            "ov",
            "uv",
            "ot",
            "ut",
            "fire",
            "i_discharge_lim_a",
            "i_charge_lim_a",
            "soc_cell_min",
            "soc_cell_mean",
            "soc_cell_max",
            "lim_code_dis",
            "lim_code_chg",
            "scale_dis_total",
            "scale_chg_total",
        ]
        self.hist = {k: deque(maxlen=self.history_max_points) for k in keys}

    def set_history_max_points(self, new_max: int) -> None:
        new_max = int(max(1000, new_max))
        if new_max == self.history_max_points:
            return
        old = self.hist
        self.history_max_points = new_max
        self._init_hist()
        for k in self.hist.keys():
            if k in old:
                self.hist[k].extend(list(old[k])[-new_max:])

    def _meas(self, x: float, sigma: float) -> float:
        if sigma <= 0.0:
            return float(x)
        return float(x + self.rng_meas.normal(0.0, sigma))

    # ---------------- FIX #2: robust mode switching ----------------
    def set_profile_mode(self, mode: str) -> None:
        mode = str(mode).strip().lower()
        new_mode = "setpoint" if mode == "setpoint" else "random"

        if new_mode == self.profile.mode:
            return

        self.profile.mode = new_mode

        if new_mode == "random":
            # Immediately arm a random segment after switching from setpoint.
            self._choose_new_request_random()
        else:
            # Stop random scheduling in setpoint mode
            self.next_switch_s = float("inf")

    def set_setpoint_a(self, val: float) -> None:
        self.profile.i_setpoint_a = float(val)

    def set_ramp_a_per_s(self, val: float) -> None:
        self.profile.ramp_a_per_s = float(max(0.0, val))

    def _choose_new_request_random(self) -> None:
        dur = float(self.rng_prof.uniform(self.segment_min_s, self.segment_max_s))
        self.next_switch_s = float(self.time_s + dur)
        mag_dis = float(self.rng_prof.uniform(0.0, self.i_discharge_max_a))
        mag_chg = float(self.rng_prof.uniform(0.0, self.i_charge_max_a))
        if bool(self.rng_prof.integers(0, 2)):
            self.i_req_a = mag_dis
        else:
            self.i_req_a = -mag_chg

    def _update_i_req_setpoint(self) -> None:
        target = float(self.profile.i_setpoint_a)
        ramp = float(self.profile.ramp_a_per_s)
        if ramp <= 0.0:
            self.i_req_a = target
            return
        delta = target - self.i_req_a
        max_step = ramp * self.dt_s
        if abs(delta) <= max_step:
            self.i_req_a = target
        else:
            self.i_req_a += float(np.sign(delta) * max_step)

    def _apply_bms_visibility_layer(
        self,
        *,
        v_cell_min_true_v: float,
        v_cell_max_true_v: float,
        t_rack_true_c: float,
        i_rack_meas_a: float,
        gas_alarm: bool,
    ) -> Tuple[float, float, float, float, bool]:
        vmin_used = float(v_cell_min_true_v)
        vmax_used = float(v_cell_max_true_v)
        t_used = float(t_rack_true_c)
        i_used = float(i_rack_meas_a)
        gas_used = bool(gas_alarm)

        # PRIORITY:
        #   BASE -> MANUAL OVERRIDES -> INJECTIONS (injection wins on conflicts)

        # manual overrides (lower priority)
        if self.ovr.enabled:
            if self.ovr.v_cell_min_override_v is not None:
                vmin_used = float(self.ovr.v_cell_min_override_v)
            if self.ovr.v_cell_max_override_v is not None:
                vmax_used = float(self.ovr.v_cell_max_override_v)
            if self.ovr.t_rack_override_c is not None:
                t_used = float(self.ovr.t_rack_override_c)
            if self.ovr.i_rack_override_a is not None:
                i_used = float(self.ovr.i_rack_override_a)
            if self.ovr.gas_alarm_override is not None:
                gas_used = bool(self.ovr.gas_alarm_override)

        # injections (highest priority)
        if self.inj.uv:
            vmin_used = min(vmin_used, float(self.uv_v_fault))
        if self.inj.ot:
            t_used = max(t_used, float(self.ot_temp_c))
        if self.inj.ut:
            t_used = min(t_used, float(self.ut_temp_c))

        # ---------------- FIX #1: OC injection sign follows I_req (deterministic) ----------------
        if self.inj.oc:
            # Direction follows operator request (I_req). If I_req is ~0, default to discharge (+).
            if self.i_req_a > 1e-9:
                sgn = 1.0
            elif self.i_req_a < -1e-9:
                sgn = -1.0
            else:
                sgn = 1.0
            i_used = float(sgn * abs(self.oc_i_fault_a))

        if self.inj.fire:
            gas_used = True

        # optional temp sensor noise (only for "used" measurement, not plant)
        if np.isfinite(self.sigma_t) and self.sigma_t > 0.0:
            t_used = self._meas(t_used, float(self.sigma_t))

        return vmin_used, vmax_used, t_used, i_used, gas_used

    def reset(self) -> None:
        self.time_s = 0.0
        self.ecm.reset(soc=self.true_init_soc)
        self.thermal.reset(t_init_c=self.t_amb_c)
        self.ekf.reset(soc_init=self.ekf_init_soc)
        self.fault_det.reset()
        self.fsm.reset(BMSState.RUN)

        self.last_res = self.ecm.step(0.0, 0.0)

        if self.profile.mode == "random":
            self._choose_new_request_random()
        else:
            self.i_req_a = 0.0
            self.next_switch_s = float("inf")

        self._init_hist()

        vmin_true = float(self.last_res.get("v_cell_min", np.nan))
        vmax_true = float(self.last_res.get("v_cell_max", np.nan))
        t_true = float(self.thermal.t_c)

        vmin_used, vmax_used, t_used, i_used, _ = self._apply_bms_visibility_layer(
            v_cell_min_true_v=vmin_true,
            v_cell_max_true_v=vmax_true,
            t_rack_true_c=t_true,
            i_rack_meas_a=0.0,
            gas_alarm=False,
        )

        self._log_sample(
            t_s=0.0,
            i_req=0.0,
            i_act=0.0,
            i_used=i_used,
            res=self.last_res,
            v_cell_min_true_v=vmin_true,
            v_cell_max_true_v=vmax_true,
            v_cell_min_used_v=vmin_used,
            v_cell_max_used_v=vmax_used,
            t_rack_true_c=t_true,
            t_rack_used_c=t_used,
            t_amb=float(self.t_amb_c),
            state=self.fsm.state,
            flags={"oc": False, "ov": False, "uv": False, "ot": False, "ut": False, "fire": False},
            curr_limits=None,
        )

    def _log_sample(
        self,
        *,
        t_s: float,
        i_req: float,
        i_act: float,
        i_used: float,
        res: Dict[str, Any],
        v_cell_min_true_v: float,
        v_cell_max_true_v: float,
        v_cell_min_used_v: float,
        v_cell_max_used_v: float,
        t_rack_true_c: float,
        t_rack_used_c: float,
        t_amb: float,
        state: BMSState,
        flags: Dict[str, Any],
        curr_limits: Optional[Dict[str, float]],
    ) -> None:
        soc_true = float(res.get("soc", np.nan))
        soc_hat = float(self.ekf.get_soc())

        self.hist["time_s"].append(float(t_s))
        self.hist["i_req_a"].append(float(i_req))
        self.hist["i_act_a"].append(float(i_act))
        self.hist["i_rack_used_a"].append(float(i_used))

        self.hist["soc_true"].append(soc_true)
        self.hist["soc_hat"].append(soc_hat)

        self.hist["v_cell_min_true_v"].append(float(v_cell_min_true_v))
        self.hist["v_cell_max_true_v"].append(float(v_cell_max_true_v))
        self.hist["v_cell_min_v"].append(float(v_cell_min_used_v))
        self.hist["v_cell_max_v"].append(float(v_cell_max_used_v))

        self.hist["t_rack_true_c"].append(float(t_rack_true_c))
        self.hist["t_rack_used_c"].append(float(t_rack_used_c))
        self.hist["t_amb_c"].append(float(t_amb))

        self.hist["state_code"].append(float(STATE_CODE_MAP[state]))

        self.hist["oc"].append(float(bool(flags.get("oc", False))))
        self.hist["ov"].append(float(bool(flags.get("ov", False))))
        self.hist["uv"].append(float(bool(flags.get("uv", False))))
        self.hist["ot"].append(float(bool(flags.get("ot", False))))
        self.hist["ut"].append(float(bool(flags.get("ut", False))))
        self.hist["fire"].append(float(bool(flags.get("fire", False))))

        # --- current limits + limiter reason (robust) ---
        if curr_limits is None:
            i_dis = np.nan
            i_chg = np.nan
            code_dis = np.nan
            code_chg = np.nan
            scale_dis = np.nan
            scale_chg = np.nan
        else:
            i_dis = float(curr_limits.get("i_discharge_max_allowed", np.nan))
            i_chg = float(curr_limits.get("i_charge_max_allowed", np.nan))
            code_dis = float(curr_limits.get("code_limit_dis", np.nan))
            code_chg = float(curr_limits.get("code_limit_chg", np.nan))
            scale_dis = float(curr_limits.get("scale_dis_total", np.nan))
            scale_chg = float(curr_limits.get("scale_chg_total", np.nan))

        self.hist["i_discharge_lim_a"].append(i_dis)
        self.hist["i_charge_lim_a"].append(i_chg)
        self.hist["lim_code_dis"].append(code_dis)
        self.hist["lim_code_chg"].append(code_chg)
        self.hist["scale_dis_total"].append(scale_dis)
        self.hist["scale_chg_total"].append(scale_chg)

        # --- cell SoC stats ---
        self.hist["soc_cell_min"].append(float(res.get("soc_cell_min", soc_true)))
        self.hist["soc_cell_mean"].append(float(res.get("soc_cell_mean", soc_true)))
        self.hist["soc_cell_max"].append(float(res.get("soc_cell_max", soc_true)))

    def step(self) -> Dict[str, float]:
        if self.profile.mode == "random":
            if self.time_s >= self.next_switch_s:
                self._choose_new_request_random()
        else:
            self._update_i_req_setpoint()

        if self.last_res is None:
            self.last_res = self.ecm.step(0.0, 0.0)

        vmin_true_prev = float(self.last_res.get("v_cell_min", np.nan))
        vmax_true_prev = float(self.last_res.get("v_cell_max", np.nan))
        t_true_prev = float(self.thermal.t_c)

        vmin_used_prev, vmax_used_prev, t_used_prev, _, _ = self._apply_bms_visibility_layer(
            v_cell_min_true_v=vmin_true_prev,
            v_cell_max_true_v=vmax_true_prev,
            t_rack_true_c=t_true_prev,
            i_rack_meas_a=0.0,
            gas_alarm=False,
        )

        curr_limits: Optional[Dict[str, float]] = None

        if self.fsm.state in (BMSState.FAULT, BMSState.EMERGENCY_SHUTDOWN):
            i_act = 0.0
        else:
            if self.use_bms_limits:
                curr_limits = compute_current_limits(
                    soc_hat=float(self.ekf.get_soc()),
                    t_rack_c=float(t_used_prev),
                    v_cell_min=float(vmin_used_prev),
                    v_cell_max=float(vmax_used_prev),
                    params=self.bms_params,
                )
                if self.i_req_a >= 0.0:
                    i_act = min(self.i_req_a, float(curr_limits["i_discharge_max_allowed"]))
                else:
                    i_act = max(self.i_req_a, -float(curr_limits["i_charge_max_allowed"]))
            else:
                i_act = float(self.i_req_a)

        self.ekf.predict(i_act, self.dt_s)

        res = self.ecm.step(i_act, self.dt_s)
        self.last_res = res

        t_true = float(self.thermal.step(float(res["p_loss"]), float(self.t_amb_c), float(self.dt_s)))

        v_meas = self._meas(float(res["v_rack"]), self.sigma_v)
        i_meas = self._meas(float(i_act), self.sigma_i)
        self.ekf.update(v_meas, i_meas)

        vmin_true = float(res.get("v_cell_min", np.nan))
        vmax_true = float(res.get("v_cell_max", np.nan))

        vmin_used, vmax_used, t_used, i_used, gas_used = self._apply_bms_visibility_layer(
            v_cell_min_true_v=vmin_true,
            v_cell_max_true_v=vmax_true,
            t_rack_true_c=t_true,
            i_rack_meas_a=float(i_meas),
            gas_alarm=False,
        )

        flags = self.fault_det.step(
            v_cell_min=vmin_used,
            v_cell_max=vmax_used,
            t_rack_c=t_used,
            i_rack_a=i_used,
            gas_alarm=gas_used,
        )
        state = self.fsm.step(flags, enable=True)

        self.time_s += self.dt_s
        self._log_sample(
            t_s=self.time_s,
            i_req=self.i_req_a,
            i_act=i_act,
            i_used=i_used,
            res=res,
            v_cell_min_true_v=vmin_true,
            v_cell_max_true_v=vmax_true,
            v_cell_min_used_v=vmin_used,
            v_cell_max_used_v=vmax_used,
            t_rack_true_c=t_true,
            t_rack_used_c=t_used,
            t_amb=float(self.t_amb_c),
            state=state,
            flags=flags,
            curr_limits=curr_limits,
        )

        return {
            "t_s": float(self.time_s),
            "i_req_a": float(self.i_req_a),
            "i_act_a": float(i_act),
            "i_rack_used_a": float(i_used),
            "soc_true": float(res.get("soc", np.nan)),
            "soc_hat": float(self.ekf.get_soc()),
            "v_cell_min_true_v": float(vmin_true),
            "v_cell_max_true_v": float(vmax_true),
            "v_cell_min_v": float(vmin_used),
            "v_cell_max_v": float(vmax_used),
            "t_rack_true_c": float(t_true),
            "t_rack_used_c": float(t_used),
            "t_amb_c": float(self.t_amb_c),
            "state_code": float(STATE_CODE_MAP[state]),
            "oc": float(bool(flags.get("oc", False))),
            "ov": float(bool(flags.get("ov", False))),
            "uv": float(bool(flags.get("uv", False))),
            "ot": float(bool(flags.get("ot", False))),
            "ut": float(bool(flags.get("ut", False))),
            "fire": float(bool(flags.get("fire", False))),
        }

    def to_npz(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(list(v), dtype=float) for k, v in self.hist.items()}


def _compute_inhibit_banner(params: Dict[str, Any], *, t_used_c: float, limits_row: Dict[str, float]) -> Tuple[Optional[str], Optional[str]]:
    if not np.isfinite(t_used_c):
        return None, None

    bms = _get(params, "bms_control", default={}) or {}
    limits = _get(params, "limits", default={}) or {}

    t_low_cutoff = _as_float(bms.get("t_low_cutoff_c", limits.get("t_min_c", np.nan)))
    t_low_derate = _as_float(bms.get("t_low_derate_start_c", np.nan))
    t_high_derate = _as_float(bms.get("t_high_derate_start_c", np.nan))
    t_high_cutoff = _as_float(bms.get("t_high_cutoff_c", limits.get("t_max_c", np.nan)))

    i_dis = float(limits_row.get("i_discharge_max_allowed", np.nan))
    i_chg = float(limits_row.get("i_charge_max_allowed", np.nan))

    if np.isfinite(i_dis) and np.isfinite(i_chg) and (i_dis <= 1e-9) and (i_chg <= 1e-9):
        if np.isfinite(t_low_cutoff) and t_used_c <= t_low_cutoff + 1e-9:
            return "warning", (
                f"Current limits are 0 because temperature is below/at low cutoff "
                f"(T_used={t_used_c:.1f}°C ≤ T_low_cutoff={t_low_cutoff:.1f}°C). "
                f"This is an operational inhibit (not necessarily a latched fault)."
            )
        if np.isfinite(t_high_cutoff) and t_used_c >= t_high_cutoff - 1e-9:
            return "warning", (
                f"Current limits are 0 because temperature is above/at high cutoff "
                f"(T_used={t_used_c:.1f}°C ≥ T_high_cutoff={t_high_cutoff:.1f}°C). "
                f"This is an operational inhibit (not necessarily a latched fault)."
            )
        return "info", "Current limits are 0 due to BMS limits (temperature / SoC / voltage). Check limits plots."

    if np.isfinite(t_low_derate) and np.isfinite(t_low_cutoff) and (t_used_c < t_low_derate) and (t_used_c > t_low_cutoff):
        return "info", (
            f"Low-temperature derating region active "
            f"(T_low_cutoff={t_low_cutoff:.1f}°C < T_used={t_used_c:.1f}°C < T_low_derate_start={t_low_derate:.1f}°C)."
        )
    if np.isfinite(t_high_derate) and np.isfinite(t_high_cutoff) and (t_used_c > t_high_derate) and (t_used_c < t_high_cutoff):
        return "info", (
            f"High-temperature derating region active "
            f"(T_high_derate_start={t_high_derate:.1f}°C < T_used={t_used_c:.1f}°C < T_high_cutoff={t_high_cutoff:.1f}°C)."
        )

    return None, None


def render_scada_playground(
    params: Dict[str, Any],
    *,
    config_path: Optional[str] = None,
    config_path_resolved: Optional[str] = None,
) -> None:
    st.header("Live SCADA (offline simulation)")

    if not config_path:
        config_path = str(st.session_state.get("active_config_path", "data/params_rack.yaml"))
    if not config_path_resolved:
        config_path_resolved = _resolve_path(config_path)

    if "scada_last_config" not in st.session_state:
        st.session_state.scada_last_config = config_path
    if st.session_state.scada_last_config != config_path:
        st.session_state.scada_last_config = config_path
        st.session_state.scada_running = False
        st.session_state.sync_autorun_toggle = True
        st.session_state.scada_event_log = deque(maxlen=200)
        st.session_state.scada_prev_flags = {"oc": 0, "ov": 0, "uv": 0, "ot": 0, "ut": 0, "fire": 0}
        st.session_state.scada_prev_state_code = None
        if "scada_sim" in st.session_state:
            del st.session_state["scada_sim"]

    _ensure_event_log()

    dt_s_param = float(_get(params, "simulation", "dt_s", default=np.nan))
    safe_dt = float(dt_s_param) if (np.isfinite(dt_s_param) and dt_s_param > 0.0) else 1.0

    _render_active_config_sidebar(params, config_path=config_path, dt_s=safe_dt)

    st.caption(f"UI build: {UI_BUILD} | Config: {config_path} | Resolved: {config_path_resolved} | dt={safe_dt:g}s")

    if "scada_running" not in st.session_state:
        st.session_state.scada_running = False

    # Auto-lock: operator intervention => setpoint
    def _lock_to_setpoint() -> None:
        st.session_state["sc_prof_mode"] = "setpoint"

    st.subheader("Controls")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Initial conditions**")
        true_soc = st.number_input("True initial SoC", 0.0, 1.0, 0.70, 0.01, key="sc_true_soc")
        ekf_soc = st.number_input("EKF initial SoC", 0.0, 1.0, 0.60, 0.01, key="sc_ekf_soc")
        t_amb = st.number_input("Ambient temperature [°C]", value=25.0, step=1.0, key="sc_t_amb")

        st.markdown("**BMS limits**")
        use_limits = st.checkbox("Use BMS current limits", value=True, key="sc_use_limits")

    with c2:
        st.markdown("**Profile**")
        profile_mode = st.selectbox("Profile mode", ["random", "setpoint"], index=0, key="sc_prof_mode")
        i_set = st.number_input("Setpoint I_req [A]", value=0.0, step=10.0, key="sc_i_set", on_change=_lock_to_setpoint)
        ramp = st.number_input("Ramp [A/s] (0 = instant)", min_value=0.0, value=0.0, step=10.0, key="sc_ramp", on_change=_lock_to_setpoint)

        st.markdown("**Random profile ranges**")
        i_dis = st.number_input("Max discharge (random) [A]", value=300.0, step=10.0, key="sc_i_dis")
        i_chg = st.number_input("Max charge (random) [A]", value=300.0, step=10.0, key="sc_i_chg")
        seg_min = st.number_input("Random segment min [s]", value=5.0, step=1.0, key="sc_seg_min")
        seg_max = st.number_input("Random segment max [s]", value=20.0, step=1.0, key="sc_seg_max")
        seed_prof = st.number_input("Profile RNG seed", value=0, step=1, key="sc_seed_prof")

    with c3:
        st.markdown("**Performance / plotting**")
        window_s = st.number_input("Plot window [s]", min_value=30.0, max_value=3600.0, value=900.0, step=30.0, key="sc_window_s")
        max_plot_points = st.number_input("Max points/plot", min_value=300, max_value=5000, value=2000, step=100, key="sc_max_plot_pts")

        marker_scope = st.selectbox(
            "Event markers on plots",
            ["Off", "Faults only", "Faults + state changes"],
            index=2,
            help="Adds vertical markers for asserted/cleared faults (and optionally state changes).",
            key="sc_marker_scope",
        )

        default_hist = int(max(5000, min(200000, round((window_s / max(safe_dt, 1e-6)) * 2.0))))
        history_max_points = st.number_input(
            "Historian max points",
            min_value=1000,
            max_value=200000,
            value=int(default_hist),
            step=1000,
            help="Memory cap: keeps only the last N samples.",
            key="sc_hist_max",
        )

        st.markdown("**Run loop**")
        steps_per_tick = st.number_input("Steps per refresh", min_value=1, max_value=500, value=5, step=1, key="sc_steps_tick")
        refresh_ms = st.number_input("Refresh interval [ms]", min_value=50, max_value=5000, value=250, step=50, key="sc_refresh_ms")

        if "scada_autorun_toggle" not in st.session_state:
            st.session_state.scada_autorun_toggle = bool(st.session_state.get("scada_running", False))
        if st.session_state.get("sync_autorun_toggle", False):
            st.session_state.scada_autorun_toggle = bool(st.session_state.get("scada_running", False))
            st.session_state.sync_autorun_toggle = False

        st.toggle("Auto-run", key="scada_autorun_toggle")
        st.session_state.scada_running = bool(st.session_state.scada_autorun_toggle)

    st.subheader("Interventions (BMS visibility layer)")

    with st.expander("Fault injections (force triggers)", expanded=False):
        d1, d2, d3, d4, d5 = st.columns([1.2, 1.2, 1.2, 1.6, 1.0])
        with d1:
            inj_uv = st.checkbox("UV injection", value=False, key="sc_inj_uv", on_change=_lock_to_setpoint)
            uv_v = st.number_input("UV: detector v_cell_min [V]", value=2.50, step=0.05, key="sc_uv_v", on_change=_lock_to_setpoint)
        with d2:
            inj_ot = st.checkbox("OT injection", value=False, key="sc_inj_ot", on_change=_lock_to_setpoint)
            ot_c = st.number_input("OT: detector T_used [°C]", value=65.0, step=1.0, key="sc_ot_c", on_change=_lock_to_setpoint)
        with d3:
            inj_ut = st.checkbox("UT injection", value=False, key="sc_inj_ut", on_change=_lock_to_setpoint)
            ut_c = st.number_input("UT: detector T_used [°C]", value=-30.0, step=1.0, key="sc_ut_c", on_change=_lock_to_setpoint)
        with d4:
            inj_oc = st.checkbox("OC injection", value=False, key="sc_inj_oc", on_change=_lock_to_setpoint)
            oc_a = st.number_input("OC: detector i_rack [A]", value=500.0, step=50.0, key="sc_oc_a", on_change=_lock_to_setpoint)
        with d5:
            inj_fire = st.checkbox("FIRE (gas alarm)", value=False, key="sc_inj_fire", on_change=_lock_to_setpoint)

    with st.expander("Manual overrides (mirror Offline segment overrides)", expanded=False):
        ovr_enable = st.checkbox("Enable manual overrides", value=False, key="sc_ovr_enable", on_change=_lock_to_setpoint)
        o1, o2 = st.columns(2)
        with o1:
            o_vmin_on = st.checkbox("Override v_cell_min (used)", value=False, disabled=not ovr_enable, key="sc_ovr_vmin_on", on_change=_lock_to_setpoint)
            o_vmin = st.number_input(
                "v_cell_min override [V]",
                value=2.80,
                step=0.05,
                disabled=(not ovr_enable or not o_vmin_on),
                key="sc_ovr_vmin",
                on_change=_lock_to_setpoint,
            )
            o_t_on = st.checkbox("Override T_rack_used", value=False, disabled=not ovr_enable, key="sc_ovr_t_on", on_change=_lock_to_setpoint)
            o_t = st.number_input(
                "T_rack_used override [°C]",
                value=30.0,
                step=1.0,
                disabled=(not ovr_enable or not o_t_on),
                key="sc_ovr_t",
                on_change=_lock_to_setpoint,
            )
        with o2:
            o_vmax_on = st.checkbox("Override v_cell_max (used)", value=False, disabled=not ovr_enable, key="sc_ovr_vmax_on", on_change=_lock_to_setpoint)
            o_vmax = st.number_input(
                "v_cell_max override [V]",
                value=3.55,
                step=0.05,
                disabled=(not ovr_enable or not o_vmax_on),
                key="sc_ovr_vmax",
                on_change=_lock_to_setpoint,
            )
            o_i_on = st.checkbox("Override i_rack (used)", value=False, disabled=not ovr_enable, key="sc_ovr_i_on", on_change=_lock_to_setpoint)
            o_i = st.number_input(
                "i_rack override [A]",
                value=0.0,
                step=50.0,
                disabled=(not ovr_enable or not o_i_on),
                key="sc_ovr_i",
                on_change=_lock_to_setpoint,
            )
        o_gas_on = st.checkbox("Override gas_alarm", value=False, disabled=not ovr_enable, key="sc_ovr_gas_on", on_change=_lock_to_setpoint)
        o_gas = st.checkbox("gas_alarm override = TRUE", value=False, disabled=(not ovr_enable or not o_gas_on), key="sc_ovr_gas", on_change=_lock_to_setpoint)

    if "scada_sim" not in st.session_state:
        st.session_state.scada_sim = LiveBmsSim(
            params,
            t_amb_c=float(t_amb),
            true_init_soc=float(true_soc),
            ekf_init_soc=float(ekf_soc),
            use_bms_limits=bool(use_limits),
            i_discharge_max_a=float(i_dis),
            i_charge_max_a=float(i_chg),
            segment_min_s=float(seg_min),
            segment_max_s=float(seg_max),
            seed_profile=int(seed_prof),
            profile=LiveProfile(mode=str(profile_mode), i_setpoint_a=float(i_set), ramp_a_per_s=float(ramp)),
            uv_v_fault=float(uv_v),
            ot_temp_c=float(ot_c),
            ut_temp_c=float(ut_c),
            oc_i_fault_a=float(oc_a),
            history_max_points=int(history_max_points),
        )

    sim: LiveBmsSim = st.session_state.scada_sim

    sim.t_amb_c = float(t_amb)
    sim.true_init_soc = _clip01(float(true_soc))
    sim.ekf_init_soc = _clip01(float(ekf_soc))
    sim.use_bms_limits = bool(use_limits)
    sim.i_discharge_max_a = float(i_dis)
    sim.i_charge_max_a = float(i_chg)
    sim.segment_min_s = float(seg_min)
    sim.segment_max_s = float(seg_max)
    sim.set_profile_mode(profile_mode)
    sim.set_setpoint_a(float(i_set))
    sim.set_ramp_a_per_s(float(ramp))
    sim.set_history_max_points(int(history_max_points))

    sim.inj.uv = bool(inj_uv)
    sim.inj.ot = bool(inj_ot)
    sim.inj.ut = bool(inj_ut)
    sim.inj.oc = bool(inj_oc)
    sim.inj.fire = bool(inj_fire)
    sim.uv_v_fault = float(uv_v)
    sim.ot_temp_c = float(ot_c)
    sim.ut_temp_c = float(ut_c)
    sim.oc_i_fault_a = float(oc_a)

    sim.ovr.enabled = bool(ovr_enable)
    sim.ovr.v_cell_min_override_v = float(o_vmin) if (ovr_enable and o_vmin_on) else None
    sim.ovr.v_cell_max_override_v = float(o_vmax) if (ovr_enable and o_vmax_on) else None
    sim.ovr.t_rack_override_c = float(o_t) if (ovr_enable and o_t_on) else None
    sim.ovr.i_rack_override_a = float(o_i) if (ovr_enable and o_i_on) else None
    sim.ovr.gas_alarm_override = bool(o_gas) if (ovr_enable and o_gas_on) else None

    st.subheader("Run controls")
    r1, r2, r3, r4, r5, r6 = st.columns([1.1, 1.1, 1.0, 1.0, 1.2, 1.6])

    with r1:
        if st.button("Start", key="sc_btn_start"):
            st.session_state.scada_running = True
            st.session_state.sync_autorun_toggle = True
            st.rerun()

    with r2:
        if st.button("Stop", key="sc_btn_stop"):
            st.session_state.scada_running = False
            st.session_state.sync_autorun_toggle = True
            st.rerun()

    with r3:
        if st.button("Step +1", key="sc_btn_step1"):
            out = sim.step()
            _update_events_from_step(out)

    with r4:
        if st.button("Step +50", key="sc_btn_step50"):
            for _ in range(50):
                out = sim.step()
                _update_events_from_step(out)

    with r5:
        if st.button("Hard reset", key="sc_btn_reset"):
            st.session_state.scada_running = False
            st.session_state.sync_autorun_toggle = True
            st.session_state.scada_event_log = deque(maxlen=200)
            st.session_state.scada_prev_flags = {"oc": 0, "ov": 0, "uv": 0, "ot": 0, "ut": 0, "fire": 0}
            st.session_state.scada_prev_state_code = None

            st.session_state.scada_sim = LiveBmsSim(
                params,
                t_amb_c=float(t_amb),
                true_init_soc=float(true_soc),
                ekf_init_soc=float(ekf_soc),
                use_bms_limits=bool(use_limits),
                i_discharge_max_a=float(i_dis),
                i_charge_max_a=float(i_chg),
                segment_min_s=float(seg_min),
                segment_max_s=float(seg_max),
                seed_profile=int(seed_prof),
                profile=LiveProfile(mode=str(profile_mode), i_setpoint_a=float(i_set), ramp_a_per_s=float(ramp)),
                uv_v_fault=float(uv_v),
                ot_temp_c=float(ot_c),
                ut_temp_c=float(ut_c),
                oc_i_fault_a=float(oc_a),
                history_max_points=int(history_max_points),
            )
            st.rerun()

    with r6:
        if st.button("Export NPZ snapshot", key="sc_btn_npz"):
            data = sim.to_npz()
            data["meta_dt_s"] = np.asarray([sim.dt_s], dtype=float)
            buf = io.BytesIO()
            np.savez(buf, **data)
            buf.seek(0)
            st.download_button(
                "Download NPZ",
                data=buf.getvalue(),
                file_name=f"live_scada_{int(time.time())}.npz",
                key="sc_dl_npz",
            )

    running = bool(st.session_state.get("scada_running", False))
    st.caption(f"Run state: {'RUNNING (auto-run)' if running else 'STOPPED'}")
    st.divider()

    do_rerun = False
    if running:
        for _ in range(int(steps_per_tick)):
            out = sim.step()
            _update_events_from_step(out)
        do_rerun = True

    arr = sim.to_npz()
    if arr["time_s"].size < 2:
        st.info("No samples yet. Use Step buttons or press Start.")
        if do_rerun:
            time.sleep(float(refresh_ms) / 1000.0)
            st.rerun()
        return

    t_end = float(arr["time_s"][-1])
    last = {k: float(arr[k][-1]) for k in arr.keys() if arr[k].size > 0}

    active_faults = [k.upper() for k in ("oc", "ov", "uv", "ot", "ut", "fire") if last.get(k, 0.0) > 0.5]
    state_last_code = int(last.get("state_code", -999))
    state_name = STATE_NAME_MAP.get(state_last_code, str(state_last_code))

    st.subheader("Status")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.metric("State", state_name)

    with k2:
        st.metric("Faults (active)", ", ".join(active_faults) if active_faults else "None")

    with k3:
        st.metric("I_req [A]", f"{last.get('i_req_a', np.nan):.1f}")
        st.metric("I_act [A]", f"{last.get('i_act_a', np.nan):.1f}")
        st.metric("I_used [A]", f"{last.get('i_rack_used_a', np.nan):.1f}")

        i_req_last = float(last.get("i_req_a", np.nan))
        lim_code = last.get("lim_code_dis", np.nan) if (np.isfinite(i_req_last) and i_req_last >= 0.0) else last.get(
            "lim_code_chg", np.nan
        )
        st.metric("Limiter", _lim_code_name(lim_code))

    with k4:
        st.metric("SoC_true", f"{last.get('soc_true', np.nan):.3f}")
        st.metric("SoC_EKF", f"{last.get('soc_hat', np.nan):.3f}")

    with k5:
        soc_true = last.get("soc_true", np.nan)
        soc_hat = last.get("soc_hat", np.nan)
        st.metric("SoC error", f"{(soc_true - soc_hat):.3f}")
        st.metric("T_used [°C]", f"{last.get('t_rack_used_c', np.nan):.1f}")

    with k6:
        st.metric("V_cell_min used [V]", f"{last.get('v_cell_min_v', np.nan):.3f}")
        st.metric("V_cell_max used [V]", f"{last.get('v_cell_max_v', np.nan):.3f}")

    if active_faults:
        severities = [_severity_for_fault(f.lower()) for f in active_faults]
        if "CRITICAL" in severities:
            st.error(f"Active alarms: {', '.join(active_faults)}")
        else:
            st.warning(f"Active alarms: {', '.join(active_faults)}")
    else:
        st.success("No active alarms.")

    if bool(use_limits):
        limits_row = {
            "i_discharge_max_allowed": float(last.get("i_discharge_lim_a", np.nan)),
            "i_charge_max_allowed": float(last.get("i_charge_lim_a", np.nan)),
        }
        kind, msg = _compute_inhibit_banner(
            params, t_used_c=float(last.get("t_rack_used_c", np.nan)), limits_row=limits_row
        )
        if kind and msg:
            if kind == "warning":
                st.warning(msg)
            elif kind == "error":
                st.error(msg)
            else:
                st.info(msg)

    with st.expander("Alarm table & event log", expanded=False):
        st.markdown("**Event log filters**")
        f1, f2, f3 = st.columns([1.2, 1.8, 1.0])

        with f1:
            severity_sel = st.multiselect(
                "Severity",
                ["CRITICAL", "WARNING", "INFO"],
                default=["CRITICAL", "WARNING", "INFO"],
                key="scada_evt_severity",
            )

        with f2:
            type_sel = st.multiselect(
                "Type",
                ["FAULT_ASSERT", "FAULT_CLEAR", "STATE_CHANGE"],
                default=["FAULT_ASSERT", "FAULT_CLEAR", "STATE_CHANGE"],
                key="scada_evt_type",
            )

        with f3:
            max_rows = st.number_input(
                "Max rows",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                key="scada_evt_max_rows",
            )

        st.markdown("**Alarm table**")
        alarm_rows = []
        meaning_map = {
            "oc": "Over-current detected (direction-dependent).",
            "ov": "Cell over-voltage (v_cell_max above limit).",
            "uv": "Cell under-voltage (v_cell_min below limit).",
            "ot": "Over-temperature (T_used above OT threshold).",
            "ut": "Under-temperature (T_used below UT threshold).",
            "fire": "Gas/smoke alarm input active.",
        }
        action_map = {
            "oc": "Limit current / check load profile.",
            "ov": "Stop charge / verify cell voltages.",
            "uv": "Stop discharge / verify SoC and cell voltages.",
            "ot": "Reduce load / increase cooling; stop if persistent.",
            "ut": "Reduce/stop charge (and possibly discharge) until temperature recovers.",
            "fire": "Emergency response per safety concept (ESD).",
        }
        for f in ("oc", "ov", "uv", "ot", "ut", "fire"):
            is_on = bool(last.get(f, 0.0) > 0.5)
            sev = _severity_for_fault(f)
            alarm_rows.append(
                {
                    "Fault": f.upper(),
                    "Active": is_on,
                    "Severity": sev,
                    "Meaning": meaning_map[f],
                    "Recommended action": action_map[f],
                }
            )
        st.dataframe(alarm_rows, use_container_width=True, hide_index=True)

        st.markdown("**Event log**")
        _ensure_event_log()
        if len(st.session_state.scada_event_log) == 0:
            st.info("Event log is empty.")
        else:
            rows = list(st.session_state.scada_event_log)
            rows_f = [r for r in rows if (r.get("severity") in severity_sel) and (r.get("type") in type_sel)]
            rows_f = rows_f[: int(max_rows)]
            st.dataframe(rows_f, use_container_width=True, hide_index=True)

    st.divider()

    t = arr["time_s"]
    t_start = max(0.0, t_end - float(window_s))
    mask = t >= t_start

    def w(k: str) -> np.ndarray:
        return arr[k][mask]

    t_w = w("time_s")
    if t_w.size == 0:
        st.info("No samples in the current window.")
        if do_rerun:
            time.sleep(float(refresh_ms) / 1000.0)
            st.rerun()
        return

    stride = int(max(1, np.ceil(t_w.size / int(max_plot_points))))
    sl = slice(None, None, stride)

    t_w = t_w[sl]
    i_req = w("i_req_a")[sl]
    i_act = w("i_act_a")[sl]
    i_used = w("i_rack_used_a")[sl]

    soc_true_s = w("soc_true")[sl]
    soc_hat_s = w("soc_hat")[sl]
    soc_err = soc_true_s - soc_hat_s

    v_min_used = w("v_cell_min_v")[sl]
    v_max_used = w("v_cell_max_v")[sl]
    v_min_true = w("v_cell_min_true_v")[sl]
    v_max_true = w("v_cell_max_true_v")[sl]

    t_true = w("t_rack_true_c")[sl]
    t_used = w("t_rack_used_c")[sl]
    t_amb_series = w("t_amb_c")[sl]

    state = w("state_code")[sl]
    oc = w("oc")[sl]
    ov = w("ov")[sl]
    uv = w("uv")[sl]
    ot = w("ot")[sl]
    ut = w("ut")[sl]
    fire = w("fire")[sl]

    i_dis_lim = w("i_discharge_lim_a")[sl]
    i_chg_lim = w("i_charge_lim_a")[sl]
    soc_spread = (w("soc_cell_max") - w("soc_cell_min"))[sl]
    v_spread = (v_max_used - v_min_used)

    event_markers = []
    if marker_scope != "Off":
        _ensure_event_log()
        all_events = list(reversed(list(st.session_state.scada_event_log)))
        allowed_types = {"FAULT_ASSERT", "FAULT_CLEAR"} if marker_scope == "Faults only" else {
            "FAULT_ASSERT",
            "FAULT_CLEAR",
            "STATE_CHANGE",
        }

        for ev in all_events:
            try:
                t_ev = float(ev.get("t_s", np.nan))
            except Exception:
                continue
            if not np.isfinite(t_ev):
                continue
            if t_ev < float(t_start) or t_ev > float(t_end):
                continue
            if ev.get("type") in allowed_types:
                event_markers.append(ev)

    def _fault_y(flag: np.ndarray, y: float) -> np.ndarray:
        return np.where(flag > 0.5, y, np.nan)

    fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True)
    ax_i, ax_soc, ax_err, ax_v, ax_temp, ax_sp, ax_state = axes

    ax_i.plot(t_w, i_req, label="I_req [A]")
    ax_i.plot(t_w, i_act, label="I_act [A]")
    ax_i.plot(t_w, i_used, linestyle=":", label="I_used [A] (BMS perceived)")
    if np.any(np.isfinite(i_dis_lim)):
        ax_i.plot(t_w, i_dis_lim, linestyle="--", label="I_dis_limit [A]")
    if np.any(np.isfinite(i_chg_lim)):
        ax_i.plot(t_w, -i_chg_lim, linestyle="--", label="-I_chg_limit [A]")
    ax_i.set_ylabel("Current")
    ax_i.grid(True)
    ax_i.legend(loc="upper right")

    ax_soc.plot(t_w, soc_true_s, label="SoC_true")
    ax_soc.plot(t_w, soc_hat_s, label="SoC_EKF")
    ax_soc.set_ylabel("SoC")
    ax_soc.grid(True)
    ax_soc.legend(loc="upper right")

    ax_err.plot(t_w, soc_err, label="SoC_true - SoC_EKF")
    ax_err.set_ylabel("SoC error")
    ax_err.grid(True)
    ax_err.legend(loc="upper right")

    ax_v.plot(t_w, v_min_used, label="V_cell_min USED [V]")
    ax_v.plot(t_w, v_max_used, label="V_cell_max USED [V]")
    if np.nanmax(np.abs(v_min_true - v_min_used)) > 1e-9 or np.nanmax(np.abs(v_max_true - v_max_used)) > 1e-9:
        ax_v.plot(t_w, v_min_true, linestyle=":", label="V_cell_min TRUE [V]")
        ax_v.plot(t_w, v_max_true, linestyle=":", label="V_cell_max TRUE [V]")
    ax_v.set_ylabel("Cell V")
    ax_v.grid(True)
    ax_v.legend(loc="upper right")

    ax_temp.plot(t_w, t_true, label="T_rack TRUE [°C]")
    ax_temp.plot(t_w, t_used, linestyle="--", label="T_rack USED [°C]")
    ax_temp.plot(t_w, t_amb_series, linestyle=":", label="T_amb [°C]")
    ax_temp.set_ylabel("Temp [°C]")
    ax_temp.grid(True)
    ax_temp.legend(loc="upper right")

    ax_sp.plot(t_w, soc_spread, label="SoC spread (max-min)")
    ax_sp2 = ax_sp.twinx()
    ax_sp2.plot(t_w, v_spread, label="V spread (max-min)", alpha=0.8)
    ax_sp.set_ylabel("SoC spread")
    ax_sp2.set_ylabel("V spread")
    ax_sp.grid(True)

    ax_state.step(t_w, state, where="post", label="State code")
    ax_state.plot(t_w, _fault_y(oc, 3.05), linestyle="--", label="OC")
    ax_state.plot(t_w, _fault_y(ov, 3.15), linestyle="--", label="OV")
    ax_state.plot(t_w, _fault_y(uv, 3.25), linestyle="--", label="UV")
    ax_state.plot(t_w, _fault_y(ot, 3.35), linestyle="--", label="OT")
    ax_state.plot(t_w, _fault_y(ut, 3.45), linestyle="--", label="UT")
    ax_state.plot(t_w, _fault_y(fire, 3.55), linestyle="--", label="FIRE")
    ax_state.set_ylabel("State / faults")
    ax_state.set_xlabel("Time [s]")
    ax_state.grid(True)
    ax_state.legend(loc="upper right")
    ax_state.set_ylim(-0.2, 3.8)

    if event_markers:
        for ev in event_markers:
            t_ev = float(ev["t_s"])
            typ = str(ev.get("type", ""))

            if typ == "FAULT_ASSERT":
                ls, a = ":", 0.8
            elif typ == "FAULT_CLEAR":
                ls, a = "--", 0.8
            else:
                ls, a = "-.", 0.6

            ax_state.axvline(t_ev, linestyle=ls, alpha=a)

    fig.suptitle("Live SCADA – Rolling window (decimated)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"t={t_end:.1f}s | state={state_name} | "
        f"T_amb={last.get('t_amb_c', np.nan):.1f}°C | T_used={last.get('t_rack_used_c', np.nan):.1f}°C | "
        f"SoC_true={last.get('soc_true', np.nan):.3f} | SoC_EKF={last.get('soc_hat', np.nan):.3f} | "
        f"I_req={last.get('i_req_a', np.nan):.1f}A | I_act={last.get('i_act_a', np.nan):.1f}A | I_used={last.get('i_rack_used_a', np.nan):.1f}A | "
        f"hist={arr['time_s'].size}/{sim.history_max_points} pts | stride={stride} | "
        f"events_in_window={len(event_markers)}"
    )

    st.divider()
    st.subheader("Documentation")

    tab1, tab2, tab3 = st.tabs(["Fault flags (OV/UV/OC/OT/UT/FIRE)", "Fault injections", "Manual overrides"])

    with tab1:
        st.markdown(
            """
**OV (Over-Voltage):** Triggered when the maximum cell voltage exceeds the configured upper threshold.  
**UV (Under-Voltage):** Triggered when the minimum cell voltage drops below the configured lower threshold.  
**OC (Over-Current):** Triggered when the rack current exceeds configured safe limits (direction-dependent for charge vs. discharge).  
**OT (Over-Temperature):** Triggered when **T_used** exceeds the configured threshold.  
**UT (Under-Temperature):** Triggered when **T_used** drops below the configured threshold.  
**FIRE (Gas alarm):** Represents a gas/smoke alarm signal (typically mapped to emergency actions).

On this page, fault flags are generated by the fault detector and then used by the FSM to transition between states.
            """
        )

    with tab2:
        st.markdown(
            """
**Fault injections** are designed for testing and demonstration. They do not necessarily modify the physical plant state.
Instead, they **force the values that the fault detector + limiter see** (the BMS perceived signals),
so you can reliably trigger a given fault and observe the FSM behavior.

- **UV injection:** Forces `v_cell_min_used` down to the selected voltage.
- **OT injection:** Forces `T_used` up to the selected temperature.
- **UT injection:** Forces `T_used` down to the selected temperature.
- **OC injection:** Forces `I_used` to the selected current magnitude. Direction follows `I_req` (default discharge if `I_req≈0`).
- **FIRE (gas alarm):** Sets `gas_alarm=True`.

**Priority rule:** if both Manual override and Injection set the same signal, Injection wins.
            """
        )

    with tab3:
        st.markdown(
            """
**Manual overrides** mirror the Offline Scenarios “segment overrides” in a Live context.

Goal: emulate measurement-layer effects and answer “What would the BMS do if it measured X?” by overriding what the
BMS logic receives.

- **Override v_cell_min / v_cell_max:** Forces the cell-voltage statistics seen by the limiter + fault detector.
- **Override T_rack_used:** Forces the rack temperature seen by the limiter + fault detector.
- **Override i_rack (used):** Forces **I_used** (rack current seen by the fault detector).
- **Override gas_alarm:** Manually sets the gas-alarm signal.

Important: Overrides affect **perceived measurements** rather than the underlying physical plant model.
That is why the plots show both TRUE and USED signals (and why `I_used` may differ from `I_act`).
            """
        )

    if do_rerun:
        time.sleep(float(refresh_ms) / 1000.0)
        st.rerun()
