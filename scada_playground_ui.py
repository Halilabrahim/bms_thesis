# scada_playground_ui.py
from __future__ import annotations

import io
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.sim_runner import build_models
from src.fsm import BMSState
from src.bms_logic import compute_current_limits

UI_BUILD = "2025-12-31_scada_ux_v1"


STATE_CODE_MAP = {
    BMSState.OFF: 0,
    BMSState.RUN: 1,
    BMSState.FAULT: 2,
    BMSState.EMERGENCY_SHUTDOWN: 3,
}
STATE_NAME_MAP = {v: k.name for k, v in STATE_CODE_MAP.items()}


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
    t_derate_start = _as_float(bms.get("t_high_derate_start_c", float("nan")))
    t_cutoff = _as_float(bms.get("t_high_cutoff_c", limits.get("t_max_c", float("nan"))))

    c_th = _as_float(thermal.get("c_th_j_per_k", float("nan")))
    r_th = _as_float(thermal.get("r_th_k_per_w", float("nan")))
    tau_s = float(c_th * r_th) if (np.isfinite(c_th) and np.isfinite(r_th)) else float("nan")
    tau_h = float(tau_s / 3600.0) if np.isfinite(tau_s) else float("nan")

    # Quick trigger suggestions (pragmatic defaults)
    oc_dis = _as_float(faults.get("oc_discharge_a", float("nan")))
    oc_chg = _as_float(faults.get("oc_charge_a", float("nan")))
    fire_temp = _as_float(faults.get("fire_temp_c", _get(params, "fire_detection", "temp_threshold_c", default=float("nan"))))

    rec_uv = (uv_thr - 0.05) if np.isfinite(uv_thr) else float("nan")
    rec_ov = (ov_thr + 0.02) if np.isfinite(ov_thr) else float("nan")
    rec_ot = (ot_thr + 5.0) if np.isfinite(ot_thr) else float("nan")
    rec_oc_dis = (oc_dis + 50.0) if np.isfinite(oc_dis) else float("nan")
    rec_oc_chg = (oc_chg + 50.0) if np.isfinite(oc_chg) else float("nan")
    rec_fire_t = (fire_temp + 1.0) if np.isfinite(fire_temp) else float("nan")

    # Extra (optional) spec section (from params_rack.yaml)
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
        if np.isfinite(t_derate_start):
            st.write(f"T derate start: {t_derate_start:.1f} °C")
        if np.isfinite(t_cutoff):
            st.write(f"T cutoff: {t_cutoff:.1f} °C")

        if np.isfinite(tau_s):
            st.write(f"Thermal time constant estimate: τ ≈ {tau_s:.0f} s (~{tau_h:.1f} h).")
            st.caption("So changing T_amb in later segments may not raise T_rack quickly. Use T_rack override or OT injection for fast logic tests.")

        # meta section (offline ile aynı mantık)
        sys_name = str(meta.get("system_name", "")).strip()
        ver = str(meta.get("version", "")).strip()
        if sys_name or ver:
            st.divider()
            st.caption("meta")
            if sys_name:
                st.write(f'system_name: "{sys_name}"')
            if ver:
                st.write(f'version: "{ver}"')

        # Useful specs (params_rack.yaml’dan)
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

        # Fault quick ref (fault tetiklemek için)
        st.divider()
        st.caption("Fault quick reference (for fast triggering)")
        rows = []
        if np.isfinite(uv_thr):
            rows.append({
                "Fault": "UV",
                "Threshold": f"{uv_thr:.2f} V",
                "Quick trigger": f"Override v_cell_min ≈ {rec_uv:.2f} V",
                "Best knob": "Manual override v_cell_min OR UV injection",
            })
        if np.isfinite(ov_thr):
            rows.append({
                "Fault": "OV",
                "Threshold": f"{ov_thr:.2f} V",
                "Quick trigger": f"Override v_cell_max ≈ {rec_ov:.2f} V",
                "Best knob": "Manual override v_cell_max",
            })
        if np.isfinite(ot_thr):
            rows.append({
                "Fault": "OT",
                "Threshold": f"{ot_thr:.1f} °C",
                "Quick trigger": f"Override t_rack ≈ {rec_ot:.1f} °C",
                "Best knob": "Manual override t_rack OR OT injection",
            })
        if np.isfinite(oc_dis) or np.isfinite(oc_chg):
            thr = []
            if np.isfinite(oc_dis):
                thr.append(f"dis>{oc_dis:.0f}A")
            if np.isfinite(oc_chg):
                thr.append(f"chg>{oc_chg:.0f}A")
            rows.append({
                "Fault": "OC",
                "Threshold": ", ".join(thr),
                "Quick trigger": f"Override i_rack ≈ {rec_oc_dis:.0f}A (dis) / {rec_oc_chg:.0f}A (chg)",
                "Best knob": "Manual override i_rack OR OC injection",
            })
        rows.append({
            "Fault": "FIRE",
            "Threshold": "gas_alarm=True",
            "Quick trigger": f"Enable FIRE injection (or gas_alarm override). Temp ref: {rec_fire_t:.1f}°C",
            "Best knob": "FIRE injection OR gas_alarm override",
        })

        st.dataframe(rows, use_container_width=True, hide_index=True)

def _ensure_event_log(maxlen: int = 200) -> None:
    if "scada_event_log" not in st.session_state:
        st.session_state.scada_event_log = deque(maxlen=maxlen)
    if "scada_prev_flags" not in st.session_state:
        st.session_state.scada_prev_flags = {"oc": 0, "ov": 0, "uv": 0, "ot": 0, "fire": 0}
    if "scada_prev_state_code" not in st.session_state:
        st.session_state.scada_prev_state_code = None


def _event_add(t_s: float, severity: str, kind: str, message: str) -> None:
    _ensure_event_log()
    st.session_state.scada_event_log.appendleft(
        {"t_s": float(t_s), "severity": severity, "type": kind, "message": message}
    )


def _severity_for_fault(fault: str) -> str:
    # Thesis-friendly default mapping (can be refined later)
    if fault in ("fire", "ot", "ov", "uv"):
        return "CRITICAL"
    if fault == "oc":
        return "WARNING"
    return "INFO"


def _update_events_from_step(step_out: Dict[str, float]) -> None:
    """
    Edge-based event logging:
    - fault asserted / cleared
    - state transitions
    """
    _ensure_event_log()
    t_s = float(step_out.get("t_s", np.nan))

    cur_flags = {
        "oc": int(step_out.get("oc", 0.0) > 0.5),
        "ov": int(step_out.get("ov", 0.0) > 0.5),
        "uv": int(step_out.get("uv", 0.0) > 0.5),
        "ot": int(step_out.get("ot", 0.0) > 0.5),
        "fire": int(step_out.get("fire", 0.0) > 0.5),
    }
    prev_flags = dict(st.session_state.scada_prev_flags)

    for k in ("oc", "ov", "uv", "ot", "fire"):
        if prev_flags.get(k, 0) == 0 and cur_flags[k] == 1:
            _event_add(
                t_s=t_s,
                severity=_severity_for_fault(k),
                kind="FAULT_ASSERT",
                message=f"{k.upper()} asserted",
            )
        if prev_flags.get(k, 0) == 1 and cur_flags[k] == 0:
            _event_add(
                t_s=t_s,
                severity="INFO",
                kind="FAULT_CLEAR",
                message=f"{k.upper()} cleared",
            )

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
            "soc_true",
            "soc_hat",
            "v_cell_min_v",
            "v_cell_max_v",
            "t_rack_c",
            "t_amb_c",
            "state_code",
            "oc",
            "ov",
            "uv",
            "ot",
            "fire",
            "i_discharge_lim_a",
            "i_charge_lim_a",
            "soc_cell_min",
            "soc_cell_mean",
            "soc_cell_max",
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

    def set_profile_mode(self, mode: str) -> None:
        mode = str(mode).strip().lower()
        self.profile.mode = "setpoint" if mode == "setpoint" else "random"

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
        vmin = float(self.last_res.get("v_cell_min", np.nan))
        vmax = float(self.last_res.get("v_cell_max", np.nan))
        self._log_sample(
            t_s=0.0,
            i_req=0.0,
            i_act=0.0,
            res=self.last_res,
            v_cell_min_used=vmin,
            v_cell_max_used=vmax,
            t_rack=float(self.thermal.t_c),
            t_amb=float(self.t_amb_c),
            state=self.fsm.state,
            flags={"oc": False, "ov": False, "uv": False, "ot": False, "fire": False},
            curr_limits=None,
        )

    def _log_sample(
        self,
        *,
        t_s: float,
        i_req: float,
        i_act: float,
        res: Dict[str, Any],
        v_cell_min_used: float,
        v_cell_max_used: float,
        t_rack: float,
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
        self.hist["soc_true"].append(soc_true)
        self.hist["soc_hat"].append(soc_hat)

        self.hist["v_cell_min_v"].append(float(v_cell_min_used))
        self.hist["v_cell_max_v"].append(float(v_cell_max_used))

        self.hist["t_rack_c"].append(float(t_rack))
        self.hist["t_amb_c"].append(float(t_amb))

        self.hist["state_code"].append(float(STATE_CODE_MAP[state]))

        self.hist["oc"].append(float(bool(flags.get("oc", False))))
        self.hist["ov"].append(float(bool(flags.get("ov", False))))
        self.hist["uv"].append(float(bool(flags.get("uv", False))))
        self.hist["ot"].append(float(bool(flags.get("ot", False))))
        self.hist["fire"].append(float(bool(flags.get("fire", False))))

        if curr_limits is None:
            self.hist["i_discharge_lim_a"].append(np.nan)
            self.hist["i_charge_lim_a"].append(np.nan)
        else:
            self.hist["i_discharge_lim_a"].append(float(curr_limits.get("i_discharge_max_allowed", np.nan)))
            self.hist["i_charge_lim_a"].append(float(curr_limits.get("i_charge_max_allowed", np.nan)))

        self.hist["soc_cell_min"].append(float(res.get("soc_cell_min", soc_true)))
        self.hist["soc_cell_mean"].append(float(res.get("soc_cell_mean", soc_true)))
        self.hist["soc_cell_max"].append(float(res.get("soc_cell_max", soc_true)))

    def step(self) -> Dict[str, float]:
        if self.profile.mode == "random":
            if self.time_s >= self.next_switch_s:
                self._choose_new_request_random()
        else:
            self._update_i_req_setpoint()

        curr_limits: Optional[Dict[str, float]] = None
        if self.fsm.state in (BMSState.FAULT, BMSState.EMERGENCY_SHUTDOWN):
            i_act = 0.0
        else:
            if self.use_bms_limits and self.last_res is not None:
                curr_limits = compute_current_limits(
                    soc_hat=float(self.ekf.get_soc()),
                    t_rack_c=float(self.thermal.t_c),
                    v_cell_min=float(self.last_res.get("v_cell_min", np.nan)),
                    v_cell_max=float(self.last_res.get("v_cell_max", np.nan)),
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

        t_rack = float(self.thermal.step(float(res["p_loss"]), float(self.t_amb_c), float(self.dt_s)))

        v_meas = self._meas(float(res["v_rack"]), self.sigma_v)
        i_meas = self._meas(float(i_act), self.sigma_i)
        self.ekf.update(v_meas, i_meas)

        v_cell_min_used = float(res.get("v_cell_min", np.nan))
        v_cell_max_used = float(res.get("v_cell_max", np.nan))
        t_for_fault = float(t_rack)
        i_for_fault = float(i_meas)
        gas_alarm = False

        if self.inj.uv:
            v_cell_min_used = min(v_cell_min_used, float(self.uv_v_fault))
        if self.inj.ot:
            t_for_fault = max(t_for_fault, float(self.ot_temp_c))
        if self.inj.oc:
            i_for_fault = float(self.oc_i_fault_a) if i_for_fault >= 0 else -float(self.oc_i_fault_a)
        if self.inj.fire:
            gas_alarm = True

        if self.ovr.enabled:
            if self.ovr.v_cell_min_override_v is not None:
                v_cell_min_used = float(self.ovr.v_cell_min_override_v)
            if self.ovr.v_cell_max_override_v is not None:
                v_cell_max_used = float(self.ovr.v_cell_max_override_v)
            if self.ovr.t_rack_override_c is not None:
                t_for_fault = float(self.ovr.t_rack_override_c)
            if self.ovr.i_rack_override_a is not None:
                i_for_fault = float(self.ovr.i_rack_override_a)
            if self.ovr.gas_alarm_override is not None:
                gas_alarm = bool(self.ovr.gas_alarm_override)

        flags = self.fault_det.step(
            v_cell_min=v_cell_min_used,
            v_cell_max=v_cell_max_used,
            t_rack_c=t_for_fault,
            i_rack_a=i_for_fault,
            gas_alarm=gas_alarm,
        )
        state = self.fsm.step(flags, enable=True)

        self.time_s += self.dt_s
        self._log_sample(
            t_s=self.time_s,
            i_req=self.i_req_a,
            i_act=i_act,
            res=res,
            v_cell_min_used=v_cell_min_used,
            v_cell_max_used=v_cell_max_used,
            t_rack=t_rack,
            t_amb=float(self.t_amb_c),
            state=state,
            flags=flags,
            curr_limits=curr_limits,
        )

        return {
            "t_s": float(self.time_s),
            "i_req_a": float(self.i_req_a),
            "i_act_a": float(i_act),
            "soc_true": float(res.get("soc", np.nan)),
            "soc_hat": float(self.ekf.get_soc()),
            "v_cell_min_v": float(v_cell_min_used),
            "v_cell_max_v": float(v_cell_max_used),
            "t_rack_c": float(t_rack),
            "t_amb_c": float(self.t_amb_c),
            "state_code": float(STATE_CODE_MAP[state]),
            "oc": float(bool(flags.get("oc", False))),
            "ov": float(bool(flags.get("ov", False))),
            "uv": float(bool(flags.get("uv", False))),
            "ot": float(bool(flags.get("ot", False))),
            "fire": float(bool(flags.get("fire", False))),
        }

    def to_npz(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(list(v), dtype=float) for k, v in self.hist.items()}


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

    # Reset state if config changes
    if "scada_last_config" not in st.session_state:
        st.session_state.scada_last_config = config_path
    if st.session_state.scada_last_config != config_path:
        st.session_state.scada_last_config = config_path
        st.session_state.scada_running = False
        st.session_state.sync_autorun_toggle = True
        st.session_state.scada_event_log = deque(maxlen=200)
        st.session_state.scada_prev_flags = {"oc": 0, "ov": 0, "uv": 0, "ot": 0, "fire": 0}
        st.session_state.scada_prev_state_code = None
        if "scada_sim" in st.session_state:
            del st.session_state["scada_sim"]

    _ensure_event_log()

    dt_s_param = float(_get(params, "simulation", "dt_s", default=np.nan))
    safe_dt = float(dt_s_param) if (np.isfinite(dt_s_param) and dt_s_param > 0.0) else 1.0

    _render_active_config_sidebar(params, config_path=config_path, dt_s=safe_dt)

    st.caption(
        f"UI build: {UI_BUILD} | Config: {config_path} | Resolved: {config_path_resolved} | dt={safe_dt:g}s"
    )

    if "scada_running" not in st.session_state:
        st.session_state.scada_running = False

    # -----------------------------
    # Controls
    # -----------------------------
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
        i_set = st.number_input("Setpoint I_req [A]", value=0.0, step=10.0, key="sc_i_set")
        ramp = st.number_input("Ramp [A/s] (0 = instant)", min_value=0.0, value=0.0, step=10.0, key="sc_ramp")

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

        # Auto-run toggle (widget key != scada_running)
        if "scada_autorun_toggle" not in st.session_state:
            st.session_state.scada_autorun_toggle = bool(st.session_state.get("scada_running", False))
        if st.session_state.get("sync_autorun_toggle", False):
            st.session_state.scada_autorun_toggle = bool(st.session_state.get("scada_running", False))
            st.session_state.sync_autorun_toggle = False

        st.toggle("Auto-run", key="scada_autorun_toggle")
        st.session_state.scada_running = bool(st.session_state.scada_autorun_toggle)

    # -----------------------------
    # Interventions
    # -----------------------------
    st.subheader("Interventions (BMS visibility)")

    with st.expander("Fault injections (force triggers)", expanded=False):
        d1, d2, d3, d4 = st.columns([1.2, 1.2, 1.6, 1.0])
        with d1:
            inj_uv = st.checkbox("UV injection", value=False, key="sc_inj_uv")
            uv_v = st.number_input("UV: detector v_cell_min [V]", value=2.50, step=0.05, key="sc_uv_v")
        with d2:
            inj_ot = st.checkbox("OT injection", value=False, key="sc_inj_ot")
            ot_c = st.number_input("OT: detector t_rack [°C]", value=65.0, step=1.0, key="sc_ot_c")
        with d3:
            inj_oc = st.checkbox("OC injection", value=False, key="sc_inj_oc")
            oc_a = st.number_input("OC: detector i_rack [A]", value=500.0, step=50.0, key="sc_oc_a")
        with d4:
            inj_fire = st.checkbox("FIRE (gas alarm)", value=False, key="sc_inj_fire")

    with st.expander("Manual overrides (mirror offline segment overrides)", expanded=False):
        ovr_enable = st.checkbox("Enable manual overrides", value=False, key="sc_ovr_enable")
        o1, o2 = st.columns(2)
        with o1:
            o_vmin_on = st.checkbox("Override v_cell_min", value=False, disabled=not ovr_enable, key="sc_ovr_vmin_on")
            o_vmin = st.number_input(
                "v_cell_min override [V]",
                value=2.80,
                step=0.05,
                disabled=(not ovr_enable or not o_vmin_on),
                key="sc_ovr_vmin",
            )
            o_t_on = st.checkbox("Override t_rack", value=False, disabled=not ovr_enable, key="sc_ovr_t_on")
            o_t = st.number_input(
                "t_rack override [°C]",
                value=30.0,
                step=1.0,
                disabled=(not ovr_enable or not o_t_on),
                key="sc_ovr_t",
            )
        with o2:
            o_vmax_on = st.checkbox("Override v_cell_max", value=False, disabled=not ovr_enable, key="sc_ovr_vmax_on")
            o_vmax = st.number_input(
                "v_cell_max override [V]",
                value=3.55,
                step=0.05,
                disabled=(not ovr_enable or not o_vmax_on),
                key="sc_ovr_vmax",
            )
            o_i_on = st.checkbox("Override i_rack", value=False, disabled=not ovr_enable, key="sc_ovr_i_on")
            o_i = st.number_input(
                "i_rack override [A]",
                value=0.0,
                step=50.0,
                disabled=(not ovr_enable or not o_i_on),
                key="sc_ovr_i",
            )
        o_gas_on = st.checkbox("Override gas_alarm", value=False, disabled=not ovr_enable, key="sc_ovr_gas_on")
        o_gas = st.checkbox("gas_alarm override = TRUE", value=False, disabled=(not ovr_enable or not o_gas_on), key="sc_ovr_gas")

    # -----------------------------
    # Create / reuse sim
    # -----------------------------
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
            oc_i_fault_a=float(oc_a),
            history_max_points=int(history_max_points),
        )

    sim: LiveBmsSim = st.session_state.scada_sim

    # Apply live-updatable knobs
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

    # Interventions
    sim.inj.uv = bool(inj_uv)
    sim.inj.ot = bool(inj_ot)
    sim.inj.oc = bool(inj_oc)
    sim.inj.fire = bool(inj_fire)
    sim.uv_v_fault = float(uv_v)
    sim.ot_temp_c = float(ot_c)
    sim.oc_i_fault_a = float(oc_a)

    sim.ovr.enabled = bool(ovr_enable)
    sim.ovr.v_cell_min_override_v = float(o_vmin) if (ovr_enable and o_vmin_on) else None
    sim.ovr.v_cell_max_override_v = float(o_vmax) if (ovr_enable and o_vmax_on) else None
    sim.ovr.t_rack_override_c = float(o_t) if (ovr_enable and o_t_on) else None
    sim.ovr.i_rack_override_a = float(o_i) if (ovr_enable and o_i_on) else None
    sim.ovr.gas_alarm_override = bool(o_gas) if (ovr_enable and o_gas_on) else None

    # -----------------------------
    # Run controls
    # -----------------------------
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
            st.session_state.scada_prev_flags = {"oc": 0, "ov": 0, "uv": 0, "ot": 0, "fire": 0}
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

    # -----------------------------
    # Auto-run step (IMPORTANT: NO rerun here)
    # -----------------------------
    do_rerun = False
    if running:
        for _ in range(int(steps_per_tick)):
            out = sim.step()
            _update_events_from_step(out)
        do_rerun = True

    # -----------------------------
    # Data snapshot for UI
    # -----------------------------
    arr = sim.to_npz()
    if arr["time_s"].size < 2:
        st.info("No samples yet. Use Step buttons or press Start.")
        if do_rerun:
            time.sleep(float(refresh_ms) / 1000.0)
            st.rerun()
        return

    # Latest values (for KPI + alarms)
    t_end = float(arr["time_s"][-1])
    last = {k: float(arr[k][-1]) for k in arr.keys() if arr[k].size > 0}

    active_faults = [k.upper() for k in ("oc", "ov", "uv", "ot", "fire") if last.get(k, 0.0) > 0.5]
    state_last_code = int(last.get("state_code", -999))
    state_name = STATE_NAME_MAP.get(state_last_code, str(state_last_code))

    # -----------------------------
    # KPI / status bar
    # -----------------------------
    st.subheader("Status")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.metric("State", state_name)
    with k2:
        st.metric("Faults (active)", ", ".join(active_faults) if active_faults else "None")
    with k3:
        st.metric("I_req [A]", f"{last.get('i_req_a', np.nan):.1f}")
        st.metric("I_act [A]", f"{last.get('i_act_a', np.nan):.1f}")
    with k4:
        soc_true = last.get("soc_true", np.nan)
        soc_hat = last.get("soc_hat", np.nan)
        st.metric("SoC_true", f"{soc_true:.3f}")
        st.metric("SoC_EKF", f"{soc_hat:.3f}")
    with k5:
        st.metric("SoC error", f"{(soc_true - soc_hat):.3f}")
        st.metric("T_rack [°C]", f"{last.get('t_rack_c', np.nan):.1f}")
    with k6:
        st.metric("V_cell_min [V]", f"{last.get('v_cell_min_v', np.nan):.3f}")
        st.metric("V_cell_max [V]", f"{last.get('v_cell_max_v', np.nan):.3f}")

    # Alarm banner
    if active_faults:
        severities = [_severity_for_fault(f.lower()) for f in active_faults]
        if "CRITICAL" in severities:
            st.error(f"Active alarms: {', '.join(active_faults)}")
        else:
            st.warning(f"Active alarms: {', '.join(active_faults)}")
    else:
        st.success("No active alarms.")

    # -----------------------------
    # Alarm table + event log
    # -----------------------------
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
        for f in ("oc", "ov", "uv", "ot", "fire"):
            is_on = bool(last.get(f, 0.0) > 0.5)
            sev = _severity_for_fault(f)
            meaning = {
                "oc": "Over-current detected (direction-dependent).",
                "ov": "Cell over-voltage (v_cell_max above limit).",
                "uv": "Cell under-voltage (v_cell_min below limit).",
                "ot": "Over-temperature (rack temperature above limit).",
                "fire": "Gas/smoke alarm input active.",
            }[f]
            action = {
                "oc": "Limit current / check load profile.",
                "ov": "Stop charge / verify cell voltages.",
                "uv": "Stop discharge / verify SoC and cell voltages.",
                "ot": "Reduce load / increase cooling; stop if persistent.",
                "fire": "Emergency response per safety concept (ESD).",
            }[f]
            alarm_rows.append(
                {"Fault": f.upper(), "Active": is_on, "Severity": sev, "Meaning": meaning, "Recommended action": action}
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

    # -----------------------------
    # Plotting (rolling window + decimation)
    # -----------------------------
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
    soc_true = w("soc_true")[sl]
    soc_hat = w("soc_hat")[sl]
    soc_err = soc_true - soc_hat
    v_min = w("v_cell_min_v")[sl]
    v_max = w("v_cell_max_v")[sl]
    t_rack = w("t_rack_c")[sl]
    t_amb_series = w("t_amb_c")[sl]
    state = w("state_code")[sl]
    oc = w("oc")[sl]
    ov = w("ov")[sl]
    uv = w("uv")[sl]
    ot = w("ot")[sl]
    fire = w("fire")[sl]

    # --- Event markers (vertical lines) ---
    event_markers = []
    if marker_scope != "Off":
        _ensure_event_log()
        all_events = list(reversed(list(st.session_state.scada_event_log)))  # oldest -> newest
        allowed_types = {"FAULT_ASSERT", "FAULT_CLEAR"} if marker_scope == "Faults only" else {"FAULT_ASSERT", "FAULT_CLEAR", "STATE_CHANGE"}

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

    i_dis_lim = w("i_discharge_lim_a")[sl]
    i_chg_lim = w("i_charge_lim_a")[sl]
    soc_spread = (w("soc_cell_max") - w("soc_cell_min"))[sl]
    v_spread = (v_max - v_min)

    def _fault_y(flag: np.ndarray, y: float) -> np.ndarray:
        return np.where(flag > 0.5, y, np.nan)

    fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True)
    ax_i, ax_soc, ax_err, ax_v, ax_temp, ax_sp, ax_state = axes

    ax_i.plot(t_w, i_req, label="I_req [A]")
    ax_i.plot(t_w, i_act, label="I_act [A]")
    if np.any(np.isfinite(i_dis_lim)):
        ax_i.plot(t_w, i_dis_lim, linestyle="--", label="I_dis_limit [A]")
    if np.any(np.isfinite(i_chg_lim)):
        ax_i.plot(t_w, -i_chg_lim, linestyle="--", label="-I_chg_limit [A]")
    ax_i.set_ylabel("Current")
    ax_i.grid(True)
    ax_i.legend(loc="upper right")

    ax_soc.plot(t_w, soc_true, label="SoC_true")
    ax_soc.plot(t_w, soc_hat, label="SoC_EKF")
    ax_soc.set_ylabel("SoC")
    ax_soc.grid(True)
    ax_soc.legend(loc="upper right")

    ax_err.plot(t_w, soc_err, label="SoC_true - SoC_EKF")
    ax_err.set_ylabel("SoC error")
    ax_err.grid(True)
    ax_err.legend(loc="upper right")

    ax_v.plot(t_w, v_min, label="V_cell_min [V]")
    ax_v.plot(t_w, v_max, label="V_cell_max [V]")
    ax_v.set_ylabel("Cell V")

    v_lo = np.nanmin(np.concatenate([v_min, v_max]))
    v_hi = np.nanmax(np.concatenate([v_min, v_max]))
    if np.isfinite(v_lo) and np.isfinite(v_hi):
        span = max(0.1, v_hi - v_lo)
        margin = 0.10 * span
        ax_v.set_ylim(v_lo - margin, v_hi + margin)

    ax_v.grid(True)
    ax_v.legend(loc="upper right")

    ax_temp.plot(t_w, t_rack, label="T_rack [°C]")
    ax_temp.plot(t_w, t_amb_series, linestyle="--", label="T_amb [°C]")
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
    ax_state.plot(t_w, _fault_y(oc, 3.10), linestyle="--", label="OC")
    ax_state.plot(t_w, _fault_y(ov, 3.20), linestyle="--", label="OV")
    ax_state.plot(t_w, _fault_y(uv, 3.30), linestyle="--", label="UV")
    ax_state.plot(t_w, _fault_y(ot, 3.40), linestyle="--", label="OT")
    ax_state.plot(t_w, _fault_y(fire, 3.50), linestyle="--", label="FIRE")
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
        f"T_amb={last.get('t_amb_c', np.nan):.1f}°C | T_rack={last.get('t_rack_c', np.nan):.1f}°C | "
        f"SoC_true={last.get('soc_true', np.nan):.3f} | SoC_EKF={last.get('soc_hat', np.nan):.3f} | "
        f"hist={arr['time_s'].size}/{sim.history_max_points} pts | stride={stride} | "
        f"events_in_window={len(event_markers)}"
    )

    # -----------------------------
    # Documentation tabs (bottom)
    # -----------------------------
    st.divider()
    st.subheader("Documentation")

    tab1, tab2, tab3 = st.tabs(["Fault flags (OV/UV/OC/OT/FIRE)", "Fault injections", "Manual overrides"])

    with tab1:
        st.markdown(
            """
**OV (Over-Voltage):** Triggered when the maximum cell voltage exceeds the configured upper threshold (`v_cell_max`).  
**UV (Under-Voltage):** Triggered when the minimum cell voltage drops below the configured lower threshold (`v_cell_min`).  
**OC (Over-Current):** Triggered when the rack current exceeds configured safe limits (direction-dependent for charge vs. discharge).  
**OT (Over-Temperature):** Triggered when the rack temperature exceeds the configured threshold (`t_rack`).  
**FIRE (Gas alarm):** Represents a gas/smoke alarm signal. In many safety concepts this is tied to **emergency actions** (e.g., ESD).

On this page, fault flags are generated by the fault detector and then used by the FSM to transition between states.
            """
        )

    with tab2:
        st.markdown(
            """
**Fault injections** are designed for testing and demonstration. They do not necessarily modify the physical plant state.
Instead, they **force the values that the fault detector “sees”**, so you can reliably trigger a given fault and observe
the FSM behavior.

- **UV injection:** Forces the detector input `v_cell_min` down to the selected voltage → UV becomes easy to trigger.
- **OT injection:** Forces the detector input `t_rack` up to the selected temperature → OT becomes easy to trigger.
- **OC injection:** Forces the detector input `i_rack` to the selected current magnitude → OC becomes easy to trigger.
- **FIRE (gas alarm):** Sets `gas_alarm=True` → FIRE becomes active.
            """
        )

    with tab3:
        st.markdown(
            """
**Manual overrides** mirror the Offline Scenarios “segment overrides” in a Live context.

Goal: emulate measurement-layer effects and answer “What would the BMS do if it measured X?” by overriding what the
fault detector receives.

- **Override v_cell_min / v_cell_max:** Forces the cell-voltage statistics seen by the fault detector.
- **Override t_rack:** Forces the rack temperature seen by the fault detector.
- **Override i_rack:** Forces the rack current seen by the fault detector.
- **Override gas_alarm:** Manually sets the gas-alarm signal.

Important: Overrides typically affect the **perceived measurements** rather than the underlying physical plant model.
            """
        )

    # -----------------------------
    # Final: schedule next refresh AFTER UI is rendered
    # -----------------------------
    if do_rerun:
        time.sleep(float(refresh_ms) / 1000.0)
        st.rerun()
