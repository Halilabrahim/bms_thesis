# scada_playground_ui.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.sim_runner import build_models
from src.fsm import BMSState
from src.bms_logic import compute_current_limits


# Visible build marker (to verify Streamlit is loading THIS file)
UI_BUILD = "2025-12-23_livecfg_v1"


STATE_CODE_MAP = {
    BMSState.OFF: 0,
    BMSState.RUN: 1,
    BMSState.FAULT: 2,
    BMSState.EMERGENCY_SHUTDOWN: 3,
}

STATE_NAME_MAP = {
    0: "OFF",
    1: "RUN",
    2: "FAULT",
    3: "EMERGENCY_SHUTDOWN",
}


def _get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safe nested dict get: _get(params, 'structure','cells_in_series_per_pack', default=16)."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _isfinite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _fmt(x: Any, unit: str = "", nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        xf = float(x)
        if not np.isfinite(xf):
            return "n/a"
        return f"{xf:.{nd}f}{unit}"
    except Exception:
        return str(x)


def _resolve_path(p: str) -> str:
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


@dataclass
class LiveInjections:
    uv: bool = False
    fire: bool = False
    ot: bool = False
    oc: bool = False


@dataclass
class LiveProfile:
    """
    How we generate requested current I_req in Live mode.

    mode:
      - "random": piecewise-constant random segments
      - "setpoint": user-defined setpoint (can change during runtime)

    ramp_a_per_s:
      - If > 0: ramp toward setpoint smoothly
      - If 0: immediate jump to setpoint
    """
    mode: str = "random"          # "random" | "setpoint"
    i_setpoint_a: float = 0.0
    ramp_a_per_s: float = 0.0     # 0 => instant


class LiveBmsSim:
    """
    Step-by-step rack simulation for a "SCADA-like" view.

    - Same build_models() as offline.
    - Injections modify ONLY fault-detector inputs.
    - Ambient temperature is a real thermal input and is applied live.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        *,
        t_amb_c: float,
        true_init_soc: float,
        ekf_init_soc: float,
        use_bms_limits: bool,
        # random profile parameters
        i_discharge_max_a: float,
        i_charge_max_a: float,
        segment_min_s: float,
        segment_max_s: float,
        seed_profile: int,
        # profile
        profile: LiveProfile,
        # injection magnitudes
        uv_v_fault: float,
        ot_temp_c: float,
        oc_i_fault_a: float,
    ) -> None:
        self.params = params
        self.models = build_models(params)

        self.ecm = self.models["ecm"]
        self.thermal = self.models["thermal"]
        self.ekf = self.models["ekf"]
        self.bms_params = self.models["bms_params"]
        self.fault_det = self.models["fault_det"]
        self.fsm = self.models["fsm"]
        self.dt_s = float(self.models["dt_s"])

        sensor_cfg = params.get("sensors", {})
        self.sigma_v = float(sensor_cfg.get("voltage_noise_std_v", 0.0))
        self.sigma_i = float(sensor_cfg.get("current_noise_std_a", 0.0))
        self.sigma_t = float(sensor_cfg.get("temp_noise_std_c", 0.0))
        self.rng_meas = np.random.default_rng(int(sensor_cfg.get("seed", 0)))

        self.use_bms_limits = bool(use_bms_limits)

        self.i_discharge_max_a = float(i_discharge_max_a)
        self.i_charge_max_a = float(i_charge_max_a)
        self.segment_min_s = float(segment_min_s)
        self.segment_max_s = float(segment_max_s)
        self.seed_profile = int(seed_profile)
        self.rng_prof = np.random.default_rng(int(seed_profile))

        self.profile = profile
        self.i_req_target_a = float(profile.i_setpoint_a)

        self.inj = LiveInjections()
        self.uv_v_fault = float(uv_v_fault)
        self.ot_temp_c = float(ot_temp_c)
        self.oc_i_fault_a = float(oc_i_fault_a)

        self.t_amb_c = float(t_amb_c)
        self.true_init_soc = _clip01(true_init_soc)
        self.ekf_init_soc = _clip01(ekf_init_soc)

        self.time_s = 0.0
        self.i_req_a = 0.0
        self.next_switch_s = 0.0
        self.last_res: Optional[Dict[str, float]] = None

        self.hist: Dict[str, List[float]] = {}
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
            "state_code",
            "oc",
            "ov",
            "uv",
            "ot",
            "fire",
            "soc_cell_min",
            "soc_cell_mean",
            "soc_cell_max",
        ]
        self.hist = {k: [] for k in keys}

    def set_profile_mode(self, mode: str) -> None:
        mode = str(mode).strip().lower()
        if mode not in ("random", "setpoint"):
            mode = "random"
        self.profile.mode = mode
        if mode == "setpoint":
            self.i_req_target_a = float(self.profile.i_setpoint_a)

    def set_setpoint(self, i_setpoint_a: float) -> None:
        self.profile.i_setpoint_a = float(i_setpoint_a)
        self.i_req_target_a = float(i_setpoint_a)

    def set_ramp(self, ramp_a_per_s: float) -> None:
        self.profile.ramp_a_per_s = float(max(0.0, ramp_a_per_s))

    def _update_i_req_setpoint(self) -> None:
        ramp = float(max(0.0, self.profile.ramp_a_per_s))
        if ramp <= 0.0:
            self.i_req_a = float(self.i_req_target_a)
            return

        max_delta = ramp * float(self.dt_s)
        delta = float(self.i_req_target_a - self.i_req_a)
        if abs(delta) <= max_delta:
            self.i_req_a = float(self.i_req_target_a)
        else:
            self.i_req_a = float(self.i_req_a + np.sign(delta) * max_delta)

    def _choose_new_request_random(self, *, t_now: float) -> None:
        dur = float(self.rng_prof.uniform(self.segment_min_s, self.segment_max_s))
        self.next_switch_s = float(t_now + dur)

        sign = +1.0 if self.rng_prof.random() < 0.65 else -1.0
        if sign > 0:
            mag = float(self.rng_prof.uniform(0.10 * self.i_discharge_max_a, self.i_discharge_max_a))
        else:
            mag = float(self.rng_prof.uniform(0.10 * self.i_charge_max_a, self.i_charge_max_a))

        self.i_req_a = float(sign * mag)

    def reset(self) -> None:
        self.time_s = 0.0

        self.ecm.reset(soc=self.true_init_soc)
        self.thermal.reset(t_init_c=self.t_amb_c)
        self.ekf.reset(soc_init=self.ekf_init_soc)
        self.fault_det.reset()
        self.fsm.reset(BMSState.RUN)

        res0 = self.ecm.step(0.0, 0.0)
        self.last_res = res0

        if self.profile.mode == "random":
            self._choose_new_request_random(t_now=0.0)
        else:
            self.i_req_a = 0.0
            self.i_req_target_a = float(self.profile.i_setpoint_a)
            self.next_switch_s = float("inf")

        self._init_hist()
        self._log_sample(
            t_s=0.0,
            i_req=0.0,
            i_act=0.0,
            res=res0,
            t_rack=self.thermal.t_c,
            state=self.fsm.state,
            flags={"oc": False, "ov": False, "uv": False, "ot": False, "fire": False},
        )

    def _meas(self, x: float, sigma: float) -> float:
        if sigma <= 0.0:
            return float(x)
        return float(x + self.rng_meas.normal(0.0, sigma))

    def _log_sample(
        self,
        *,
        t_s: float,
        i_req: float,
        i_act: float,
        res: Dict[str, float],
        t_rack: float,
        state: BMSState,
        flags: Dict[str, bool],
    ) -> None:
        self.hist["time_s"].append(float(t_s))
        self.hist["i_req_a"].append(float(i_req))
        self.hist["i_act_a"].append(float(i_act))
        self.hist["soc_true"].append(float(res["soc"]))
        self.hist["soc_hat"].append(float(self.ekf.get_soc()))
        self.hist["v_cell_min_v"].append(float(res.get("v_cell_min", np.nan)))
        self.hist["v_cell_max_v"].append(float(res.get("v_cell_max", np.nan)))
        self.hist["t_rack_c"].append(float(t_rack))
        self.hist["state_code"].append(int(STATE_CODE_MAP[state]))
        for k in ("oc", "ov", "uv", "ot", "fire"):
            self.hist[k].append(1.0 if bool(flags.get(k, False)) else 0.0)

        self.hist["soc_cell_min"].append(float(res.get("soc_cell_min", res["soc"])))
        self.hist["soc_cell_mean"].append(float(res.get("soc_cell_mean", res["soc"])))
        self.hist["soc_cell_max"].append(float(res.get("soc_cell_max", res["soc"])))

    def step(self) -> Dict[str, float]:
        if self.profile.mode == "random":
            if self.time_s >= self.next_switch_s:
                self._choose_new_request_random(t_now=self.time_s)
        else:
            self._update_i_req_setpoint()

        if self.fsm.state in (BMSState.FAULT, BMSState.EMERGENCY_SHUTDOWN):
            i_act = 0.0
        else:
            if self.use_bms_limits:
                assert self.last_res is not None
                curr_limits = compute_current_limits(
                    soc_hat=self.ekf.get_soc(),
                    t_rack_c=self.thermal.t_c,
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

        t_rack = float(self.thermal.step(res["p_loss"], self.t_amb_c, self.dt_s))

        v_meas = self._meas(float(res["v_rack"]), self.sigma_v)
        i_meas = self._meas(float(i_act), self.sigma_i)
        _t_meas = self._meas(float(t_rack), self.sigma_t)

        self.ekf.update(v_meas, i_meas)
        soc_hat = float(self.ekf.get_soc())

        v_cell_min_used = float(res.get("v_cell_min", np.nan))
        v_cell_max_used = float(res.get("v_cell_max", np.nan))
        t_for_fault = float(t_rack)
        i_for_fault = float(i_meas)
        gas_alarm = False

        if self.inj.uv:
            v_cell_min_used = min(v_cell_min_used, self.uv_v_fault)
        if self.inj.ot:
            t_for_fault = max(t_for_fault, self.ot_temp_c)
        if self.inj.oc:
            i_for_fault = float(self.oc_i_fault_a)
        if self.inj.fire:
            gas_alarm = True

        flags = self.fault_det.step(
            v_cell_min=v_cell_min_used,
            v_cell_max=v_cell_max_used,
            t_rack_c=t_for_fault,
            i_rack_a=i_for_fault,
            gas_alarm=gas_alarm,
        )
        state = self.fsm.step(flags, enable=True)

        self.time_s = float(self.time_s + self.dt_s)

        res_for_log = dict(res)
        res_for_log["v_cell_min"] = v_cell_min_used
        res_for_log["v_cell_max"] = v_cell_max_used

        self._log_sample(
            t_s=self.time_s,
            i_req=self.i_req_a,
            i_act=i_act,
            res=res_for_log,
            t_rack=t_rack,
            state=state,
            flags=flags,
        )

        return {
            "t": self.time_s,
            "i_req": self.i_req_a,
            "i_act": i_act,
            "soc_true": float(res["soc"]),
            "soc_hat": soc_hat,
            "v_cell_min": v_cell_min_used,
            "v_cell_max": v_cell_max_used,
            "t_rack": t_rack,
            "state_code": int(STATE_CODE_MAP[state]),
            "oc": float(bool(flags.get("oc", False))),
            "ov": float(bool(flags.get("ov", False))),
            "uv": float(bool(flags.get("uv", False))),
            "ot": float(bool(flags.get("ot", False))),
            "fire": float(bool(flags.get("fire", False))),
        }

    def to_npz(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=float) for k, v in self.hist.items()}

    def save_npz(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **self.to_npz())
        return out_path


def render_scada_playground(
    params: Dict[str, Any],
    *,
    config_path: Optional[str] = None,
    config_path_resolved: Optional[str] = None,
) -> None:
    # Keep your original header text (so you immediately notice if this file loaded)
    st.header("Live SCADA Playground (Offline)")

    # Determine config path even if app does NOT pass it (fallback)
    if not config_path:
        config_path = str(st.session_state.get("active_config_path", "data/params_rack.yaml"))
    if not config_path_resolved:
        config_path_resolved = _resolve_path(config_path)

    dt_s = float(_get(params, "simulation", "dt_s", default=np.nan))
    st.caption(f"UI build: {UI_BUILD} | Active config: {config_path} | Resolved: {config_path_resolved} | dt={_fmt(dt_s, ' s', 3)}")

    # ---------------- explainers ----------------
    with st.expander("What is Live mode doing? ", expanded=True):
        st.markdown(
            """
**Live = step-by-step simulation**  
This page advances the rack model by **dt seconds** per step and updates the charts.

**Two input modes**
- **Random profile**: the app generates random current requests (`I_req`) in time segments.
- **Manual setpoint**: you choose `I_req` yourself and can change it while it runs.

**Important: what can change without Reset**
- `T_amb` (ambient temperature) is a **plant input** → can be changed live.
- `UV/OT/OC/FIRE toggles` are **fault-detector injections** → can be changed live.
- `true_init_soc / ekf_init_soc` are **initial conditions** → you must press **Reset** to apply.
            """
        )

    with st.expander("Operator guide (how to read this screen)", expanded=True):
        st.markdown(
            """
**Signal meanings**
- **I_req [A]**: Requested rack current profile (setpoint).
- **I_act [A]**: Actual rack current after BMS limiting and/or shutdown.
- **Sign convention**: **+ current = discharge**, **– current = charge**.
- **SoC true**: The plant (ECM) internal SoC.
- **SoC EKF**: The estimator’s SoC (EKF), based on voltage + current measurements.
- **V_cell min/max**: Minimum and maximum cell voltage in the rack (statistics across series cells).
- **State code**: BMS finite state machine state (0=OFF, 1=RUN, 2=FAULT, 3=EMERGENCY_SHUTDOWN).

**What is a “segment”?**
- In Live mode, the simulator generates **piecewise-constant current segments**.
  Each segment lasts a random duration and then a new current is requested.

**Fault injections (toggles)**
- UV / OT / OC / FIRE toggles affect **only the fault detector inputs**.
- This is intentional: you can validate detection + FSM actions without corrupting the plant physics.
- When a fault is latched, the FSM can force **I_act → 0 A** (shutdown behaviour).

**Live vs Reset**
- Ambient temperature and injection toggles apply **immediately** (next simulation step).
- “True/EKF initial SoC” are **initial conditions** and apply after **Reset**.
            """
        )

    # ---------------- Reference rack: Reference vs Calculated ----------------
    with st.expander("Reference rack model (Huawei LUNA-inspired) used in this simulation", expanded=False):
        sys_name = _get(params, "meta", "system_name", default="(unknown)")
        chem = _get(params, "chemistry", "name", default="(unknown)")
        cell_v = _get(params, "chemistry", "nominal_cell_voltage_v", default=np.nan)
        cell_ah = _get(params, "chemistry", "cell_capacity_ah", default=np.nan)

        s_pack = _get(params, "structure", "cells_in_series_per_pack", default=np.nan)
        p_pack = _get(params, "structure", "cells_in_parallel_per_pack", default=np.nan)
        packs_series = _get(params, "structure", "packs_in_series_per_rack", default=np.nan)
        racks_ess = _get(params, "structure", "racks_in_ess", default=np.nan)

        pack_v_ref = _get(params, "pack", "rated_voltage_v", default=np.nan)
        rack_v_ref = _get(params, "rack", "nominal_voltage_v", default=np.nan)
        pack_kwh_ref = _get(params, "pack", "nominal_energy_kwh", default=np.nan)
        rack_kwh_ref = _get(params, "rack", "nominal_energy_kwh", default=np.nan)

        cell_v_min = _get(params, "pack", "cell_voltage_min_v", default=_get(params, "limits", "v_cell_min_v", default=np.nan))
        cell_v_max = _get(params, "pack", "cell_voltage_max_v", default=np.nan)
        rack_v_min_ref = _get(params, "rack", "v_min_v", default=_get(params, "limits", "v_rack_min_v", default=np.nan))
        rack_v_max_ref = _get(params, "rack", "v_max_v", default=_get(params, "limits", "v_rack_max_v", default=np.nan))

        rated_bus_v = _get(params, "hardware", "rated_operating_voltage_v", default=np.nan)
        p_rack_kw = _get(params, "hardware", "rated_power_per_rack_kw", default=np.nan)
        i_rack_ref = _get(params, "hardware", "rated_current_per_rack_a", default=np.nan)

        # Calculated
        pack_v_calc = (float(s_pack) * float(cell_v)) if _isfinite(s_pack) and _isfinite(cell_v) else np.nan
        pack_ah_calc = (float(p_pack) * float(cell_ah)) if _isfinite(p_pack) and _isfinite(cell_ah) else np.nan
        pack_kwh_calc = (pack_v_calc * pack_ah_calc / 1000.0) if _isfinite(pack_v_calc) and _isfinite(pack_ah_calc) else np.nan

        rack_v_calc = (pack_v_calc * float(packs_series)) if _isfinite(pack_v_calc) and _isfinite(packs_series) else np.nan
        rack_kwh_calc = (pack_kwh_calc * float(packs_series)) if _isfinite(pack_kwh_calc) and _isfinite(packs_series) else np.nan

        rack_v_min_calc = (float(cell_v_min) * float(s_pack) * float(packs_series)) if _isfinite(cell_v_min) and _isfinite(s_pack) and _isfinite(packs_series) else np.nan
        rack_v_max_calc = (float(cell_v_max) * float(s_pack) * float(packs_series)) if _isfinite(cell_v_max) and _isfinite(s_pack) and _isfinite(packs_series) else np.nan

        i_rack_from_pv = (float(p_rack_kw) * 1000.0 / float(rated_bus_v)) if _isfinite(p_rack_kw) and _isfinite(rated_bus_v) and float(rated_bus_v) > 0 else np.nan

        st.markdown(
            """
This project uses a **Huawei LUNA2000-2.0MWH-1HX inspired rack** as a realistic reference.
The goal is not to copy proprietary internals, but to simulate a **credible, utility-scale rack topology**
for BMS logic validation and thesis-ready results.

Below, **Reference** values come from YAML, while **Calculated** values are derived from the cell + topology.
            """
        )

        df_ref = pd.DataFrame(
            [
                ["System name", sys_name, sys_name],
                ["Chemistry", chem, chem],
                ["Nominal cell voltage [V]", _fmt(cell_v, "", 3), _fmt(cell_v, "", 3)],
                ["Cell capacity [Ah]", _fmt(cell_ah, "", 1), _fmt(cell_ah, "", 1)],
                ["Pack topology", f"{s_pack}S{p_pack}P", f"{s_pack}S{p_pack}P"],
                ["Packs in series per rack", str(packs_series), str(packs_series)],
                ["Pack rated voltage [V]", _fmt(pack_v_ref, "", 1), _fmt(pack_v_calc, "", 1)],
                ["Pack nominal energy [kWh]", _fmt(pack_kwh_ref, "", 2), _fmt(pack_kwh_calc, "", 3)],
                ["Rack nominal voltage [V]", _fmt(rack_v_ref, "", 1), _fmt(rack_v_calc, "", 1)],
                ["Rack nominal energy [kWh]", _fmt(rack_kwh_ref, "", 1), _fmt(rack_kwh_calc, "", 3)],
                ["Rack V_min [V]", _fmt(rack_v_min_ref, "", 1), _fmt(rack_v_min_calc, "", 1)],
                ["Rack V_max [V]", _fmt(rack_v_max_ref, "", 1), _fmt(rack_v_max_calc, "", 1)],
                ["Rated rack current [A]", _fmt(i_rack_ref, "", 1), _fmt(i_rack_from_pv, "", 1)],
                ["Racks per ESS", str(racks_ess), str(racks_ess)],
            ],
            columns=["Item", "Reference", "Calculated"],
        )
        st.table(df_ref)

        st.caption("Optional: place an image at `assets/huawei_rack.png` to show it here.")
        here = Path(__file__).resolve().parent
        candidates = [
            here / "assets" / "huawei_rack.png",
            here / "assets" / "huawei_container.png",
            Path("assets/huawei_rack.png"),
            Path("assets/huawei_container.png"),
        ]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is not None:
            st.image(str(img_path), caption="Reference rack/container visual (local asset).", use_container_width=True)
        else:
            st.info("No local image found yet (assets/huawei_rack.png). This is optional.")

    # ---------------- settings ----------------
    with st.expander("Simulation settings", expanded=True):
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            t_amb = st.number_input("Ambient temperature T_amb [°C] (LIVE)", value=25.0, step=1.0)
            use_limits = st.checkbox("Apply BMS current limits", value=True)

        with c2:
            true_soc = st.number_input(
                "True initial SoC (plant) [0..1] (applies on Reset)",
                min_value=0.0, max_value=1.0, value=0.90, step=0.01
            )
            ekf_soc = st.number_input(
                "EKF initial SoC [0..1] (applies on Reset)",
                min_value=0.0, max_value=1.0, value=0.90, step=0.01
            )
            st.caption("Initial conditions require Reset.")

        with c3:
            mode_ui = st.radio("Input mode", ["Random profile", "Manual setpoint"], horizontal=True)
            profile_mode = "random" if mode_ui == "Random profile" else "setpoint"

            if profile_mode == "setpoint":
                i_set = st.number_input("Manual I_req setpoint [A] (LIVE)", value=0.0, step=10.0)
                ramp = st.number_input("Setpoint ramp [A/s] (0=instant)", min_value=0.0, value=0.0, step=10.0)
            else:
                i_set = 0.0
                ramp = 0.0
                st.caption("Manual setpoint inputs appear when you select Manual setpoint.")

        with c4:
            if profile_mode == "random":
                i_dis = st.number_input("Max discharge request [A]", value=320.0, step=10.0)
                i_chg = st.number_input("Max charge request [A] (abs)", value=160.0, step=10.0)
                seg_min = st.number_input("Random segment min length [s]", value=30.0, step=5.0)
                seg_max = st.number_input("Random segment max length [s]", value=180.0, step=10.0)
                seed = st.number_input("Profile seed", value=1, step=1)
                st.caption("Seed affects future segments. For a clean start, press Reset.")
            else:
                i_dis, i_chg, seg_min, seg_max, seed = 320.0, 160.0, 30.0, 180.0, 1

    with st.expander("Fault injections (toggles)", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            inj_uv = st.checkbox("UV injection", value=False)
            uv_v = st.number_input("Forced V_cell_min during UV [V]", value=2.0, step=0.1)
        with d2:
            inj_ot = st.checkbox("OT injection", value=False)
            ot_c = st.number_input("Forced rack temperature during OT [°C]", value=100.0, step=5.0)
        with d3:
            inj_oc = st.checkbox("OC injection", value=False)
            oc_a = st.number_input("Current seen by detector during OC [A]", value=500.0, step=50.0)
        with d4:
            inj_fire = st.checkbox("FIRE (gas alarm)", value=False)

    # ---------------- live controls ----------------
    cA, cB, cC, cD = st.columns([1.2, 1.2, 1.2, 2.4])
    with cA:
        auto_run = st.checkbox("Auto-run (Play)")
    with cB:
        steps_per_tick = st.number_input("Steps per refresh", min_value=1, max_value=500, value=5, step=1)
    with cC:
        refresh_ms = st.number_input("Refresh interval [ms]", min_value=50, max_value=5000, value=250, step=50)
    with cD:
        window_s = st.number_input("Plot window [s]", min_value=30.0, max_value=3600.0, value=900.0, step=30.0)

    # ---------------- create sim ----------------
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
            seed_profile=int(seed),
            profile=LiveProfile(mode=profile_mode, i_setpoint_a=float(i_set), ramp_a_per_s=float(ramp)),
            uv_v_fault=float(uv_v),
            ot_temp_c=float(ot_c),
            oc_i_fault_a=float(oc_a),
        )

    sim: LiveBmsSim = st.session_state.scada_sim

    # ---------------- live updates ----------------
    sim.use_bms_limits = bool(use_limits)
    sim.t_amb_c = float(t_amb)

    sim.set_profile_mode(profile_mode)
    sim.set_setpoint(float(i_set))
    sim.set_ramp(float(ramp))

    sim.i_discharge_max_a = float(i_dis)
    sim.i_charge_max_a = float(i_chg)
    sim.segment_min_s = float(seg_min)
    sim.segment_max_s = float(seg_max)
    if int(seed) != int(sim.seed_profile):
        sim.seed_profile = int(seed)
        sim.rng_prof = np.random.default_rng(int(seed))

    sim.inj.uv = bool(inj_uv)
    sim.inj.ot = bool(inj_ot)
    sim.inj.oc = bool(inj_oc)
    sim.inj.fire = bool(inj_fire)
    sim.uv_v_fault = float(uv_v)
    sim.ot_temp_c = float(ot_c)
    sim.oc_i_fault_a = float(oc_a)

    # ---------------- buttons ----------------
    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 2])
    with b1:
        if st.button("Reset"):
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
                seed_profile=int(seed),
                profile=LiveProfile(mode=profile_mode, i_setpoint_a=float(i_set), ramp_a_per_s=float(ramp)),
                uv_v_fault=float(uv_v),
                ot_temp_c=float(ot_c),
                oc_i_fault_a=float(oc_a),
            )
            st.rerun()
    with b2:
        if st.button("+1 step"):
            sim.step()
    with b3:
        if st.button("+50"):
            for _ in range(50):
                sim.step()
    with b4:
        if st.button("+100"):
            for _ in range(100):
                sim.step()
    with b5:
        out_path = st.text_input("Save path (.npz)", value="results/live/live_scada_run_01.npz")
        if st.button("Save NPZ"):
            p = sim.save_npz(out_path)
            st.success(f"Saved to: {p}")

    # ---------------- plots ----------------
    arr = sim.to_npz()
    t = arr["time_s"]
    if t.size < 2:
        st.info("Simulation is ready. Click +1 step (or enable Auto-run).")
        return

    t_end = float(t[-1])
    t_start = max(0.0, t_end - float(window_s))
    mask = t >= t_start

    def _w(key: str) -> np.ndarray:
        return arr[key][mask]

    t_w = t[mask]
    i_req = _w("i_req_a")
    i_act = _w("i_act_a")
    soc_true = _w("soc_true")
    soc_hat = _w("soc_hat")
    soc_err = soc_true - soc_hat
    v_min = _w("v_cell_min_v")
    v_max = _w("v_cell_max_v")
    soc_sp = _w("soc_cell_max") - _w("soc_cell_min")
    v_sp = v_max - v_min
    state = _w("state_code")
    oc = _w("oc")
    ov = _w("ov")
    uv = _w("uv")
    ot = _w("ot")
    fire = _w("fire")

    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    ax_i, ax_soc, ax_err, ax_v, ax_sp, ax_state = axes
    ax_sp2 = ax_sp.twinx()

    ax_i.plot(t_w, i_req, label="I_req [A]")
    ax_i.plot(t_w, i_act, "--", label="I_act [A]")
    ax_i.set_ylabel("Current [A]")
    ax_i.grid(True)
    ax_i.legend(loc="upper right")

    ax_soc.plot(t_w, soc_true, label="SoC true")
    ax_soc.plot(t_w, soc_hat, "--", label="SoC EKF")
    ax_soc.set_ylabel("SoC [-]")
    ax_soc.grid(True)
    ax_soc.legend(loc="upper right")

    ax_err.plot(t_w, soc_err, label="SoC error (true - EKF)")
    ax_err.axhline(0.0, linestyle="--", linewidth=0.8)
    ax_err.set_ylabel("SoC err [-]")
    ax_err.grid(True)
    ax_err.legend(loc="upper right")

    ax_v.plot(t_w, v_min, label="V_cell min")
    ax_v.plot(t_w, v_max, label="V_cell max")
    ax_v.set_ylabel("Cell V [V]")
    ax_v.grid(True)
    ax_v.legend(loc="upper right")

    ax_sp.plot(t_w, soc_sp, label="SoC spread (max-min)")
    ax_sp2.plot(t_w, v_sp, "--", label="V spread (max-min)")
    ax_sp.set_ylabel("SoC spread [-]")
    ax_sp2.set_ylabel("V spread [V]")
    ax_sp.grid(True)
    h1, l1 = ax_sp.get_legend_handles_labels()
    h2, l2 = ax_sp2.get_legend_handles_labels()
    ax_sp.legend(h1 + h2, l1 + l2, loc="upper right")

    ax_state.step(t_w, state, where="post", label="BMS state code")
    ax_state.step(t_w, oc, where="post", label="OC")
    ax_state.step(t_w, ov, where="post", label="OV")
    ax_state.step(t_w, uv, where="post", label="UV")
    ax_state.step(t_w, ot, where="post", label="OT")
    ax_state.step(t_w, fire, where="post", label="FIRE")
    ax_state.set_ylabel("State / faults")
    ax_state.set_xlabel("Time [s]")
    ax_state.grid(True)
    ax_state.legend(loc="upper right")
    ax_state.set_ylim(-0.2, 3.8)

    fig.suptitle("Live SCADA (Offline) – Rolling window", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"t={t_end:.1f}s | state={STATE_NAME_MAP.get(int(arr['state_code'][-1]), str(int(arr['state_code'][-1])))} | "
        f"mode={'RANDOM' if sim.profile.mode=='random' else 'SETPOINT'} | "
        f"T_amb={sim.t_amb_c:.1f}°C | "
        f"inj: UV={'ON' if sim.inj.uv else 'off'}, OT={'ON' if sim.inj.ot else 'off'}, "
        f"OC={'ON' if sim.inj.oc else 'off'}, FIRE={'ON' if sim.inj.fire else 'off'}"
    )

    if auto_run:
        for _ in range(int(steps_per_tick)):
            sim.step()
        time.sleep(float(refresh_ms) / 1000.0)
        st.rerun()
