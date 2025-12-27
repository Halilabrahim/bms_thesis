"""live_scada_matplotlib.py

Offline, real-time "SCADA-like" dashboard for a BESS rack simulation.

Run from your project root (where `src/` is importable):
    python live_scada_matplotlib.py

Optional args:
    python live_scada_matplotlib.py --window-s 900 --max-time-s 3600 --out results/live/live_run_01.npz

Keyboard controls (focus the plot window):
    u : toggle UV injection (forces v_cell_min down to --uv-v)
    f : toggle FIRE injection (gas alarm)
    t : toggle OT injection (forces rack temp up to --ot-c)
    c : toggle OC injection (forces current seen by detector to --oc-a)
    r : hard reset (restart simulation from t=0)
    s : save snapshot (.npz) to --out
    q : quit (saves then exits)

Notes
- UV/OT/OC injections affect ONLY the *fault detector inputs* (like your manual runner)
  so you can test detection + FSM behavior without breaking the plant model.
- The BMS action is visible as I_act tracking / clipping / going to zero on FAULT/EMERGENCY.

IMPORTANT FIX (your blank screen issue)
- Matplotlib FuncAnimation must be kept in a variable (otherwise it can be garbage collected).
- Also we draw an initial frame immediately after seeding so you see data at t=0.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Deque, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Ensure project root + src are importable even if run from another working dir
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.config import load_params
from src.sim_runner import build_models
from src.fsm import BMSState
from src.bms_logic import compute_current_limits


STATE_CODE_MAP = {
    BMSState.OFF: 0,
    BMSState.RUN: 1,
    BMSState.FAULT: 2,
    BMSState.EMERGENCY_SHUTDOWN: 3,
}


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class LiveInjections:
    uv: bool = False
    fire: bool = False
    ot: bool = False
    oc: bool = False


class LiveBmsSim:
    """Holds the models and advances the simulation one step at a time."""

    def __init__(
        self,
        params: Dict[str, Any],
        *,
        t_amb_c: float = 25.0,
        true_init_soc: float = 0.90,
        ekf_init_soc: float = 0.90,
        use_bms_limits: bool = True,
        i_discharge_max_a: float = 320.0,
        i_charge_max_a: float = 160.0,
        segment_min_s: float = 30.0,
        segment_max_s: float = 180.0,
        seed_profile: int = 1,
        # injection magnitudes
        uv_v_fault: float = 2.0,
        ot_temp_c: float = 100.0,
        oc_i_fault_a: float = 500.0,
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

        # noise
        sensor_cfg = params.get("sensors", {})
        self.sigma_v = float(sensor_cfg.get("voltage_noise_std_v", 0.0))
        self.sigma_i = float(sensor_cfg.get("current_noise_std_a", 0.0))
        self.sigma_t = float(sensor_cfg.get("temp_noise_std_c", 0.0))
        self.rng_meas = np.random.default_rng(int(sensor_cfg.get("seed", 0)))

        # profile
        self.use_bms_limits = bool(use_bms_limits)
        self.i_discharge_max_a = float(i_discharge_max_a)
        self.i_charge_max_a = float(i_charge_max_a)
        self.segment_min_s = float(segment_min_s)
        self.segment_max_s = float(segment_max_s)
        self.rng_prof = np.random.default_rng(int(seed_profile))

        # injections
        self.inj = LiveInjections()
        self.uv_v_fault = float(uv_v_fault)
        self.ot_temp_c = float(ot_temp_c)
        self.oc_i_fault_a = float(oc_i_fault_a)

        # stateful
        self.t_amb_c = float(t_amb_c)
        self.true_init_soc = _clip01(true_init_soc)
        self.ekf_init_soc = _clip01(ekf_init_soc)

        self.time_s = 0.0
        self.i_req_a = 0.0
        self.next_switch_s = 0.0

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
            "soc_hat_pred",
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

    def reset(self) -> None:
        self.time_s = 0.0

        self.ecm.reset(soc=self.true_init_soc)
        self.thermal.reset(t_init_c=self.t_amb_c)
        self.ekf.reset(soc_init=self.ekf_init_soc)
        self.fault_det.reset()
        self.fsm.reset(BMSState.RUN)

        res0 = self.ecm.step(0.0, 0.0)
        self._choose_new_request(t_now=0.0)

        self._init_hist()
        self._log(
            t_s=0.0,
            i_req=0.0,
            i_act=0.0,
            soc_true=res0["soc"],
            soc_hat=self.ekf.get_soc(),
            soc_hat_pred=self.ekf.get_soc(),
            v_cell_min=res0.get("v_cell_min", np.nan),
            v_cell_max=res0.get("v_cell_max", np.nan),
            t_rack=self.thermal.t_c,
            state=self.fsm.state,
            flags={"oc": False, "ov": False, "uv": False, "ot": False, "fire": False},
            soc_cell_min=res0.get("soc_cell_min", res0["soc"]),
            soc_cell_mean=res0.get("soc_cell_mean", res0["soc"]),
            soc_cell_max=res0.get("soc_cell_max", res0["soc"]),
        )

    def _choose_new_request(self, *, t_now: float) -> None:
        dur = float(self.rng_prof.uniform(self.segment_min_s, self.segment_max_s))
        self.next_switch_s = float(t_now + dur)

        sign = +1.0 if self.rng_prof.random() < 0.65 else -1.0
        if sign > 0:
            mag = float(self.rng_prof.uniform(0.10 * self.i_discharge_max_a, self.i_discharge_max_a))
        else:
            mag = float(self.rng_prof.uniform(0.10 * self.i_charge_max_a, self.i_charge_max_a))
        self.i_req_a = float(sign * mag)

    def _meas(self, x: float, sigma: float) -> float:
        if sigma <= 0.0:
            return float(x)
        return float(x + self.rng_meas.normal(0.0, sigma))

    def _log(
        self,
        *,
        t_s: float,
        i_req: float,
        i_act: float,
        soc_true: float,
        soc_hat: float,
        soc_hat_pred: float,
        v_cell_min: float,
        v_cell_max: float,
        t_rack: float,
        state: BMSState,
        flags: Dict[str, bool],
        soc_cell_min: float,
        soc_cell_mean: float,
        soc_cell_max: float,
    ) -> None:
        self.hist["time_s"].append(float(t_s))
        self.hist["i_req_a"].append(float(i_req))
        self.hist["i_act_a"].append(float(i_act))
        self.hist["soc_true"].append(float(soc_true))
        self.hist["soc_hat"].append(float(soc_hat))
        self.hist["soc_hat_pred"].append(float(soc_hat_pred))
        self.hist["v_cell_min_v"].append(float(v_cell_min))
        self.hist["v_cell_max_v"].append(float(v_cell_max))
        self.hist["t_rack_c"].append(float(t_rack))
        self.hist["state_code"].append(int(STATE_CODE_MAP[state]))
        for k in ("oc", "ov", "uv", "ot", "fire"):
            self.hist[k].append(1.0 if bool(flags.get(k, False)) else 0.0)
        self.hist["soc_cell_min"].append(float(soc_cell_min))
        self.hist["soc_cell_mean"].append(float(soc_cell_mean))
        self.hist["soc_cell_max"].append(float(soc_cell_max))

    def step(self) -> Dict[str, float]:
        if self.time_s >= self.next_switch_s:
            self._choose_new_request(t_now=self.time_s)

        if self.fsm.state in (BMSState.FAULT, BMSState.EMERGENCY_SHUTDOWN):
            i_act = 0.0
        else:
            if self.use_bms_limits:
                peek = self.ecm.step(0.0, 0.0)
                curr_limits = compute_current_limits(
                    soc_hat=self.ekf.get_soc(),
                    t_rack_c=self.thermal.t_c,
                    v_cell_min=float(peek.get("v_cell_min", np.nan)),
                    v_cell_max=float(peek.get("v_cell_max", np.nan)),
                    params=self.bms_params,
                )
                if self.i_req_a >= 0.0:
                    i_act = min(self.i_req_a, float(curr_limits["i_discharge_max_allowed"]))
                else:
                    i_act = max(self.i_req_a, -float(curr_limits["i_charge_max_allowed"]))
            else:
                i_act = float(self.i_req_a)

        self.ekf.predict(i_act, self.dt_s)
        soc_hat_pred = float(self.ekf.get_soc())

        res = self.ecm.step(i_act, self.dt_s)
        t_rack = float(self.thermal.step(res["p_loss"], self.t_amb_c, self.dt_s))

        v_meas = self._meas(res["v_rack"], self.sigma_v)
        i_meas = self._meas(i_act, self.sigma_i)
        _t_meas = self._meas(t_rack, self.sigma_t)

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

        self._log(
            t_s=self.time_s,
            i_req=self.i_req_a,
            i_act=i_act,
            soc_true=float(res["soc"]),
            soc_hat=soc_hat,
            soc_hat_pred=soc_hat_pred,
            v_cell_min=v_cell_min_used,
            v_cell_max=v_cell_max_used,
            t_rack=t_rack,
            state=state,
            flags=flags,
            soc_cell_min=float(res.get("soc_cell_min", res["soc"])),
            soc_cell_mean=float(res.get("soc_cell_mean", res["soc"])),
            soc_cell_max=float(res.get("soc_cell_max", res["soc"])),
        )

        return {
            "t": self.time_s,
            "i_req": self.i_req_a,
            "i_act": i_act,
            "soc_true": float(res["soc"]),
            "soc_hat": soc_hat,
            "soc_hat_pred": soc_hat_pred,
            "v_cell_min": v_cell_min_used,
            "v_cell_max": v_cell_max_used,
            "t_rack": t_rack,
            "state_code": int(STATE_CODE_MAP[state]),
            "oc": float(bool(flags.get("oc", False))),
            "ov": float(bool(flags.get("ov", False))),
            "uv": float(bool(flags.get("uv", False))),
            "ot": float(bool(flags.get("ot", False))),
            "fire": float(bool(flags.get("fire", False))),
            "soc_cell_min": float(res.get("soc_cell_min", res["soc"])),
            "soc_cell_mean": float(res.get("soc_cell_mean", res["soc"])),
            "soc_cell_max": float(res.get("soc_cell_max", res["soc"])),
        }

    def save(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrs = {k: np.asarray(v, dtype=float) for k, v in self.hist.items()}
        np.savez(out_path, **arrs)
        return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-s", type=float, default=900.0, help="Plot window length [s]")
    ap.add_argument("--max-time-s", type=float, default=0.0, help="Stop after this sim time [s] (0 = unlimited)")
    ap.add_argument("--out", type=str, default="results/live/live_run_01.npz", help="Output .npz path")

    ap.add_argument("--t-amb", type=float, default=25.0)
    ap.add_argument("--true-soc", type=float, default=0.90)
    ap.add_argument("--ekf-soc", type=float, default=0.90)
    ap.add_argument("--no-bms-limits", action="store_true")

    ap.add_argument("--i-dis-max", type=float, default=320.0)
    ap.add_argument("--i-chg-max", type=float, default=160.0)
    ap.add_argument("--seg-min", type=float, default=30.0)
    ap.add_argument("--seg-max", type=float, default=180.0)
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--uv-v", type=float, default=2.0)
    ap.add_argument("--ot-c", type=float, default=100.0)
    ap.add_argument("--oc-a", type=float, default=500.0)

    args = ap.parse_args()
    params = load_params()

    sim = LiveBmsSim(
        params,
        t_amb_c=args.t_amb,
        true_init_soc=args.true_soc,
        ekf_init_soc=args.ekf_soc,
        use_bms_limits=not args.no_bms_limits,
        i_discharge_max_a=args.i_dis_max,
        i_charge_max_a=args.i_chg_max,
        segment_min_s=args.seg_min,
        segment_max_s=args.seg_max,
        seed_profile=args.seed,
        uv_v_fault=args.uv_v,
        ot_temp_c=args.ot_c,
        oc_i_fault_a=args.oc_a,
    )

    dt = sim.dt_s
    maxlen = max(10, int(args.window_s / dt) + 2)

    win: Dict[str, Deque[float]] = {
        "t": deque(maxlen=maxlen),
        "i_req": deque(maxlen=maxlen),
        "i_act": deque(maxlen=maxlen),
        "soc_true": deque(maxlen=maxlen),
        "soc_hat": deque(maxlen=maxlen),
        "soc_err": deque(maxlen=maxlen),
        "v_min": deque(maxlen=maxlen),
        "v_max": deque(maxlen=maxlen),
        "soc_spread": deque(maxlen=maxlen),
        "v_spread": deque(maxlen=maxlen),
        "t_rack": deque(maxlen=maxlen),
        "state": deque(maxlen=maxlen),
        "oc": deque(maxlen=maxlen),
        "ov": deque(maxlen=maxlen),
        "uv": deque(maxlen=maxlen),
        "ot": deque(maxlen=maxlen),
        "fire": deque(maxlen=maxlen),
    }

    def _seed_from_hist() -> None:
        t0 = sim.hist["time_s"][-1]
        i_req0 = sim.hist["i_req_a"][-1]
        i_act0 = sim.hist["i_act_a"][-1]
        soc_t0 = sim.hist["soc_true"][-1]
        soc_h0 = sim.hist["soc_hat"][-1]
        vmin0 = sim.hist["v_cell_min_v"][-1]
        vmax0 = sim.hist["v_cell_max_v"][-1]
        tr0 = sim.hist["t_rack_c"][-1]
        st0 = sim.hist["state_code"][-1]
        oc0 = sim.hist["oc"][-1]
        ov0 = sim.hist["ov"][-1]
        uv0 = sim.hist["uv"][-1]
        ot0 = sim.hist["ot"][-1]
        fi0 = sim.hist["fire"][-1]

        win["t"].append(t0)
        win["i_req"].append(i_req0)
        win["i_act"].append(i_act0)
        win["soc_true"].append(soc_t0)
        win["soc_hat"].append(soc_h0)
        win["soc_err"].append(soc_t0 - soc_h0)
        win["v_min"].append(vmin0)
        win["v_max"].append(vmax0)
        win["soc_spread"].append(sim.hist["soc_cell_max"][-1] - sim.hist["soc_cell_min"][-1])
        win["v_spread"].append(vmax0 - vmin0)
        win["t_rack"].append(tr0)
        win["state"].append(st0)
        win["oc"].append(oc0)
        win["ov"].append(ov0)
        win["uv"].append(uv0)
        win["ot"].append(ot0)
        win["fire"].append(fi0)

    _seed_from_hist()

    fig, axes = plt.subplots(6, 1, figsize=(14, 9), sharex=True)
    ax_i, ax_soc, ax_err, ax_v, ax_sp, ax_state = axes
    ax_sp2 = ax_sp.twinx()

    (ln_i_req,) = ax_i.plot([], [], label="I_req [A]")
    (ln_i_act,) = ax_i.plot([], [], "--", label="I_act [A]")
    ax_i.set_ylabel("Current [A]")
    ax_i.grid(True)
    ax_i.legend(loc="upper right")
    ax_i.set_ylim(-1.25 * args.i_chg_max, 1.25 * args.i_dis_max)

    (ln_soc_true,) = ax_soc.plot([], [], label="SoC true")
    (ln_soc_hat,) = ax_soc.plot([], [], "--", label="SoC EKF")
    ax_soc.set_ylabel("SoC [-]")
    ax_soc.grid(True)
    ax_soc.legend(loc="upper right")
    ax_soc.set_ylim(-0.02, 1.02)

    (ln_soc_err,) = ax_err.plot([], [], label="SoC error (true - EKF)")
    ax_err.axhline(0.0, linestyle="--", linewidth=0.8)
    ax_err.set_ylabel("SoC err [-]")
    ax_err.grid(True)
    ax_err.legend(loc="upper right")
    ax_err.set_ylim(-0.05, 0.05)

    (ln_v_min,) = ax_v.plot([], [], label="V_cell min")
    (ln_v_max,) = ax_v.plot([], [], label="V_cell max")
    ax_v.set_ylabel("Cell V [V]")
    ax_v.grid(True)
    ax_v.legend(loc="upper right")
    ax_v.set_ylim(1.8, 3.8)

    (ln_soc_sp,) = ax_sp.plot([], [], label="SoC spread (max-min)")
    (ln_v_sp,) = ax_sp2.plot([], [], "--", label="V spread (max-min)")
    ax_sp.set_ylabel("SoC spread [-]")
    ax_sp2.set_ylabel("V spread [V]")
    ax_sp.grid(True)
    h1, l1 = ax_sp.get_legend_handles_labels()
    h2, l2 = ax_sp2.get_legend_handles_labels()
    ax_sp.legend(h1 + h2, l1 + l2, loc="upper right")
    ax_sp.set_ylim(0.0, 0.10)
    ax_sp2.set_ylim(0.0, 1.5)

    (ln_state,) = ax_state.plot([], [], label="BMS state code")
    (ln_oc,) = ax_state.plot([], [], label="OC")
    (ln_ov,) = ax_state.plot([], [], label="OV")
    (ln_uv,) = ax_state.plot([], [], label="UV")
    (ln_ot,) = ax_state.plot([], [], label="OT")
    (ln_fire,) = ax_state.plot([], [], label="FIRE")
    ax_state.set_ylabel("State / faults")
    ax_state.set_xlabel("Time [s]")
    ax_state.grid(True)
    ax_state.legend(loc="upper right")
    ax_state.set_ylim(-0.2, 3.8)

    info = ax_i.text(
        0.01,
        0.95,
        "",
        transform=ax_i.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    def _refresh_lines() -> None:
        x = np.asarray(win["t"], dtype=float)

        ln_i_req.set_data(x, win["i_req"])
        ln_i_act.set_data(x, win["i_act"])

        ln_soc_true.set_data(x, win["soc_true"])
        ln_soc_hat.set_data(x, win["soc_hat"])
        ln_soc_err.set_data(x, win["soc_err"])

        ln_v_min.set_data(x, win["v_min"])
        ln_v_max.set_data(x, win["v_max"])

        ln_soc_sp.set_data(x, win["soc_spread"])
        ln_v_sp.set_data(x, win["v_spread"])

        ln_state.set_data(x, win["state"])
        ln_oc.set_data(x, win["oc"])
        ln_ov.set_data(x, win["ov"])
        ln_uv.set_data(x, win["uv"])
        ln_ot.set_data(x, win["ot"])
        ln_fire.set_data(x, win["fire"])

        if x.size >= 1:
            x_max = float(x[-1])
            x_min = max(0.0, x_max - float(args.window_s))
            for ax in axes:
                ax.set_xlim(x_min, max(x_min + 1e-6, x_max))

        if len(win["soc_err"]) > 5:
            m = float(np.max(np.abs(np.asarray(win["soc_err"], dtype=float))))
            m = max(m, 1e-4)
            m = min(max(m * 1.4, 0.002), 0.20)
            ax_err.set_ylim(-m, m)

    def _save_and_close(reason: str) -> None:
        out = sim.save(args.out)
        print(f"Saved: {out} ({reason})")
        plt.close(fig)

    def _update(_frame: int):
        sample = sim.step()

        win["t"].append(sample["t"])
        win["i_req"].append(sample["i_req"])
        win["i_act"].append(sample["i_act"])
        win["soc_true"].append(sample["soc_true"])
        win["soc_hat"].append(sample["soc_hat"])
        win["soc_err"].append(sample["soc_true"] - sample["soc_hat"])
        win["v_min"].append(sample["v_cell_min"])
        win["v_max"].append(sample["v_cell_max"])
        win["soc_spread"].append(sample["soc_cell_max"] - sample["soc_cell_min"])
        win["v_spread"].append(sample["v_cell_max"] - sample["v_cell_min"])
        win["t_rack"].append(sample["t_rack"])
        win["state"].append(sample["state_code"])
        win["oc"].append(sample["oc"])
        win["ov"].append(sample["ov"])
        win["uv"].append(sample["uv"])
        win["ot"].append(sample["ot"])
        win["fire"].append(sample["fire"])

        _refresh_lines()

        info.set_text(
            " ".join(
                [
                    f"t={sample['t']:.1f}s",
                    f"state={int(sample['state_code'])}",
                    f"inj:[UV={'ON' if sim.inj.uv else 'off'}",
                    f"OT={'ON' if sim.inj.ot else 'off'}",
                    f"OC={'ON' if sim.inj.oc else 'off'}",
                    f"FIRE={'ON' if sim.inj.fire else 'off'}]",
                ]
            )
        )

        if args.max_time_s and sample["t"] >= float(args.max_time_s):
            _save_and_close("Reached --max-time-s")

        return (
            ln_i_req,
            ln_i_act,
            ln_soc_true,
            ln_soc_hat,
            ln_soc_err,
            ln_v_min,
            ln_v_max,
            ln_soc_sp,
            ln_v_sp,
            ln_state,
            ln_oc,
            ln_ov,
            ln_uv,
            ln_ot,
            ln_fire,
            info,
        )

    def _on_key(event) -> None:
        k = (event.key or "").lower()
        if k == "u":
            sim.inj.uv = not sim.inj.uv
        elif k == "f":
            sim.inj.fire = not sim.inj.fire
        elif k == "t":
            sim.inj.ot = not sim.inj.ot
        elif k == "c":
            sim.inj.oc = not sim.inj.oc
        elif k == "r":
            sim.reset()
            for dq in win.values():
                dq.clear()
            _seed_from_hist()
            _refresh_lines()
            fig.canvas.draw_idle()
        elif k == "s":
            out = sim.save(args.out)
            print(f"Saved snapshot: {out}")
        elif k == "q":
            _save_and_close("Quit (q)")

    def _on_close(_event) -> None:
        try:
            out = sim.save(args.out)
            print(f"Saved on window close: {out}")
        except Exception as e:
            print(f"Could not save on close: {e}")

    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("close_event", _on_close)

    fig.suptitle("Live BMS SCADA (offline)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    # Draw initial data immediately (no empty dashboard).
    _refresh_lines()
    info.set_text(
        " ".join(
            [
                f"t={win['t'][-1]:.1f}s",
                f"state={int(win['state'][-1])}",
                f"inj:[UV={'ON' if sim.inj.uv else 'off'}",
                f"OT={'ON' if sim.inj.ot else 'off'}",
                f"OC={'ON' if sim.inj.oc else 'off'}",
                f"FIRE={'ON' if sim.inj.fire else 'off'}]",
            ]
        )
    )
    fig.canvas.draw_idle()

    interval_ms = max(20, int(1000.0 * sim.dt_s))

    # CRITICAL: keep a reference to the animation, otherwise it may stop (blank screen).
    ani = FuncAnimation(fig, _update, interval=interval_ms, blit=False, cache_frame_data=False)
    fig._ani = ani  # pin to the figure to avoid GC in some environments

    plt.show()


if __name__ == "__main__":
    main()
