# pages/1_Offline_Scenarios.py
from __future__ import annotations

import io
import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from src.scenarios import load_scenarios, Scenario
from src.sim_runner import run_scenario
from src.config import load_params, PROFILES
from src.metrics import (
    compute_safety_metrics,
    compute_operational_metrics,
    compute_estimation_metrics,
)
from run_manual_bms_scenario import Segment, FaultConfig, run_manual_profile


# ---------------- Cache / helpers ----------------
@st.cache_data
def get_params(cfg_path: str) -> Dict[str, Any]:
    return load_params(cfg_path)


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "None"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _none_to_nan(x: Optional[float]) -> float:
    return np.nan if x is None else float(x)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive dict update.
    Modifies dst in-place and returns it.
    """
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k, None), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _build_piecewise_profile(t: np.ndarray, segments: List[Segment], field: str) -> np.ndarray:
    """
    Reconstruct a piecewise-constant profile for a given segment field (e.g., t_amb_c)
    aligned with time vector t (seconds).
    """
    out = np.zeros_like(t, dtype=float)
    acc = 0.0
    for seg in segments:
        dur = float(seg.duration_s)
        t0, t1 = acc, acc + dur
        acc = t1
        mask = (t >= t0) & (t < t1)
        out[mask] = float(getattr(seg, field))
    if len(segments) > 0:
        out[t >= acc] = float(getattr(segments[-1], field))
    return out


def _get_q_nom_ah(params: Dict[str, Any]) -> float:
    # Prefer electrical_model.q_nom_ah, fallback to chemistry.cell_capacity_ah
    em = params.get("electrical_model", {}) or {}
    ch = params.get("chemistry", {}) or {}
    q = em.get("q_nom_ah", None)
    if q is None:
        q = ch.get("cell_capacity_ah", 0.0)
    try:
        return float(q)
    except Exception:
        return 0.0


def _compute_soc_cc(
    *,
    t: np.ndarray,
    i_act: np.ndarray,
    dt_s: float,
    q_nom_ah: float,
    soc0: float,
) -> np.ndarray:
    """
    CC-only SoC reference starting from soc0 (EKF init), using applied current i_act.
    Sign convention in this project:
      +I => discharge => SoC decreases
      -I => charge    => SoC increases
    """
    n = int(len(t))
    soc = np.zeros(n, dtype=float)
    soc[0] = float(soc0)
    denom = float(q_nom_ah) * 3600.0
    if denom <= 0:
        return soc

    for k in range(1, n):
        ds = -float(i_act[k - 1]) * float(dt_s) / denom
        soc[k] = float(np.clip(soc[k - 1] + ds, 0.0, 1.0))
    return soc


def _attach_demo_series_if_possible(
    result: Dict[str, Any],
    *,
    params: Dict[str, Any],
    dt_s: float,
    ekf_init_soc: Optional[float],
) -> None:
    """
    Adds helper series for plotting/reporting without touching core sim code.
    - soc_cc: coulomb-counting only SoC reference starting from EKF init SoC.
    """
    try:
        if ekf_init_soc is None:
            return
        t = np.asarray(result.get("time_s", []), dtype=float)
        if len(t) == 0:
            return
        i_act = np.asarray(result.get("i_act_a", result.get("i_req_a", [])), dtype=float)
        if len(i_act) != len(t):
            return

        q_nom_ah = _get_q_nom_ah(params)
        if q_nom_ah <= 0:
            return

        soc_cc = _compute_soc_cc(
            t=t,
            i_act=i_act,
            dt_s=float(dt_s),
            q_nom_ah=float(q_nom_ah),
            soc0=float(ekf_init_soc),
        )
        # store as list for JSON-ish compatibility
        result["soc_cc"] = soc_cc.tolist()
        # store meta scalars too (helpful for report)
        result["meta_ekf_init_soc"] = [float(ekf_init_soc)]
        result["meta_q_nom_ah_used"] = [float(q_nom_ah)]
    except Exception:
        # keep UI robust; no hard fail
        return


def make_manual_npz_bytes(
    result: dict,
    *,
    segments: List[Segment],
    fault_inj: FaultConfig,
    use_limits: bool,
    true_init_soc: float,
    ekf_init_soc: float,
    dt_s: float,
    ekf_overrides: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Create an in-memory .npz for the manual run.
    Includes raw simulation arrays + meta so you can reproduce the run conditions later.
    """
    export = {k: np.asarray(v) for k, v in result.items()}

    # ---- Meta ----
    export["meta_dt_s"] = np.asarray([float(dt_s)], dtype=float)
    export["meta_true_init_soc"] = np.asarray([float(true_init_soc)], dtype=float)
    export["meta_ekf_init_soc"] = np.asarray([float(ekf_init_soc)], dtype=float)
    export["meta_use_bms_limits"] = np.asarray([1 if use_limits else 0], dtype=int)

    if ekf_overrides:
        # store as string to avoid nested arrays complexity
        export["meta_ekf_overrides_jsonish"] = np.asarray([str(ekf_overrides)], dtype=object)

    # ---- Segment definition ----
    export["segments_duration_s"] = np.asarray([float(s.duration_s) for s in segments], dtype=float)
    export["segments_current_a"] = np.asarray([float(s.current_a) for s in segments], dtype=float)
    export["segments_t_amb_c"] = np.asarray([float(s.t_amb_c) for s in segments], dtype=float)

    # ---- Segment overrides (NaN if None) ----
    export["segments_v_cell_min_override_v"] = np.asarray(
        [_none_to_nan(getattr(s, "v_cell_min_override_v", None)) for s in segments], dtype=float
    )
    export["segments_v_cell_max_override_v"] = np.asarray(
        [_none_to_nan(getattr(s, "v_cell_max_override_v", None)) for s in segments], dtype=float
    )
    export["segments_t_rack_override_c"] = np.asarray(
        [_none_to_nan(getattr(s, "t_rack_override_c", None)) for s in segments], dtype=float
    )
    export["segments_i_rack_override_a"] = np.asarray(
        [_none_to_nan(getattr(s, "i_rack_override_a", None)) for s in segments], dtype=float
    )
    export["segments_gas_alarm"] = np.asarray(
        [1 if bool(getattr(s, "gas_alarm", False)) else 0 for s in segments], dtype=int
    )

    # ---- Fault injection meta ----
    export["fault_uv_time_s"] = np.asarray([_none_to_nan(getattr(fault_inj, "uv_time_s", None))], dtype=float)
    export["fault_uv_v_fault"] = np.asarray([float(getattr(fault_inj, "uv_v_fault", 2.0))], dtype=float)
    export["fault_uv_duration_s"] = np.asarray([float(getattr(fault_inj, "uv_duration_s", 0.0))], dtype=float)

    export["fault_ot_time_s"] = np.asarray([_none_to_nan(getattr(fault_inj, "ot_time_s", None))], dtype=float)
    export["fault_ot_temp_c"] = np.asarray([float(getattr(fault_inj, "ot_temp_c", 100.0))], dtype=float)
    export["fault_ot_duration_s"] = np.asarray([float(getattr(fault_inj, "ot_duration_s", 0.0))], dtype=float)

    export["fault_oc_time_s"] = np.asarray([_none_to_nan(getattr(fault_inj, "oc_time_s", None))], dtype=float)
    export["fault_oc_i_fault_a"] = np.asarray([float(getattr(fault_inj, "oc_i_fault_a", 800.0))], dtype=float)
    export["fault_oc_duration_s"] = np.asarray([float(getattr(fault_inj, "oc_duration_s", 0.0))], dtype=float)

    export["fault_fire_time_s"] = np.asarray([_none_to_nan(getattr(fault_inj, "fire_time_s", None))], dtype=float)
    export["fault_fire_duration_s"] = np.asarray([float(getattr(fault_inj, "fire_duration_s", 0.0))], dtype=float)

    buf = io.BytesIO()
    np.savez_compressed(buf, **export)
    return buf.getvalue()


# ---------------- Presets ----------------
@dataclass
class Preset:
    key: str
    title: str
    description: str
    expected: str
    segments: List[Dict[str, Any]]
    faults: Dict[str, Any]
    use_limits: bool
    true_init_soc: float
    ekf_init_soc: float
    ekf_overrides: Optional[Dict[str, Any]] = None  # runtime overrides (demo), does not touch YAML


def _presets(dt_s: float) -> List[Preset]:
    return [
        Preset(
            key="uv_step",
            title="UV injection (step)",
            description="Force V_cell_min below UV threshold at a given time (sensor-level injection).",
            expected=(
                "Expected: UV flag triggers at t ≈ uv_time_s + (debounce_steps-1)*dt. "
                "State goes to FAULT, I_act drops to 0 after the transition."
            ),
            segments=[
                {"duration_s": 1800.0, "current_a": 250.0, "t_amb_c": 25.0},
                {"duration_s": 10.0, "current_a": 0.0, "t_amb_c": 25.0},
            ],
            faults={
                "uv_enable": True, "uv_time_s": 1200.0, "uv_v_fault": 2.0, "uv_duration_s": 0.0,
                "ot_enable": False, "oc_enable": False, "fire_enable": False,
            },
            use_limits=True,
            true_init_soc=0.90,
            ekf_init_soc=0.90,
        ),
        Preset(
            key="ot_step",
            title="OT injection (step)",
            description="Force T_rack above OT threshold at a given time (sensor-level injection).",
            expected=(
                "Expected: OT triggers at t ≈ ot_time_s + (debounce_steps-1)*dt. "
                "Because emergency includes OT, state likely goes to EMERGENCY_SHUTDOWN. I_act becomes 0."
            ),
            segments=[
                {"duration_s": 600.0, "current_a": 250.0, "t_amb_c": 25.0},
                {"duration_s": 10.0, "current_a": 0.0, "t_amb_c": 25.0},
            ],
            faults={
                "uv_enable": False,
                "ot_enable": True, "ot_time_s": 200.0, "ot_temp_c": 100.0, "ot_duration_s": 0.0,
                "oc_enable": False,
                "fire_enable": False,
            },
            use_limits=True,
            true_init_soc=0.80,
            ekf_init_soc=0.80,
        ),
        Preset(
            key="oc_step",
            title="OC injection (step)",
            description="Force I_rack measurement above OC threshold at a given time (sensor-level injection).",
            expected=(
                "Expected: OC triggers at t ≈ oc_time_s + (debounce_steps-1)*dt. "
                "Emergency includes OC, so EMERGENCY_SHUTDOWN is typical. I_act becomes 0."
            ),
            segments=[
                {"duration_s": 600.0, "current_a": 250.0, "t_amb_c": 25.0},
                {"duration_s": 10.0, "current_a": 0.0, "t_amb_c": 25.0},
            ],
            faults={
                "uv_enable": False,
                "ot_enable": False,
                "oc_enable": True, "oc_time_s": 200.0, "oc_i_fault_a": 800.0, "oc_duration_s": 0.0,
                "fire_enable": False,
            },
            use_limits=True,
            true_init_soc=0.80,
            ekf_init_soc=0.80,
        ),
        Preset(
            key="fire_step",
            title="FIRE injection (gas alarm step)",
            description="Force gas alarm (fire) at a given time (sensor-level injection).",
            expected=(
                "Expected: FIRE triggers at t ≈ fire_time_s + (debounce_steps-1)*dt (depending on detector debounce). "
                "Emergency includes FIRE => EMERGENCY_SHUTDOWN. I_act becomes 0."
            ),
            segments=[
                {"duration_s": 600.0, "current_a": 0.0, "t_amb_c": 25.0},
                {"duration_s": 10.0, "current_a": 0.0, "t_amb_c": 25.0},
            ],
            faults={
                "uv_enable": False,
                "ot_enable": False,
                "oc_enable": False,
                "fire_enable": True, "fire_time_s": 200.0, "fire_duration_s": 0.0,
            },
            use_limits=True,
            true_init_soc=0.50,
            ekf_init_soc=0.50,
        ),
        Preset(
            key="oc_pulse_debounce",
            title="OC debounce (short pulse should NOT trigger)",
            description="Send a short OC pulse shorter than debounce window; OC should remain false.",
            expected=(
                "Expected: OC stays false (t_oc_s=None), fault_any remains 0, state stays RUN. "
                "If this triggers, your debounce settings or dt assumptions differ."
            ),
            segments=[
                {"duration_s": 600.0, "current_a": 250.0, "t_amb_c": 25.0},
                {"duration_s": 10.0, "current_a": 0.0, "t_amb_c": 25.0},
            ],
            faults={
                "uv_enable": False,
                "ot_enable": False,
                "oc_enable": True, "oc_time_s": 200.0, "oc_i_fault_a": 800.0, "oc_duration_s": 1.0,
                "fire_enable": False,
            },
            use_limits=True,
            true_init_soc=0.80,
            ekf_init_soc=0.80,
        ),
        Preset(
            key="override_vs_true_voltage",
            title="Segment override vs true (voltage)",
            description="Override V_cell_min/max during a segment. Used vs true should diverge.",
            expected=(
                "Expected: V_cell (used) diverges from V_cell (true) during the override segment. "
                "Fault flags follow the USED signals (because BMS sees those)."
            ),
            segments=[
                {"duration_s": 300.0, "current_a": 250.0, "t_amb_c": 25.0},
                {"duration_s": 60.0, "current_a": 250.0, "t_amb_c": 25.0,
                 "v_override_enable": True, "vmin_override": 2.0, "vmax_override": 3.2},
                {"duration_s": 60.0, "current_a": 0.0, "t_amb_c": 25.0},
            ],
            faults={"uv_enable": False, "ot_enable": False, "oc_enable": False, "fire_enable": False},
            use_limits=True,
            true_init_soc=0.90,
            ekf_init_soc=0.90,
        ),
        Preset(
            key="ekf_convergence",
            title="EKF demo: init mismatch vs CC-only reference",
            description=(
                "Run a normal discharge while EKF starts with wrong SoC. "
                "We also plot a CC-only reference (soc_cc) starting from EKF init to make the mismatch visible."
            ),
            expected=(
                "Expected: EKF SoC (post) tracks SoC_true closely, while CC-only reference remains offset "
                "when EKF init != true init. No safety faults should trigger."
            ),
            segments=[
                {"duration_s": 1200.0, "current_a": 200.0, "t_amb_c": 25.0},
                {"duration_s": 10.0, "current_a": 0.0, "t_amb_c": 25.0},
            ],
            faults={"uv_enable": False, "ot_enable": False, "oc_enable": False, "fire_enable": False},
            use_limits=True,
            true_init_soc=0.50,
            ekf_init_soc=0.80,
            # Optional: slow down the immediate correction if needed for nicer plots
            ekf_overrides={
                "estimation": {
                    "ekf": {
                        # smaller P0 -> trust init more; larger R -> trust voltage less
                        "initial_covariance": {"soc": 1.0e-6},
                        "r_measurement": {"v_terminal": 0.0025},
                        # keep some process noise so P can grow and allow correction over time if your model is too "perfect"
                        "q_process": {"soc": 1.0e-8},
                    }
                }
            },
        ),
    ]


# ---------------- Preset pass/fail evaluation ----------------
def _within_tol(actual: Optional[float], expected: float, tol: float) -> bool:
    if actual is None:
        return False
    return abs(float(actual) - float(expected)) <= float(tol)


def _first_index_at_or_after(t: np.ndarray, t0: float) -> int:
    return int(np.searchsorted(t, float(t0), side="left"))


def evaluate_preset_run(
    *,
    preset: Preset,
    result: Dict[str, Any],
    dt_s: float,
    debounce_steps: int,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Returns:
      overall_pass: bool
      checks: list of dict rows for a dataframe
    """
    checks: List[Dict[str, Any]] = []

    safety = compute_safety_metrics(result, float(dt_s))

    t = np.asarray(result["time_s"], dtype=float)
    i_act = np.asarray(result["i_act_a"], dtype=float)
    state = np.asarray(result["state_code"], dtype=int)

    tol_time = max(0.51 * float(dt_s), 1e-9)  # ~1-sample tolerance

    def add_check(name: str, ok: bool, details: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "details": details})

    def check_shutdown_after(trigger_t: Optional[float], name: str) -> None:
        if trigger_t is None:
            add_check(f"{name}: I_act -> 0", False, "No trigger time (None).")
            add_check(f"{name}: state >= FAULT", False, "No trigger time (None).")
            return

        idx = _first_index_at_or_after(t, float(trigger_t) + float(dt_s))
        if idx >= len(t):
            idx = len(t) - 1

        i_tail = i_act[idx:]
        state_tail = state[idx:]

        ok_i0 = bool(np.nanmax(np.abs(i_tail)) < 1e-6)
        ok_state = bool(np.nanmax(state_tail) >= 2)  # FAULT=2, EMERGENCY=3

        add_check(f"{name}: I_act -> 0", ok_i0, f"max|I_act| after trigger: {float(np.nanmax(np.abs(i_tail))):.3e} A")
        add_check(f"{name}: state >= FAULT", ok_state, f"max state after trigger: {int(np.nanmax(state_tail))}")

    key = preset.key

    if key in ["uv_step", "ot_step", "oc_step", "fire_step"]:
        shift = (int(debounce_steps) - 1) * float(dt_s)

        if key == "uv_step":
            exp = float(preset.faults.get("uv_time_s", 0.0)) + shift
            act = safety.get("t_uv_s", None)
            ok = _within_tol(act, exp, tol_time)
            add_check("UV trigger time", ok, f"expected≈{exp:.3f}s, actual={fmt(act)}s, tol={tol_time:.3f}s")
            check_shutdown_after(act, "UV")

        if key == "ot_step":
            exp = float(preset.faults.get("ot_time_s", 0.0)) + shift
            act = safety.get("t_ot_s", None)
            ok = _within_tol(act, exp, tol_time)
            add_check("OT trigger time", ok, f"expected≈{exp:.3f}s, actual={fmt(act)}s, tol={tol_time:.3f}s")
            check_shutdown_after(act, "OT")

        if key == "oc_step":
            exp = float(preset.faults.get("oc_time_s", 0.0)) + shift
            act = safety.get("t_oc_s", None)
            ok = _within_tol(act, exp, tol_time)
            add_check("OC trigger time", ok, f"expected≈{exp:.3f}s, actual={fmt(act)}s, tol={tol_time:.3f}s")
            check_shutdown_after(act, "OC")

        if key == "fire_step":
            exp = float(preset.faults.get("fire_time_s", 0.0)) + shift
            act = safety.get("t_fire_s", None)
            ok = _within_tol(act, exp, tol_time)
            add_check("FIRE trigger time", ok, f"expected≈{exp:.3f}s, actual={fmt(act)}s, tol={tol_time:.3f}s")
            check_shutdown_after(act, "FIRE")

    elif key == "oc_pulse_debounce":
        act_oc = safety.get("t_oc_s", None)
        act_any = safety.get("t_fault_any_s", None)
        ok = (act_oc is None) and (act_any is None)
        add_check("Debounce pulse should NOT trigger", ok, f"t_oc_s={fmt(act_oc)}, t_fault_any_s={fmt(act_any)}")

    elif key == "override_vs_true_voltage":
        v_min = np.asarray(result["v_cell_min_v"], dtype=float)
        v_max = np.asarray(result["v_cell_max_v"], dtype=float)
        v_min_true = np.asarray(result.get("v_cell_min_true_v", v_min), dtype=float)
        v_max_true = np.asarray(result.get("v_cell_max_true_v", v_max), dtype=float)

        diff = np.nanmax(np.abs(v_min - v_min_true)) + np.nanmax(np.abs(v_max - v_max_true))
        ok = bool(diff > 1e-3)
        add_check("Used vs true voltage must diverge", ok, f"max combined divergence ≈ {diff:.6f} V")

    elif key == "ekf_convergence":
        soc_true = np.asarray(result["soc_true"], dtype=float)
        soc_hat = np.asarray(result["soc_hat"], dtype=float)

        safety_any = safety.get("t_fault_any_s", None)
        ok_no_fault = (safety_any is None)
        add_check("No safety faults", ok_no_fault, f"t_fault_any_s={fmt(safety_any)}")

        # We want to show mismatch is present (by design), but soc_hat[0] is often posterior at t=0.
        init_mismatch = abs(float(preset.true_init_soc) - float(preset.ekf_init_soc))
        add_check(
            "Init mismatch configured",
            bool(init_mismatch >= 0.05),
            f"|true_init - ekf_init| = {init_mismatch:.3f} (target >= 0.05)"
        )

        # EKF tracking quality at end
        err_post = soc_true - soc_hat
        e_end = float(abs(err_post[-1])) if len(err_post) else float("nan")
        ok_end = np.isfinite(e_end) and (e_end < 0.05)
        add_check("EKF end error small", ok_end, f"|e_end|={e_end:.4f} (target < 0.05)")

        # If CC-only series exists, show EKF improvement vs CC-only
        if "soc_cc" in result:
            soc_cc = np.asarray(result["soc_cc"], dtype=float)
            err_cc = soc_true - soc_cc
            e_cc_end = float(abs(err_cc[-1])) if len(err_cc) else float("nan")
            ok_improve = np.isfinite(e_cc_end) and np.isfinite(e_end) and (e_end < 0.5 * e_cc_end)
            add_check("EKF improves vs CC-only", ok_improve, f"|e_end|={e_end:.4f}, |e_cc_end|={e_cc_end:.4f} (target: EKF < 50% of CC)")
        else:
            add_check("CC-only reference available", False, "soc_cc not found in result (plot will be less illustrative).")

    overall = all(bool(r["pass"]) for r in checks) if checks else True
    return overall, checks


def _init_ui_state_defaults() -> None:
    """
    Ensure all keys exist BEFORE any widgets are instantiated.
    This avoids StreamlitAPIException when presets/reset write into session_state.
    """
    st.session_state.setdefault("n_segments", 2)
    st.session_state.setdefault("use_limits", True)

    st.session_state.setdefault("true_init_soc_ui", 0.50)
    st.session_state.setdefault("ekf_init_soc_ui", 0.80)

    # EKF demo overrides (manual)
    st.session_state.setdefault("ekf_override_enable", False)
    st.session_state.setdefault("ekf_override_p0_soc", 1.0e-6)
    st.session_state.setdefault("ekf_override_r_v", 0.0025)
    st.session_state.setdefault("ekf_override_q_soc", 1.0e-8)

    st.session_state.setdefault("plot_show_soc_cc", True)

    st.session_state.setdefault("uv_enable", False)
    st.session_state.setdefault("uv_time", 1200.0)
    st.session_state.setdefault("uv_v_fault", 2.0)
    st.session_state.setdefault("uv_dur", 0.0)

    st.session_state.setdefault("ot_enable", False)
    st.session_state.setdefault("ot_time", 200.0)
    st.session_state.setdefault("ot_temp", 100.0)
    st.session_state.setdefault("ot_dur", 0.0)

    st.session_state.setdefault("oc_enable", False)
    st.session_state.setdefault("oc_time", 200.0)
    st.session_state.setdefault("oc_i_fault", 800.0)
    st.session_state.setdefault("oc_dur", 0.0)

    st.session_state.setdefault("fire_enable", False)
    st.session_state.setdefault("fire_time", 200.0)
    st.session_state.setdefault("fire_dur", 0.0)

    st.session_state.setdefault("offline_last_preset_key", None)


def _reset_to_safe_defaults(max_segments: int = 6, *, reset_initial_conditions: bool = True) -> None:
    """
    Reset ALL segment inputs (duration/current/ambient + overrides) to a safe, normal-operation baseline.
    Intended to be called from callbacks (on_click) so session_state can be written safely.
    """
    for i in range(max_segments):
        st.session_state[f"dur_{i}"] = 600.0 if i == 0 else 10.0
        st.session_state[f"curr_{i}"] = 250.0 if i == 0 else 0.0
        st.session_state[f"tamb_{i}"] = 25.0

        st.session_state[f"seg_{i}_t_override_enable"] = False
        st.session_state[f"seg_{i}_t_override_val"] = 25.0

        st.session_state[f"seg_{i}_v_override_enable"] = False
        st.session_state[f"seg_{i}_vmin_override_val"] = 3.20
        st.session_state[f"seg_{i}_vmax_override_val"] = 3.20

        st.session_state[f"seg_{i}_i_override_enable"] = False
        st.session_state[f"seg_{i}_i_override_val"] = 0.0

        st.session_state[f"seg_{i}_gas_alarm"] = False

    st.session_state["n_segments"] = 2

    st.session_state["uv_enable"] = False
    st.session_state["uv_time"] = 1200.0
    st.session_state["uv_v_fault"] = 2.0
    st.session_state["uv_dur"] = 0.0

    st.session_state["ot_enable"] = False
    st.session_state["ot_time"] = 200.0
    st.session_state["ot_temp"] = 100.0
    st.session_state["ot_dur"] = 0.0

    st.session_state["oc_enable"] = False
    st.session_state["oc_time"] = 200.0
    st.session_state["oc_i_fault"] = 800.0
    st.session_state["oc_dur"] = 0.0

    st.session_state["fire_enable"] = False
    st.session_state["fire_time"] = 200.0
    st.session_state["fire_dur"] = 0.0

    st.session_state["use_limits"] = True

    if reset_initial_conditions:
        st.session_state["true_init_soc_ui"] = 0.50
        st.session_state["ekf_init_soc_ui"] = 0.80

    # EKF override defaults (manual)
    st.session_state["ekf_override_enable"] = False
    st.session_state["ekf_override_p0_soc"] = 1.0e-6
    st.session_state["ekf_override_r_v"] = 0.0025
    st.session_state["ekf_override_q_soc"] = 1.0e-8

    st.session_state["plot_show_soc_cc"] = True

    # reset preset tracking
    st.session_state["offline_last_preset_key"] = None


def _apply_preset_to_ui(p: Preset) -> None:
    _reset_to_safe_defaults(max_segments=6, reset_initial_conditions=False)

    segs = p.segments or []
    st.session_state["n_segments"] = int(min(6, max(1, len(segs))))

    for i, s in enumerate(segs[:6]):
        st.session_state[f"dur_{i}"] = float(s.get("duration_s", 10.0))
        st.session_state[f"curr_{i}"] = float(s.get("current_a", 0.0))
        st.session_state[f"tamb_{i}"] = float(s.get("t_amb_c", 25.0))

        if bool(s.get("v_override_enable", False)):
            st.session_state[f"seg_{i}_v_override_enable"] = True
            st.session_state[f"seg_{i}_vmin_override_val"] = float(s.get("vmin_override", 3.2))
            st.session_state[f"seg_{i}_vmax_override_val"] = float(s.get("vmax_override", 3.2))

    f = p.faults or {}
    st.session_state["uv_enable"] = bool(f.get("uv_enable", False))
    st.session_state["uv_time"] = float(f.get("uv_time_s", 1200.0))
    st.session_state["uv_v_fault"] = float(f.get("uv_v_fault", 2.0))
    st.session_state["uv_dur"] = float(f.get("uv_duration_s", 0.0))

    st.session_state["ot_enable"] = bool(f.get("ot_enable", False))
    st.session_state["ot_time"] = float(f.get("ot_time_s", 200.0))
    st.session_state["ot_temp"] = float(f.get("ot_temp_c", 100.0))
    st.session_state["ot_dur"] = float(f.get("ot_duration_s", 0.0))

    st.session_state["oc_enable"] = bool(f.get("oc_enable", False))
    st.session_state["oc_time"] = float(f.get("oc_time_s", 200.0))
    st.session_state["oc_i_fault"] = float(f.get("oc_i_fault_a", 800.0))
    st.session_state["oc_dur"] = float(f.get("oc_duration_s", 0.0))

    st.session_state["fire_enable"] = bool(f.get("fire_enable", False))
    st.session_state["fire_time"] = float(f.get("fire_time_s", 200.0))
    st.session_state["fire_dur"] = float(f.get("fire_duration_s", 0.0))

    st.session_state["use_limits"] = bool(p.use_limits)
    st.session_state["true_init_soc_ui"] = float(p.true_init_soc)
    st.session_state["ekf_init_soc_ui"] = float(p.ekf_init_soc)

    # If preset includes EKF overrides, enable them for demo runs by default
    if p.ekf_overrides:
        st.session_state["ekf_override_enable"] = True
        ekf = (p.ekf_overrides.get("estimation", {}).get("ekf", {}) if isinstance(p.ekf_overrides, dict) else {}) or {}
        st.session_state["ekf_override_p0_soc"] = float((ekf.get("initial_covariance", {}) or {}).get("soc", 1.0e-6))
        st.session_state["ekf_override_r_v"] = float((ekf.get("r_measurement", {}) or {}).get("v_terminal", 0.0025))
        st.session_state["ekf_override_q_soc"] = float((ekf.get("q_process", {}) or {}).get("soc", 1.0e-8))


def _set_notice(kind: str, msg: str) -> None:
    st.session_state["_notice_kind"] = kind
    st.session_state["_notice_msg"] = msg


def _pop_notice() -> Optional[tuple]:
    kind = st.session_state.pop("_notice_kind", None)
    msg = st.session_state.pop("_notice_msg", None)
    if kind and msg:
        return kind, msg
    return None


def _segments_from_preset(p: Preset) -> List[Segment]:
    seg_objs: List[Segment] = []
    for s in (p.segments or []):
        vmin = float(s.get("vmin_override")) if bool(s.get("v_override_enable", False)) else None
        vmax = float(s.get("vmax_override")) if bool(s.get("v_override_enable", False)) else None
        seg_objs.append(
            Segment(
                duration_s=float(s.get("duration_s", 10.0)),
                current_a=float(s.get("current_a", 0.0)),
                t_amb_c=float(s.get("t_amb_c", 25.0)),
                v_cell_min_override_v=vmin,
                v_cell_max_override_v=vmax,
                t_rack_override_c=None,
                i_rack_override_a=None,
                gas_alarm=bool(s.get("gas_alarm", False)),
            )
        )
    return seg_objs


def _faults_from_preset(p: Preset) -> FaultConfig:
    f = p.faults or {}
    return FaultConfig(
        uv_time_s=float(f.get("uv_time_s")) if bool(f.get("uv_enable", False)) else None,
        uv_v_fault=float(f.get("uv_v_fault", 2.0)),
        uv_duration_s=float(f.get("uv_duration_s", 0.0)),

        ot_time_s=float(f.get("ot_time_s")) if bool(f.get("ot_enable", False)) else None,
        ot_temp_c=float(f.get("ot_temp_c", 100.0)),
        ot_duration_s=float(f.get("ot_duration_s", 0.0)),

        oc_time_s=float(f.get("oc_time_s")) if bool(f.get("oc_enable", False)) else None,
        oc_i_fault_a=float(f.get("oc_i_fault_a", 800.0)),
        oc_duration_s=float(f.get("oc_duration_s", 0.0)),

        fire_time_s=float(f.get("fire_time_s")) if bool(f.get("fire_enable", False)) else None,
        fire_duration_s=float(f.get("fire_duration_s", 0.0)),
    )


def _build_ekf_override_dict_from_ui() -> Optional[Dict[str, Any]]:
    if not bool(st.session_state.get("ekf_override_enable", False)):
        return None
    try:
        p0 = float(st.session_state.get("ekf_override_p0_soc", 1.0e-6))
        rv = float(st.session_state.get("ekf_override_r_v", 0.0025))
        qs = float(st.session_state.get("ekf_override_q_soc", 1.0e-8))
        return {
            "estimation": {
                "ekf": {
                    "initial_covariance": {"soc": p0},
                    "r_measurement": {"v_terminal": rv},
                    "q_process": {"soc": qs},
                }
            }
        }
    except Exception:
        return None


# ---------------- Page ----------------
_init_ui_state_defaults()

st.title("Offline Scenarios")
st.caption("Preset tests + manual builder for offline BMS validation (single rack).")

# Ensure profile state exists even if user opens this page directly
profiles = list(PROFILES.keys())
default_profile = "LUNA" if "LUNA" in PROFILES else profiles[0]

if "active_profile" not in st.session_state:
    st.session_state.active_profile = default_profile

# Allow switching profile directly on this page
st.sidebar.selectbox("Rack profile", profiles, key="active_profile")

active = st.session_state.active_profile
st.session_state.active_config_path = PROFILES[active]["params"]
st.session_state.active_scenarios_path = PROFILES[active]["scenarios"]

cfg_path = st.session_state.active_config_path
params = get_params(cfg_path)

st.sidebar.caption(f"Config: {cfg_path}")
st.sidebar.caption(f"Scenarios: {st.session_state.active_scenarios_path}")

dt_s = float(params["simulation"]["dt_s"])

limits = params["limits"]
faults_yaml = params["faults"]
ctrl = params.get("bms_control", {}) or {}

UV_THRESHOLD = float(faults_yaml["uv_cell_v"])
OV_THRESHOLD = float(faults_yaml["ov_cell_v"])
SOFT_MAX_V = float(limits["v_cell_max_v"])

OT_THRESHOLD = float(faults_yaml.get("ot_rack_c", 50.0))
T_DERATE_START = float(ctrl.get("t_high_derate_start_c", 45.0))
T_CUTOFF = float(ctrl.get("t_high_cutoff_c", 55.0))

V_CELL_MIN_OP = float(limits.get("v_cell_min_v", 2.5))
V_CELL_MAX_OP = float(limits.get("v_cell_max_v", 3.5))

# debounce
fire_cfg: Dict[str, Any] = {}
if isinstance(faults_yaml.get("fire_detection", None), dict):
    fire_cfg.update(faults_yaml.get("fire_detection", {}) or {})
if isinstance(params.get("fire_detection", None), dict):
    fire_cfg.update(params.get("fire_detection", {}) or {})
DEBOUNCE_STEPS = int(fire_cfg.get("debounce_steps", faults_yaml.get("debounce_steps", 3)))

therm = params.get("thermal_model", {}) or {}
try:
    tau_s = float(therm.get("c_th_j_per_k", 0.0)) * float(therm.get("r_th_k_per_w", 0.0))
except Exception:
    tau_s = 0.0

# Persist last run
for k, default in [
    ("offline_last_result", None),
    ("offline_last_npz_bytes", None),
    ("offline_last_npz_name", "offline_result.npz"),
    ("offline_last_segments", None),
    ("offline_last_fault_inj", None),
    ("offline_last_use_limits", True),
    ("offline_last_preset_key", None),
    ("offline_last_true_init_soc", None),
    ("offline_last_ekf_init_soc", None),
    ("offline_last_ekf_overrides", None),
]:
    if k not in st.session_state:
        st.session_state[k] = default

# Sidebar summary (no editable widgets -> avoids preset conflict)
st.sidebar.subheader("Active config")
st.sidebar.markdown(f"**dt_s**: {dt_s} s")
st.sidebar.markdown(f"**debounce_steps**: {DEBOUNCE_STEPS}")
st.sidebar.markdown(
    f"**UV threshold**: {UV_THRESHOLD:.2f} V\n\n"
    f"**OV threshold**: {OV_THRESHOLD:.2f} V\n\n"
    f"**Soft max cell voltage**: {SOFT_MAX_V:.2f} V\n\n"
    f"**OT threshold**: {OT_THRESHOLD:.1f} °C\n\n"
    f"**T derate start**: {T_DERATE_START:.1f} °C\n\n"
    f"**T cutoff**: {T_CUTOFF:.1f} °C"
)
if tau_s > 0:
    st.sidebar.caption(
        f"Thermal time constant estimate: τ ≈ {tau_s:.0f} s (~{tau_s/3600:.1f} h). "
        "So changing T_amb in later segments may not raise T_rack quickly. "
        "Use T_rack override or OT injection for fast logic tests."
    )

st.info(
    "Segment overrides and injections change **what the BMS sees** (limits + fault detector). "
    "They do **not** force the plant physics. This is intentional for operator-style validation."
)

# ---------------- YAML Scenario Runner (active profile) ----------------
st.header("YAML Scenario Runner (Active profile)")
st.caption(
    "Runs scenarios from the active profile YAML file using src.sim_runner.run_scenario(). "
    "This is independent from the Manual Builder below."
)

scenarios_path = st.session_state.get("active_scenarios_path", None)
if not scenarios_path:
    st.warning("No scenarios YAML path found for the active profile.")
else:
    try:
        yaml_scenarios = load_scenarios(scenarios_path)
    except Exception as e:
        st.error("Failed to load scenarios YAML for the active profile.")
        st.code(f"{scenarios_path}\n\n{e}")
        yaml_scenarios = []

    if not yaml_scenarios:
        st.warning("No scenarios found in the active YAML file.")
    else:
        scenario_ids = [s.id for s in yaml_scenarios]
        selected_id = st.selectbox(
            "Select a scenario ID",
            options=scenario_ids,
            index=0,
            key="yaml_selected_scenario_id",
        )
        scn = next(s for s in yaml_scenarios if s.id == selected_id)

        with st.expander("Scenario details", expanded=True):
            st.markdown(f"**ID:** {scn.id}")
            st.markdown(f"**Description:** {getattr(scn, 'description', '') or ''}")
            st.markdown(f"**Profile type:** {getattr(scn, 'profile_type', '') or getattr(getattr(scn, 'profile', None), 'type', '')}")
            st.markdown(f"**Max time [s]:** {getattr(scn, 'max_time_s', None)}")
            st.markdown(f"**Stop on emergency:** {bool(getattr(scn, 'stop_on_emergency', False))}")

        run_yaml = st.button("Run selected YAML scenario", use_container_width=True, key="run_yaml_btn")

        if run_yaml:
            with st.spinner("Running YAML scenario..."):
                result = run_scenario(scn, params)

            st.session_state["offline_last_result"] = result
            st.session_state["offline_last_segments"] = getattr(scn, "segments", None)
            st.session_state["offline_last_fault_inj"] = None
            st.session_state["offline_last_use_limits"] = bool(getattr(scn, "use_bms_limits", True))
            st.session_state["offline_last_preset_key"] = None

            st.session_state["offline_last_true_init_soc"] = getattr(scn, "true_init_soc", None)
            st.session_state["offline_last_ekf_init_soc"] = getattr(scn, "ekf_init_soc", None)
            st.session_state["offline_last_ekf_overrides"] = None

            # Add CC-only series if we have ekf init
            _attach_demo_series_if_possible(
                result,
                params=params,
                dt_s=float(dt_s),
                ekf_init_soc=st.session_state["offline_last_ekf_init_soc"],
            )

            st.success(f"YAML scenario finished: {scn.id}. Scroll down for metrics and plots.")


# ---------------- Quick Tests (Preset library) ----------------
st.header("Quick Tests (Preset library)")

preset_list = _presets(dt_s)
preset_map = {p.title: p for p in preset_list}
preset_by_key = {p.key: p for p in preset_list}

cpt1, cpt2 = st.columns([2, 1])
with cpt1:
    preset_title = st.selectbox(
        "Select a preset",
        options=[p.title for p in preset_list],
        index=0,
        key="preset_select",
    )
    preset = preset_map[preset_title]
    st.write(preset.description)
    st.warning(preset.expected)


def _cb_load_preset() -> None:
    p = preset_map[st.session_state["preset_select"]]
    _apply_preset_to_ui(p)
    _set_notice("success", "Preset loaded into the manual builder below.")


def _cb_reset() -> None:
    _reset_to_safe_defaults(max_segments=6, reset_initial_conditions=True)
    _set_notice("success", "Reset completed (safe defaults loaded).")


def _cb_run_preset_now() -> None:
    p = preset_map[st.session_state["preset_select"]]

    segs = _segments_from_preset(p)
    fault_inj = _faults_from_preset(p)

    # Build runtime params (optional EKF overrides)
    params_run = copy.deepcopy(params)
    ekf_ov = p.ekf_overrides if isinstance(p.ekf_overrides, dict) else None
    if ekf_ov:
        _deep_update(params_run, ekf_ov)

    with st.spinner("Running preset scenario..."):
        result = run_manual_profile(
            segments=segs,
            faults=fault_inj,
            params=params_run,
            use_bms_limits=bool(p.use_limits),
            true_init_soc=float(p.true_init_soc),
            ekf_init_soc=float(p.ekf_init_soc),
        )

    st.session_state["offline_last_result"] = result
    st.session_state["offline_last_segments"] = segs
    st.session_state["offline_last_fault_inj"] = fault_inj
    st.session_state["offline_last_use_limits"] = bool(p.use_limits)
    st.session_state["offline_last_preset_key"] = p.key

    st.session_state["offline_last_true_init_soc"] = float(p.true_init_soc)
    st.session_state["offline_last_ekf_init_soc"] = float(p.ekf_init_soc)
    st.session_state["offline_last_ekf_overrides"] = ekf_ov

    # Attach CC-only series for plots and reporting
    _attach_demo_series_if_possible(
        result,
        params=params_run,
        dt_s=float(dt_s),
        ekf_init_soc=float(p.ekf_init_soc),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state["offline_last_npz_name"] = f"offline_preset_{p.key}_{ts}.npz"
    st.session_state["offline_last_npz_bytes"] = make_manual_npz_bytes(
        result,
        segments=segs,
        fault_inj=fault_inj,
        use_limits=bool(p.use_limits),
        true_init_soc=float(p.true_init_soc),
        ekf_init_soc=float(p.ekf_init_soc),
        dt_s=float(dt_s),
        ekf_overrides=ekf_ov,
    )

    _set_notice("success", f"Preset run finished: {p.title}. Scroll down for PASS/FAIL + plots.")


with cpt2:
    st.button("Run selected preset now (PASS/FAIL)", use_container_width=True, on_click=_cb_run_preset_now)
    st.button("Load preset into builder", use_container_width=True, on_click=_cb_load_preset)
    st.button("Reset to normal operation (safe defaults)", use_container_width=True, on_click=_cb_reset)

notice = _pop_notice()
if notice:
    kind, msg = notice
    if kind == "success":
        st.success(msg)
    elif kind == "warning":
        st.warning(msg)
    else:
        st.info(msg)

st.divider()

# ---------------- Manual builder ----------------
st.header("Manual Builder")

st.subheader("Initial conditions (affects output)")
cic1, cic2 = st.columns(2)
with cic1:
    true_init_soc_ui = st.number_input(
        "True initial SoC (ECM) [0..1]",
        min_value=0.0, max_value=1.0,
        value=float(st.session_state.get("true_init_soc_ui", 0.50)),
        step=0.01, key="true_init_soc_ui"
    )
with cic2:
    ekf_init_soc_ui = st.number_input(
        "EKF initial SoC estimate [0..1]",
        min_value=0.0, max_value=1.0,
        value=float(st.session_state.get("ekf_init_soc_ui", 0.80)),
        step=0.01, key="ekf_init_soc_ui"
    )
st.caption(
    "Sign convention: **+I = discharge (SoC decreases)**, **-I = charge (SoC increases)**. "
    "Example EKF demo: True=0.50, EKF=0.80."
)

with st.expander("EKF demo overrides (optional, per-run only)", expanded=False):
    st.caption(
        "If your EKF snaps to SoC_true at t=0 (very common with perfect OCV model), "
        "enable this to slow the correction for a more illustrative convergence plot."
    )
    st.checkbox("Enable EKF overrides for this run", key="ekf_override_enable")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("P0(soc) initial covariance", min_value=0.0, value=float(st.session_state.get("ekf_override_p0_soc", 1.0e-6)), format="%.2e", key="ekf_override_p0_soc")
    with c2:
        st.number_input("R(v_terminal) measurement noise", min_value=0.0, value=float(st.session_state.get("ekf_override_r_v", 0.0025)), format="%.4f", key="ekf_override_r_v")
    with c3:
        st.number_input("Q(soc) process noise", min_value=0.0, value=float(st.session_state.get("ekf_override_q_soc", 1.0e-8)), format="%.2e", key="ekf_override_q_soc")

st.checkbox("Show CC-only SoC reference (soc_cc) on plots", key="plot_show_soc_cc")

n_segments = st.number_input(
    "Number of segments",
    min_value=1, max_value=6,
    value=int(st.session_state.get("n_segments", 2)),
    step=1,
    key="n_segments",
)

segments: List[Segment] = []
for i in range(int(n_segments)):
    st.subheader(f"Segment {i + 1}")
    c1, c2, c3 = st.columns(3)

    with c1:
        dur = st.number_input(
            f"Duration [s] (segment {i + 1})",
            min_value=1.0,
            value=float(st.session_state.get(f"dur_{i}", 600.0 if i == 0 else 10.0)),
            key=f"dur_{i}",
        )
    with c2:
        curr = st.number_input(
            f"Current [A] (segment {i + 1})",
            value=float(st.session_state.get(f"curr_{i}", 250.0 if i == 0 else 0.0)),
            step=10.0,
            key=f"curr_{i}",
        )
    with c3:
        tamb = st.number_input(
            f"Ambient temperature [°C] (segment {i + 1})",
            value=float(st.session_state.get(f"tamb_{i}", 25.0)),
            key=f"tamb_{i}",
        )

    with st.expander(f"Advanced overrides (BMS inputs) — segment {i + 1}", expanded=False):
        st.caption(
            f"Operating ranges (from YAML): V_cell ≈ [{V_CELL_MIN_OP:.2f}, {V_CELL_MAX_OP:.2f}] V, "
            f"T cutoff ≈ {T_CUTOFF:.1f} °C, OT threshold ≈ {OT_THRESHOLD:.1f} °C."
        )

        cA, cB = st.columns(2)

        with cA:
            t_ov_en = st.checkbox(
                "Override rack temperature seen by BMS",
                value=bool(st.session_state.get(f"seg_{i}_t_override_enable", False)),
                key=f"seg_{i}_t_override_enable",
            )
            t_ov_val = None
            if t_ov_en:
                t_ov_val = st.number_input(
                    "T_rack override [°C]",
                    value=float(st.session_state.get(f"seg_{i}_t_override_val", 25.0)),
                    step=5.0,
                    key=f"seg_{i}_t_override_val",
                )

        with cB:
            i_ov_en = st.checkbox(
                "Override rack current measurement seen by BMS",
                value=bool(st.session_state.get(f"seg_{i}_i_override_enable", False)),
                key=f"seg_{i}_i_override_enable",
            )
            i_ov_val = None
            if i_ov_en:
                i_ov_val = st.number_input(
                    "I_rack override [A]",
                    value=float(st.session_state.get(f"seg_{i}_i_override_val", 0.0)),
                    step=10.0,
                    key=f"seg_{i}_i_override_val",
                )

        v_ov_en = st.checkbox(
            "Override cell voltage min/max seen by BMS",
            value=bool(st.session_state.get(f"seg_{i}_v_override_enable", False)),
            key=f"seg_{i}_v_override_enable",
        )
        vmin_ov, vmax_ov = None, None
        if v_ov_en:
            cv1, cv2 = st.columns(2)
            with cv1:
                vmin_ov = st.number_input(
                    "V_cell_min override [V]",
                    value=float(st.session_state.get(f"seg_{i}_vmin_override_val", 3.20)),
                    step=0.01,
                    key=f"seg_{i}_vmin_override_val",
                )
            with cv2:
                vmax_ov = st.number_input(
                    "V_cell_max override [V]",
                    value=float(st.session_state.get(f"seg_{i}_vmax_override_val", 3.20)),
                    step=0.01,
                    key=f"seg_{i}_vmax_override_val",
                )

        gas_alarm = st.checkbox(
            "Force gas alarm (fire) for this segment",
            value=bool(st.session_state.get(f"seg_{i}_gas_alarm", False)),
            key=f"seg_{i}_gas_alarm",
        )

    segments.append(
        Segment(
            duration_s=float(dur),
            current_a=float(curr),
            t_amb_c=float(tamb),
            v_cell_min_override_v=None if vmin_ov is None else float(vmin_ov),
            v_cell_max_override_v=None if vmax_ov is None else float(vmax_ov),
            t_rack_override_c=None if t_ov_val is None else float(t_ov_val),
            i_rack_override_a=None if i_ov_val is None else float(i_ov_val),
            gas_alarm=bool(gas_alarm),
        )
    )

# ---------------- Fault injections ----------------
st.header("Fault injections (optional)")

col_uv, col_ot, col_oc, col_fire = st.columns(4)

with col_uv:
    st.markdown("**UV fault (V_cell_min)**")
    uv_enable = st.checkbox("Enable UV", value=bool(st.session_state.get("uv_enable", False)), key="uv_enable")
    uv_time_s, uv_v_fault, uv_dur_s = None, 2.0, 0.0
    if uv_enable:
        uv_time_s = st.number_input("UV time [s]", min_value=0.0, value=float(st.session_state.get("uv_time", 1200.0)), key="uv_time")
        uv_v_fault = st.number_input("Vmin during UV [V]", min_value=0.0, value=float(st.session_state.get("uv_v_fault", 2.0)), key="uv_v_fault")
        uv_dur_s = st.number_input("UV duration [s] (0=step)", min_value=0.0, value=float(st.session_state.get("uv_dur", 0.0)), key="uv_dur")

with col_ot:
    st.markdown("**OT fault (T_rack)**")
    ot_enable = st.checkbox("Enable OT", value=bool(st.session_state.get("ot_enable", False)), key="ot_enable")
    ot_time_s, ot_temp_c, ot_dur_s = None, 100.0, 0.0
    if ot_enable:
        ot_time_s = st.number_input("OT time [s]", min_value=0.0, value=float(st.session_state.get("ot_time", 200.0)), key="ot_time")
        ot_temp_c = st.number_input("T during OT [°C]", min_value=-50.0, value=float(st.session_state.get("ot_temp", 100.0)), step=5.0, key="ot_temp")
        ot_dur_s = st.number_input("OT duration [s] (0=step)", min_value=0.0, value=float(st.session_state.get("ot_dur", 0.0)), key="ot_dur")

with col_oc:
    st.markdown("**OC fault (I_rack meas.)**")
    oc_enable = st.checkbox("Enable OC", value=bool(st.session_state.get("oc_enable", False)), key="oc_enable")
    oc_time_s, oc_i_fault_a, oc_dur_s = None, 800.0, 0.0
    if oc_enable:
        oc_time_s = st.number_input("OC time [s]", min_value=0.0, value=float(st.session_state.get("oc_time", 200.0)), key="oc_time")
        oc_i_fault_a = st.number_input("I during OC [A]", value=float(st.session_state.get("oc_i_fault", 800.0)), step=50.0, key="oc_i_fault")
        oc_dur_s = st.number_input("OC duration [s] (0=step)", min_value=0.0, value=float(st.session_state.get("oc_dur", 0.0)), key="oc_dur")

with col_fire:
    st.markdown("**FIRE (gas alarm)**")
    fire_enable = st.checkbox("Enable FIRE", value=bool(st.session_state.get("fire_enable", False)), key="fire_enable")
    fire_time_s, fire_dur_s = None, 0.0
    if fire_enable:
        fire_time_s = st.number_input("FIRE time [s]", min_value=0.0, value=float(st.session_state.get("fire_time", 200.0)), key="fire_time")
        fire_dur_s = st.number_input("FIRE duration [s] (0=step)", min_value=0.0, value=float(st.session_state.get("fire_dur", 0.0)), key="fire_dur")

fault_inj = FaultConfig(
    uv_time_s=uv_time_s,
    uv_v_fault=float(uv_v_fault),
    uv_duration_s=float(uv_dur_s),

    ot_time_s=ot_time_s,
    ot_temp_c=float(ot_temp_c),
    ot_duration_s=float(ot_dur_s),

    oc_time_s=oc_time_s,
    oc_i_fault_a=float(oc_i_fault_a),
    oc_duration_s=float(oc_dur_s),

    fire_time_s=fire_time_s,
    fire_duration_s=float(fire_dur_s),
)

use_limits = st.checkbox("Apply BMS current limits", value=bool(st.session_state.get("use_limits", True)), key="use_limits")

# ---------------- Run simulation ----------------
run_col1, run_col2 = st.columns([1, 2])
with run_col1:
    run_btn = st.button("Run simulation", key="run_btn", use_container_width=True)
with run_col2:
    st.caption("Tip: For quick fault logic tests, use injections/overrides. For plant-driven faults, increase current and run longer.")

if run_btn:
    # Runtime params copy (optional EKF overrides)
    params_run = copy.deepcopy(params)
    ekf_ov_ui = _build_ekf_override_dict_from_ui()
    if ekf_ov_ui:
        _deep_update(params_run, ekf_ov_ui)

    with st.spinner("Running offline BMS simulation..."):
        result = run_manual_profile(
            segments=segments,
            faults=fault_inj,
            params=params_run,
            use_bms_limits=bool(use_limits),
            true_init_soc=float(true_init_soc_ui),
            ekf_init_soc=float(ekf_init_soc_ui),
        )

    st.session_state["offline_last_result"] = result
    st.session_state["offline_last_segments"] = segments
    st.session_state["offline_last_fault_inj"] = fault_inj
    st.session_state["offline_last_use_limits"] = bool(use_limits)
    st.session_state["offline_last_preset_key"] = None

    st.session_state["offline_last_true_init_soc"] = float(true_init_soc_ui)
    st.session_state["offline_last_ekf_init_soc"] = float(ekf_init_soc_ui)
    st.session_state["offline_last_ekf_overrides"] = ekf_ov_ui

    # Attach CC-only series for plots and reporting
    _attach_demo_series_if_possible(
        result,
        params=params_run,
        dt_s=float(dt_s),
        ekf_init_soc=float(ekf_init_soc_ui),
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state["offline_last_npz_name"] = f"offline_result_{ts}.npz"
    st.session_state["offline_last_npz_bytes"] = make_manual_npz_bytes(
        result,
        segments=segments,
        fault_inj=fault_inj,
        use_limits=bool(use_limits),
        true_init_soc=float(true_init_soc_ui),
        ekf_init_soc=float(ekf_init_soc_ui),
        dt_s=float(dt_s),
        ekf_overrides=ekf_ov_ui,
    )

    st.success("Simulation finished. You can download the .npz below.")

# Export
if st.session_state.get("offline_last_npz_bytes", None) is not None:
    st.download_button(
        label="Download result (.npz)",
        data=st.session_state["offline_last_npz_bytes"],
        file_name=st.session_state["offline_last_npz_name"],
        mime="application/octet-stream",
        key="dl_offline_npz",
    )

# ---------------- Plot + metrics ----------------
if st.session_state.get("offline_last_result", None) is not None:
    result = st.session_state["offline_last_result"]
    segs_used = st.session_state.get("offline_last_segments", None)
    use_limits_used = bool(st.session_state.get("offline_last_use_limits", True))

    t = np.array(result["time_s"], dtype=float)
    i_req = np.array(result["i_req_a"], dtype=float)
    i_act = np.array(result["i_act_a"], dtype=float)

    soc_true = np.array(result["soc_true"], dtype=float)
    soc_hat = np.array(result["soc_hat"], dtype=float)

    show_soc_cc = bool(st.session_state.get("plot_show_soc_cc", True))
    soc_cc = None
    if show_soc_cc and ("soc_cc" in result):
        try:
            soc_cc = np.array(result["soc_cc"], dtype=float)
        except Exception:
            soc_cc = None

    soc_min = np.array(result.get("soc_cell_min", soc_true), dtype=float)
    soc_max = np.array(result.get("soc_cell_max", soc_true), dtype=float)

    v_min = np.array(result["v_cell_min_v"], dtype=float)
    v_max = np.array(result["v_cell_max_v"], dtype=float)

    v_min_true = np.array(result.get("v_cell_min_true_v", v_min), dtype=float)
    v_max_true = np.array(result.get("v_cell_max_true_v", v_max), dtype=float)

    t_rack_true = np.array(result.get("t_rack_c", np.full_like(t, np.nan)), dtype=float)
    t_rack_used = np.array(result.get("t_rack_used_c", t_rack_true), dtype=float)

    state = np.array(result["state_code"], dtype=int)
    oc = np.array(result["oc"], dtype=bool)
    ov = np.array(result["ov"], dtype=bool)
    uv = np.array(result["uv"], dtype=bool)
    ot = np.array(result["ot"], dtype=bool)
    fire = np.array(result["fire"], dtype=bool)

    if len(t) == 0 or not np.all(np.isfinite(t)):
        st.error("Invalid time vector returned from simulation.")
        st.stop()

    if np.any(np.diff(t) <= 0):
        st.warning(
            "Time vector is NOT strictly increasing. "
            "This typically indicates duplicate timestamps (e.g., two samples at t=0). "
            "Please check the runner logging."
        )

    dt = float(params["simulation"]["dt_s"])
    safety = compute_safety_metrics(result, dt)
    oper = compute_operational_metrics(result, dt)
    est = compute_estimation_metrics(result)

    st.subheader("Summary metrics")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown("**Safety**")
        st.write(f"t_fault_any_s : {fmt(safety.get('t_fault_any_s'))}")
        st.write(f"t_emergency_s : {fmt(safety.get('t_emergency_s'))}")
        st.write(f"t_uv_s        : {fmt(safety.get('t_uv_s'))}")
        st.write(f"t_ov_s        : {fmt(safety.get('t_ov_s'))}")
        st.write(f"t_oc_s        : {fmt(safety.get('t_oc_s'))}")
        st.write(f"t_ot_s        : {fmt(safety.get('t_ot_s'))}")
        st.write(f"t_fire_s      : {fmt(safety.get('t_fire_s'))}")
    with mcol2:
        st.markdown("**Operational / thermal**")
        st.write(f"Energy_discharge_kWh : {fmt(oper.get('energy_discharge_kwh'))}")
        st.write(f"soc_delta            : {fmt(oper.get('soc_delta'))}")
        st.write(f"t_discharge_s        : {fmt(oper.get('t_discharge_s'))}")
        st.write(f"t_max_c              : {fmt(oper.get('t_max_c'))}")
        st.write(f"delta_t_c            : {fmt(oper.get('delta_t_c'))}")
    with mcol3:
        st.markdown("**SoC estimation**")
        st.write(f"RMSE SoC             : {fmt(est.get('rmse_soc'), nd=5)}")
        st.write(f"Max abs error SoC    : {fmt(est.get('max_abs_error_soc'), nd=5)}")
        st.write(f"p95 abs error SoC    : {fmt(est.get('p95_abs_error_soc'), nd=5)}")

    # -------- PASS/FAIL (only when last run was a preset run) --------
    last_preset_key = st.session_state.get("offline_last_preset_key", None)
    if last_preset_key is not None and last_preset_key in preset_by_key:
        pr = preset_by_key[last_preset_key]
        overall_ok, rows = evaluate_preset_run(
            preset=pr,
            result=result,
            dt_s=float(dt_s),
            debounce_steps=int(DEBOUNCE_STEPS),
        )

        st.subheader("Preset validation (PASS/FAIL)")
        if overall_ok:
            st.success(f"PASS: {pr.title}")
        else:
            st.error(f"FAIL: {pr.title}")
        st.dataframe(rows, use_container_width=True)
        st.caption("Note: trigger-time checks use ~±1 sample tolerance based on dt.")

    if safety.get("t_fault_any_s", None) is None:
        st.success("No safety faults were triggered during this scenario.")
    else:
        msgs = []
        for key, name in [("t_uv_s", "UV"), ("t_ov_s", "OV"), ("t_oc_s", "OC"), ("t_ot_s", "OT"), ("t_fire_s", "FIRE")]:
            if safety.get(key, None) is not None:
                msgs.append(f"{name} at t = {fmt(safety[key])} s")
        st.error("Faults triggered: " + ", ".join(msgs))

    soc_err = soc_true - soc_hat
    soc_spread = soc_max - soc_min
    v_spread = v_max - v_min
    fault_any = (oc | ov | uv | ot | fire)

    soc_err_cc = None
    if soc_cc is not None and len(soc_cc) == len(soc_true):
        soc_err_cc = soc_true - soc_cc

    def first_time(mask: np.ndarray) -> Optional[float]:
        idx = np.where(mask)[0]
        return None if len(idx) == 0 else float(t[idx[0]])

    t_uv = first_time(uv)
    t_ov = first_time(ov)
    t_oc = first_time(oc)
    t_ot = first_time(ot)
    t_fire = first_time(fire)

    derate_mask = (np.abs(i_req - i_act) > 1e-6) if use_limits_used else np.zeros_like(i_req, dtype=bool)
    derate_start_idx = np.where((~derate_mask[:-1]) & (derate_mask[1:]))[0] + 1
    derate_start_times = t[derate_start_idx] if len(derate_start_idx) else np.array([])

    state_change_idx = np.where(state[1:] != state[:-1])[0] + 1
    state_change_times = t[state_change_idx] if len(state_change_idx) else np.array([])

    if segs_used is not None:
        t_amb_series = _build_piecewise_profile(t, segs_used, "t_amb_c")
    else:
        t_amb_series = np.full_like(t, np.nan)

    fig, axes = plt.subplots(7, 1, figsize=(10, 15), sharex=True)

    def vline(ax, x: Optional[float], label: str, style: str = "--"):
        if x is None:
            return
        ax.axvline(x, linestyle=style, linewidth=1.2, label=label)

    ax = axes[0]
    ax.plot(t, i_req, label="I_req [A]")
    ax.plot(t, i_act, "--", label="I_act [A]")
    for j, x in enumerate(derate_start_times):
        ax.axvline(x, linestyle=":", linewidth=1.0, label="Derating start" if j == 0 else "_nolegend_")
    ax.set_ylabel("Current [A]")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(t, soc_true, label="SoC true (rack)")
    ax.plot(t, soc_hat, "--", label="SoC EKF (post)")
    if soc_cc is not None:
        ax.plot(t, soc_cc, ":", label="SoC CC-only (from EKF init)")
    ax.set_ylabel("SoC [-]")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(t, soc_err, label="SoC error (true - EKF post)")
    if soc_err_cc is not None:
        ax.plot(t, soc_err_cc, ":", label="SoC error (true - CC-only)")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("SoC error [-]")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[3]
    ax.plot(t, v_min, label="V_cell min (used)")
    ax.plot(t, v_max, label="V_cell max (used)")
    if np.nanmax(np.abs(v_min_true - v_min)) > 1e-9 or np.nanmax(np.abs(v_max_true - v_max)) > 1e-9:
        ax.plot(t, v_min_true, ":", label="V_cell min (true)")
        ax.plot(t, v_max_true, ":", label="V_cell max (true)")
    ax.axhline(UV_THRESHOLD, linestyle="--", label="UV threshold")
    ax.axhline(OV_THRESHOLD, linestyle="--", label="OV threshold")
    ax.axhline(SOFT_MAX_V, linestyle=":", label="Soft max voltage")
    vline(ax, t_uv, "UV trigger")
    vline(ax, t_ov, "OV trigger")
    ax.set_ylabel("Cell voltage [V]")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[4]
    ax.plot(t, t_rack_true, label="T_rack true [°C]")
    ax.plot(t, t_rack_used, "--", label="T_rack used by BMS [°C]")
    if np.any(np.isfinite(t_amb_series)):
        ax.plot(t, t_amb_series, ":", label="T_amb [°C]")
    ax.axhline(OT_THRESHOLD, linestyle="--", label="OT threshold")
    ax.axhline(T_DERATE_START, linestyle=":", label="T derate start")
    ax.axhline(T_CUTOFF, linestyle="-.", label="T cutoff")
    vline(ax, t_ot, "OT trigger")
    ax.set_ylabel("Temperature [°C]")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[5]
    ax.plot(t, soc_spread, label="SoC spread (max-min)")
    ax.set_ylabel("SoC spread [-]")
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(t, v_spread, "--", label="V spread (max-min)")
    ax2.set_ylabel("V spread [V]")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    ax = axes[6]
    ax.step(t, state, where="post", label="BMS state code")
    ax.step(t, fault_any.astype(int), where="post", label="fault_any")
    for j, x in enumerate(state_change_times):
        ax.axvline(x, linestyle=":", linewidth=0.8, label="State change" if j == 0 else "_nolegend_")
    vline(ax, t_oc, "OC trigger")
    vline(ax, t_ot, "OT trigger")
    vline(ax, t_fire, "FIRE trigger")
    ax.set_ylabel("State / fault_any")
    ax.set_xlabel("Time [s]")
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    events = []
    for name, tx in [("UV", t_uv), ("OV", t_ov), ("OC", t_oc), ("OT", t_ot), ("FIRE", t_fire)]:
        if tx is not None:
            events.append({"event": name, "t_s": float(tx)})

    if events:
        st.subheader("Event timeline")
        st.dataframe(events, use_container_width=True)

with st.expander("How to read the plots (recommended)", expanded=False):
    st.markdown(
        """
**Currents**
- `I_req` is the requested current (from your segment profile).
- `I_act` is the applied current after BMS derating/limits and FSM state.
- Sign convention: **positive = discharging**, **negative = charging**.

**EKF vs CC-only**
- `SoC EKF (post)` is the EKF estimate you already log (after measurement update).
- `SoC CC-only` is a reference computed in this UI: it starts from EKF init SoC and integrates current only.
  It is shown to make the initial mismatch visible even if EKF corrects quickly.

**Used vs True**
- `V_cell min/max (true)`: computed from the plant model.
- `V_cell min/max (used)`: the signals used by BMS logic after overrides/injections.
- If you apply overrides/injections, **used can differ from true**.

**State timeline**
- `state_code`: OFF=0, RUN=1, FAULT=2, EMERGENCY=3
- `fault_any`: 1 if any of UV/OV/OT/UT/OC/FIRE is true (debounced)

**Derating markers**
- Vertical lines mark times where `I_act != I_req` (limits active).

**Trigger timing**
- With `debounce_steps=N` and `dt=dt_s`, expect first trigger at:
  `t_trigger ≈ t_inject + (N-1)*dt_s`
"""
    )
