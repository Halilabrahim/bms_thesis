# export_plots_and_report.py
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.config import load_params, PROFILES
from src.scenarios import load_scenarios, Scenario
from src.sim_runner import run_scenario
from src.metrics import (
    compute_safety_metrics,
    compute_operational_metrics,
    compute_estimation_metrics,
)
from src.bms_logic import BMSControlParams, compute_current_limits


# -------------------------
# Utils
# -------------------------
def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _f(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _series(hist: Dict[str, List[Any]], key: str) -> Optional[np.ndarray]:
    v = hist.get(key, None)
    if v is None or not isinstance(v, list) or len(v) == 0:
        return None
    try:
        return np.asarray(v, dtype=float)
    except Exception:
        # fallback: best effort conversion
        out = []
        for x in v:
            try:
                out.append(float(x))
            except Exception:
                out.append(np.nan)
        return np.asarray(out, dtype=float)


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "None"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _default_profile() -> str:
    if "LUNA" in PROFILES:
        return "LUNA"
    return list(PROFILES.keys())[0]


def _filter_scenarios(items: List[Scenario], only_csv: Optional[str]) -> List[Scenario]:
    if not only_csv:
        return items
    wanted = {s.strip() for s in only_csv.split(",") if s.strip()}
    return [sc for sc in items if sc.id in wanted]


# -------------------------
# Build BMS params from YAML (robust, matches your run_scenarios_with_metrics.py)
# -------------------------
def _build_bms_params(params_yaml: Dict[str, Any]) -> BMSControlParams:
    bc = params_yaml.get("bms_control", {}) or {}
    lim = params_yaml.get("limits", {}) or {}

    def opt(key: str) -> Optional[float]:
        return _f(bc.get(key), None)

    return BMSControlParams(
        i_charge_max_a=float(_f(lim.get("i_charge_max_a"), bc.get("i_charge_max_a")) or 0.0),
        i_discharge_max_a=float(_f(lim.get("i_discharge_max_a"), bc.get("i_discharge_max_a")) or 0.0),

        soc_low_cutoff=float(_f(bc.get("soc_low_cutoff"), 0.0) or 0.0),
        soc_low_derate_start=float(_f(bc.get("soc_low_derate_start"), 0.0) or 0.0),
        soc_high_cutoff=float(_f(bc.get("soc_high_cutoff"), 1.0) or 1.0),
        soc_high_derate_start=float(_f(bc.get("soc_high_derate_start"), 1.0) or 1.0),

        # legacy low-temp fallback
        t_low_cutoff_c=float(_f(bc.get("t_low_cutoff_c"), lim.get("t_min_c")) or 0.0),
        t_low_derate_start_c=float(_f(bc.get("t_low_derate_start_c"), 0.0) or 0.0),

        t_high_cutoff_c=float(_f(bc.get("t_high_cutoff_c"), lim.get("t_max_c")) or 0.0),
        t_high_derate_start_c=float(_f(bc.get("t_high_derate_start_c"), 0.0) or 0.0),

        # split low-temp (optional)
        t_low_cutoff_discharge_c=opt("t_low_cutoff_discharge_c"),
        t_low_derate_start_discharge_c=opt("t_low_derate_start_discharge_c"),
        t_low_cutoff_charge_c=opt("t_low_cutoff_charge_c"),
        t_low_derate_start_charge_c=opt("t_low_derate_start_charge_c"),

        v_cell_min_v=float(_f(lim.get("v_cell_min_v"), 0.0) or 0.0),
        v_cell_max_v=float(_f(lim.get("v_cell_max_v"), 0.0) or 0.0),
        v_margin_v=float(_f(bc.get("v_margin_v"), 0.05) or 0.05),
        v_margin_low_v=_f(bc.get("v_margin_low_v"), None),
        v_margin_high_v=_f(bc.get("v_margin_high_v"), None),
    )


def compute_bms_debug_series(hist: Dict[str, List[Any]], params_yaml: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Guaranteed series for:
      scale_t_low_dis, scale_t_low_chg, scale_t_high,
      scale_dis_total, scale_chg_total,
      code_limit_dis, code_limit_chg,
      i_dis_allowed_A, i_chg_allowed_A
    If your sim already logs them -> use.
    Else -> compute from logged soc_hat, t_rack_used_c, v_cell_min_v, v_cell_max_v.
    """
    # If already logged, prefer them
    existing_keys = [
        "scale_t_low_dis", "scale_t_low_chg", "scale_t_high",
        "scale_dis_total", "scale_chg_total",
        "code_limit_dis", "code_limit_chg",
        "i_discharge_max_allowed", "i_charge_max_allowed",
        "i_discharge_max_allowed_a", "i_charge_max_allowed_a",
    ]
    any_exist = any((k in hist and isinstance(hist[k], list) and len(hist[k]) > 0) for k in existing_keys)

    out: Dict[str, np.ndarray] = {}

    if any_exist:
        # robust mapping
        out["scale_t_low_dis"] = _series(hist, "scale_t_low_dis") or np.array([])
        out["scale_t_low_chg"] = _series(hist, "scale_t_low_chg") or np.array([])
        out["scale_t_high"] = _series(hist, "scale_t_high") or np.array([])
        out["scale_dis_total"] = _series(hist, "scale_dis_total") or np.array([])
        out["scale_chg_total"] = _series(hist, "scale_chg_total") or np.array([])
        out["code_limit_dis"] = _series(hist, "code_limit_dis") or np.array([])
        out["code_limit_chg"] = _series(hist, "code_limit_chg") or np.array([])

        i_dis = _series(hist, "i_discharge_max_allowed") or _series(hist, "i_discharge_max_allowed_a")
        i_chg = _series(hist, "i_charge_max_allowed") or _series(hist, "i_charge_max_allowed_a")
        out["i_dis_allowed_A"] = i_dis if i_dis is not None else np.array([])
        out["i_chg_allowed_A"] = i_chg if i_chg is not None else np.array([])

        # If empty because keys not present, fall back to compute below
        if all(v.size > 0 for v in out.values()):
            return out

    # Compute fallback
    t = _series(hist, "time_s")
    soc_hat = _series(hist, "soc_hat")
    t_used = _series(hist, "t_rack_used_c") or _series(hist, "t_rack_meas_c") or _series(hist, "t_rack_c")
    vmin = _series(hist, "v_cell_min_v") or _series(hist, "v_cell_min_true_v")
    vmax = _series(hist, "v_cell_max_v") or _series(hist, "v_cell_max_true_v")

    if t is None or soc_hat is None or t_used is None or vmin is None or vmax is None:
        # Can't compute (missing signals)
        z = np.zeros(len(hist.get("time_s", [])), dtype=float)
        out = {
            "scale_t_low_dis": z.copy(),
            "scale_t_low_chg": z.copy(),
            "scale_t_high": z.copy(),
            "scale_dis_total": z.copy(),
            "scale_chg_total": z.copy(),
            "code_limit_dis": z.copy(),
            "code_limit_chg": z.copy(),
            "i_dis_allowed_A": z.copy(),
            "i_chg_allowed_A": z.copy(),
        }
        return out

    bms_params = _build_bms_params(params_yaml)

    n = len(t)
    scale_t_low_dis = np.zeros(n)
    scale_t_low_chg = np.zeros(n)
    scale_t_high = np.zeros(n)
    scale_dis_total = np.zeros(n)
    scale_chg_total = np.zeros(n)
    code_limit_dis = np.zeros(n)
    code_limit_chg = np.zeros(n)
    i_dis_allowed = np.zeros(n)
    i_chg_allowed = np.zeros(n)

    for k in range(n):
        lim = compute_current_limits(
            soc_hat=float(soc_hat[k]),
            t_rack_c=float(t_used[k]),
            v_cell_min=float(vmin[k]),
            v_cell_max=float(vmax[k]),
            params=bms_params,
        )
        scale_t_low_dis[k] = float(lim.get("scale_t_low_dis", np.nan))
        scale_t_low_chg[k] = float(lim.get("scale_t_low_chg", np.nan))
        scale_t_high[k] = float(lim.get("scale_t_high", np.nan))
        scale_dis_total[k] = float(lim.get("scale_dis_total", np.nan))
        scale_chg_total[k] = float(lim.get("scale_chg_total", np.nan))
        code_limit_dis[k] = float(lim.get("code_limit_dis", np.nan))
        code_limit_chg[k] = float(lim.get("code_limit_chg", np.nan))
        i_dis_allowed[k] = float(lim.get("i_discharge_max_allowed", np.nan))
        i_chg_allowed[k] = float(lim.get("i_charge_max_allowed", np.nan))

    out = {
        "scale_t_low_dis": scale_t_low_dis,
        "scale_t_low_chg": scale_t_low_chg,
        "scale_t_high": scale_t_high,
        "scale_dis_total": scale_dis_total,
        "scale_chg_total": scale_chg_total,
        "code_limit_dis": code_limit_dis,
        "code_limit_chg": code_limit_chg,
        "i_dis_allowed_A": i_dis_allowed,
        "i_chg_allowed_A": i_chg_allowed,
    }
    return out


# -------------------------
# Plotters
# -------------------------
def _save_fig(fig: plt.Figure, path: str, dpi: int = 180) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_current(hist: Dict[str, List[Any]], dbg: Dict[str, np.ndarray], title: str) -> plt.Figure:
    t = _series(hist, "time_s") or np.array([])
    i_req = _series(hist, "i_req_a") or np.array([])
    i_act = _series(hist, "i_act_a") or np.array([])

    i_dis_allowed = dbg.get("i_dis_allowed_A", np.array([]))
    i_chg_allowed = dbg.get("i_chg_allowed_A", np.array([]))

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, i_req, label="i_req (A)")
    ax.plot(t, i_act, label="i_act (A)")

    if i_dis_allowed.size == t.size and t.size > 0:
        ax.plot(t, i_dis_allowed, label="+i_dis_allowed (A)", linestyle="--")
    if i_chg_allowed.size == t.size and t.size > 0:
        ax.plot(t, -i_chg_allowed, label="-i_chg_allowed (A)", linestyle="--")

    ax.set_title(f"{title} — Current Tracking & Allowed Envelope")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("current (A)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig


def plot_soc(hist: Dict[str, List[Any]], title: str) -> plt.Figure:
    t = _series(hist, "time_s") or np.array([])
    soc_true = _series(hist, "soc_true") or np.array([])
    soc_hat = _series(hist, "soc_hat") or np.array([])

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, soc_true, label="SoC true")
    ax.plot(t, soc_hat, label="SoC EKF")
    if t.size == soc_true.size == soc_hat.size and t.size > 0:
        ax.plot(t, soc_true - soc_hat, label="SoC error (true-hat)", linestyle="--")

    ax.set_title(f"{title} — SoC (True vs EKF)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("SoC")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig


def plot_derating_scales(dbg: Dict[str, np.ndarray], hist: Dict[str, List[Any]], title: str) -> plt.Figure:
    t = _series(hist, "time_s") or np.array([])
    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    for k, lab in [
        ("scale_t_low_dis", "scale_t_low_dis"),
        ("scale_t_low_chg", "scale_t_low_chg"),
        ("scale_t_high", "scale_t_high"),
        ("scale_dis_total", "scale_dis_total"),
        ("scale_chg_total", "scale_chg_total"),
    ]:
        y = dbg.get(k, np.array([]))
        if y.size == t.size and t.size > 0:
            ax.plot(t, y, label=lab)

    ax.set_title(f"{title} — Derating Scales")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("scale (0..1)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    return fig


def plot_limiter_codes(dbg: Dict[str, np.ndarray], hist: Dict[str, List[Any]], title: str) -> plt.Figure:
    t = _series(hist, "time_s") or np.array([])
    code_dis = dbg.get("code_limit_dis", np.array([]))
    code_chg = dbg.get("code_limit_chg", np.array([]))

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    if code_dis.size == t.size and t.size > 0:
        ax.step(t, code_dis, where="post", label="code_limit_dis")
    if code_chg.size == t.size and t.size > 0:
        ax.step(t, code_chg, where="post", label="code_limit_chg")

    ax.set_title(f"{title} — Dominant Limiter Codes")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("code")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig


def plot_voltage(hist: Dict[str, List[Any]], title: str) -> plt.Figure:
    t = _series(hist, "time_s") or np.array([])
    v_rack = _series(hist, "v_rack_v") or np.array([])
    v_rack_meas = _series(hist, "v_rack_meas_v") or np.array([])

    vmin_used = _series(hist, "v_cell_min_v") or np.array([])
    vmax_used = _series(hist, "v_cell_max_v") or np.array([])
    vmin_true = _series(hist, "v_cell_min_true_v") or np.array([])
    vmax_true = _series(hist, "v_cell_max_true_v") or np.array([])

    fig = plt.figure(figsize=(10.5, 6.2))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, v_rack, label="v_rack")
    if v_rack_meas.size == t.size and t.size > 0:
        ax1.plot(t, v_rack_meas, label="v_rack_meas", linestyle="--")
    ax1.set_title(f"{title} — Rack Voltage")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("V")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(2, 1, 2)
    if vmin_true.size == t.size and t.size > 0:
        ax2.plot(t, vmin_true, label="v_cell_min_true")
    if vmax_true.size == t.size and t.size > 0:
        ax2.plot(t, vmax_true, label="v_cell_max_true")
    if vmin_used.size == t.size and t.size > 0:
        ax2.plot(t, vmin_used, label="v_cell_min_used", linestyle="--")
    if vmax_used.size == t.size and t.size > 0:
        ax2.plot(t, vmax_used, label="v_cell_max_used", linestyle="--")

    ax2.set_title(f"{title} — Cell Min/Max (True vs Used)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("V")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", ncol=2)

    fig.tight_layout()
    return fig


def plot_temperature(hist: Dict[str, List[Any]], title: str) -> plt.Figure:
    t = _series(hist, "time_s") or np.array([])
    t_true = _series(hist, "t_rack_c") or np.array([])
    t_meas = _series(hist, "t_rack_meas_c") or np.array([])
    t_used = _series(hist, "t_rack_used_c") or np.array([])

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    if t_true.size == t.size and t.size > 0:
        ax.plot(t, t_true, label="t_rack (model)")
    if t_meas.size == t.size and t.size > 0:
        ax.plot(t, t_meas, label="t_rack_meas", linestyle="--")
    if t_used.size == t.size and t.size > 0:
        ax.plot(t, t_used, label="t_rack_used (for faults)", linestyle=":")

    ax.set_title(f"{title} — Temperature")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("°C")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return fig


def plot_fsm_and_faults(hist: Dict[str, List[Any]], title: str) -> plt.Figure:
    t = _series(hist, "time_s") or np.array([])
    state = _series(hist, "state_code") or np.array([])

    oc = _series(hist, "oc") or np.array([])
    ov = _series(hist, "ov") or np.array([])
    uv = _series(hist, "uv") or np.array([])
    ot = _series(hist, "ot") or np.array([])
    ut = _series(hist, "ut") or np.array([])
    fire = _series(hist, "fire") or np.array([])

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    if state.size == t.size and t.size > 0:
        ax.step(t, state, where="post", label="FSM state_code")

    # plot fault flags as impulses at different y-levels
    def _imp(y: np.ndarray, level: float, label: str) -> None:
        if y.size == t.size and t.size > 0:
            idx = np.where(y > 0.5)[0]
            if idx.size > 0:
                ax.scatter(t[idx], np.full(idx.size, level), s=10, label=label)

    _imp(oc, 3.4, "OC")
    _imp(ov, 3.2, "OV")
    _imp(uv, 3.0, "UV")
    _imp(ot, 2.8, "OT")
    _imp(ut, 2.6, "UT")
    _imp(fire, 2.4, "FIRE")

    ax.set_title(f"{title} — FSM & Fault Flags")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("state / flags")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=3)
    return fig


# -------------------------
# PDF report pages (matplotlib text/table)
# -------------------------
def _table_page(pdf: PdfPages, title: str, rows: List[Tuple[str, str]], subtitle: Optional[str] = None) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    ax.text(0.0, 0.97, title, fontsize=16, fontweight="bold", transform=ax.transAxes, va="top")
    if subtitle:
        ax.text(0.0, 0.93, subtitle, fontsize=10, transform=ax.transAxes, va="top")

    # Build table
    cell_text = [[k, v] for (k, v) in rows]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=["Field", "Value"],
        colLoc="left",
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.25)

    pdf.savefig(fig)
    plt.close(fig)


def _title_page(pdf: PdfPages, title: str, lines: List[str]) -> None:
    fig = plt.figure(figsize=(11.0, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.text(0.0, 0.90, title, fontsize=20, fontweight="bold", transform=ax.transAxes, va="top")
    y = 0.80
    for ln in lines:
        ax.text(0.0, y, ln, fontsize=11, transform=ax.transAxes, va="top")
        y -= 0.04
    pdf.savefig(fig)
    plt.close(fig)


def _scenario_metrics_rows(sc: Scenario, safety: Dict[str, Any], oper: Dict[str, Any], est: Dict[str, Any]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    rows.append(("Scenario ID", sc.id))
    rows.append(("Description", getattr(sc, "description", "")))

    rows.append(("", ""))
    rows.append(("Safety t_fault_any_s", _fmt(safety.get("t_fault_any_s"))))
    rows.append(("Safety t_emergency_s", _fmt(safety.get("t_emergency_s"))))
    rows.append(("t_oc_s / t_ov_s / t_uv_s", f"{_fmt(safety.get('t_oc_s'))} / {_fmt(safety.get('t_ov_s'))} / {_fmt(safety.get('t_uv_s'))}"))
    rows.append(("t_ot_s / t_ut_s / t_fire_s", f"{_fmt(safety.get('t_ot_s'))} / {_fmt(safety.get('t_ut_s'))} / {_fmt(safety.get('t_fire_s'))}"))

    rows.append(("", ""))
    rows.append(("v_rack_min / max (V)", f"{_fmt(oper.get('v_rack_min_v'))} / {_fmt(oper.get('v_rack_max_v'))}"))
    rows.append(("i_rack_min / max (A)", f"{_fmt(oper.get('i_rack_min_a'))} / {_fmt(oper.get('i_rack_max_a'))}"))
    rows.append(("t_rack_max (°C)", _fmt(oper.get("t_rack_max_c"))))
    rows.append(("delta_t (°C)", _fmt(oper.get("delta_t_c"))))
    rows.append(("energy_net (kWh)", _fmt(oper.get("energy_net_kwh"))))
    rows.append(("energy_abs (kWh)", _fmt(oper.get("energy_abs_kwh"))))
    rows.append(("soc_delta", _fmt(oper.get("soc_delta"))))
    rows.append(("t_total (s)", _fmt(oper.get("t_total_s"))))

    rows.append(("", ""))
    rows.append(("RMSE SoC", _fmt(est.get("rmse_soc"), nd=5)))
    rows.append(("Max abs error SoC", _fmt(est.get("max_abs_error_soc"), nd=5)))
    rows.append(("P95 abs error SoC", _fmt(est.get("p95_abs_error_soc"), nd=5)))
    return rows


def _params_summary_rows(params: Dict[str, Any], profile: str, params_path: str, scenarios_path: str) -> List[Tuple[str, str]]:
    em = params.get("electrical_model", {}) or {}
    struct = params.get("structure", {}) or {}
    lim = params.get("limits", {}) or {}
    therm = params.get("thermal_model", {}) or {}
    bc = params.get("bms_control", {}) or {}
    faults = params.get("faults", {}) or {}

    n_series = int(struct.get("packs_in_series_per_rack", 0) or 0) * int(struct.get("cells_in_series_per_pack", 0) or 0)

    rows = [
        ("Active profile", profile),
        ("Params YAML", params_path),
        ("Scenarios YAML", scenarios_path),
        ("", ""),
        ("Structure: packs_in_series_per_rack", str(struct.get("packs_in_series_per_rack"))),
        ("Structure: cells_in_series_per_pack", str(struct.get("cells_in_series_per_pack"))),
        ("Derived: n_series_cells", str(n_series)),
        ("", ""),
        ("Electrical: q_nom_ah", str(em.get("q_nom_ah"))),
        ("Electrical: r0_ohm", str(em.get("r0_ohm"))),
        ("Electrical: r1_ohm / c1_f", f"{em.get('r1_ohm')} / {em.get('c1_f')}"),
        ("Electrical: r2_ohm / c2_f", f"{em.get('r2_ohm')} / {em.get('c2_f')}"),
        ("", ""),
        ("Limits: i_discharge_max_a", str(lim.get("i_discharge_max_a"))),
        ("Limits: i_charge_max_a", str(lim.get("i_charge_max_a"))),
        ("Limits: v_cell_min_v / v_cell_max_v", f"{lim.get('v_cell_min_v')} / {lim.get('v_cell_max_v')}"),
        ("Limits: t_min_c / t_max_c", f"{lim.get('t_min_c')} / {lim.get('t_max_c')}"),
        ("", ""),
        ("BMS: soc_low_cutoff / start", f"{bc.get('soc_low_cutoff')} / {bc.get('soc_low_derate_start')}"),
        ("BMS: soc_high_cutoff / start", f"{bc.get('soc_high_cutoff')} / {bc.get('soc_high_derate_start')}"),
        ("BMS: t_low_cutoff_c / start (legacy)", f"{bc.get('t_low_cutoff_c')} / {bc.get('t_low_derate_start_c')}"),
        ("BMS: t_low_cutoff_discharge_c / start", f"{bc.get('t_low_cutoff_discharge_c')} / {bc.get('t_low_derate_start_discharge_c')}"),
        ("BMS: t_low_cutoff_charge_c / start", f"{bc.get('t_low_cutoff_charge_c')} / {bc.get('t_low_derate_start_charge_c')}"),
        ("BMS: t_high_cutoff_c / start", f"{bc.get('t_high_cutoff_c')} / {bc.get('t_high_derate_start_c')}"),
        ("BMS: v_margin_v (or split)", f"{bc.get('v_margin_v')} (low={bc.get('v_margin_low_v')}, high={bc.get('v_margin_high_v')})"),
        ("", ""),
        ("Thermal: c_th_j_per_k", str(therm.get("c_th_j_per_k"))),
        ("Thermal: r_th_k_per_w", str(therm.get("r_th_k_per_w"))),
        ("Thermal: t_init_c", str(therm.get("t_init_c"))),
        ("", ""),
        ("Faults: ov_cell_v / uv_cell_v", f"{faults.get('ov_cell_v')} / {faults.get('uv_cell_v')}"),
        ("Faults: ot_rack_c / ut_rack_c", f"{faults.get('ot_rack_c')} / {faults.get('ut_rack_c')}"),
        ("Faults: oc_discharge_a / oc_charge_a", f"{faults.get('oc_discharge_a')} / {faults.get('oc_charge_a')}"),
    ]
    return rows


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export plots + generate a single PDF report for scenarios.")
    p.add_argument("--profile", choices=list(PROFILES.keys()), default=_default_profile())
    p.add_argument("--params", default=None, help="Override params YAML path")
    p.add_argument("--scenarios", default=None, help="Override scenarios YAML path")
    p.add_argument("--only", default=None, help="Comma-separated scenario IDs")
    p.add_argument("--out", default="out_reports", help="Output root directory")
    p.add_argument("--dpi", type=int, default=180, help="PNG DPI")
    p.add_argument("--no_pdf", action="store_true", help="Do not generate PDF, only PNGs + metrics")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    profile = args.profile
    params_path = args.params or PROFILES[profile]["params"]
    scenarios_path = args.scenarios or PROFILES[profile]["scenarios"]

    params = load_params(params_path)
    dt_s = float(params["simulation"]["dt_s"])

    scenarios = load_scenarios(scenarios_path)
    scenarios = _filter_scenarios(scenarios, args.only)

    if not scenarios:
        print("No scenarios selected/found.")
        return

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.out, profile, ts)
    _mkdir(out_root)

    # Save meta
    meta = {
        "profile": profile,
        "params_path": params_path,
        "scenarios_path": scenarios_path,
        "dt_s": dt_s,
        "generated_at": ts,
        "scenario_ids": [sc.id for sc in scenarios],
    }
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Metrics summary (CSV + JSON)
    metrics_rows: List[Dict[str, Any]] = []

    pdf_path = os.path.join(out_root, f"BMS_Report_{profile}_{ts}.pdf")
    pdf = None if args.no_pdf else PdfPages(pdf_path)

    try:
        if pdf is not None:
            _title_page(
                pdf,
                title=f"BMS Offline Simulation Report — {profile}",
                lines=[
                    f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Params: {params_path}",
                    f"Scenarios: {scenarios_path}",
                    "",
                    "Contents:",
                    "  1) Rack & BMS configuration summary",
                    "  2) Scenario test plan (IDs + descriptions)",
                    "  3) Results: metrics + plots per scenario",
                ],
            )

            _table_page(
                pdf,
                title="Rack & BMS Configuration Summary",
                subtitle="Extracted directly from the active params YAML used in simulations.",
                rows=_params_summary_rows(params, profile, params_path, scenarios_path),
            )

            # Scenario list page
            scen_rows = [("Scenario ID", "Description")]
            for sc in scenarios:
                scen_rows.append((sc.id, getattr(sc, "description", "")))
            _table_page(
                pdf,
                title="Scenario Test Plan",
                subtitle="Scenarios executed in this report.",
                rows=scen_rows[1:],  # table_page adds its own header
            )

        # Run all scenarios
        for sc in scenarios:
            print(f"Running: {sc.id}")
            sc_dir = os.path.join(out_root, sc.id)
            _mkdir(sc_dir)

            hist = run_scenario(sc, params)

            safety = compute_safety_metrics(hist, dt_s)
            oper = compute_operational_metrics(hist, dt_s)
            est = compute_estimation_metrics(hist)

            # store scenario metrics
            row = {
                "scenario_id": sc.id,
                "description": getattr(sc, "description", ""),
                **{f"safety_{k}": safety.get(k) for k in safety.keys()},
                **{f"oper_{k}": oper.get(k) for k in oper.keys()},
                **{f"est_{k}": est.get(k) for k in est.keys()},
            }
            metrics_rows.append(row)

            with open(os.path.join(sc_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"safety": safety, "operational": oper, "estimation": est}, f, indent=2)

            # compute debug series (always available)
            dbg = compute_bms_debug_series(hist, params)

            # ---- plots (PNG + PDF) ----
            figs = []
            figs.append(plot_current(hist, dbg, sc.id))
            figs.append(plot_soc(hist, sc.id))
            figs.append(plot_derating_scales(dbg, hist, sc.id))
            figs.append(plot_limiter_codes(dbg, hist, sc.id))
            figs.append(plot_voltage(hist, sc.id))
            figs.append(plot_temperature(hist, sc.id))
            figs.append(plot_fsm_and_faults(hist, sc.id))

            png_names = [
                "01_current.png",
                "02_soc.png",
                "03_derating_scales.png",
                "04_limiter_codes.png",
                "05_voltage.png",
                "06_temperature.png",
                "07_fsm_faults.png",
            ]
            for fig, name in zip(figs, png_names):
                _save_fig(fig, os.path.join(sc_dir, name), dpi=args.dpi)

            if pdf is not None:
                # Metrics page
                _table_page(
                    pdf,
                    title=f"Results — {sc.id}",
                    subtitle="Key metrics computed from the simulation time-series.",
                    rows=_scenario_metrics_rows(sc, safety, oper, est),
                )

                # Add plots into PDF as pages
                for name in png_names:
                    img_path = os.path.join(sc_dir, name)
                    fig = plt.figure(figsize=(11.0, 8.5))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.axis("off")
                    ax.set_title(f"{sc.id} — {name}", fontsize=12)
                    img = plt.imread(img_path)
                    ax.imshow(img)
                    pdf.savefig(fig)
                    plt.close(fig)

        # Write metrics summary to root
        with open(os.path.join(out_root, "metrics_summary.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_rows, f, indent=2)

        # CSV
        csv_path = os.path.join(out_root, "metrics_summary.csv")
        if metrics_rows:
            keys = sorted(metrics_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(metrics_rows)

        print(f"\nDone. Output folder:\n  {out_root}")
        if pdf is not None:
            print(f"PDF report:\n  {pdf_path}")

    finally:
        if pdf is not None:
            pdf.close()


if __name__ == "__main__":
    main()
