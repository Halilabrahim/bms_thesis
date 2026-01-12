# run_scenarios_with_metrics.py
from __future__ import annotations

import argparse
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from src.bms_logic import BMSControlParams, compute_current_limits
from src.config import load_params, PROFILES
from src.scenarios import load_scenarios, Scenario
from src.sim_runner import run_scenario
from src.metrics import (
    compute_safety_metrics,
    compute_operational_metrics,
    compute_estimation_metrics,
)


# -------------------------
# Small helpers
# -------------------------
def _f(x: Any, default: Optional[float] = None) -> Optional[float]:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "None"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _last(result: Dict[str, List[Any]], keys: List[str], default: Optional[float] = None) -> Optional[float]:
    """Try multiple keys, return last sample as float."""
    for k in keys:
        if k in result and isinstance(result[k], list) and len(result[k]) > 0:
            return _f(result[k][-1], default)
    return default


def _get_series(result: Dict[str, List[Any]], keys: List[str]) -> Optional[List[float]]:
    """Try multiple candidate keys and return the first matching time-series."""
    for k in keys:
        if k in result and isinstance(result[k], list) and len(result[k]) > 0:
            try:
                return [float(x) for x in result[k]]
            except Exception:
                return result[k]
    return None


def _stats(xs: List[float]) -> Tuple[float, float, float]:
    xs_f = [float(x) for x in xs]
    return (min(xs_f), sum(xs_f) / len(xs_f), max(xs_f))


def _fmt_stats(xs: Optional[List[float]]) -> str:
    if not xs:
        return "n/a"
    mn, av, mx = _stats(xs)
    return f"min={mn:.3f}, avg={av:.3f}, max={mx:.3f}"


def _mode_code(codes: List[float]) -> float:
    """Returns the most common limiter code (including 0.0 if dominant)."""
    c = Counter(int(round(float(x))) for x in codes)
    return float(c.most_common(1)[0][0]) if c else 0.0


def _mode_nonzero_code(codes: List[float]) -> float:
    """Returns most common non-zero limiter code; 0.0 if none."""
    c = Counter(int(round(float(x))) for x in codes if int(round(float(x))) != 0)
    return float(c.most_common(1)[0][0]) if c else 0.0


def _active_pct_and_seconds(codes: Optional[List[float]], dt_s: float) -> Tuple[Optional[float], Optional[float], int, int]:
    """
    Limiter active stats where 'active' means code != 0.
    Returns: (pct, seconds, n_active, n_total)
    """
    if not codes:
        return None, None, 0, 0
    n_total = len(codes)
    n_active = sum(1 for x in codes if int(round(float(x))) != 0)
    pct = 100.0 * float(n_active) / float(n_total) if n_total > 0 else 0.0
    sec = float(n_active) * float(dt_s)
    return pct, sec, n_active, n_total


# -------------------------
# BMS debug snapshot fallback
# -------------------------
def _build_bms_params(params_yaml: Dict[str, Any]) -> BMSControlParams:
    bc = params_yaml.get("bms_control", {}) or {}
    lim = params_yaml.get("limits", {}) or {}

    def opt(key: str) -> Optional[float]:
        return _f(bc.get(key), None)

    return BMSControlParams(
        # Rated current limits (prefer limits.* if present)
        i_charge_max_a=float(_f(lim.get("i_charge_max_a"), bc.get("i_charge_max_a")) or 0.0),
        i_discharge_max_a=float(_f(lim.get("i_discharge_max_a"), bc.get("i_discharge_max_a")) or 0.0),

        # SoC limits
        soc_low_cutoff=float(_f(bc.get("soc_low_cutoff"), 0.0) or 0.0),
        soc_low_derate_start=float(_f(bc.get("soc_low_derate_start"), 0.0) or 0.0),
        soc_high_cutoff=float(_f(bc.get("soc_high_cutoff"), 1.0) or 1.0),
        soc_high_derate_start=float(_f(bc.get("soc_high_derate_start"), 1.0) or 1.0),

        # Legacy low-temp fallback
        t_low_cutoff_c=float(_f(bc.get("t_low_cutoff_c"), lim.get("t_min_c")) or 0.0),
        t_low_derate_start_c=float(_f(bc.get("t_low_derate_start_c"), 0.0) or 0.0),

        # High-temp limits
        t_high_cutoff_c=float(_f(bc.get("t_high_cutoff_c"), lim.get("t_max_c")) or 0.0),
        t_high_derate_start_c=float(_f(bc.get("t_high_derate_start_c"), 0.0) or 0.0),

        # Split low-temp (optional)
        t_low_cutoff_discharge_c=opt("t_low_cutoff_discharge_c"),
        t_low_derate_start_discharge_c=opt("t_low_derate_start_discharge_c"),
        t_low_cutoff_charge_c=opt("t_low_cutoff_charge_c"),
        t_low_derate_start_charge_c=opt("t_low_derate_start_charge_c"),

        # Voltage limits for derating
        v_cell_min_v=float(_f(lim.get("v_cell_min_v"), 0.0) or 0.0),
        v_cell_max_v=float(_f(lim.get("v_cell_max_v"), 0.0) or 0.0),
        v_margin_v=float(_f(bc.get("v_margin_v"), 0.05) or 0.05),
        v_margin_low_v=_f(bc.get("v_margin_low_v"), None),
        v_margin_high_v=_f(bc.get("v_margin_high_v"), None),
    )


def compute_bms_debug_snapshot(result: Dict[str, List[Any]], params_yaml: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Compute BMS debug scalars from the last sample (fallback when not logged in result).
    """
    soc_hat = _last(result, ["soc_hat"])
    t_rack = _last(result, ["t_rack_c", "t_cell_c", "t_max_c"])

    # Prefer explicit cell min/max if available
    vmin = _last(result, ["v_cell_min_v", "v_cell_min", "vmin_cell_v"])
    vmax = _last(result, ["v_cell_max_v", "v_cell_max", "vmax_cell_v"])

    # If cell min/max are not logged, approximate from rack voltage / series count
    if vmin is None or vmax is None:
        v_rack = _last(result, ["v_rack_v", "v_dc_v", "v_pack_v"])
        struct = params_yaml.get("structure", {}) or {}
        n_series = int(struct.get("cells_in_series_per_pack", 0) or 0) * int(struct.get("packs_in_series_per_rack", 0) or 0)
        if v_rack is not None and n_series > 0:
            v_cell_est = float(v_rack) / float(n_series)
            if vmin is None:
                vmin = v_cell_est
            if vmax is None:
                vmax = v_cell_est

    if soc_hat is None or t_rack is None or vmin is None or vmax is None:
        return None

    bms_params = _build_bms_params(params_yaml)
    return compute_current_limits(
        soc_hat=float(soc_hat),
        t_rack_c=float(t_rack),
        v_cell_min=float(vmin),
        v_cell_max=float(vmax),
        params=bms_params,
    )


# -------------------------
# CLI glue
# -------------------------
def _default_profile() -> str:
    if "LUNA" in PROFILES:
        return "LUNA"
    return list(PROFILES.keys())[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scenario set and print BMS safety/operational/estimation metrics."
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default=_default_profile(),
        help="Select rack profile from src.config.PROFILES",
    )
    parser.add_argument("--params", default=None, help="Override params YAML path")
    parser.add_argument("--scenarios", default=None, help="Override scenarios YAML path")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated scenario IDs to run (e.g. GP_S1_nominal_25C_1C,GP_S12_peak_25C_320A_5min)",
    )
    parser.add_argument("--list", action="store_true", help="List scenarios and exit")
    return parser.parse_args()


def filter_scenarios(items: List[Scenario], only_csv: Optional[str]) -> List[Scenario]:
    if not only_csv:
        return items
    wanted = {s.strip() for s in only_csv.split(",") if s.strip()}
    return [sc for sc in items if sc.id in wanted]


def _print_bms_debug(result: Dict[str, List[Any]], params_yaml: Dict[str, Any], dt_s: float) -> None:
    """
    Print BMS derating debug. If time-series exist in result, show stats + code modes.
    Otherwise, compute snapshot from last sample and print single values.
    """
    print("\nBMS debug (derating/limits):")

    # Try time-series first (if sim logs them)
    scale_t_low_dis = _get_series(result, ["scale_t_low_dis", "scale_t_low_discharge"])
    scale_t_low_chg = _get_series(result, ["scale_t_low_chg", "scale_t_low_charge"])
    scale_t_high = _get_series(result, ["scale_t_high"])

    scale_dis_total = _get_series(result, ["scale_dis_total"])
    scale_chg_total = _get_series(result, ["scale_chg_total"])

    code_dis = _get_series(result, ["code_limit_dis"])
    code_chg = _get_series(result, ["code_limit_chg"])

    i_dis_allowed = _get_series(result, ["i_discharge_max_allowed", "i_discharge_max_allowed_a"])
    i_chg_allowed = _get_series(result, ["i_charge_max_allowed", "i_charge_max_allowed_a"])

    have_any_series = any([
        scale_t_low_dis, scale_t_low_chg, scale_t_high,
        scale_dis_total, scale_chg_total,
        code_dis, code_chg,
        i_dis_allowed, i_chg_allowed,
    ])

    def _p(name: str, val: str) -> None:
        print(f"  {name:<28}: {val}")

    if have_any_series:
        _p("scale_t_low_dis", _fmt_stats(scale_t_low_dis))
        _p("scale_t_low_chg", _fmt_stats(scale_t_low_chg))
        _p("scale_t_high", _fmt_stats(scale_t_high))
        _p("scale_dis_total", _fmt_stats(scale_dis_total))
        _p("scale_chg_total", _fmt_stats(scale_chg_total))

        if code_dis:
            _p("code_limit_dis (mode)", f"{_mode_code(code_dis):.0f}")
            _p("code_limit_dis (mode!=0)", f"{_mode_nonzero_code(code_dis):.0f}")
            _p("code_limit_dis (last)", f"{int(round(code_dis[-1]))}")

            pct, sec, n_act, n_tot = _active_pct_and_seconds(code_dis, dt_s)
            _p("dis limiter active", f"{pct:.1f}% ({n_act}/{n_tot}) => {sec:.1f} s")
        else:
            _p("code_limit_dis", "n/a")
            _p("dis limiter active", "n/a")

        if code_chg:
            _p("code_limit_chg (mode)", f"{_mode_code(code_chg):.0f}")
            _p("code_limit_chg (mode!=0)", f"{_mode_nonzero_code(code_chg):.0f}")
            _p("code_limit_chg (last)", f"{int(round(code_chg[-1]))}")

            pct, sec, n_act, n_tot = _active_pct_and_seconds(code_chg, dt_s)
            _p("chg limiter active", f"{pct:.1f}% ({n_act}/{n_tot}) => {sec:.1f} s")
        else:
            _p("code_limit_chg", "n/a")
            _p("chg limiter active", "n/a")

        _p("i_dis_allowed_A", _fmt_stats(i_dis_allowed))
        _p("i_chg_allowed_A", _fmt_stats(i_chg_allowed))
        return

    # Fallback snapshot from last sample
    dbg = compute_bms_debug_snapshot(result, params_yaml)
    if not dbg:
        _p("debug", "n/a (missing signals in result)")
        return

    _p("scale_t_low_dis (last)", fmt(dbg.get("scale_t_low_dis"), nd=6))
    _p("scale_t_low_chg (last)", fmt(dbg.get("scale_t_low_chg"), nd=6))
    _p("scale_t_high (last)", fmt(dbg.get("scale_t_high"), nd=6))
    _p("scale_dis_total (last)", fmt(dbg.get("scale_dis_total"), nd=6))
    _p("scale_chg_total (last)", fmt(dbg.get("scale_chg_total"), nd=6))
    _p("code_limit_dis (last)", f"{int(round(float(dbg.get('code_limit_dis', 0.0))))}")
    _p("code_limit_chg (last)", f"{int(round(float(dbg.get('code_limit_chg', 0.0))))}")
    _p("i_dis_allowed_A (last)", fmt(dbg.get("i_discharge_max_allowed"), nd=3))
    _p("i_chg_allowed_A (last)", fmt(dbg.get("i_charge_max_allowed"), nd=3))


def main() -> None:
    args = parse_args()

    profile = args.profile
    params_path = args.params or PROFILES[profile]["params"]
    scenarios_path = args.scenarios or PROFILES[profile]["scenarios"]

    print(f"\n=== Active profile: {profile} ===")
    print(f"Params    : {params_path}")
    print(f"Scenarios : {scenarios_path}")

    params = load_params(params_path)
    dt_s = float(params["simulation"]["dt_s"])

    scenarios = load_scenarios(scenarios_path)
    scenarios = filter_scenarios(scenarios, args.only)

    if args.list:
        print("\nScenario IDs:")
        for sc in scenarios:
            print(f"  - {sc.id}: {sc.description}")
        return

    if not scenarios:
        print("\nNo scenarios selected/found. Use --list to see available IDs.")
        return

    for sc in scenarios:
        print("\n" + "=" * 80)
        print(f"Scenario: {sc.id}")
        print(f"  {sc.description}")

        result = run_scenario(sc, params)

        safety = compute_safety_metrics(result, dt_s)
        oper = compute_operational_metrics(result, dt_s)
        est = compute_estimation_metrics(result)

        print("\nSafety metrics:")
        print(f"  t_fault_any_s        : {fmt(safety.get('t_fault_any_s'))}")
        print(f"  t_emergency_s        : {fmt(safety.get('t_emergency_s'))}")
        print(f"  t_oc_s               : {fmt(safety.get('t_oc_s'))}")
        print(f"  t_ov_s               : {fmt(safety.get('t_ov_s'))}")
        print(f"  t_uv_s               : {fmt(safety.get('t_uv_s'))}")
        print(f"  t_ot_s               : {fmt(safety.get('t_ot_s'))}")
        print(f"  t_ut_s               : {fmt(safety.get('t_ut_s'))}")
        print(f"  t_fire_s             : {fmt(safety.get('t_fire_s'))}")

        print("\nOperational metrics:")
        print(f"  v_rack_min_v         : {fmt(oper.get('v_rack_min_v'))}")
        print(f"  v_rack_max_v         : {fmt(oper.get('v_rack_max_v'))}")
        print(f"  i_rack_min_a         : {fmt(oper.get('i_rack_min_a'))}")
        print(f"  i_rack_max_a         : {fmt(oper.get('i_rack_max_a'))}")
        print(f"  t_rack_max_c         : {fmt(oper.get('t_rack_max_c'))}")
        print(f"  delta_t_c            : {fmt(oper.get('delta_t_c'))}")
        print(f"  energy_charge_kwh    : {fmt(oper.get('energy_charge_kwh'))}")
        print(f"  energy_net_kwh       : {fmt(oper.get('energy_net_kwh'))}")
        print(f"  energy_abs_kwh       : {fmt(oper.get('energy_abs_kwh'))}")
        print(f"  t_charge_s           : {fmt(oper.get('t_charge_s'))}")
        print(f"  energy_discharge_kwh : {fmt(oper.get('energy_discharge_kwh'))}")
        print(f"  soc_delta            : {fmt(oper.get('soc_delta'))}")
        print(f"  t_total_s            : {fmt(oper.get('t_total_s'))}")
        print(f"  t_discharge_s        : {fmt(oper.get('t_discharge_s'))}")

        print("\nEstimation (SoC) metrics:")
        print(f"  rmse_soc             : {fmt(est.get('rmse_soc'), nd=5)}")
        print(f"  max_abs_error_soc    : {fmt(est.get('max_abs_error_soc'), nd=5)}")
        print(f"  p95_abs_error_soc    : {fmt(est.get('p95_abs_error_soc'), nd=5)}")

        # Per-scenario BMS debug
        _print_bms_debug(result, params, dt_s)


if __name__ == "__main__":
    main()
