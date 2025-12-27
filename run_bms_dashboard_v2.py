# run_bms_dashboard_v2.py
# Offline dashboard plotter for BMS scenario .npz files
#
# Improvements vs v1:
# - Supports both preferred and legacy fault-flag keys
# - Adds SoC error subplot
# - Adds spread subplot (SoC spread + V spread)
# - Uses step plots for discrete state/fault signals (better alignment)
# - Adds vertical markers at first fault detection times

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt


def _first_true_time(t: np.ndarray, flag: np.ndarray):
    idx = np.where(flag > 0.5)[0]
    return None if idx.size == 0 else float(t[idx[0]])


def plot_bms_dashboard(filepath: str) -> None:
    """
    Load a rack simulation (.npz) and plot a dashboard with:
      1) Currents (I_req / I_act)
      2) SoC (true / EKF + cell min/mean/max)
      3) SoC error (true - EKF)
      4) Cell voltages (min/max)
      5) Spread panel (SoC spread + V spread)
      6) State code + fault flags

    Vertical markers are drawn at first detection of each fault, if present.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    data = np.load(filepath)

    # --- Required signals ---
    t = np.asarray(data["time_s"], dtype=float)

    i_req = np.asarray(data["i_req_a"], dtype=float)
    i_act = np.asarray(data["i_act_a"], dtype=float)

    soc_true = np.asarray(data["soc_true"], dtype=float)
    soc_hat = np.asarray(data["soc_hat"], dtype=float)

    soc_min = np.asarray(data["soc_cell_min"], dtype=float) if "soc_cell_min" in data.files else soc_true
    soc_mean = np.asarray(data["soc_cell_mean"], dtype=float) if "soc_cell_mean" in data.files else soc_true
    soc_max = np.asarray(data["soc_cell_max"], dtype=float) if "soc_cell_max" in data.files else soc_true

    v_min = np.asarray(data["v_cell_min_v"], dtype=float) if "v_cell_min_v" in data.files else np.full_like(t, np.nan)
    v_max = np.asarray(data["v_cell_max_v"], dtype=float) if "v_cell_max_v" in data.files else np.full_like(t, np.nan)

    t_rack = np.asarray(data["t_rack_c"], dtype=float) if "t_rack_c" in data.files else np.full_like(t, np.nan)

    state_code = np.asarray(data["state_code"], dtype=int) if "state_code" in data.files else np.zeros_like(t, dtype=int)

    # --- Fault flags (support both key conventions) ---
    faults = {}

    preferred = ["oc", "ov", "uv", "ot", "fire"]
    legacy = ["fault_oc", "fault_ov", "fault_uv", "fault_ot", "fault_fire"]

    for k in preferred:
        if k in data.files:
            faults[k] = np.asarray(data[k], dtype=float)

    if len(faults) == 0:
        for k in legacy:
            if k in data.files:
                faults[k] = np.asarray(data[k], dtype=float)

    # --- Styling meta ---
    fault_meta = {
        "oc": {"label": "OC", "color": "tab:orange"},
        "ov": {"label": "OV", "color": "tab:green"},
        "uv": {"label": "UV", "color": "tab:red"},
        "ot": {"label": "OT", "color": "tab:purple"},
        "fire": {"label": "FIRE", "color": "tab:brown"},
        "fault_oc": {"label": "OC", "color": "tab:orange"},
        "fault_ov": {"label": "OV", "color": "tab:green"},
        "fault_uv": {"label": "UV", "color": "tab:red"},
        "fault_ot": {"label": "OT", "color": "tab:purple"},
        "fault_fire": {"label": "FIRE", "color": "tab:brown"},
    }

    # --- Derived signals ---
    soc_err = soc_true - soc_hat
    soc_spread = soc_max - soc_min
    v_spread = v_max - v_min

    # --- Fault times ---
    fault_times = {}
    for name, arr in faults.items():
        t_f = _first_true_time(t, arr)
        if t_f is not None:
            fault_times[name] = t_f

    # --- Plot layout ---
    fig, axes = plt.subplots(6, 1, figsize=(13.5, 8.5), sharex=True)
    ax_i, ax_soc, ax_err, ax_v, ax_spread, ax_state = axes

    # 1) Currents
    ax_i.plot(t, i_req, label="I_req [A]")
    ax_i.plot(t, i_act, "--", label="I_act [A]")
    ax_i.set_ylabel("Current [A]")
    ax_i.grid(True)
    ax_i.legend(loc="upper right")

    # 2) SoC
    ax_soc.plot(t, soc_true, label="SoC true (rack)")
    ax_soc.plot(t, soc_hat, "--", label="SoC EKF")
    ax_soc.plot(t, soc_min, ":", label="cell SoC min")
    ax_soc.plot(t, soc_mean, ":", label="cell SoC mean")
    ax_soc.plot(t, soc_max, ":", label="cell SoC max")
    ax_soc.set_ylabel("SoC [-]")
    ax_soc.grid(True)
    ax_soc.legend(loc="upper right")

    # 3) SoC error
    ax_err.plot(t, soc_err, label="SoC error (true - EKF)")
    ax_err.axhline(0.0, linestyle="--", linewidth=1.0)
    ax_err.set_ylabel("SoC err [-]")
    ax_err.grid(True)
    ax_err.legend(loc="upper right")

    # 4) Cell voltages
    ax_v.plot(t, v_min, label="V_cell min")
    ax_v.plot(t, v_max, label="V_cell max")
    ax_v.set_ylabel("Cell V [V]")
    ax_v.grid(True)
    ax_v.legend(loc="upper right")

    # 5) Spread panel
    ax_spread.plot(t, soc_spread, label="SoC spread (max-min)")
    ax_spread.set_ylabel("SoC spread [-]")
    ax_spread.grid(True)

    ax_spread2 = ax_spread.twinx()
    ax_spread2.plot(t, v_spread, "--", label="V spread (max-min)")
    ax_spread2.set_ylabel("V spread [V]")

    l1, lab1 = ax_spread.get_legend_handles_labels()
    l2, lab2 = ax_spread2.get_legend_handles_labels()
    ax_spread.legend(l1 + l2, lab1 + lab2, loc="upper right")

    # 6) State + faults (step for discrete signals)
    ax_state.step(t, state_code, where="post", label="BMS state code")
    for name, arr in faults.items():
        meta = fault_meta.get(name, {})
        lbl = meta.get("label", name)
        col = meta.get("color", None)
        ax_state.step(t, (arr > 0.5).astype(int), where="post", label=lbl, color=col)

    ax_state.set_ylabel("State / faults")
    ax_state.set_xlabel("Time [s]")
    ax_state.grid(True)
    ax_state.legend(loc="upper right")

    # --- Vertical markers for fault times on all axes ---
    for name, t_f in fault_times.items():
        meta = fault_meta.get(name, {})
        col = meta.get("color", "k")
        lbl = meta.get("label", name)

        for ax in axes:
            ax.axvline(t_f, color=col, linestyle="--", alpha=0.5)

        y_max = ax_state.get_ylim()[1]
        ax_state.text(
            t_f,
            y_max * 0.97,
            lbl,
            color=col,
            rotation=90,
            va="top",
            ha="right",
        )

    fig.suptitle(f"BMS dashboard – {filepath.name}", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    out_png = filepath.with_suffix(".png")
    plt.savefig(out_png, dpi=150)
    print(f"Dashboard saved to: {out_png}")

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        fname = "results/random/R1_random_25C.npz"
    plot_bms_dashboard(fname)
