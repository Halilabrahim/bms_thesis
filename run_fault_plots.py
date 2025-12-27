import matplotlib.pyplot as plt

from src.config import load_params
from src.scenarios import load_scenarios
from src.sim_runner import run_scenario


def plot_fault_scenario(scenario_id: str, save_figs: bool = False) -> None:
    """Run one fault scenario and plot SoC, current, state & fire flag."""
    params = load_params()
    scenarios = load_scenarios()

    # İstenen ID'yi senaryolardan bul
    id_to_sc = {sc.id: sc for sc in scenarios}
    if scenario_id not in id_to_sc:
        raise ValueError(f"Scenario '{scenario_id}' not found in scenarios.yaml")

    sc = id_to_sc[scenario_id]
    result = run_scenario(sc, params)

    t = result["time_s"]
    soc_true = result["soc_true"]
    soc_hat = result["soc_hat"]
    i_req = result["i_req_a"]
    i_act = result["i_act_a"]
    state_code = result["state_code"]
    fire_flag = result["fire"]

    # --- SoC plot ---
    fig1, ax1 = plt.subplots()
    ax1.plot(t, soc_true, label="True SoC")
    ax1.plot(t, soc_hat, "--", label="Estimated SoC")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("SoC [-]")
    ax1.set_title(f"SoC – {sc.id}")
    ax1.grid(True)
    ax1.legend()

    # --- Current plot ---
    fig2, ax2 = plt.subplots()
    ax2.plot(t, i_req, label="Requested")
    ax2.plot(t, i_act, "--", label="Actual (BMS)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Current [A]")
    ax2.set_title(f"Current – {sc.id}")
    ax2.grid(True)
    ax2.legend()

    # --- BMS state plot ---
    fig3, ax3 = plt.subplots()
    ax3.step(t, state_code, where="post")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("State code (0 OFF, 1 RUN, 2 FAULT, 3 EMERGENCY)")
    ax3.set_title(f"BMS state – {sc.id}")
    ax3.grid(True)

    # --- Fire flag plot (özellikle S7 için anlamlı) ---
    fig4, ax4 = plt.subplots()
    ax4.step(t, fire_flag, where="post")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Fire flag [-]")
    ax4.set_title(f"Fire detection – {sc.id}")
    ax4.grid(True)

    if save_figs:
        import pathlib

        out_dir = pathlib.Path("results/figures")
        out_dir.mkdir(parents=True, exist_ok=True)
        fig1.savefig(out_dir / f"SoC_{sc.id}.png", dpi=150)
        fig2.savefig(out_dir / f"Current_{sc.id}.png", dpi=150)
        fig3.savefig(out_dir / f"State_{sc.id}.png", dpi=150)
        fig4.savefig(out_dir / f"Fire_{sc.id}.png", dpi=150)

    # show hepsini
    plt.show()


def main():
    # İstersen buraya liste koyup hepsini çizdirebilirsin:
    fault_ids = ["S4_OC_fault_25C", "S5_UV_fault_25C", "S6_OT_fault_45C", "S7_FIRE_event_25C"]
    for fid in fault_ids:
        plot_fault_scenario(fid)

    # Şimdilik sadece S7'yi çalıştır:
    plot_fault_scenario("S7_FIRE_event_25C", save_figs=False)


if __name__ == "__main__":
    main()
