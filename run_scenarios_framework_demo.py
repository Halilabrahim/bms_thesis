from src.config import load_params
from src.scenarios import load_scenarios
from src.sim_runner import run_scenario
import matplotlib.pyplot as plt


def main():
    params = load_params()
    scenarios = load_scenarios()

    # Örnek olarak: S1 ve S4'ü çalıştır
    ids_to_run = ["S1_nominal_25C_1C", "S4_OC_fault_25C"]

    for sc in scenarios:
        if sc.id not in ids_to_run:
            continue

        print(f"Running scenario {sc.id}: {sc.description}")
        result = run_scenario(sc, params)

        # Basit özet
        print(f"  Final SoC: {result['soc_true'][-1]:.3f}")
        print(f"  Final state code: {result['state_code'][-1]}")

        # Birkaç grafik çizelim (SoC, current, state)
        fig, ax = plt.subplots()
        ax.plot(result["time_s"], result["soc_true"], label="True SoC")
        ax.plot(result["time_s"], result["soc_hat"], "--", label="Estimated SoC")
        ax.set_title(f"SoC – {sc.id}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("SoC [-]")
        ax.grid(True)
        ax.legend()

        fig2, ax2 = plt.subplots()
        ax2.plot(result["time_s"], result["i_req_a"], label="Requested")
        ax2.plot(result["time_s"], result["i_act_a"], "--", label="Actual (BMS)")
        ax2.set_title(f"Current – {sc.id}")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Current [A]")
        ax2.grid(True)
        ax2.legend()

        fig3, ax3 = plt.subplots()
        ax3.step(result["time_s"], result["state_code"], where="post")
        ax3.set_title(f"BMS state – {sc.id}")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("State code (0 OFF,1 RUN,2 FAULT,3 EMERGENCY)")
        ax3.grid(True)

        plt.show()


if __name__ == "__main__":
    main()
