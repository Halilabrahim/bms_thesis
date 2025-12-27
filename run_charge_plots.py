import matplotlib.pyplot as plt

from src.config import load_params
from src.scenarios import load_scenarios
from src.sim_runner import run_scenario


def main():
    # Parametreleri ve senaryoları yükle
    params = load_params()
    scenarios = {sc.id: sc for sc in load_scenarios()}

    # Şarj ve referans senaryoları
    scenario_ids = [
        "S1_nominal_25C_1C",   # referans: 1C deşarj
        "S8_CC_charge_25C",    # 0.5C şarj, 25°C
        "S9_CC_charge_0C",     # 0.5C şarj, 0°C
    ]

    for sid in scenario_ids:
        sc = scenarios[sid]
        result = run_scenario(sc, params)

        t = result["time_s"]
        soc_true = result["soc_true"]
        soc_hat = result["soc_hat"]
        i_req = result["i_req_a"]
        i_act = result["i_act_a"]

        # --- SoC grafiği ---
        plt.figure()
        plt.plot(t, soc_true, label="True SoC")
        plt.plot(t, soc_hat, "--", label="Estimated SoC")
        plt.xlabel("Time [s]")
        plt.ylabel("SoC [-]")
        plt.title(f"SoC – {sid}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # --- Akım grafiği ---
        plt.figure()
        plt.plot(t, i_req, label="Requested")
        plt.plot(t, i_act, "--", label="Actual (BMS)")
        plt.xlabel("Time [s]")
        plt.ylabel("Current [A]")
        plt.title(f"Current – {sid}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
