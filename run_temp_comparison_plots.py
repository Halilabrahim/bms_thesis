from src.config import load_params
from src.scenarios import load_scenarios
from src.sim_runner import run_scenario
import matplotlib.pyplot as plt


def main():
    params = load_params()
    scenarios = load_scenarios()

    # Sadece S1, S2, S3'ü kullan
    ids = ["S1_nominal_25C_1C", "S2_nominal_0C_1C", "S3_nominal_45C_1C"]
    label_map = {
        "S1_nominal_25C_1C": "25°C ambient",
        "S2_nominal_0C_1C": "0°C ambient",
        "S3_nominal_45C_1C": "45°C ambient",
    }

    results = {}
    for sc in scenarios:
        if sc.id in ids:
            print(f"Running {sc.id}")
            res = run_scenario(sc, params)
            results[sc.id] = res

    # --- SoC vs time (üç senaryoyu üst üste) ---
    fig_soc, ax_soc = plt.subplots()
    for sid in ids:
        res = results[sid]
        ax_soc.plot(res["time_s"], res["soc_true"], label=label_map[sid])
    ax_soc.set_title("Rack SoC for different ambient temperatures")
    ax_soc.set_xlabel("Time [s]")
    ax_soc.set_ylabel("SoC [-]")
    ax_soc.grid(True)
    ax_soc.legend()

    # --- Temperature vs time ---
    fig_t, ax_t = plt.subplots()
    for sid in ids:
        res = results[sid]
        ax_t.plot(res["time_s"], res["t_rack_c"], label=label_map[sid])
    ax_t.set_title("Rack temperature for different ambient temperatures")
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel("Rack temperature [°C]")
    ax_t.grid(True)
    ax_t.legend()

    plt.show()


if __name__ == "__main__":
    main()
