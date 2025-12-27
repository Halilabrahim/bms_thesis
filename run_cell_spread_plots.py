# run_cell_spread_plots.py
import pathlib

import matplotlib.pyplot as plt

from src.config import load_params
from src.scenarios import load_scenarios, Scenario
from src.sim_runner import run_scenario

FIG_DIR = pathlib.Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def run_and_plot_cell_spread(sc: Scenario, params):
    """
    Verilen senaryo için simülasyonu çalıştırır ve
    hücre SoC / gerilim spread grafiklerini kaydeder.
    """
    print(f"Running cell spread plots for {sc.id} – {sc.description}")

    result = run_scenario(sc, params)

    t = result["time_s"]
    soc_min = result["soc_cell_min"]
    soc_max = result["soc_cell_max"]
    soc_mean = result["soc_cell_mean"]
    v_min = result["v_cell_min_v"]
    v_max = result["v_cell_max_v"]

    # --- SoC spread ---
    plt.figure(figsize=(7, 4))
    plt.plot(t, soc_min, label="Min cell SoC")
    plt.plot(t, soc_mean, label="Mean cell SoC")
    plt.plot(t, soc_max, label="Max cell SoC")
    plt.xlabel("Time [s]")
    plt.ylabel("SoC [-]")
    plt.title(f"Cell SoC spread – {sc.id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"soc_cell_spread_{sc.id}.png", dpi=150)
    plt.close()

    # --- Voltage spread ---
    plt.figure(figsize=(7, 4))
    plt.plot(t, v_min, label="Min cell voltage")
    plt.plot(t, v_max, label="Max cell voltage")
    plt.xlabel("Time [s]")
    plt.ylabel("Cell voltage [V]")
    plt.title(f"Cell voltage spread – {sc.id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"v_cell_spread_{sc.id}.png", dpi=150)
    plt.close()


def main():
    # Tüm senaryoları oku
    params = load_params()
    scenarios = load_scenarios()

    # Hepsi için sırayla grafik üret
    for sc in scenarios:
        run_and_plot_cell_spread(sc, params)


if __name__ == "__main__":
    main()
