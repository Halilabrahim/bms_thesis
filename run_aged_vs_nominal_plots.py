# run_aged_vs_nominal_plots.py

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_params
from src.scenarios import load_scenarios
from src.sim_runner import run_scenario


def get_scenario_by_id(scenarios, scen_id: str):
    for sc in scenarios:
        if sc.id == scen_id:
            return sc
    raise ValueError(f"Scenario with id '{scen_id}' not found.")


def main():
    # --- Parametreleri ve senaryoları yükle ---
    params = load_params()
    scenarios = load_scenarios()

    sc_nom = get_scenario_by_id(scenarios, "S1_nominal_25C_1C")
    sc_aged = get_scenario_by_id(scenarios, "S10_nominal_25C_1C_AGED")

    # --- Simülasyonları çalıştır ---
    res_nom = run_scenario(sc_nom, params)
    res_aged = run_scenario(sc_aged, params)

    t_nom = res_nom["time_s"]
    t_aged = res_aged["time_s"]

    soc_nom_true = res_nom["soc_true"]
    soc_nom_hat = res_nom["soc_hat"]

    soc_aged_true = res_aged["soc_true"]
    soc_aged_hat = res_aged["soc_hat"]

    i_nom = res_nom["i_act_a"]
    i_aged = res_aged["i_act_a"]

    t_rack_nom = res_nom["t_rack_c"]
    t_rack_aged = res_aged["t_rack_c"]

    v_nom = res_nom["v_rack_v"]
    v_aged = res_aged["v_rack_v"]

    # SoC hatası (EKF – true)
    soc_err_nom = [h - t for h, t in zip(soc_nom_hat, soc_nom_true)]
    soc_err_aged = [h - t for h, t in zip(soc_aged_hat, soc_aged_true)]

    # Kayıt klasörü
    fig_dir = Path("results") / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Şekil 1: SoC karşılaştırması ---
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t_nom, soc_nom_true, label="S1 – True SoC (nominal)")
    ax1.plot(t_nom, soc_nom_hat, "--", label="S1 – Estimated SoC (nominal)")

    ax1.plot(t_aged, soc_aged_true, label="S10 – True SoC (aged)")
    ax1.plot(t_aged, soc_aged_hat, "--", label="S10 – Estimated SoC (aged)")

    ax1.set_title("SoC comparison – S1 vs S10 (aged pack)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("SoC [-]")
    ax1.grid(True)
    ax1.legend(loc="best")
    fig1.tight_layout()
    fig1.savefig(fig_dir / "aged_vs_nominal_soc.png", dpi=150)

    # --- Şekil 2: SoC tahmin hatası ---
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t_nom, soc_err_nom, label="S1 – SoC error (hat - true)")
    ax2.plot(t_aged, soc_err_aged, label="S10 – SoC error (hat - true, aged)")

    ax2.set_title("SoC estimation error – S1 vs S10")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("SoC error [-]")
    ax2.grid(True)
    ax2.legend(loc="best")
    fig2.tight_layout()
    fig2.savefig(fig_dir / "aged_vs_nominal_soc_error.png", dpi=150)

    # --- Şekil 3: Akım ve terminal gerilimi ---
    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax3a.plot(t_nom, i_nom, label="S1 – I_act [A]")
    ax3a.plot(t_aged, i_aged, label="S10 – I_act [A] (aged)")
    ax3a.set_ylabel("Current [A]")
    ax3a.set_title("Rack current – S1 vs S10")
    ax3a.grid(True)
    ax3a.legend(loc="best")

    ax3b.plot(t_nom, v_nom, label="S1 – V_rack [V]")
    ax3b.plot(t_aged, v_aged, label="S10 – V_rack [V] (aged)")
    ax3b.set_xlabel("Time [s]")
    ax3b.set_ylabel("Voltage [V]")
    ax3b.set_title("Rack terminal voltage – S1 vs S10")
    ax3b.grid(True)
    ax3b.legend(loc="best")

    fig3.tight_layout()
    fig3.savefig(fig_dir / "aged_vs_nominal_current_voltage.png", dpi=150)

    # --- Şekil 4: Sıcaklık karşılaştırması ---
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(t_nom, t_rack_nom, label="S1 – T_rack [°C]")
    ax4.plot(t_aged, t_rack_aged, label="S10 – T_rack [°C] (aged)")
    ax4.set_title("Rack temperature – S1 vs S10")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Temperature [°C]")
    ax4.grid(True)
    ax4.legend(loc="best")
    fig4.tight_layout()
    fig4.savefig(fig_dir / "aged_vs_nominal_temperature.png", dpi=150)

    print(f"Figures saved to: {fig_dir.resolve()}")


if __name__ == "__main__":
    main()
