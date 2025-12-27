# run_ess_pv_demo.py

import numpy as np
import matplotlib.pyplot as plt

from src.ess_container import build_ess_container_from_yaml


def main():
    # 1 konteynerlik Huawei ESS inşa et
    ess = build_ess_container_from_yaml("data/params_rack.yaml")
    ess.reset(t_amb_c=25.0)

    dt_s = ess.dt_s

    # --- Senaryo parametreleri ---
    sim_hours = 6.0                # 6 saatlik basit bir gündüz senaryosu
    t_end_s = int(sim_hours * 3600)
    n_steps = int(t_end_s / dt_s)

    p_pv_peak_kw = 3000.0          # 3 MW tepe PV gücü (tek konteynerlik blok varsayımı)
    p_grid_target_kw = 1500.0      # Şebekeye sabit 1.5 MW vermek istiyoruz (peak shaving)

    t_values = []
    p_pv = []
    p_ess = []
    p_grid = []
    soc_mean = []
    soc_min = []
    t_rack = []

    for k in range(n_steps):
        t_s = k * dt_s

        # 1) Basit PV profili (sinüs benzeri çan eğrisi)
        #    t=0 ve t=t_end'de 0, ortada ~p_pv_peak_kw
        x = t_s / t_end_s
        p_pv_kw = max(0.0, np.sin(np.pi * x)) * p_pv_peak_kw

        # 2) Şebekeye sabit güç verme hedefi:
        #    P_grid_target = P_pv + P_ess  ->  P_ess_req = P_grid_target - P_pv
        p_ess_req_kw = p_grid_target_kw - p_pv_kw

        # Gücü akıma çevir (pozitif = deşarj, negatif = şarj)
        v_dc = ess.params.dc_bus_nominal_v
        if v_dc <= 0.0:
            raise ValueError("DC nominal gerilim sıfır olamaz.")
        i_ess_req_a = p_ess_req_kw * 1000.0 / v_dc

        # 3) ESS adımı (BMS limitleri burada devreye giriyor)
        res = ess.step(i_ess_req_a=i_ess_req_a, t_amb_c=25.0)

        # Gerçek BESS gücü
        p_ess_kw = res["p_ess_kw"]

        # 4) Şebekeye giden gerçek güç
        p_grid_kw = p_pv_kw + p_ess_kw

        # Log
        t_values.append(t_s / 3600.0)  # saat olarak
        p_pv.append(p_pv_kw)
        p_ess.append(p_ess_kw)
        p_grid.append(p_grid_kw)
        soc_mean.append(res["soc_rack_mean"])
        soc_min.append(res["soc_rack_min"])
        t_rack.append(res["t_rack_mean_c"])

    # ------------------ Plotlar ------------------

    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Güçler
    axs[0].plot(t_values, p_pv, label="P_PV [kW]")
    axs[0].plot(t_values, p_ess, label="P_ESS (Container) [kW]")
    axs[0].plot(t_values, p_grid, label="P_grid (PV+ESS) [kW]")
    axs[0].axhline(p_grid_target_kw, linestyle="--", label="P_grid target [kW]")
    axs[0].set_ylabel("Power [kW]")
    axs[0].legend()
    axs[0].grid(True)

    # SOC
    axs[1].plot(t_values, soc_mean, label="SOC mean")
    axs[1].plot(t_values, soc_min, label="SOC min")
    axs[1].set_ylabel("SOC [-]")
    axs[1].legend()
    axs[1].grid(True)

    # Sıcaklık
    axs[2].plot(t_values, t_rack, label="T_rack average [°C]")
    axs[2].set_ylabel("Temperature [°C]")
    axs[2].set_xlabel("Time [h]")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
