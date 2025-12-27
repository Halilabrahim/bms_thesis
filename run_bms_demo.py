from src.config import load_params
from src.battery_model import RackParams, RackECMModel
from src.thermal_model import ThermalParams, RackThermalModel
from src.estimator import SoCEstimatorEKF, EKFParams
from src.bms_logic import BMSControlParams, compute_current_limits
import matplotlib.pyplot as plt


def main():
    # --- Load config ---
    params = load_params()

    em = params["electrical_model"]
    struct = params["structure"]
    limits = params["limits"]
    therm = params["thermal_model"]
    sim_cfg = params["simulation"]
    ekf_cfg = params["estimation"]["ekf"]
    ctrl_cfg = params["bms_control"]

    # --- Build rack & thermal models ---
    rack_params = RackParams(
        q_nom_ah=float(em["q_nom_ah"]),
        r0_ohm=float(em["r0_ohm"]),
        r1_ohm=float(em["r1_ohm"]),
        c1_f=float(em["c1_f"]),
        r2_ohm=float(em["r2_ohm"]),
        c2_f=float(em["c2_f"]),
        n_packs=int(struct["packs_in_series_per_rack"]),
        cells_in_series_per_pack=int(struct["cells_in_series_per_pack"]),
        v_cell_min_v=float(limits["v_cell_min_v"]),
        v_cell_max_v=float(limits["v_cell_max_v"]),
    )
    ecm = RackECMModel(rack_params)

    thermal_params = ThermalParams(
        c_th_j_per_k=float(therm["c_th_j_per_k"]),
        r_th_k_per_w=float(therm["r_th_k_per_w"]),
        t_init_c=float(therm["t_init_c"]),
    )
    thermal = RackThermalModel(thermal_params)

    # --- EKF params & object ---
    n_series_cells = rack_params.n_packs * rack_params.cells_in_series_per_pack

    ekf_params = EKFParams(
        q_nom_ah=rack_params.q_nom_ah,
        r0_ohm=rack_params.r0_ohm,
        r1_ohm=rack_params.r1_ohm,
        c1_f=rack_params.c1_f,
        r2_ohm=rack_params.r2_ohm,
        c2_f=rack_params.c2_f,
        n_series_cells=n_series_cells,
        q_process=ekf_cfg["q_process"],
        r_meas_v=float(ekf_cfg["r_measurement"]["v_terminal"]),
        p0=ekf_cfg["initial_covariance"],
    )
    ekf = SoCEstimatorEKF(ekf_params, soc_init=1.0)

    # --- BMS control params ---
    bms_params = BMSControlParams(
        i_charge_max_a=float(limits["i_charge_max_a"]),
        i_discharge_max_a=float(limits["i_discharge_max_a"]),
        soc_low_cutoff=float(ctrl_cfg["soc_low_cutoff"]),
        soc_low_derate_start=float(ctrl_cfg["soc_low_derate_start"]),
        soc_high_cutoff=float(ctrl_cfg["soc_high_cutoff"]),
        soc_high_derate_start=float(ctrl_cfg["soc_high_derate_start"]),
        t_low_cutoff_c=float(ctrl_cfg["t_low_cutoff_c"]),
        t_low_derate_start_c=float(ctrl_cfg["t_low_derate_start_c"]),
        t_high_cutoff_c=float(ctrl_cfg["t_high_cutoff_c"]),
        t_high_derate_start_c=float(ctrl_cfg["t_high_derate_start_c"]),
        v_cell_min_v=float(limits["v_cell_min_v"]),
        v_cell_max_v=float(limits["v_cell_max_v"]),
        v_margin_v=float(ctrl_cfg["v_margin_v"]),
    )

    # --- Simulation setup ---
    dt_s = float(sim_cfg["dt_s"])      # 1 s
    t_end_s = float(sim_cfg["t_end_s"])  # 3600 s
    t_amb_c = 25.0
    i_req_dis_a = 320.0   # 1C discharge request

    times = []
    soc_true = []
    soc_hat_list = []
    v_rack_list = []
    t_rack_list = []
    i_req_list = []
    i_act_list = []

    # Reset models
    ecm.reset(soc=1.0)
    thermal.reset(t_init_c=t_amb_c)
    ekf.reset(soc_init=1.0)

    # İlk adım için başlangıç ölçümleri (dt=0, I=0)
    res = ecm.step(0.0, 0.0)
    t_rack_c = t_amb_c
    soc_hat = ekf.get_soc()

    n_steps = int(t_end_s / dt_s)
    for k in range(n_steps):
        t_s = k * dt_s

        # --- BMS current limits (SoC_hat, T, cell voltages) ---
        curr_limits = compute_current_limits(
            soc_hat=soc_hat,
            t_rack_c=t_rack_c,
            v_cell_min=res["v_cell_min"],
            v_cell_max=res["v_cell_max"],
            params=bms_params,
        )

        i_dis_max = curr_limits["i_discharge_max_allowed"]

        # Requested vs allowed current
        i_rack_a = min(i_req_dis_a, i_dis_max)
        # (ileride charge için negatif akım da ekleriz)

        # --- EKF prediction + true model + update ---
        ekf.predict(i_rack_a, dt_s)
        res = ecm.step(i_rack_a, dt_s)
        t_rack_c = thermal.step(res["p_loss"], t_amb_c, dt_s)
        ekf.update(res["v_rack"], i_rack_a)
        soc_hat = ekf.get_soc()

        # --- Log ---
        times.append(t_s)
        soc_true.append(res["soc"])
        soc_hat_list.append(soc_hat)
        v_rack_list.append(res["v_rack"])
        t_rack_list.append(t_rack_c)
        i_req_list.append(i_req_dis_a)
        i_act_list.append(i_rack_a)

    # --- Plots ---

    # 1) Current: requested vs actual
    fig0, ax0 = plt.subplots()
    ax0.plot(times, i_req_list, label="Requested current")
    ax0.plot(times, i_act_list, "--", label="Actual current (BMS limited)")
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Current [A]")
    ax0.grid(True)
    ax0.legend()

    # 2) Rack voltage
    fig1, ax1 = plt.subplots()
    ax1.plot(times, v_rack_list)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Rack voltage [V]")
    ax1.grid(True)

    # 3) SoC true vs estimated
    fig2, ax2 = plt.subplots()
    ax2.plot(times, soc_true, label="True SoC")
    ax2.plot(times, soc_hat_list, "--", label="Estimated SoC (EKF)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("SoC [-]")
    ax2.grid(True)
    ax2.legend()

    # 4) Rack temperature
    fig3, ax3 = plt.subplots()
    ax3.plot(times, t_rack_list)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Rack temperature [°C]")
    ax3.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
