from src.config import load_params
from src.battery_model import RackParams, RackECMModel
from src.thermal_model import ThermalParams, RackThermalModel
from src.estimator import SoCEstimatorEKF, EKFParams
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

    # --- Build rack & thermal parameter objects ---
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

    # --- EKF parameter object ---
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

    # --- Simple simulation setup ---
    dt_s = float(sim_cfg["dt_s"])
    t_end_s = float(sim_cfg["t_end_s"])
    t_amb_c = 25.0
    i_rack_a = 160.0   # ~0.5C discharge

    times = []
    socs = []
    soc_hats = []
    v_racks = []
    t_racks = []

    ecm.reset(soc=1.0)
    thermal.reset(t_init_c=t_amb_c)
    ekf.reset(soc_init=1.0)

    n_steps = int(t_end_s / dt_s)
    for k in range(n_steps):
        t_s = k * dt_s

        # EKF prediction step
        ekf.predict(i_rack_a, dt_s)

        # True model step
        res = ecm.step(i_rack_a, dt_s)
        t_rack_c = thermal.step(res["p_loss"], t_amb_c, dt_s)

        # EKF measurement update with true rack voltage
        ekf.update(res["v_rack"], i_rack_a)
        soc_hat = ekf.get_soc()

        times.append(t_s)
        socs.append(res["soc"])
        soc_hats.append(soc_hat)
        v_racks.append(res["v_rack"])
        t_racks.append(t_rack_c)

    # --- Plots ---
    # 1) Rack voltage
    fig, ax1 = plt.subplots()
    ax1.plot(times, v_racks)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Rack voltage [V]")
    ax1.grid(True)

    # 2) SoC true vs estimated
    fig2, ax2 = plt.subplots()
    ax2.plot(times, socs, label="True SoC")
    ax2.plot(times, soc_hats, "--", label="Estimated SoC (EKF)")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("SoC [-]")
    ax2.grid(True)
    ax2.legend()

    # 3) Rack temperature
    fig3, ax3 = plt.subplots()
    ax3.plot(times, t_racks)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Rack temperature [°C]")
    ax3.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
