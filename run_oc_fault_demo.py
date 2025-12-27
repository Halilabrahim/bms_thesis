from src.config import load_params
from src.battery_model import RackParams, RackECMModel
from src.thermal_model import ThermalParams, RackThermalModel
from src.faults import FaultThresholds, FaultDetector
from src.fsm import BMSStateMachine, BMSState
import matplotlib.pyplot as plt


def main():
    params = load_params()

    em = params["electrical_model"]
    struct = params["structure"]
    limits = params["limits"]
    therm = params["thermal_model"]
    sim_cfg = params["simulation"]
    faults_cfg = params["faults"]

    # --- Models: rack + thermal ---
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

    # --- Fault detector & FSM ---
    dt_s = float(sim_cfg["dt_s"])
    t_end_s = float(sim_cfg["t_end_s"])

    thr = FaultThresholds(
        ov_cell_v=float(faults_cfg["ov_cell_v"]),
        uv_cell_v=float(faults_cfg["uv_cell_v"]),
        ot_rack_c=float(faults_cfg["ot_rack_c"]),
        ut_rack_c=float(faults_cfg["ut_rack_c"]),
        oc_discharge_a=float(faults_cfg["oc_discharge_a"]),
        oc_charge_a=float(faults_cfg["oc_charge_a"]),
        fire_temp_c=float(faults_cfg["fire_temp_c"]),
        fire_dTdt_c_per_s=float(faults_cfg["fire_dTdt_c_per_s"]),
        debounce_steps=int(faults_cfg["debounce_steps"]),
    )
    fault_det = FaultDetector(thr, dt_s)
    fsm = BMSStateMachine()

    # --- Fault scenario setup ---
    t_amb_c = 25.0
    i_fault_a = 500.0   # deliberately above oc_discharge_a

    times = []
    currents = []
    socs = []
    temps = []
    oc_flags = []
    emergency_flags = []
    state_codes = []

    state_code_map = {
        BMSState.OFF: 0,
        BMSState.RUN: 1,
        BMSState.FAULT: 2,
        BMSState.EMERGENCY_SHUTDOWN: 3,
    }

    ecm.reset(soc=1.0)
    thermal.reset(t_init_c=t_amb_c)
    fault_det.reset()
    fsm.reset(BMSState.RUN)

    n_steps = int(t_end_s / dt_s)
    for k in range(n_steps):
        t_s = k * dt_s

        # Current applied depends on state: EMERGENCY -> 0 A
        if fsm.state == BMSState.EMERGENCY_SHUTDOWN:
            i_rack_a = 0.0
        else:
            i_rack_a = i_fault_a

        # True model
        res = ecm.step(i_rack_a, dt_s)
        t_rack_c = thermal.step(res["p_loss"], t_amb_c, dt_s)

        # Fault detection & FSM update
        flags = fault_det.step(
            v_cell_min=res["v_cell_min"],
            v_cell_max=res["v_cell_max"],
            t_rack_c=t_rack_c,
            i_rack_a=i_rack_a,
        )
        state = fsm.step(flags, enable=True)

        # Log
        times.append(t_s)
        currents.append(i_rack_a)
        socs.append(res["soc"])
        temps.append(t_rack_c)
        oc_flags.append(flags["oc"])
        emergency_flags.append(flags["emergency"])
        state_codes.append(state_code_map[state])

    # Detection times (konsola yazalım)
    t_oc = next((t for t, f in zip(times, oc_flags) if f), None)
    t_em = next((t for t, f in zip(times, emergency_flags) if f), None)
    print(f"OC fault latched at: {t_oc} s")
    print(f"Emergency shutdown at: {t_em} s")
    print(f"Final state: {fsm.state}")

    # --- Plots ---

    # 1) Current profile
    fig0, ax0 = plt.subplots()
    ax0.plot(times, currents)
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Current [A]")
    ax0.set_title("Over-current fault scenario")
    ax0.grid(True)

    # 2) SoC
    fig1, ax1 = plt.subplots()
    ax1.plot(times, socs)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("SoC [-]")
    ax1.grid(True)

    # 3) Temperature
    fig2, ax2 = plt.subplots()
    ax2.plot(times, temps)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Rack temperature [°C]")
    ax2.grid(True)

    # 4) State code (0..3)
    fig3, ax3 = plt.subplots()
    ax3.step(times, state_codes, where="post")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("BMS state code (0 OFF, 1 RUN, 2 FAULT, 3 EMERGENCY)")
    ax3.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
