# src/ess_container.py

from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import yaml

from src.battery_model import RackParams, RackECMModel
from src.thermal_model import ThermalParams, RackThermalModel
from src.estimator import SoCEstimatorEKF, EKFParams
from src.bms_logic import BMSControlParams, compute_current_limits
from src.faults import FaultThresholds, FaultDetector
from src.fsm import BMSStateMachine, BMSState


# -----------------------------
# Tek bir rack için "stack" (ECM + termal + EKF + fault + FSM)
# -----------------------------

@dataclass
class RackStack:
    ecm: RackECMModel
    thermal: RackThermalModel
    ekf: SoCEstimatorEKF
    fault_det: FaultDetector
    fsm: BMSStateMachine

    soc_hat: float = 1.0
    last_res: Dict[str, float] = field(default_factory=dict)

    def reset(self, t_amb_c: float) -> None:
        # SOC = 100% ile başlat
        self.ecm.reset(soc=1.0)
        self.thermal.reset(t_init_c=t_amb_c)
        self.ekf.reset(soc_init=1.0)
        self.fault_det.reset()
        self.fsm.reset(BMSState.RUN)
        # başlangıçta "0 akım" adımı ile bir voltaj değeri alalım
        self.last_res = self.ecm.step(0.0, 0.0)
        self.soc_hat = self.ekf.get_soc()


# -----------------------------
# ESS (konteyner) parametreleri
# -----------------------------

@dataclass
class ESSContainerParams:
    n_racks: int                     # 1 konteynerdeki rack sayısı (Huawei için 6)
    rack_params: RackParams
    thermal_params: ThermalParams
    ekf_params: EKFParams
    bms_params: BMSControlParams
    fault_thresholds: FaultThresholds
    rack_nominal_energy_kwh: float   # tek rack’in nominal enerjisi (≈344 kWh)
    dc_bus_nominal_v: float          # DC LV panel nominal gerilimi (≈1200 V)
    dt_s: float


# -----------------------------
# ESS (konteyner) modeli
# -----------------------------

class ESSContainerModel:
    """
    1 konteyner = n_racks adet Huawei rack.
    Pozitif akım: deşarj (şebekeye güç verme).
    Negatif akım: şarj (PV'den bataryaya güç alma).
    """

    def __init__(self, params: ESSContainerParams):
        self.params = params
        self.dt_s = params.dt_s

        self.racks: List[RackStack] = []
        for _ in range(params.n_racks):
            ecm = RackECMModel(params.rack_params)
            thermal = RackThermalModel(params.thermal_params)
            ekf = SoCEstimatorEKF(params.ekf_params, soc_init=1.0)
            fault_det = FaultDetector(params.fault_thresholds, params.dt_s)
            fsm = BMSStateMachine()
            self.racks.append(RackStack(ecm, thermal, ekf, fault_det, fsm))

        # toplu izleme değişkenleri
        self.energy_throughput_kwh = 0.0  # |P| entegre – lifecycle için kullanılabilir

    # ---- yardımcı hesaplar ----

    @property
    def rated_energy_kwh(self) -> float:
        return self.params.n_racks * self.params.rack_nominal_energy_kwh

    def reset(self, t_amb_c: float = 25.0) -> None:
        self.energy_throughput_kwh = 0.0
        for r in self.racks:
            r.reset(t_amb_c)

    # ---- ana adım fonksiyonu ----

    def step(self, i_ess_req_a: float, t_amb_c: float) -> Dict[str, Any]:
        """
        Tek zaman adımı.
        i_ess_req_a: konteyner DC tarafında istenen akım (+deşarj / -şarj).
        t_amb_c: ortam sıcaklığı.
        """
        n = len(self.racks)

        # 1) BMS limitleri için "en kötü" rack'i kullan (min SOC, max sıcaklık/voltaj)
        soc_min = min(r.soc_hat for r in self.racks)
        t_max = max(r.thermal.t_c for r in self.racks)

        # son step'teki hücre voltajları (yoksa nominale düş)
        v_cell_min = min(r.last_res.get("v_cell_min", 3.2) for r in self.racks)
        v_cell_max = max(r.last_res.get("v_cell_max", 3.2) for r in self.racks)

        curr_limits = compute_current_limits(
            soc_hat=soc_min,
            t_rack_c=t_max,
            v_cell_min=v_cell_min,
            v_cell_max=v_cell_max,
            params=self.params.bms_params,
        )

        # 2) İstenen akımı BMS limitlerine göre saturate et
        if i_ess_req_a >= 0.0:
            i_ess_act = min(i_ess_req_a, curr_limits["i_discharge_max_allowed"])
        else:
            i_ess_act = max(i_ess_req_a, -curr_limits["i_charge_max_allowed"])

        # 3) Akımı rack'ler arasında eşit paylaştır (Huawei "no bias current" varsayımı)
        i_rack_a = i_ess_act / max(n, 1)

        socs = []
        soc_hats = []
        temps = []
        v_rack_list = []
        rack_faults = []
        rack_states = []

        total_p_kw = 0.0

        for r in self.racks:
            # EKF prediction
            r.ekf.predict(i_rack_a, self.dt_s)

            # gerçek ECM ve termal
            res = r.ecm.step(i_rack_a, self.dt_s)
            t_rack_c = r.thermal.step(res["p_loss"], t_amb_c, self.dt_s)

            # sensör gürültüsü eklemedik – direkt gerçek değerleri kullanalım
            v_meas_v = res["v_rack"]
            i_meas_a = i_rack_a
            t_meas_c = t_rack_c

            # EKF update
            r.ekf.update(v_meas_v, i_meas_a)
            r.soc_hat = r.ekf.get_soc()
            r.last_res = res

            # fault detection + FSM
            flags = r.fault_det.step(
                v_cell_min=res["v_cell_min"],
                v_cell_max=res["v_cell_max"],
                t_rack_c=t_meas_c,
                i_rack_a=i_meas_a,
                gas_alarm=False,
            )
            state = r.fsm.step(flags, enable=True)

            # listelere ekle
            socs.append(res["soc"])
            soc_hats.append(r.soc_hat)
            temps.append(t_rack_c)
            v_rack_list.append(res["v_rack"])
            rack_faults.append(flags)
            rack_states.append(state)

            # güç hesabı (pozitif = deşarj)
            total_p_kw += res["v_rack"] * i_rack_a / 1000.0

        # enerji throughput
        self.energy_throughput_kwh += abs(total_p_kw) * self.dt_s / 3600.0

        return {
            "i_ess_req_a": i_ess_req_a,
            "i_ess_act_a": i_ess_act,
            "i_rack_a": i_rack_a,
            "p_ess_kw": total_p_kw,
            "soc_rack_min": float(np.min(socs)),
            "soc_rack_max": float(np.max(socs)),
            "soc_rack_mean": float(np.mean(socs)),
            "soc_hat_mean": float(np.mean(soc_hats)),
            "t_rack_max_c": float(np.max(temps)),
            "t_rack_mean_c": float(np.mean(temps)),
            "v_rack_mean_v": float(np.mean(v_rack_list)),
            "curr_limits": curr_limits,
            "rack_faults": rack_faults,
            "rack_states": rack_states,
            "energy_throughput_kwh": self.energy_throughput_kwh,
        }


# -----------------------------
# YAML'dan Huawei konteyneri kuran yardımcı fonksiyon
# -----------------------------

def build_ess_container_from_yaml(
    yaml_path: str = "data/params_rack.yaml",
) -> ESSContainerModel:
    """
    params_rack.yaml içinden Huawei LUNA tabanlı
    1 konteyner = 6 rack ESS modelini kurar.
    """

    yaml_file = Path(yaml_path)
    with yaml_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    em = cfg["electrical_model"]
    struct = cfg["structure"]
    therm = cfg["thermal_model"]
    limits = cfg["limits"]
    ekf_cfg = cfg["estimation"]["ekf"]
    bms_cfg = cfg["bms_control"]
    faults_cfg = cfg["faults"]
    rack_cfg = cfg["rack"]
    hw_cfg = cfg["hardware"]
    sim_cfg = cfg["simulation"]

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

    thermal_params = ThermalParams(
        c_th_j_per_k=float(therm["c_th_j_per_k"]),
        r_th_k_per_w=float(therm["r_th_k_per_w"]),
        t_init_c=float(therm["t_init_c"]),
    )

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

    bms_params = BMSControlParams(
        i_charge_max_a=float(limits["i_charge_max_a"]),
        i_discharge_max_a=float(limits["i_discharge_max_a"]),
        soc_low_cutoff=float(bms_cfg["soc_low_cutoff"]),
        soc_low_derate_start=float(bms_cfg["soc_low_derate_start"]),
        soc_high_cutoff=float(bms_cfg["soc_high_cutoff"]),
        soc_high_derate_start=float(bms_cfg["soc_high_derate_start"]),
        t_low_cutoff_c=float(bms_cfg["t_low_cutoff_c"]),
        t_low_derate_start_c=float(bms_cfg["t_low_derate_start_c"]),
        t_high_cutoff_c=float(bms_cfg["t_high_cutoff_c"]),
        t_high_derate_start_c=float(bms_cfg["t_high_derate_start_c"]),
        v_cell_min_v=float(limits["v_cell_min_v"]),
        v_cell_max_v=float(limits["v_cell_max_v"]),
        v_margin_v=float(bms_cfg["v_margin_v"]),
    )

    fault_thr = FaultThresholds(
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

    ess_params = ESSContainerParams(
        n_racks=int(struct.get("racks_in_ess", 6)),
        rack_params=rack_params,
        thermal_params=thermal_params,
        ekf_params=ekf_params,
        bms_params=bms_params,
        fault_thresholds=fault_thr,
        rack_nominal_energy_kwh=float(rack_cfg["nominal_energy_kwh"]),
        dc_bus_nominal_v=float(hw_cfg["rated_operating_voltage_v"]),
        dt_s=float(sim_cfg["dt_s"]),
    )

    return ESSContainerModel(ess_params)
