from dataclasses import dataclass
from typing import Dict


@dataclass
class FaultThresholds:
    ov_cell_v: float
    uv_cell_v: float
    ot_rack_c: float
    ut_rack_c: float
    oc_discharge_a: float
    oc_charge_a: float
    fire_temp_c: float
    fire_dTdt_c_per_s: float
    debounce_steps: int


class FaultDetector:
    """
    Threshold-based fault detection with simple debouncing.

    Flags:
      - ov: over-voltage (cell)
      - uv: under-voltage (cell)
      - ot: over-temperature
      - ut: under-temperature
      - oc: over-current
      - fire: fire event (gas alarm OR high T / high dT/dt)
      - any_fault: any of ov/uv/ot/ut/oc
      - emergency: fire OR severe faults (ot or oc)
    """

    def __init__(self, thresholds: FaultThresholds, dt_s: float):
        self.th = thresholds
        self.dt_s = dt_s
        self.reset()

    def reset(self) -> None:
        self.prev_temp = None
        self.counters = {name: 0 for name in ["ov", "uv", "ot", "ut", "oc", "fire"]}

    def step(
        self,
        v_cell_min: float,
        v_cell_max: float,
        t_rack_c: float,
        i_rack_a: float,
        gas_alarm: bool = False,   # <-- YENİ
    ) -> Dict[str, bool]:
        """
        Update fault flags based on current measurements.

        Returns a dict of boolean flags.
        """
        # Raw conditions from measurements
        cond_ov = v_cell_max > self.th.ov_cell_v
        cond_uv = v_cell_min < self.th.uv_cell_v
        cond_ot = t_rack_c > self.th.ot_rack_c
        cond_ut = t_rack_c < self.th.ut_rack_c

        if i_rack_a >= 0.0:
            cond_oc = i_rack_a > self.th.oc_discharge_a
        else:
            cond_oc = -i_rack_a > self.th.oc_charge_a

        # ---- Fire condition: gas alarm OR temperature-based ----
        cond_fire = gas_alarm  # gaz sensörü direkt tetikleyebilir

        if self.prev_temp is not None and self.dt_s > 0.0:
            dTdt = (t_rack_c - self.prev_temp) / self.dt_s
            if (
                t_rack_c > self.th.fire_temp_c
                or dTdt > self.th.fire_dTdt_c_per_s
            ):
                cond_fire = True

        self.prev_temp = t_rack_c

        raw = {
            "ov": cond_ov,
            "uv": cond_uv,
            "ot": cond_ot,
            "ut": cond_ut,
            "oc": cond_oc,
            "fire": cond_fire,
        }

        # Debounce & latch
        flags = {
            "ov": False,
            "uv": False,
            "ot": False,
            "ut": False,
            "oc": False,
            "fire": False,
            "any_fault": False,
            "emergency": False,
        }

        for name, cond in raw.items():
            if cond:
                self.counters[name] += 1
            else:
                self.counters[name] = 0

            if self.counters[name] >= self.th.debounce_steps:
                flags[name] = True

        flags["any_fault"] = (
            flags["ov"] or flags["uv"] or flags["ot"] or flags["ut"] or flags["oc"]
        )
        # Emergency: fire OR severe faults
        flags["emergency"] = flags["fire"] or flags["ot"] or flags["oc"]

        return flags
