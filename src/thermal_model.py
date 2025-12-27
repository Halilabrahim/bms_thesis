from dataclasses import dataclass


@dataclass
class ThermalParams:
    """Lumped thermal model parameters for the rack."""
    c_th_j_per_k: float     # thermal capacitance [J/K]
    r_th_k_per_w: float     # thermal resistance [K/W]
    t_init_c: float         # initial temperature [°C]


class RackThermalModel:
    """
    Simple lumped RC thermal model for the rack.

    dT/dt = (T_amb - T) / (R_th * C_th) + P_loss / C_th
    """

    def __init__(self, params: ThermalParams):
        self.p = params
        self.t_c = params.t_init_c

    def reset(self, t_init_c: float = None) -> None:
        """Reset temperature to initial value or given value."""
        if t_init_c is None:
            self.t_c = self.p.t_init_c
        else:
            self.t_c = t_init_c

    def step(self, p_loss_w: float, t_amb_c: float, dt_s: float) -> float:
        """
        Advance the thermal model by one time step.

        Parameters
        ----------
        p_loss_w : float
            Internal loss power in watts (from ECM).
        t_amb_c : float
            Ambient temperature in °C.
        dt_s : float
            Time step in seconds.

        Returns
        -------
        float
            New rack temperature in °C.
        """
        p = self.p
        dT_dt = ((t_amb_c - self.t_c) /
                 (p.r_th_k_per_w * p.c_th_j_per_k)) + (p_loss_w / p.c_th_j_per_k)
        self.t_c += dT_dt * dt_s
        return self.t_c
