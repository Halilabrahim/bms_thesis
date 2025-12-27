from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.battery_model import ocv_lfp


@dataclass
class EKFParams:
    q_nom_ah: float
    r0_ohm: float
    r1_ohm: float
    c1_f: float
    r2_ohm: float
    c2_f: float
    n_series_cells: int
    q_process: Dict[str, float]
    r_meas_v: float
    p0: Dict[str, float]


class SoCEstimatorEKF:
    """
    Simple EKF for rack-level SoC estimation using the same 2-RC ECM structure.

    State vector:
        x = [SoC, v1, v2]^T

    Measurement:
        z = V_rack = OCV(SoC) * N_series_cells - v1 - v2 - I * R0
    """

    def __init__(self, params: EKFParams, soc_init: float = 1.0):
        self.p = params
        self.q_nom_c = params.q_nom_ah * 3600.0

        # Process noise covariance Q (3x3)
        self.Q = np.diag([
            float(params.q_process["soc"]),
            float(params.q_process["v_rc1"]),
            float(params.q_process["v_rc2"]),
        ])

        # Measurement noise covariance R (1x1)
        self.R = np.array([[float(params.r_meas_v)]], dtype=float)

        # Initial covariance P (3x3)
        self.P = np.diag([
            float(params.p0["soc"]),
            float(params.p0["v_rc1"]),
            float(params.p0["v_rc2"]),
        ])

        # State vector x (3x1)
        soc_init = max(0.0, min(1.0, soc_init))
        self.x = np.array([[soc_init], [0.0], [0.0]], dtype=float)

    # ---------- helpers ----------

    @staticmethod
    def _clip_soc(soc: float) -> float:
        return max(0.0, min(1.0, soc))

    @staticmethod
    def _d_ocv_d_soc(soc: float) -> float:
        """Numerical derivative of the OCV-SoC curve."""
        soc = max(0.0, min(1.0, soc))
        eps = 1e-4
        s1 = min(1.0, soc + eps)
        s0 = max(0.0, soc - eps)
        if s1 == s0:
            return 0.0
        return (ocv_lfp(s1) - ocv_lfp(s0)) / (s1 - s0)

    # ---------- EKF core ----------

    def reset(self, soc_init: float = 1.0) -> None:
        soc_init = self._clip_soc(soc_init)
        self.x = np.array([[soc_init], [0.0], [0.0]], dtype=float)
        # P'yi istersen burda da resetleyebilirsin:
        # şimdilik ilk haliyle bırakıyoruz.

    def predict(self, i_rack_a: float, dt_s: float) -> None:
        """
        Time update (prediction step) of the EKF.
        """
        soc, v1, v2 = self.x[:, 0]

        # --- nonlinear state update (Euler) ---
        soc_pred = soc - i_rack_a * dt_s / self.q_nom_c
        soc_pred = self._clip_soc(soc_pred)

        if self.p.c1_f > 0.0 and self.p.r1_ohm > 0.0:
            dv1_dt = - v1 / (self.p.r1_ohm * self.p.c1_f) + i_rack_a / self.p.c1_f
            v1_pred = v1 + dv1_dt * dt_s
        else:
            v1_pred = 0.0

        if self.p.c2_f > 0.0 and self.p.r2_ohm > 0.0:
            dv2_dt = - v2 / (self.p.r2_ohm * self.p.c2_f) + i_rack_a / self.p.c2_f
            v2_pred = v2 + dv2_dt * dt_s
        else:
            v2_pred = 0.0

        self.x = np.array([[soc_pred], [v1_pred], [v2_pred]], dtype=float)

        # --- Jacobian F = ∂f/∂x (3x3) ---
        F = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0 - dt_s / (self.p.r1_ohm * self.p.c1_f) if self.p.c1_f > 0.0 else 1.0, 0.0],
            [0.0, 0.0, 1.0 - dt_s / (self.p.r2_ohm * self.p.c2_f) if self.p.c2_f > 0.0 else 1.0],
        ])

        # --- Covariance prediction ---
        self.P = F @ self.P @ F.T + self.Q

    def update(self, v_meas_v: float, i_rack_a: float) -> None:
        """
        Measurement update of the EKF using terminal rack voltage.
        """
        soc, v1, v2 = self.x[:, 0]

        # Predicted measurement (rack terminal voltage)
        v_ocv_cell = ocv_lfp(soc)
        v_ocv_rack = v_ocv_cell * self.p.n_series_cells
        v_pred = v_ocv_rack - v1 - v2 - i_rack_a * self.p.r0_ohm

        # Innovation
        y = v_meas_v - v_pred  # scalar

        # Measurement Jacobian H (1x3)
        d_ocv_d_soc = self._d_ocv_d_soc(soc) * self.p.n_series_cells
        H = np.array([[d_ocv_d_soc, -1.0, -1.0]])  # 1x3

        # Innovation covariance S (1x1)
        S = H @ self.P @ H.T + self.R  # 1x1
        S_inv = np.linalg.inv(S)

        # Kalman gain K (3x1)
        K = self.P @ H.T @ S_inv

        # State update
        self.x = self.x + K * y
        self.x[0, 0] = self._clip_soc(self.x[0, 0])  # SoC saturate

        # Covariance update
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    # ---------- accessors ----------

    def get_soc(self) -> float:
        """Return current SoC estimate (0..1)."""
        return float(self.x[0, 0])

    def get_state(self):
        """(optional) Return full EKF state for logging."""
        return {
            "soc_hat": float(self.x[0, 0]),
            "v1_hat": float(self.x[1, 0]),
            "v2_hat": float(self.x[2, 0]),
        }
