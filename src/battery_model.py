# src/battery_model.py

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class RackParams:
    """Electrical configuration and ECM parameters for the rack."""
    q_nom_ah: float
    r0_ohm: float
    r1_ohm: float
    c1_f: float
    r2_ohm: float
    c2_f: float
    n_packs: int
    cells_in_series_per_pack: int
    v_cell_min_v: float
    v_cell_max_v: float

    # Hücre varyasyonu
    cell_param_spread_q_pct: float = 0.0
    cell_param_spread_r0_pct: float = 0.0
    cell_random_seed: int = 0

    # SoH / aging bilgileri (sadece kayıt amaçlı)
    soh_capacity: float = 1.0      # 1.0 = %100 kapasite, 0.8 = %80 vb.
    soh_r0_factor: float = 1.0     # 1.0 = nominal R0, 1.5 = 1.5x R0 vb.


def ocv_lfp(soc: float) -> float:
    """
    Very rough OCV–SoC curve for an LFP cell (nominal 3.2 V).
    soc: 0.0–1.0
    returns: cell OCV in volts.
    """
    soc = max(0.0, min(1.0, float(soc)))

    # Piecewise linear approximation:
    if soc <= 0.05:
        # knee at very low SoC
        return 2.5 + soc / 0.05 * (3.2 - 2.5)
    elif soc <= 0.90:
        # plateau region
        return 3.2 + (soc - 0.05) / 0.85 * (3.35 - 3.2)
    else:
        # top knee
        return 3.35 + (soc - 0.90) / 0.10 * (3.45 - 3.35)


class RackECMModel:
    """
    2-RC equivalent circuit model of a rack of LFP packs with
    simple cell-to-cell variation in capacity and ohmic resistance.

    Positive current = discharge.
    """

    def __init__(self, params: RackParams):
        self.p = params
        self.n_series_cells = (
            params.n_packs * params.cells_in_series_per_pack
        )

        # Nominal capacity of the string (and each cell, since 1P)
        # !!! ÖNEMLİ: Hücre kapasitesini N'e BÖLMÜYORUZ !!!
        self.q_nom_c = params.q_nom_ah * 3600.0  # [C]

        rng = np.random.default_rng(params.cell_random_seed)
        q_spread = params.cell_param_spread_q_pct / 100.0
        r0_spread = params.cell_param_spread_r0_pct / 100.0

        # Hücre kapasite varyasyonu (her hücrenin kapasitesi Q_nom * (1 ± spread))
        if q_spread > 0.0:
            eps_q = rng.uniform(-q_spread, q_spread, size=self.n_series_cells)
            self.q_cell_c = self.q_nom_c * (1.0 + eps_q)
        else:
            self.q_cell_c = np.full(self.n_series_cells, self.q_nom_c)

        # Toplam R0'u hücrelere dağıt, üstüne ±% spread ekle
        base_r0_cell = params.r0_ohm / self.n_series_cells
        if r0_spread > 0.0:
            eps_r0 = rng.uniform(-r0_spread, r0_spread, size=self.n_series_cells)
            self.r0_cell = base_r0_cell * (1.0 + eps_r0)
        else:
            self.r0_cell = np.full(self.n_series_cells, base_r0_cell)

        # Dinamik durumlar
        self.soc_cells = np.ones(self.n_series_cells)  # her hücre SoC
        self.v1 = 0.0   # RC1 gerilimi (rack level)
        self.v2 = 0.0   # RC2 gerilimi (rack level)

    def reset(self, soc: float = 1.0) -> None:
        """Reset model states."""
        soc = float(np.clip(soc, 0.0, 1.0))
        self.soc_cells[:] = soc
        self.v1 = 0.0
        self.v2 = 0.0

    def step(self, i_rack_a: float, dt_s: float) -> Dict[str, float]:
        """
        Advance the ECM by one time step.

        Parameters
        ----------
        i_rack_a : float
            Rack current in A (positive = discharge).
        dt_s : float
            Time step in seconds.

        Returns
        -------
        dict
            {
              "v_rack": rack terminal voltage [V],
              "soc": mean state of charge [0..1],
              "v_cell_min": min cell voltage [V],
              "v_cell_max": max cell voltage [V],
              "p_loss": ohmic loss power [W],
              "soc_cell_min": min cell SoC [-],
              "soc_cell_max": max cell SoC [-],
              "soc_cell_mean": mean cell SoC [-],
            }
        """
        p = self.p

        # --- 1) Hücre SoC dinamiği (Coulomb counting, her hücre için ayrı) ---
        if dt_s > 0.0:
            d_soc = - i_rack_a * dt_s / self.q_cell_c
            self.soc_cells = np.clip(self.soc_cells + d_soc, 0.0, 1.0)

        soc_mean = float(self.soc_cells.mean())
        soc_min = float(self.soc_cells.min())
        soc_max = float(self.soc_cells.max())

        # --- 2) RC branch dinamiği (rack seviyesi, lumped) ---
        if p.c1_f > 0.0 and p.r1_ohm > 0.0:
            dv1_dt = - self.v1 / (p.r1_ohm * p.c1_f) + i_rack_a / p.c1_f
            self.v1 += dv1_dt * dt_s

        if p.c2_f > 0.0 and p.r2_ohm > 0.0:
            dv2_dt = - self.v2 / (p.r2_ohm * p.c2_f) + i_rack_a / p.c2_f
            self.v2 += dv2_dt * dt_s

        # --- 3) OCV ve terminaller ---
        # Hücre bazlı OCV
        ocv_cells = np.array([ocv_lfp(float(s)) for s in self.soc_cells])
        v_rack_ocv_v = float(ocv_cells.sum())

        # Rack terminal gerilimi (Thevenin lumped)
        v_rack_v = (
            v_rack_ocv_v
            - self.v1
            - self.v2
            - i_rack_a * p.r0_ohm
        )

        # Hücre gerilimlerinin yaklaşık hesabı:
        # RC gerilimlerini ve ohmik düşüşü hücrelere paylaştır.
        v1_cell = self.v1 / self.n_series_cells
        v2_cell = self.v2 / self.n_series_cells
        v_drop_ohmic_cell = i_rack_a * self.r0_cell

        v_cells = ocv_cells - v1_cell - v2_cell - v_drop_ohmic_cell
        v_cell_min_v = float(v_cells.min())
        v_cell_max_v = float(v_cells.max())

        # --- 4) Kayıplar (sadece R0) ---
        p_loss_w = (i_rack_a ** 2) * p.r0_ohm

        return {
            "v_rack": v_rack_v,
            "soc": soc_mean,
            "v_cell_min": v_cell_min_v,
            "v_cell_max": v_cell_max_v,
            "p_loss": p_loss_w,
            "soc_cell_min": soc_min,
            "soc_cell_max": soc_max,
            "soc_cell_mean": soc_mean,
        }
