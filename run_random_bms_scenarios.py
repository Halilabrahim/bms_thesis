import os
import yaml
import numpy as np

from src.config import load_params
from src.scenarios import Scenario
from src.sim_runner import run_scenario


def main():
    # Mevcut run_scenarios_with_metrics ile aynı parametre dosyasını yükle
    params = load_params()

    os.makedirs(os.path.join("results", "random"), exist_ok=True)

    scenarios = []
    # Örnek: 3 farklı random profil, farklı seed ile
    for idx, seed in enumerate([1, 2, 3], start=1):
        sc = Scenario(
            id=f"R{idx}_random_25C",
            description=f"Random charge/discharge profile at 25°C (seed={seed})",
            max_time_s=2 * 3600.0,          # 2 saat
            profile_type="random_current",
            t_amb_c=25.0,
            use_bms_limits=True,
            i_discharge_max_a=320.0,        # ~1C deşarj sınırı
            i_charge_max_a=160.0,           # ~0.5C şarj sınırı
            segment_min_s=120.0,            # her segment en az 2 dk
            segment_max_s=900.0,            # en fazla 15 dk
            random_seed=seed,
        )
        scenarios.append(sc)

    for sc in scenarios:
        print("=" * 60)
        print(f"Running random scenario {sc.id}: {sc.description}")
        result = run_scenario(sc, params)

        out_path = os.path.join("results", "random", f"{sc.id}.npz")
        np.savez(out_path, **result)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
