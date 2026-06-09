# extend_n.py — does JEPA's energy decodability degrade slower than the memorizer as N grows?
#   probes linear + nonlinear energy R^2 for all three models at N=3..8 (zero-shot from N=3).
import os
import torch

from config import DataCfg, ModelCfg
from data import NBodyDataset, generate
from models.jepa import JEPADynamics
from models.baselines import RegressionDynamics, HNNDynamics
from eval import linear_probe, nonlinear_probe

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
SPECS = {"Regression": RegressionDynamics, "HNN": HNNDynamics, "JEPA": JEPADynamics}


def main():
    cfg = ModelCfg()
    models = {}
    for name, cls in SPECS.items():
        m = cls(cfg).to(DEV)
        m.load_state_dict(torch.load(f"results/{name.lower()}_model.pt", map_location=DEV, weights_only=True))
        m.eval()
        models[name] = m

    lines = ["N   | Regression L/NL   | HNN L/NL          | JEPA L/NL"]
    print(lines[-1])
    for n in NS:
        path = f"nbody_data_N{n}.npz"
        if not os.path.exists(path):
            print(f"  generating N={n} data...")
            generate(DataCfg(N=n, num_trajectories=200), filepath=path)
        ds = NBodyDataset(filepath=path, cfg=DataCfg(N=n))
        cells = {}
        for name, m in models.items():
            cells[name] = (linear_probe(m, ds), nonlinear_probe(m, ds))
        row = (f"N={n} | {cells['Regression'][0]:.3f} / {cells['Regression'][1]:.3f}   "
               f"| {cells['HNN'][0]:.3f} / {cells['HNN'][1]:.3f}   "
               f"| {cells['JEPA'][0]:.3f} / {cells['JEPA'][1]:.3f}")
        lines.append(row)
        print(row)

    with open("results/extrapolation_probes.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nsaved results/extrapolation_probes.txt")


if __name__ == "__main__":
    main()
