# seed_preview.py — scan a large pool of seeds, rank by JEPA tracking error, show the best.
#   PREVIEW_N=3 PREVIEW_SEEDS=256 uv run python seed_preview.py
import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DataCfg, ModelCfg
from data import NBodyDataset
from models.jepa import JEPADynamics
from viz import get_decoder, physical_positions
from viz_hidden import predict_hidden

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = int(os.environ.get("PREVIEW_N", "3"))
N_SEEDS = int(os.environ.get("PREVIEW_SEEDS", "256"))  # large pool, ranked by error
FRAME_DT, N_FRAMES = 0.25, 40
GOLD, TRUTH_C, JEPA_C, BG = "#e0a85a", "#dfe6f0", "#5fb4ff", "#0a0a14"


def main():
    jepa = JEPADynamics(ModelCfg()).to(DEV)
    jepa.load_state_dict(torch.load("results/jepa_model.pt", map_location=DEV, weights_only=True))
    jepa.eval()
    ds = NBodyDataset(filepath=f"nbody_data_N{N}.npz", cfg=DataCfg(N=N))
    decoder = get_decoder(jepa, ds, DEV, N)
    stride = max(1, round(FRAME_DT / ds.cfg.dt))
    n_seeds = min(N_SEEDS, ds.states.shape[0])

    rows_data = []
    for i in range(n_seeds):
        full = ds.normalize(ds.states[i, :stride * N_FRAMES:stride].to(DEV))
        mass = ds.masses[i].unsqueeze(0).to(DEV)
        vis_idx = list(range(N - 1))
        vis_traj, true_hidden = full[:, vis_idx], full[:, N - 1]
        with torch.no_grad():
            pred = predict_hidden(jepa, decoder, vis_traj, true_hidden[0], mass[:, vis_idx], FRAME_DT)
        pred_p = physical_positions(pred.unsqueeze(1), ds)[:, 0]
        true_p = physical_positions(true_hidden.unsqueeze(1), ds)[:, 0]
        vis_p = physical_positions(vis_traj, ds)
        err = np.linalg.norm(pred_p - true_p, axis=1).mean()
        rows_data.append((i, err, vis_p, true_p, pred_p))

    rows_data.sort(key=lambda r: r[1])  # best tracking first
    print(f"--- N={N}: scanned {n_seeds} seeds, ranked by mean tracking error ---")
    for rank, (i, me, *_ ) in enumerate(rows_data[:20]):
        print(f"  #{rank+1:2d}  seed {i:4d}   err {me:.3f}")
    K = min(12, len(rows_data))
    cols = 4
    rows = math.ceil(K / cols)
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    fig.patch.set_facecolor(BG)
    for k in range(K):
        i, me, vis_p, true_p, pred_p = rows_data[k]
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        ax.set_facecolor(BG)
        for b in range(vis_p.shape[1]):
            ax.plot(vis_p[:, b, 0], vis_p[:, b, 1], vis_p[:, b, 2], color=GOLD, lw=1.2, alpha=0.8)
        ax.plot(true_p[:, 0], true_p[:, 1], true_p[:, 2], color=TRUTH_C, lw=1.5, ls="--", alpha=0.7)
        ax.plot(pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], color=JEPA_C, lw=2.2, alpha=0.95)
        ax.scatter(*pred_p[0], color=JEPA_C, s=25)
        ax.set_title(f"seed {i}  ·  err {me:.2f}", color="white", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
    plt.tight_layout()
    out = f"results/_seed_preview_N{N}.png"
    plt.savefig(out, dpi=90, facecolor=fig.get_facecolor())
    print(f"{out}  (best seed by error: {rows_data[0][0]}, err {rows_data[0][1]:.3f})")


if __name__ == "__main__":
    main()
