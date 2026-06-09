# export_seeds.py — export specific hidden-body seeds for preview/comparison.
#   exports results/seed_candidates/manim_N{N}_seed{s}.npz for each seed in SEEDS.
#   render one with:  MANIM_N=3 MANIM_NPZ=results/seed_candidates/manim_N3_seed215.npz \
#                     uv run manim -ql --disable_caching -o seed215 manim_scene.py HiddenOrbit3D
import os
import torch
import numpy as np
from scipy.interpolate import CubicSpline

from config import DataCfg, ModelCfg
from data import NBodyDataset
from models.jepa import JEPADynamics
from viz import get_decoder, physical_positions
import manim_export as me

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = int(os.environ.get("PREVIEW_N", "3"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "215,190,94,64,9,122,33,168").split(",")]
FRAME_DT, N_FRAMES = 0.25, 40


def main():
    out_dir = "results/seed_candidates"
    os.makedirs(out_dir, exist_ok=True)
    jepa = JEPADynamics(ModelCfg()).to(DEV)
    jepa.load_state_dict(torch.load("results/jepa_model.pt", map_location=DEV, weights_only=True))
    jepa.eval()
    ds = NBodyDataset(filepath=f"nbody_data_N{N}.npz", cfg=DataCfg(N=N))
    decoder = get_decoder(jepa, ds, DEV, N)
    stride = max(1, round(FRAME_DT / ds.cfg.dt))
    n_hi = N_FRAMES * stride

    for s in SEEDS:
        pred_lo, err_lo = me._predict_seed(ds, jepa, decoder, N, FRAME_DT, N_FRAMES, s)
        full_hi = ds.normalize(ds.states[s, :n_hi].to(DEV))
        vis_hi = physical_positions(full_hi[:, list(range(N - 1))], ds)
        true_hi = physical_positions(full_hi[:, N - 1].unsqueeze(1), ds)[:, 0]
        t_lo = np.arange(len(pred_lo)) * FRAME_DT
        t_hi = np.arange(n_hi) * ds.cfg.dt
        pred_hi = np.stack([CubicSpline(t_lo, pred_lo[:, d])(t_hi) for d in range(3)], axis=1)
        err_hi = np.linalg.norm(pred_hi - true_hi, axis=1)
        out = f"{out_dir}/manim_N{N}_seed{s}.npz"
        np.savez(out, vis=vis_hi, true_hidden=true_hi, pred_hidden=pred_hi, err=err_hi, frame_dt=ds.cfg.dt)
        print(f"  {out}  (mean err {err_lo.mean():.3f})")


if __name__ == "__main__":
    main()
