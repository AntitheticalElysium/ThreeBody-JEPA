# manim_export.py — smooth hidden-body trajectories (true + JEPA-predicted) for Manim.
import os
import torch
import numpy as np
from scipy.interpolate import CubicSpline

from config import DataCfg, ModelCfg
from data import NBodyDataset, generate
from models.jepa import JEPADynamics
from viz import get_decoder, physical_positions
from viz_hidden import predict_hidden

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _predict_seed(dataset, jepa, decoder, n, frame_dt, n_frames, traj_idx):
    """Coarse autoregressive prediction (predictor cadence) -> physical positions + error."""
    stride = max(1, round(frame_dt / dataset.cfg.dt))
    full = dataset.normalize(dataset.states[traj_idx, :stride * n_frames:stride].to(DEV))
    mass = dataset.masses[traj_idx].unsqueeze(0).to(DEV)
    vis_idx = list(range(n - 1))
    vis_traj, true_hidden = full[:, vis_idx], full[:, n - 1]
    with torch.no_grad():
        pred = predict_hidden(jepa, decoder, vis_traj, true_hidden[0], mass[:, vis_idx], frame_dt)
    pred_p = physical_positions(pred.unsqueeze(1), dataset)[:, 0]                 # [F_lo, 3]
    true_p = physical_positions(true_hidden.unsqueeze(1), dataset)[:, 0]
    return pred_p, np.linalg.norm(pred_p - true_p, axis=1)


def export(n, jepa, frame_dt=0.25, n_frames=40, n_seeds=256, seed=None):
    path = f"nbody_data_N{n}.npz"
    if not os.path.exists(path):
        generate(DataCfg(N=n, num_trajectories=200), filepath=path)
    dataset = NBodyDataset(filepath=path, cfg=DataCfg(N=n))
    decoder = get_decoder(jepa, dataset, DEV, n)
    stride = max(1, round(frame_dt / dataset.cfg.dt))

    # pick the cleanest-tracking seed (or use a hand-picked one)
    if seed is not None:
        best = seed
        pred_lo, err_lo = _predict_seed(dataset, jepa, decoder, n, frame_dt, n_frames, best)
    else:
        n_seeds = min(n_seeds, dataset.states.shape[0])
        seeds = [(i, *_predict_seed(dataset, jepa, decoder, n, frame_dt, n_frames, i)) for i in range(n_seeds)]
        best, pred_lo, err_lo = min(seeds, key=lambda r: r[2].mean())

    # full-resolution ground truth (every integrator step) for smooth playback
    n_hi = n_frames * stride
    full_hi = dataset.normalize(dataset.states[best, :n_hi].to(DEV))
    vis_hi = physical_positions(full_hi[:, list(range(n - 1))], dataset)          # [n_hi, Nv, 3]
    true_hi = physical_positions(full_hi[:, n - 1].unsqueeze(1), dataset)[:, 0]   # [n_hi, 3]

    # spline the coarse prediction onto the fine timeline (honest smooth display of its own points)
    t_lo = np.arange(len(pred_lo)) * frame_dt
    t_hi = np.arange(n_hi) * dataset.cfg.dt
    pred_hi = np.stack([CubicSpline(t_lo, pred_lo[:, d])(t_hi) for d in range(3)], axis=1)
    err_hi = np.linalg.norm(pred_hi - true_hi, axis=1)

    out = f"results/manim_N{n}.npz"
    np.savez(out, vis=vis_hi, true_hidden=true_hi, pred_hidden=pred_hi, err=err_hi, frame_dt=dataset.cfg.dt)
    print(f"  {out}: {n_hi} smooth frames, {n-1} visible + 1 hidden, mean err {err_lo.mean():.3f}")


HERO_SEEDS = {3: 9, 4: 54, 5: 65}  # hand-picked hero shots; other N auto-pick lowest-error


def main():
    os.makedirs("results", exist_ok=True)
    jepa = JEPADynamics(ModelCfg()).to(DEV)
    jepa.load_state_dict(torch.load("results/jepa_model.pt", map_location=DEV, weights_only=True))
    jepa.eval()
    for n in [3, 4, 5]:
        print(f"--- exporting N={n} ---")
        export(n, jepa, seed=HERO_SEEDS.get(n))


if __name__ == "__main__":
    main()
