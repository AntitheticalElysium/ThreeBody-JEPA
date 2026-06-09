# viz_hidden.py
import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from config import DataCfg, ModelCfg
from data import NBodyDataset, generate
from models.jepa import JEPADynamics
from viz import get_decoder, physical_positions

MODEL_CFG = ModelCfg()


def predict_hidden(jepa, decoder, vis_traj, hidden0, mass_vis, frame_dt):
    """Autoregressively predict the hidden body from the visible bodies' true states."""
    device = vis_traj.device
    dt = torch.tensor([[frame_dt]], device=device)
    h = hidden0.clone()
    out = [h]
    for i in range(vis_traj.shape[0] - 1):
        ctx = vis_traj[i].unsqueeze(0)               # [1, Nv, 2*n_dim]
        pred = jepa.predict_final(ctx, mass_vis, h.view(1, 1, -1), dt)
        h = decoder(pred)[0, 0]
        out.append(h)
    return torch.stack(out)                          # [F, 4]


def build(n, jepa, dataset, device, frame_dt=0.25, n_frames=48, fps=10, traj_idx=0):
    decoder = get_decoder(jepa, dataset, device, n)
    stride = max(1, round(frame_dt / dataset.cfg.dt))
    full = dataset.normalize(dataset.states[traj_idx, :stride * n_frames:stride].to(device)) # [F, N, 4]
    n_frames = full.shape[0] # clamp to frames actually available in the seed trajectory
    mass = dataset.masses[traj_idx].unsqueeze(0).to(device)

    hidden_idx = n - 1
    vis_idx = list(range(n - 1))
    vis_traj = full[:, vis_idx]                      # [F, Nv, 4]
    true_hidden = full[:, hidden_idx]                # [F, 4]

    with torch.no_grad():
        pred_hidden = predict_hidden(jepa, decoder, vis_traj, true_hidden[0], mass[:, vis_idx], frame_dt)

    nd = dataset.n_dim
    is3d = nd == 3
    p_vis = physical_positions(vis_traj, dataset)            # [F, Nv, nd]
    p_true = physical_positions(true_hidden.unsqueeze(1), dataset)[:, 0]   # [F, nd]
    p_pred = physical_positions(pred_hidden.unsqueeze(1), dataset)[:, 0]
    t_axis = np.arange(n_frames) * frame_dt

    allp = np.concatenate([p_vis.reshape(-1, nd), p_true, p_pred], axis=0)
    lim = np.nanpercentile(np.abs(allp), 99) * 1.5

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d" if is3d else None)
    axe = fig.add_subplot(1, 2, 2)
    fig.suptitle(f"JEPA infers the hidden body from gravity alone  (N={n})", fontsize=17, fontweight="bold")

    def set_line(line, pts):
        line.set_data(pts[:, 0], pts[:, 1])
        if is3d: line.set_3d_properties(pts[:, 2])

    def set_pts(scat, pts):
        if is3d: scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        else: scat.set_offsets(pts)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
    if is3d: ax.set_zlim(-lim, lim); ax.set_zticks([])
    else: ax.set_aspect("equal")

    vis_colors = plt.cm.Greys(np.linspace(0.5, 0.8, n - 1))
    line_args = ([], [], []) if is3d else ([], [])
    vis_trails = [ax.plot(*line_args, lw=1.5, color=vis_colors[b])[0] for b in range(n - 1)]
    vis_heads = ax.scatter(*[p_vis[0, :, i] for i in range(nd)], s=70, color=vis_colors, edgecolors="k", zorder=5)
    true_trail, = ax.plot(*line_args, lw=1.2, color="gray", ls="--", alpha=0.6, label="hidden body (truth)")
    pred_trail, = ax.plot(*line_args, lw=2.6, color="#118ab2", label="JEPA prediction")
    pred_head = ax.scatter(*[p_pred[0:1, i] for i in range(nd)], s=90, color="#118ab2", edgecolors="white", zorder=6)
    ax.legend(loc="upper right", fontsize=11)

    err = np.linalg.norm(p_pred - p_true, axis=1)
    axe.set_title("Hidden-body position error", fontsize=12)
    axe.set_xlabel("time"); axe.set_ylabel("distance from truth")
    axe.set_xlim(0, t_axis[-1]); axe.set_ylim(0, err.max() * 1.1 + 1e-6)
    axe.grid(True, ls="--", alpha=0.4)
    el_pred, = axe.plot([], [], lw=2.6, color="#118ab2", label="JEPA")
    axe.legend(loc="upper left", fontsize=11)

    def update(f):
        for b in range(n - 1):
            set_line(vis_trails[b], p_vis[:f + 1, b])
        set_pts(vis_heads, p_vis[f])
        set_line(true_trail, p_true[:f + 1])
        set_line(pred_trail, p_pred[:f + 1])
        set_pts(pred_head, p_pred[f][None, :])
        el_pred.set_data(t_axis[:f + 1], err[:f + 1])
        if is3d: ax.view_init(elev=22, azim=0.8 * f) # slow rotation
        return []

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    out = f"results/jepa_hidden_N{n}.mp4"
    try:
        ani.save(out, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception as e:
        out = f"results/jepa_hidden_N{n}.gif"
        print(f"  ffmpeg failed ({e}); gif instead")
        ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=[3])
    parser.add_argument("--traj", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("results", exist_ok=True)

    jepa = JEPADynamics(MODEL_CFG).to(device)
    jepa.load_state_dict(torch.load("results/jepa_model.pt", map_location=device, weights_only=True))
    jepa.eval()

    for n in args.ns:
        path = f"nbody_data_N{n}.npz"
        if not os.path.exists(path):
            generate(DataCfg(N=n, num_trajectories=200), filepath=path)
        dataset = NBodyDataset(filepath=path, cfg=DataCfg(N=n))
        print(f"\n--- Rendering hidden-body N={n} ---")
        build(n, jepa, dataset, device, traj_idx=args.traj)


if __name__ == "__main__":
    main()
