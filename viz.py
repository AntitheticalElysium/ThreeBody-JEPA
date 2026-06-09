# viz.py
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from config import DataCfg, ModelCfg
from data import NBodyDataset, generate
from models.jepa import JEPADynamics
from models.baselines import RegressionDynamics, HNNDynamics
from models.decoder import LatentDecoder
from models.rollout_predictor import RolloutPredictor
from eval import rollout, rollout_latent, get_physical_energy

MODEL_CFG = ModelCfg()
COLORS = {"Regression": "#ef476f", "JEPA": "#118ab2", "HNN": "#06d6a0"}
ORDER = ["Regression", "JEPA", "HNN"]
SUBTITLE = {"Regression": "no physics (floor)", "JEPA": "latent prediction", "HNN": "Hamiltonian (ceiling)"}


def load_models(device):
    specs = {"Regression": RegressionDynamics, "JEPA": JEPADynamics, "HNN": HNNDynamics}
    models = {}
    for name, cls in specs.items():
        ckpt = f"results/{name.lower()}_model.pt"
        m = cls(MODEL_CFG).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        m.eval()
        models[name] = m
    return models


def get_decoder(jepa, dataset, device, n, max_steps=2500):
    """Post-hoc latent->state readout, fit per-N on the frozen JEPA target encoder."""
    path = f"results/jepa_decoder_N{n}.pt"
    decoder = LatentDecoder(MODEL_CFG).to(device)
    if os.path.exists(path):
        decoder.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        decoder.eval()
        return decoder

    print(f"  Training latent decoder for N={n}...")
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = 0
    while steps < max_steps: # capped: a trivial per-body map, dataset size irrelevant
        for state_ctx, _, _, mass in loader:
            state_ctx, mass = state_ctx.to(device), mass.to(device)
            with torch.no_grad():
                h = jepa.target_encoder(state_ctx, mass) # LN'd final rep (canonical)
            opt.zero_grad()
            F.mse_loss(decoder(h), state_ctx).backward()
            opt.step()
            steps += 1
            if steps >= max_steps:
                break
    decoder.eval()
    torch.save(decoder.state_dict(), path)
    return decoder


def physical_positions(states_normed, dataset):
    pos = states_normed[..., :dataset.n_dim] * dataset.pos_std + dataset.pos_mean
    return pos.cpu().numpy() # [frames, N, n_dim]


def energy_series(states_normed, mass, dataset):
    e = [get_physical_energy(s.unsqueeze(0), mass, dataset).item() for s in states_normed]
    return np.array(e)


def build_animation(n, models, dataset, device, frame_dt=0.25, n_frames=60, fps=10):
    mass = dataset.masses[0].unsqueeze(0).to(device)
    s0 = dataset.normalize(dataset.states[0, 0].unsqueeze(0).to(device))

    decoder = get_decoder(models["JEPA"], dataset, device, n)
    rp = RolloutPredictor(MODEL_CFG).to(device)
    rp.load_state_dict(torch.load("results/rollout_predictor.pt", map_location=device, weights_only=True))
    rp.eval()

    # ground truth from the seed trajectory (available for the first stride*frames steps)
    stride = max(1, round(frame_dt / dataset.cfg.dt))
    gt = dataset.normalize(dataset.states[0, :stride * n_frames:stride].to(device))
    gt_pos = physical_positions(gt, dataset)

    trajs = {}
    for name, model in models.items():
        if name == "JEPA":
            traj = rollout_latent(model, rp, decoder, s0, mass, frame_dt, n_frames)
        else:
            traj = rollout(model, s0, mass, frame_dt, n_frames)
        trajs[name] = physical_positions(traj, dataset)

    # mean per-body position error vs ground truth (while the seed trajectory lasts)
    gt_len = len(gt_pos)
    perr = {nm: np.linalg.norm(trajs[nm][:gt_len] - gt_pos, axis=2).mean(axis=1) for nm in ORDER}

    # axis limits from ground truth, padded; escapers leave frame (that's the point)
    lim = np.nanpercentile(np.abs(gt_pos), 99) * 1.8
    lim = max(lim, 1.0)

    nd = dataset.n_dim
    is3d = nd == 3
    line_args = ([], [], []) if is3d else ([], [])

    def set_line(line, pts):
        line.set_data(pts[:, 0], pts[:, 1])
        if is3d: line.set_3d_properties(pts[:, 2])

    def set_pts(scat, pts):
        if is3d: scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        else: scat.set_offsets(pts)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.28, wspace=0.18)
    traj_ax = [fig.add_subplot(gs[0, c], projection="3d" if is3d else None) for c in range(3)]
    energy_ax = fig.add_subplot(gs[1, :])
    fig.suptitle(f"Trained on N=3  →  zero-shot rollout at N={n}", fontsize=17, fontweight="bold")

    body_colors = plt.cm.viridis(np.linspace(0, 1, n))
    t_axis = np.arange(n_frames + 1) * frame_dt

    # static setup
    artists = {}
    for c, name in enumerate(ORDER):
        ax = traj_ax[c]
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xticks([]); ax.set_yticks([])
        if is3d: ax.set_zlim(-lim, lim); ax.set_zticks([])
        else: ax.set_aspect("equal")
        ax.set_title(f"{name}\n{SUBTITLE[name]}", color=COLORS[name], fontsize=13, fontweight="bold")
        trails = [ax.plot(*line_args, lw=1.5, color=body_colors[b], alpha=0.8)[0] for b in range(n)]
        heads = ax.scatter(*[trajs[name][0, :, i] for i in range(nd)], s=60,
                           color=body_colors, edgecolors="white", zorder=5, linewidths=0.8)
        gt_trails = [ax.plot(*line_args, lw=1.0, color="gray", alpha=0.25, ls="--")[0] for b in range(n)]
        artists[name] = (trails, heads, gt_trails)

    err_t = np.arange(gt_len) * frame_dt
    energy_ax.set_title("Trajectory error vs ground truth (chaos makes all diverge eventually)", fontsize=12)
    energy_ax.set_xlabel("time"); energy_ax.set_ylabel("mean position error")
    energy_ax.set_xlim(0, t_axis[-1])
    energy_ax.set_ylim(0, max(perr[nm].max() for nm in ORDER) * 1.1 + 1e-6)
    energy_ax.grid(True, ls="--", alpha=0.4)
    err_lines = {name: energy_ax.plot([], [], lw=2.5, color=COLORS[name], label=name)[0] for name in ORDER}
    energy_ax.legend(loc="upper left", ncol=3)

    def update(f):
        for c, name in enumerate(ORDER):
            trails, heads, gt_trails = artists[name]
            traj = trajs[name]
            for b in range(n):
                set_line(trails[b], traj[:f + 1, b])
                if f < len(gt_pos):
                    set_line(gt_trails[b], gt_pos[:f + 1, b])
            set_pts(heads, traj[f])
            if is3d: traj_ax[c].view_init(elev=22, azim=0.8 * f)
            err_lines[name].set_data(err_t[:min(f + 1, gt_len)], perr[name][:min(f + 1, gt_len)])
        return []

    ani = FuncAnimation(fig, update, frames=n_frames + 1, interval=1000 / fps, blit=False)
    out = f"results/rollout_N{n}.mp4"
    try:
        ani.save(out, writer=FFMpegWriter(fps=fps, bitrate=3000))
    except Exception as e:
        out = f"results/rollout_N{n}.gif"
        print(f"  ffmpeg failed ({e}); writing gif instead")
        ani.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Render side-by-side rollout animations.")
    parser.add_argument("--ns", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("results", exist_ok=True)

    models = load_models(device)
    for n in args.ns:
        path = f"nbody_data_N{n}.npz"
        if not os.path.exists(path):
            generate(DataCfg(N=n, num_trajectories=200), filepath=path)
        dataset = NBodyDataset(filepath=path, cfg=DataCfg(N=n))
        print(f"\n--- Rendering N={n} ---")
        build_animation(n, models, dataset, device, n_frames=args.frames)


if __name__ == "__main__":
    main()
