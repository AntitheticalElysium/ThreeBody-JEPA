# train_rollout.py
import os
import math
import argparse
import torch
import torch.nn.functional as F

from config import DataCfg, ModelCfg
from data import NBodyDataset
from models.jepa import JEPADynamics
from models.rollout_predictor import RolloutPredictor

MCFG = ModelCfg()


def ln(z):
    return F.layer_norm(z, (z.size(-1),))


def main():
    parser = argparse.ArgumentParser(description="Train the latent rollout predictor (V-JEPA-2-AC style).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--delta", type=int, default=5, help="rollout step in integrator steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    ds = NBodyDataset("nbody_data_N3.npz", DataCfg(N=3))

    # frozen encoder from the trained masked JEPA (Stage 1)
    jepa = JEPADynamics(MCFG).to(device)
    jepa.load_state_dict(torch.load("results/jepa_model.pt", map_location=device, weights_only=True))
    encoder = jepa.target_encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    predictor = RolloutPredictor(MCFG).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=3e-4, weight_decay=1e-4)

    states = ds.states.to(device)
    masses = ds.masses.to(device)
    num_traj, num_steps, N, _ = states.shape
    d = args.delta
    dt_val = d * ds.cfg.dt

    bs = 256
    iters = (num_traj * num_steps) // bs
    total = args.epochs * iters
    warmup = max(1, int(0.05 * total))
    step = 0

    print(f"--- Training RolloutPredictor | delta={d} steps ({dt_val:.2f}t) | {iters} iters/epoch ---")
    predictor.train()
    for epoch in range(1, args.epochs + 1):
        ep_loss = 0.0
        for _ in range(iters):
            traj = torch.randint(0, num_traj, (bs,), device=device)
            t = torch.randint(0, num_steps - 2 * d, (bs,), device=device)
            mass = masses[traj]
            s0 = ds.normalize(states[traj, t])
            s1 = ds.normalize(states[traj, t + d])
            s2 = ds.normalize(states[traj, t + 2 * d])
            dt = torch.full((bs, 1), dt_val, device=device)

            with torch.no_grad():
                z0 = encoder(s0, mass) # encoder.forward returns the LN'd final rep
                h1 = encoder(s1, mass)
                h2 = encoder(s2, mass)

            pred1 = predictor(z0, dt)            # teacher-forced 1-step
            jloss = F.l1_loss(pred1, h1)
            pred2 = predictor(pred1, dt)         # feed prediction back (grads flow): 2-step rollout
            sloss = F.l1_loss(pred2, h2)
            loss = jloss + sloss

            lr = (3e-4 * (step + 1) / warmup) if step < warmup else \
                 1e-6 + 0.5 * (3e-4 - 1e-6) * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            step += 1
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | Avg Loss: {ep_loss / iters:.5f}")

    os.makedirs("results", exist_ok=True)
    torch.save(predictor.state_dict(), "results/rollout_predictor.pt")
    print("Saved results/rollout_predictor.pt")


if __name__ == "__main__":
    main()
