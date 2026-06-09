# eval.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from config import DataCfg, ModelCfg
from data import NBodyDataset
from models.baselines import RegressionDynamics, HNNDynamics
from models.jepa import JEPADynamics

def get_physical_energy(state_normed: torch.Tensor, mass: torch.Tensor, dataset: NBodyDataset):
    """
    Reconstructs the raw physical energy from normalized states.
    state_normed: [B, N, 4]
    mass: [B, N]
    """
    # unnormalize
    nd = dataset.n_dim
    state = state_normed.clone()
    state[..., :nd] = state[..., :nd] * dataset.pos_std + dataset.pos_mean
    state[..., nd:] = state[..., nd:] * dataset.vel_std + dataset.vel_mean

    pos = state[..., :nd] # [B, N, n_dim]
    vel = state[..., nd:] # [B, N, n_dim]
    
    # kinetic (0.5 * m * v^2)
    K = 0.5 * torch.sum(mass.unsqueeze(-1) * vel**2, dim=(1, 2)) # [B]
    
    # potential  (- G * m1 * m2 / r)
    U = torch.zeros_like(K)
    N = mass.shape[1]
    for i in range(N):
        for j in range(i + 1, N):
            dist = torch.norm(pos[:, i, :] - pos[:, j, :], dim=-1) + 1e-8
            U -= (mass[:, i] * mass[:, j]) / dist
            
    return K + U # [B]

def _eval_encoder(model, state, mass):
    # JEPA reports on the EMA target encoder (V-JEPA eval protocol)
    if isinstance(model, JEPADynamics):
        return model.target_encoder(state, mass)
    return model.encoder(state, mass)

@torch.no_grad()
def _collect_features(model, dataset: NBodyDataset, batch_size=256):
    """Mean-pooled latents and true energies over the whole dataset."""
    model.eval()
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    Z, E = [], []
    for batch in loader:
        state_ctx, _, _, mass = [b.to(device) for b in batch]
        z = _eval_encoder(model, state_ctx, mass).mean(dim=1) # mean-pool over bodies
        Z.append(z.cpu())
        E.append(get_physical_energy(state_ctx, mass, dataset).cpu())

    X = torch.cat(Z, dim=0).double()
    Y = torch.cat(E, dim=0).unsqueeze(1).double()
    return X, Y

def _split_standardize(X, Y, train_frac=0.8, seed=0):
    n = X.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_tr = int(n * train_frac)
    tr, te = idx[:n_tr], idx[n_tr:]
    # standardize on train stats; R^2 is scale-invariant
    Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-8
    Ym, Ys = Y[tr].mean(), Y[tr].std() + 1e-8
    Xn, Yn = (X - Xm) / Xs, (Y - Ym) / Ys
    return Xn[tr], Yn[tr], Xn[te], Yn[te]

def _r2(Y_true, Y_pred):
    ss_res = torch.sum((Y_true - Y_pred) ** 2)
    ss_tot = torch.sum((Y_true - Y_true.mean()) ** 2)
    return (1 - ss_res / ss_tot).item()

def linear_probe(model, dataset: NBodyDataset, batch_size=256):
    """Held-out R^2 of energy linearly decoded from the frozen mean-pooled latent."""
    X, Y = _collect_features(model, dataset, batch_size)
    Xtr, Ytr, Xte, Yte = _split_standardize(X, Y)

    Xtr_b = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, dtype=torch.float64)], dim=1)
    A = Xtr_b.T @ Xtr_b + 1e-6 * torch.eye(Xtr_b.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(A, Xtr_b.T @ Ytr)

    Xte_b = torch.cat([Xte, torch.ones(Xte.shape[0], 1, dtype=torch.float64)], dim=1)
    return _r2(Yte, Xte_b @ W)

def nonlinear_probe(model, dataset: NBodyDataset, batch_size=256, steps=800):
    """Held-out R^2 of energy decoded by a small MLP (V-JEPA reps need a nonlinear head)."""
    X, Y = _collect_features(model, dataset, batch_size)
    Xtr, Ytr, Xte, Yte = _split_standardize(X, Y)

    device = next(model.parameters()).device
    Xtr, Ytr, Xte, Yte = [t.float().to(device) for t in (Xtr, Ytr, Xte, Yte)]
    probe = nn.Sequential(
        nn.Linear(Xtr.shape[1], 128), nn.GELU(), nn.Linear(128, 1)
    ).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)

    probe.train()
    for _ in range(steps):
        opt.zero_grad()
        F.mse_loss(probe(Xtr), Ytr).backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        return _r2(Yte.double(), probe(Xte).double())

@torch.no_grad()
def conservation_drift(model, dataset: NBodyDataset, n_seeds=20, frame_dt=0.25, n_frames=40):
    """
    Median relative energy drift (std(E)/|E0|) over a state-space rollout, averaged
    across seeds. Uses the shared rollout primitive so each model steps at its trained
    resolution (HNN sub-steps, regression one frame_dt). Regression/HNN only.
    """
    model.eval()
    device = next(model.parameters()).device
    n_seeds = min(n_seeds, dataset.states.shape[0])

    drifts = []
    for i in range(n_seeds):
        s0 = dataset.normalize(dataset.states[i, 0].unsqueeze(0).to(device))
        mass = dataset.masses[i].unsqueeze(0).to(device)
        traj = rollout(model, s0, mass, frame_dt, n_frames)
        e = torch.tensor([get_physical_energy(st.unsqueeze(0), mass, dataset).item() for st in traj])
        e0 = e[0].abs() + 1e-8
        drifts.append((e.std() / e0).item())

    return float(np.nanmedian(drifts))

@torch.no_grad()
def rollout(model, state0, mass, frame_dt, n_frames, substep=0.05):
    """
    Autoregressive state-space rollout for regression / HNN (JEPA uses rollout_latent).
    HNN sub-steps at `substep` (its trained resolution); regression advances one frame_dt.
    """
    model.eval()
    device = state0.device
    s = state0
    traj = [s.squeeze(0)]

    frame_dt_t = torch.tensor([[frame_dt]], device=device)
    sub_dt_t = torch.tensor([[substep]], device=device)
    n_sub = max(1, round(frame_dt / substep))

    for _ in range(n_frames):
        if isinstance(model, HNNDynamics):
            for _ in range(n_sub):
                s = model.step_forward(s, sub_dt_t, mass)
        else:
            s = model.step_forward(s, frame_dt_t, mass)
        traj.append(s.squeeze(0))

    return torch.stack(traj) # [n_frames+1, N, 4]

@torch.no_grad()
def rollout_latent(jepa, rollout_predictor, decoder, state0, mass, frame_dt, n_frames):
    """V-JEPA-2-style latent rollout: roll forward in latent space, decode only for display."""
    jepa.eval(); rollout_predictor.eval()
    device = state0.device
    dt = torch.tensor([[frame_dt]], device=device)
    z = jepa.target_encoder(state0, mass) # LN'd final rep (canonical latent)
    traj = [state0.squeeze(0)]
    for _ in range(n_frames):
        z = rollout_predictor(z, dt)
        traj.append(decoder(z).squeeze(0))
    return torch.stack(traj)