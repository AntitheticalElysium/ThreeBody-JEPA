# models/baselines.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelCfg
from models.encoder import GraphEncoder, MLP

class RegressionDynamics(nn.Module):
    """
    Predicts the next normalized state directly.
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.encoder = GraphEncoder(cfg)
        self.dt_embed = MLP(1, cfg.embed_dim, cfg.embed_dim)
        # decoder maps node embeddings back to state features (x, y, vx, vy)
        self.decoder = MLP(cfg.embed_dim, cfg.embed_dim, 4)

    def forward(self, state_ctx, state_tgt, dt, mass):
        # Encode context
        z = self.encoder(state_ctx, mass) # [B, N, D]
        # Condition on time offset
        dt_emb = self.dt_embed(dt).unsqueeze(1) # [B, 1, D]
        z_t = z + dt_emb
        # Predict next state
        state_pred = self.decoder(z_t) # [B, N, 4]
        # MSE Loss in normalized state space
        loss = F.mse_loss(state_pred, state_tgt)
        return loss, state_pred

    @torch.no_grad()
    def step_forward(self, state, dt, mass):
        """Used for rollouts during evaluation."""
        z = self.encoder(state, mass)
        dt_emb = self.dt_embed(dt).unsqueeze(1)
        return self.decoder(z + dt_emb)


class HNNDynamics(nn.Module):
    """
    Learns a scalar Hamiltonian (Energy basically) and uses autograd to enforce dx/dt = M * grad(H).
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.encoder = GraphEncoder(cfg)
        # H_head outputs a single Hamiltonian scalar for the whole system
        self.H_head = MLP(cfg.embed_dim, cfg.embed_dim, 1)

    def forward(self, state_ctx, state_tgt, dt, mass):
        B, N, _ = state_ctx.shape
        
        # Prepare canonical coordinates with gradients enabled
        # q = positions, p = momentum (mass * velocity)
        q = state_ctx[..., :2].clone().requires_grad_(True)
        v = state_ctx[..., 2:].clone()
        p = (v * mass.unsqueeze(-1)).clone().requires_grad_(True)
        
        # reconstruct the state to feed into the encoder
        v_recon = p / mass.unsqueeze(-1)
        state_in = torch.cat([q, v_recon], dim=-1)

        # Compute the scalar Hamiltonian
        z = self.encoder(state_in, mass) # [B, N, D]
        z_sys = z.sum(dim=1)             # [B, D] 
        H = self.H_head(z_sys)           # [B, 1]
        
        dH_dq, dH_dp = torch.autograd.grad(
            H.sum(), [q, p], create_graph=True
        )
        
        # Hamilton's equations
        dq_dt = dH_dp           # velocity
        dp_dt = -dH_dq          # force
        dv_dt = dp_dt / mass.unsqueeze(-1)
        
        dstate_dt = torch.cat([dq_dt, dv_dt], dim=-1) # [B, N, 4]
        
        dt_expanded = dt.view(-1, 1, 1)
        dstate_dt_target = (state_tgt - state_ctx) / dt_expanded
        
        loss = F.mse_loss(dstate_dt, dstate_dt_target)
        return loss, dstate_dt

    def step_forward(self, state, dt, mass):
        """Euler integration step for rollouts."""
        q = state[..., :2].clone().requires_grad_(True)
        p = (state[..., 2:] * mass.unsqueeze(-1)).clone().requires_grad_(True)
        v_recon = p / mass.unsqueeze(-1)
        state_in = torch.cat([q, v_recon], dim=-1)
        
        with torch.enable_grad():
            z = self.encoder(state_in, mass)
            H = self.H_head(z.sum(dim=1))
            dH_dq, dH_dp = torch.autograd.grad(H.sum(), [q, p])
            
        dq_dt = dH_dp
        dv_dt = -dH_dq / mass.unsqueeze(-1)
        dstate_dt = torch.cat([dq_dt, dv_dt], dim=-1)
        
        return state + dstate_dt * dt.view(-1, 1, 1)