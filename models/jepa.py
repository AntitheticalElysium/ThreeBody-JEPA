# models/jepa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelCfg, DataCfg, TrainCfg
from models.encoder import GraphEncoder
from models.predictor import Predictor

class JEPADynamics(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.encoder = GraphEncoder(cfg) # see curr state
        self.predictor = Predictor(cfg) # predict future representation
        self.input_noise = cfg.input_noise
        self.dense_weight = cfg.dense_weight
        self.mask_frac = cfg.mask_frac
        
        # produces training target
        self.target_encoder = copy.deepcopy(self.encoder)
        # only updates via ema for stability
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, state_ctx, state_tgt, dt, mass):
        """Returns the smooth L1 loss and the masked target embeddings for logging."""
        B, N, _ = state_ctx.shape
        k = min(N - 1, max(1, round(self.mask_frac * N))) # masked (target) bodies; keep >=1 visible
        perm = torch.randperm(N, device=state_ctx.device)
        tgt_idx, ctx_idx = perm[:k], perm[k:]

        # GNS-style input noise: make encoder/predictor robust to the drift they meet in rollout
        if self.training and self.input_noise > 0:
            state_ctx = state_ctx + torch.randn_like(state_ctx) * self.input_noise

        # multi-level features of the visible (context) bodies
        hier_ctx = self.encoder.forward_hier(state_ctx[:, ctx_idx], mass[:, ctx_idx])

        with torch.no_grad():
            h = self.target_encoder.forward_hier(state_tgt, mass) # [B, N, L*D], per-level LN'd

        x_context, x_pred = self.predictor(hier_ctx, state_ctx[:, ctx_idx], state_ctx[:, tgt_idx], dt)
        # V-JEPA 2.1: prediction loss (masked) + dense context loss (visible bodies), L1
        loss = F.l1_loss(x_pred, h[:, tgt_idx]) + self.dense_weight * F.l1_loss(x_context, h[:, ctx_idx])
        return loss, h[:, tgt_idx]

    def predict_final(self, state_ctx, mass, state_tgt, dt):
        """Inference: predicted LayerNorm'd final-level rep of the masked bodies (for decode)."""
        hier_ctx = self.encoder.forward_hier(state_ctx, mass)
        _, x_pred = self.predictor(hier_ctx, state_ctx, state_tgt, dt)
        return x_pred[..., -self.encoder.embed_dim:] # last level = LN'd final layer
        
    @torch.no_grad()
    def step_ema(self, momentum: float):
        for q_param, k_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k_param.data.mul_(momentum).add_((1.0 - momentum) * q_param.data)