# models/rollout_predictor.py
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelCfg
from models.encoder import MLP

class RolloutPredictor(nn.Module):
    """
    Latent dynamics model (V-JEPA-2-AC style): map all bodies' current latents to their
    next latents, Δ ahead. Operates purely in latent space so predictions can be fed back
    for multi-step rollout. Output is LayerNorm'd (normalize_reps) to keep the latent space
    consistent across feedback steps and block the shrink-to-mean collapse.
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        D = cfg.embed_dim
        self.dt_embed = MLP(1, D, D)
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=4, dim_feedforward=D * 2, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.pred_depth)
        self.out = MLP(D, D, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, z, dt):
        # z [B, N, D] (LayerNorm'd latents), dt [B, 1] -> [B, N, D]
        dt_emb = self.dt_embed(dt).unsqueeze(1)
        h = self.transformer(z + dt_emb)
        return self.norm(self.out(h))
