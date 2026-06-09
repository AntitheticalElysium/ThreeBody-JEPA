# models/decoder.py
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelCfg
from models.encoder import MLP

class LatentDecoder(nn.Module):
    """Post-hoc latent->state map so JEPA can roll out in state space (V-JEPA has no decoder)."""
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.net = MLP(cfg.embed_dim, cfg.embed_dim, 2 * cfg.n_dim, num_layers=3)

    def forward(self, z):
        # z: [B, N, D] -> [B, N, 2*n_dim]
        return self.net(z)
