# models/predictor.py
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelCfg
from models.encoder import MLP

class Predictor(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        
        # embed the scalar delta-t into the same dimension as the nodes
        self.dt_embed = MLP(1, self.embed_dim, self.embed_dim)
        
        # self-attention across the N bodies.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=self.embed_dim * 2,
            batch_first=True,
            activation="gelu"
        )
        self.relational = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, z_ctx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        z_ctx: [B, N, D] (Context node embeddings)
        dt:    [B, 1] (Time step offset to predict into the future)
        Returns: [B, N, D]
        """
        # Embed time offset
        dt_emb = self.dt_embed(dt)  # [B, D]
        # Add time embedding to all nodes
        # Broadcasts [B, D] to [B, 1, D], which adds to [B, N, D]
        z_in = z_ctx + dt_emb.unsqueeze(1) 
        # Predict interactions
        z_pred = self.relational(z_in) # [B, N, D]

        return z_pred