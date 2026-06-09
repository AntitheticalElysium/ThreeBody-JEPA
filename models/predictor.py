# models/predictor.py
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelCfg
from models.encoder import MLP

class Predictor(nn.Module):
    """
    Body-masked predictor (V-JEPA 2.1). Context input is the visible bodies' multi-level
    features (L*D) fused by an MLP. It outputs an L*D vector for masked bodies (x_pred) and
    for visible bodies (x_context), matching the L LayerNorm'd encoder levels (channel-concat
    deep self-supervision). Predictor outputs are not normalized (normalize_predictor=false).
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        D = cfg.embed_dim
        self.L = cfg.num_layers if cfg.deep_supervision else 1

        self.embed = MLP(D * self.L, D, D)        # fuse multi-level context input
        self.state_embed = MLP(2 * cfg.n_dim, D, D)
        self.dt_embed = MLP(1, D, D)
        self.mask_token = nn.Parameter(torch.zeros(D))
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=4, dim_feedforward=D * 2, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.pred_depth)
        self.proj = nn.Linear(D, D * self.L)      # masked-body prediction (L levels)
        self.proj_ctx = nn.Linear(D, D * self.L)  # visible-body prediction (dense loss)

    def forward(self, hier_ctx, state_ctx_t, state_tgt_t, dt):
        # hier_ctx [B,Nc,L*D]; state_*_t [B,*,2*n_dim]; dt [B,1]
        dt_emb = self.dt_embed(dt).unsqueeze(1)
        ctx = self.embed(hier_ctx) + self.state_embed(state_ctx_t) + dt_emb
        q = self.mask_token.view(1, 1, -1) + self.state_embed(state_tgt_t) + dt_emb

        Nc = ctx.size(1)
        shared = self.transformer(torch.cat([ctx, q], dim=1))
        return self.proj_ctx(shared[:, :Nc]), self.proj(shared[:, Nc:]) # x_context, x_pred  [.., L*D]
