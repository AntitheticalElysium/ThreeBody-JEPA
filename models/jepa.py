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
        
        # produces training target
        self.target_encoder = copy.deepcopy(self.encoder)
        # only updates via ema for stability
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(self, state_ctx, state_tgt, dt, mass):
        """Returns the smooth L1 loss and the target embeddings for logging."""
        with torch.no_grad():
            h = self.target_encoder(state_tgt, mass) # [B, N, D]
            # empirical trick to stabilize target variance
            h = F.layer_norm(h, (h.size(-1),)) 
            
        z = self.encoder(state_ctx, mass) # [B, N, D]
        z_pred = self.predictor(z, dt)    # [B, N, D]
        
        loss = F.smooth_l1_loss(z_pred, h)
        return loss, h
        
    @torch.no_grad()
    def step_ema(self, momentum: float):
        for q_param, k_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            k_param.data.mul_(momentum).add_((1.0 - momentum) * q_param.data)