# models/encoder.py
import torch
import torch.nn as nn
import sys
import os

# FIXME: to run as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelCfg

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.SiLU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class GraphEncoder(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.num_layers = cfg.num_layers

        # Node input: [vx, vy, mass] -> 3 features
        self.node_in_mlp = MLP(3, self.embed_dim, self.embed_dim)
        
        # Edge input: [rel_x, rel_y, distance, node_i_dim, node_j_dim] -> 3 + 2*D
        self.edge_mlps = nn.ModuleList([
            MLP(self.embed_dim * 2 + 3, self.embed_dim, self.embed_dim) 
            for _ in range(self.num_layers)
        ])
        
        # Node update input: [node_i_dim, message_agg_dim] -> 2*D
        self.node_mlps = nn.ModuleList([
            MLP(self.embed_dim * 2, self.embed_dim, self.embed_dim) 
            for _ in range(self.num_layers)
        ])

    def forward(self, state: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """
        state: [B, N, 4] where features are (x, y, vx, vy)
        mass:  [B, N]
        Returns: [B, N, D]
        """
        B, N, _ = state.shape

        pos = state[..., :2]  # [B, N, 2]
        vel = state[..., 2:]  # [B, N, 2]

        # Compute relative positions and distances
        # relative position of body1 to bodyN
        rel_pos = pos.unsqueeze(2) - pos.unsqueeze(1) # [B, N, N, 2]
        # straight-line distance from body1 to bodyN
        dist = torch.sqrt(torch.sum(rel_pos**2, dim=-1, keepdim=True) + 1e-8) # [B, N, N, 1]
        edge_attr = torch.cat([rel_pos, dist], dim=-1) # [B, N, N, 3]
        
        # bundle each body's velocities and mass
        mass_expanded = mass.unsqueeze(-1) # [B, N, 1]
        node_attr = torch.cat([vel, mass_expanded], dim=-1) # [B, N, 3]
        # get the initial embedding from the mlp
        x = self.node_in_mlp(node_attr) # [B, N, D]
        
        # mask to zero out self-edges (1 == N)
        mask = ~torch.eye(N, dtype=torch.bool, device=x.device) # [N, N]
        mask = mask.view(1, N, N, 1) # [1, N, N, 1]
        
        # Passing messages (forces here) between bodies
        for i in range(self.num_layers):
            # Broadcast nodes to form all pairs: [B, N, N, D]
            x_i = x.unsqueeze(2).expand(-1, -1, N, -1) # Source nodes
            x_j = x.unsqueeze(1).expand(-1, N, -1, -1) # Target nodes
            
            # Compute messages: m_ij
            edge_inputs = torch.cat([x_i, x_j, edge_attr], dim=-1) # [B, N, N, 2D + 3]
            m_ij = self.edge_mlps[i](edge_inputs) # [B, N, N, D]
            
            # Mask out self-interactions
            m_ij = m_ij * mask
            
            # Aggregate: sum over j (target nodes)
            m_i = m_ij.sum(dim=2) # [B, N, D]
            
            # Update: apply node MLP and residual connection
            node_inputs = torch.cat([x, m_i], dim=-1) # [B, N, 2D]
            dx = self.node_mlps[i](node_inputs) # [B, N, D]
            
            x = x + dx # [B, N, D]
            
        return x
