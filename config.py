# config.py
from dataclasses import dataclass

@dataclass
class DataCfg:
    N: int = 3 # Liu Cixin :) 
    num_trajectories: int = 100 # start small for testing
    integration_span: float = 10.0 # full trajectory length
    dt: float = 0.05 # sampling res
    # prediction horizon
    dt_min_steps: int = 5        
    dt_max_steps: int = 20

@dataclass
class ModelCfg:
    embed_dim: int = 128
    num_layers: int = 4 
    # TODO

@dataclass
class TrainCfg:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4
    ema_momentum_start: float = 0.996 # JEPA specifically needs momentum for teacher / student embeddings to avoid representation collapse