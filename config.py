# config.py
from dataclasses import dataclass

@dataclass
class DataCfg:
    N: int = 3 # Liu Cixin :)
    n_dim: int = 3 # spatial dimensions (2D planar or 3D)
    num_trajectories: int = 600
    integration_span: float = 10.0 # full trajectory length
    dt: float = 0.05 # sampling res
    # prediction horizon
    dt_min_steps: int = 5
    dt_max_steps: int = 20

@dataclass
class ModelCfg:
    embed_dim: int = 192 # scaling sweep sweet spot (best extrapolation; larger overfits)
    num_layers: int = 4
    n_dim: int = 3 # spatial dimensions; state is 2*n_dim features (pos + vel)
    pred_depth: int = 8 # predictor transformer depth
    input_noise: float = 0.0 # GNS-style input noise to stabilize autoregressive rollout
    dense_weight: float = 0.0 # V-JEPA 2.1 dense loss (ablated off: hurts at this scale)
    deep_supervision: bool = False # V-JEPA 2.1 deep supervision (ablated off: hurts at this scale)
    mask_frac: float = 0.34 # fraction of bodies masked (V-JEPA masks ~90%; higher = harder)

@dataclass
class TrainCfg:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 3e-4
    ema_momentum_start: float = 0.996 # JEPA specifically needs momentum for teacher / student embeddings to avoid representation collapse