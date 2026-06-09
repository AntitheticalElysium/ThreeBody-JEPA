# train.py
import os
import math
import argparse
import dataclasses
import torch
from torch.utils.data import DataLoader

from config import DataCfg, ModelCfg, TrainCfg
from data import NBodyDataset
from models.jepa import JEPADynamics
from models.baselines import RegressionDynamics, HNNDynamics

def get_model(objective: str, cfg: ModelCfg, device: torch.device):
    """Instantiates the chosen objective wrapped around the shared graph backbone."""
    if objective == "jepa":
        return JEPADynamics(cfg).to(device)
    elif objective == "regression":
        return RegressionDynamics(cfg).to(device)
    elif objective == "hnn":
        return HNNDynamics(cfg).to(device)
    else:
        raise ValueError(f"Unknown objective: {objective}")

def main():
    parser = argparse.ArgumentParser(description="Train N-Body models.")
    parser.add_argument("--objective", type=str, required=True, choices=["jepa", "regression", "hnn"])
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--dt-min-steps", type=int, default=None, help="Override prediction horizon")
    parser.add_argument("--dt-max-steps", type=int, default=None, help="Override prediction horizon")
    parser.add_argument("--out-tag", type=str, default=None, help="Checkpoint name (default: objective)")
    parser.add_argument("--input-noise", type=float, default=0.0, help="GNS-style input noise (JEPA)")
    parser.add_argument("--dense-weight", type=float, default=0.0, help="V-JEPA 2.1 dense loss weight (JEPA)")
    parser.add_argument("--deep-supervision", type=int, default=0, help="V-JEPA 2.1 multi-layer fusion (1/0)")
    parser.add_argument("--embed-dim", type=int, default=192, help="model width")
    parser.add_argument("--num-layers", type=int, default=4, help="encoder message-passing layers")
    parser.add_argument("--pred-depth", type=int, default=8, help="predictor transformer depth")
    parser.add_argument("--data-path", type=str, default=None, help="override training data .npz")
    parser.add_argument("--mask-frac", type=float, default=0.34, help="fraction of bodies masked (JEPA)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    args = parser.parse_args()

    # Hardware acceleration setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load configurations
    data_cfg = DataCfg()
    model_cfg = ModelCfg(embed_dim=args.embed_dim, num_layers=args.num_layers, pred_depth=args.pred_depth,
                         input_noise=args.input_noise, dense_weight=args.dense_weight,
                         deep_supervision=bool(args.deep_supervision), mask_frac=args.mask_frac)
    train_cfg = TrainCfg()
    
    if args.epochs is not None:
        train_cfg.epochs = args.epochs

    # HNN's derivative target needs adjacent steps; JEPA/regression keep the long horizon
    ds_cfg = data_cfg
    if args.dt_min_steps is not None:
        ds_cfg = dataclasses.replace(data_cfg, dt_min_steps=args.dt_min_steps, dt_max_steps=args.dt_max_steps)
    elif args.objective == "hnn":
        ds_cfg = dataclasses.replace(data_cfg, dt_min_steps=1, dt_max_steps=1)

    # Data Loading (train on N=3, matching the scaling.py convention)
    data_path = args.data_path or f"nbody_data_N{data_cfg.N}.npz"
    if not os.path.exists(data_path):
        from data import generate
        print(f"Data not found. Generating {data_path}...")
        generate(data_cfg, filepath=data_path)

    dataset = NBodyDataset(filepath=data_path, cfg=ds_cfg)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)

    # Setup Model & Optimizer (paper's 0.04->0.4 WD over-regularizes a model this small)
    model = get_model(args.objective, model_cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=args.weight_decay)

    os.makedirs("results", exist_ok=True)
    
    # Pre-calculate steps for EMA schedule (only used if objective == 'jepa')
    total_steps = len(loader) * train_cfg.epochs
    global_step = 0
    m_start = train_cfg.ema_momentum_start

    # lr warmup + cosine decay (the transformer predictor diverges under a flat lr)
    warmup_steps = max(1, int(0.05 * total_steps))
    base_lr, min_lr = train_cfg.lr, 1e-6
    def lr_at(step):
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


    # dense-loss warmup (V-JEPA 2.1): 0 for first 30%, linear ramp to target over 30-60%, hold
    dw_s0, dw_s1 = int(0.3 * total_steps), int(0.6 * total_steps)
    def dense_at(step):
        if step < dw_s0: return 0.0
        if step >= dw_s1: return args.dense_weight
        return args.dense_weight * (step - dw_s0) / max(1, dw_s1 - dw_s0)

    print(f"\n--- Starting Training [{args.objective.upper()}] ---")
    print(f"Epochs: {train_cfg.epochs} | Batch Size: {train_cfg.batch_size} | Steps/Epoch: {len(loader)}")
    
    model.train()
    for epoch in range(1, train_cfg.epochs + 1):
        epoch_loss = 0.0
        
        for batch in loader:
            # Move all tensors to device
            state_ctx, state_tgt, dt, mass = [b.to(device) for b in batch]

            for g in optimizer.param_groups:
                g["lr"] = lr_at(global_step)
            if args.objective == "jepa":
                model.dense_weight = dense_at(global_step)

            optimizer.zero_grad()

            # All models return (loss, predictions/latents)
            loss, _ = model(state_ctx, state_tgt, dt, mass)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Schedule and step EMA target network
            if args.objective == "jepa":
                # Linear ramp from m_start (0.996) -> 1.0
                m_current = m_start + (1.0 - m_start) * (global_step / total_steps)
                model.step_ema(momentum=m_current)

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:03d}/{train_cfg.epochs:03d} | Avg Loss: {avg_loss:.5f}")

    # Save Checkpoint
    ckpt_path = f"results/{args.out_tag or args.objective}_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    
    print(f"\n Training complete.")
    print(f"Checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    main()