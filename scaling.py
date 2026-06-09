# scaling.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import DataCfg, ModelCfg
from data import generate, NBodyDataset
from models.jepa import JEPADynamics
from models.baselines import RegressionDynamics, HNNDynamics
from eval import linear_probe, nonlinear_probe, conservation_drift

def main():
    print("=== N-Body JEPA Scaling Experiment ===")
    
    # Setup configs
    model_cfg = ModelCfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Generate Test Datasets across the scaling sweep
    ns = [3, 4, 5]
    datasets = {}
    for n in ns:
        path = f"nbody_data_N{n}.npz"
        if not os.path.exists(path):
            print(f"Generating N={n} data (200 trajectories)...")
            # Generate slightly larger test sets for smooth evaluation curves
            cfg = DataCfg(N=n, num_trajectories=200) 
            generate(cfg, filepath=path)
        datasets[n] = NBodyDataset(filepath=path, cfg=DataCfg(N=n))

    # 2. Load Models (Trained on N=3)
    models = {
        "Regression (Floor)": RegressionDynamics(model_cfg).to(device),
        "HNN (Ceiling)": HNNDynamics(model_cfg).to(device),
        "JEPA (Hypothesis)": JEPADynamics(model_cfg).to(device)
    }

    print("\nLoading checkpoints...")
    for name in models.keys():
        file_name = name.split()[0].lower() # 'regression', 'hnn', 'jepa'
        ckpt = f"results/{file_name}_model.pt"
        if os.path.exists(ckpt):
            # Load weights safely
            models[name].load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            models[name].eval()
            print(f"  [OK] {ckpt} loaded.")
        else:
            print(f"\nError: {ckpt} missing.")
            print(f"Run: uv run python train.py --objective {file_name} --epochs 20")
            return

    # 3. Evaluate Metrics
    results_lin = {k: [] for k in models.keys()}
    results_nl = {k: [] for k in models.keys()}
    results_drift = {"Regression (Floor)": [], "HNN (Ceiling)": []} # JEPA is latent-only

    for n in ns:
        print(f"\n--- Evaluating Zero-Shot on N={n} ---")
        dataset = datasets[n]

        for name, model in models.items():
            # Energy decodability: linear (Alain-Bengio) and nonlinear (V-JEPA) probes
            r2_lin = linear_probe(model, dataset, batch_size=256)
            r2_nl = nonlinear_probe(model, dataset, batch_size=256)
            results_lin[name].append(r2_lin)
            results_nl[name].append(r2_nl)
            print(f"  {name:20s} | Linear R^2: {r2_lin:.4f} | Nonlinear R^2: {r2_nl:.4f}")

            # Conservation drift (state-space predictors only)
            if name in results_drift:
                drift = conservation_drift(model, dataset)
                results_drift[name].append(drift)
                print(f"  {name:20s} | Energy Drift: {drift:.4e}")

    # 4. Plotting the Headline Result
    os.makedirs("results", exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    colors = {"Regression (Floor)": "#ef476f", "HNN (Ceiling)": "#06d6a0", "JEPA (Hypothesis)": "#118ab2"}
    markers = {"Regression (Floor)": "X", "HNN (Ceiling)": "^", "JEPA (Hypothesis)": "o"}

    # Plot 1: Linear decodability
    for name, r2s in results_lin.items():
        ax1.plot(ns, r2s, marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=8)
    ax1.set_title("Linear Energy Decodability vs Bodies")
    ax1.set_xlabel("Number of Bodies (N)")
    ax1.set_ylabel("Linear Probe $R^2$ (Higher is better)")
    ax1.set_xticks(ns)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Nonlinear decodability (is the physics present at all?)
    for name, r2s in results_nl.items():
        ax2.plot(ns, r2s, marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=8)
    ax2.set_title("Nonlinear Energy Decodability vs Bodies")
    ax2.set_xlabel("Number of Bodies (N)")
    ax2.set_ylabel("MLP Probe $R^2$ (Higher is better)")
    ax2.set_xticks(ns)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Conservation manifold (energy drift vs N)
    for name, drifts in results_drift.items():
        ax3.plot(ns, drifts, marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=8)
    ax3.set_title("Constraint Manifold: Energy Conservation vs Bodies")
    ax3.set_xlabel("Number of Bodies (N)")
    ax3.set_ylabel("Relative energy drift std(E)/|E₀| (Lower is better)")
    ax3.set_yscale('log') # HNN is orders of magnitude better
    ax3.set_xticks(ns)
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = "results/scaling_curves.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ EXPERIMENT COMPLETE. Headline plots saved to: {save_path}")

if __name__ == "__main__":
    main()