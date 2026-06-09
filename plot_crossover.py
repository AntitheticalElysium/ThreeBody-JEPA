# plot_crossover.py — the headline figure: energy decodability vs N, the JEPA/floor crossover.
#   reads results/extrapolation_probes.txt (from extend_n.py), writes assets/extrapolation_crossover.png
import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG = "#0a0a14", "#dfe6f0"
C = {"Regression": "#ef476f", "HNN": "#06d6a0", "JEPA": "#118ab2"}


def load(path="results/extrapolation_probes.txt"):
    ns, reg, hnn, jepa = [], [], [], []
    for line in open(path):
        if not re.match(r"\s*N=\d+", line):
            continue
        n = int(re.match(r"\s*N=(\d+)", line).group(1))
        parts = line.split("|")
        pair = lambda s: [float(x) for x in s.strip().split("/")]
        ns.append(n); reg.append(pair(parts[1])); hnn.append(pair(parts[2])); jepa.append(pair(parts[3]))
    return np.array(ns), np.array(reg), np.array(hnn), np.array(jepa)


def main():
    os.makedirs("assets", exist_ok=True)
    ns, reg, hnn, jepa = load()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=BG)
    titles = ["Linear probe", "Nonlinear probe"]
    for k, ax in enumerate(axes):
        ax.set_facecolor(BG)
        for name, arr in [("Regression", reg), ("HNN", hnn), ("JEPA", jepa)]:
            ax.plot(ns, arr[:, k], marker="o", ms=5, lw=2.2, color=C[name],
                    label={"Regression": "Regression (floor)", "HNN": "HNN (ceiling)", "JEPA": "JEPA"}[name])
        # crossover: first N where JEPA passes Regression and stays ahead
        ahead = jepa[:, k] > reg[:, k]
        xover = next((ns[i] for i in range(len(ns)) if ahead[i] and ahead[i:].mean() > 0.8), None)
        if xover is not None:
            ax.axvline(xover, color=FG, ls=":", lw=1, alpha=0.4)
            ax.text(xover + 0.1, 0.92, f"JEPA overtakes\nthe floor (N={xover})", color=FG,
                    fontsize=9, alpha=0.75, va="top")
        ax.axvspan(2.5, 4.5, color=FG, alpha=0.05)
        ax.text(3, 0.02, "trained here", color=FG, fontsize=8, alpha=0.5, ha="center")
        ax.set_title(titles[k], color=FG, fontsize=13)
        ax.set_xlabel("number of bodies N", color=FG)
        ax.set_ylabel("energy decodability  (R²)", color=FG)
        ax.set_xticks(ns); ax.set_ylim(0, 1.0)
        ax.tick_params(colors=FG); ax.grid(True, ls="--", alpha=0.15)
        for s in ax.spines.values():
            s.set_color(FG); s.set_alpha(0.3)
        ax.legend(facecolor=BG, edgecolor="none", labelcolor=FG, fontsize=10)
    fig.suptitle("Trained on N=3. The memorizer wins in-distribution, then collapses; JEPA generalizes.",
                 color=FG, fontsize=14, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "assets/extrapolation_crossover.png"
    plt.savefig(out, dpi=150, facecolor=BG)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
