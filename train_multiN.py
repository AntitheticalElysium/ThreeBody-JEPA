# train_multiN.py — "ceiling" reference: JEPA trained on N=3..6 mixed (NOT the extrapolation model).
import math
import random
import torch
from torch.utils.data import DataLoader

from config import DataCfg, ModelCfg, TrainCfg
from data import NBodyDataset
from models.jepa import JEPADynamics

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NS = [3, 4, 5, 6]
TOTAL_STEPS = 30000


def main():
    cfg = ModelCfg()  # 192/4/8, light recipe — same as the hero model
    tcfg = TrainCfg()
    loaders = {n: DataLoader(NBodyDataset(f"nbody_data_N{n}.npz", DataCfg(N=n)),
                             batch_size=tcfg.batch_size, shuffle=True, drop_last=True) for n in NS}
    iters = {n: iter(loaders[n]) for n in NS}

    model = JEPADynamics(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=1e-4)
    warmup = max(1, int(0.05 * TOTAL_STEPS))

    def lr_at(s):
        if s < warmup:
            return tcfg.lr * (s + 1) / warmup
        p = (s - warmup) / max(1, TOTAL_STEPS - warmup)
        return 1e-6 + 0.5 * (tcfg.lr - 1e-6) * (1 + math.cos(math.pi * p))

    print(f"--- Training multi-N ceiling (N={NS}, {TOTAL_STEPS} steps) ---")
    model.train()
    running = 0.0
    for step in range(TOTAL_STEPS):
        n = random.choice(NS)
        try:
            batch = next(iters[n])
        except StopIteration:
            iters[n] = iter(loaders[n]); batch = next(iters[n])
        sc, st, dt, mass = [b.to(DEV) for b in batch]

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad()
        loss, _ = model(sc, st, dt, mass)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.step_ema(momentum=tcfg.ema_momentum_start + (1.0 - tcfg.ema_momentum_start) * step / TOTAL_STEPS)
        running += loss.item()
        if (step + 1) % 5000 == 0:
            print(f"step {step+1:6d}/{TOTAL_STEPS} | loss {running/5000:.5f}")
            running = 0.0

    torch.save(model.state_dict(), "results/jepa_multiN_model.pt")
    print("Saved results/jepa_multiN_model.pt")


if __name__ == "__main__":
    main()
