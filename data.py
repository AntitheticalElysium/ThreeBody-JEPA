# data.py
import os
import rebound
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from config import DataCfg

def generate(cfg: DataCfg, filepath="nbody_data.npz"):
    """
    Generates N-body trajectories using REBOUND's exact IAS15 integrator.
    Rejects trajectories that fly apart to keep the data bounded.
    """
    if os.path.exists(filepath):
        print(f"Dataset {filepath} already exists. Skipping generation.")
        return
    
    print(f"Generating {cfg.num_trajectories} trajectories for N={cfg.N}...")
    num_steps = int(cfg.integration_span / cfg.dt)

    all_states = [] # trajectory data
    all_energies = [] # energy over time
    all_angmoms = [] # angular momentum over time

    valid_count = 0
    while valid_count < cfg.num_trajectories:
        sim = rebound.Simulation()
        sim.integrator = "ias15"

        # Planar n-body problem with random positions and velocities
        for _ in range(cfg.N):
            sim.add(
                m=1.0, 
                x=np.random.uniform(-1, 1), 
                y=np.random.uniform(-1, 1),
                vx=np.random.uniform(-1, 1), 
                vy=np.random.uniform(-1, 1),
                z=0.0, vz=0.0 # Force 2D
            )

        sim.move_to_com() # without this, center-of-mass momentum drifts trivially

        states = []
        energies = []
        ang_moms = []
        ejected = False

        for step in range(num_steps):
            time = step * cfg.dt
            sim.integrate(time)
            # [N, 4] -> x, y, vx, vy
            state = [[p.x, p.y, p.vx, p.vy] for p in sim.particles]

            # if any particle gets too far, reject trajectory
            positions = np.array(state)[:, :2]
            if np.max(np.linalg.norm(positions, axis=1)) > 5.0:
                ejected = True
                break

            states.append(state)
            energies.append(sim.energy())
            ang_moms.append(sim.angular_momentum())

        if not ejected:
            all_states.append(states)
            all_energies.append(energies)
            all_angmoms.append(ang_moms)

            valid_count += 1
            if valid_count % 10 == 0:
                print(f"Generated {valid_count}/{cfg.num_trajectories}")
    
    # states: [num_traj, num_steps, N, 4]
    np.savez(
        filepath, 
        states=np.array(all_states, dtype=np.float32),
        energies=np.array(all_energies, dtype=np.float64),
        ang_moms=np.array(all_angmoms, dtype=np.float64)
    )
    print(f"Saved to {filepath}")

class NBodyDataset(Dataset):
    def __init__(self, filepath="nbody_data.npz", cfg: DataCfg = DataCfg()):
        data = np.load(filepath)
        self.states = torch.from_numpy(data['states'])  # [B_traj, T, N, 4]
        self.cfg = cfg
        
        self.num_traj, self.num_steps, self.N, self.F = self.states.shape
        
        # mass was set to 1.0 in generate()
        self.masses = torch.ones((self.num_traj, self.N), dtype=torch.float32)

        pos = self.states[..., :2]
        vel = self.states[..., 2:]
        # learn dataset distribution for normalization
        self.pos_mean, self.pos_std = pos.mean(), pos.std()
        self.vel_mean, self.vel_std = vel.mean(), vel.std()
        # avoid division by zero
        self.pos_std = self.pos_std if self.pos_std > 1e-6 else 1.0
        self.vel_std = self.vel_std if self.vel_std > 1e-6 else 1.0

    def normalize(self, state: torch.Tensor) -> torch.Tensor:
        """state shape: [..., 4]"""
        normed = state.clone()
        normed[..., :2] = (normed[..., :2] - self.pos_mean) / self.pos_std
        normed[..., 2:] = (normed[..., 2:] - self.vel_mean) / self.vel_std
        return normed
    
    def __len__(self):
        # sample random trajectories and random time-steps per epoch
        # TODO: length is num_traj * (num_steps - max_dt) for now
        return self.num_traj * (self.num_steps - self.cfg.dt_max_steps)
    
    def __getitem__(self, idx):
        # decode 1D idx into (trajectory_idx, start_time_idx)
        valid_steps_per_traj = self.num_steps - self.cfg.dt_max_steps

        traj_idx = idx // valid_steps_per_traj
        t_idx = idx % valid_steps_per_traj
        
        # Sample a random future delta_t
        dt_steps = torch.randint(self.cfg.dt_min_steps, self.cfg.dt_max_steps + 1, (1,)).item()
        
        state_ctx = self.normalize(self.states[traj_idx, t_idx])
        state_tgt = self.normalize(self.states[traj_idx, t_idx + dt_steps])
        
        dt_scalar = torch.tensor([dt_steps * self.cfg.dt], dtype=torch.float32)
        mass = self.masses[traj_idx]
        
        return state_ctx, state_tgt, dt_scalar, mass