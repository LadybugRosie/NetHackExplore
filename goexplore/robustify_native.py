"""Phase 2 robustify for the REAL NetHack engine.

Turns the Go-Explore archive's best trajectories into a PufferLib policy:

  2a  bc_pretrain_native  — behavioral cloning of (chars_obs -> action) pairs
                            harvested by replaying the archive's best
                            trajectories on the deterministic native env. The
                            policy IS PufferLib's nethack policy (MinGRU +
                            DefaultEncoder/Decoder), so the weights load straight
                            into `puffer train`.

  2b  ppo_finetune_native — `puffer train nethack --slowly --load-model-path`,
                            i.e. PufferLib's PPO initialized from the BC weights,
                            using the env's native depth/scout/score reward.

The native env's `chars` observation (1659 bytes) is byte-identical to PufferLib's
default training observation, so demos transfer with no feature engineering.

Requires the float32 build (`bash build.sh nethack --float`) so the torch backend
(which loads torch state_dicts) is available. Run inside the PufferLib venv with
scripts/env.sh sourced.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from .cells import game_depth, game_score
from .native_env import NH_GRID, NativeNLE


# --------------------------------------------------------------------------- #
# Demonstration harvesting
# --------------------------------------------------------------------------- #
def select_demo_trajectories(archive, top_k: int = 32):
    """Deepest / highest-scoring cells make the best demonstrations."""
    entries = sorted(archive.cells.values(),
                     key=lambda e: (e.depth, e.score), reverse=True)
    return [e.traj for e in entries[:top_k] if e.traj]


def collect_demos(trajectories, game_seed: int, disp: int = 0x222):
    """Replay each trajectory on the deterministic native env, emitting
    (chars_obs, action) sequences. obs is the state BEFORE the action."""
    env = NativeNLE(core=game_seed, disp=disp, include_chars=True)
    demos = []
    for traj in trajectories:
        env.seed(game_seed, disp, False)
        obs = env.reset()
        obs_seq, act_seq = [], []
        for a in traj:
            obs_seq.append(np.frombuffer(obs["chars"], dtype=np.uint8).copy())
            act_seq.append(int(a))
            obs, _, done, _ = env.step(a)
            if done:
                break
        if obs_seq:
            demos.append((np.asarray(obs_seq, dtype=np.uint8),
                          np.asarray(act_seq, dtype=np.int64)))
    env.close()
    return demos


# --------------------------------------------------------------------------- #
# Phase 2a: behavioral cloning into PufferLib's policy
# --------------------------------------------------------------------------- #
def _build_puffer_policy(device):
    """Construct PufferLib's nethack policy (same arch `puffer train` builds, so
    the saved state_dict loads via --load-model-path)."""
    import pufferlib._C as _C
    import pufferlib.models
    from pufferlib.pufferl import load_config

    # load_config() parses sys.argv; shield it from our own CLI args.
    _argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        args = load_config("nethack")
    finally:
        sys.argv = _argv
    pk = dict(args["policy"])
    net = getattr(pufferlib.models, args["torch"]["network"])(**pk)
    # Obtain obs_size / act_sizes from a tiny throwaway vec (authoritative).
    va = dict(args)
    va["vec"] = dict(args["vec"]); va["vec"].update(total_agents=2, num_threads=1, num_buffers=1)
    vec = _C.create_vec(va, _C.gpu)
    obs_size, act_sizes = vec.obs_size, list(vec.act_sizes)
    enc = getattr(pufferlib.models, args["torch"]["encoder"])(obs_size, pk["hidden_size"])
    dec = getattr(pufferlib.models, args["torch"]["decoder"])(act_sizes, pk["hidden_size"])
    policy = pufferlib.models.Policy(enc, dec, net).to(device)
    return policy, obs_size, act_sizes


def bc_pretrain_native(archive, *, game_seed: int = 0, disp: int = 0x222,
                       top_k: int = 32, epochs: int = 300, lr: float = 3e-4,
                       max_grad_norm: float = 1.0,
                       checkpoint: str = "bc_init.pt", device: str | None = None):
    """Behaviorally clone the archive's best trajectories into PufferLib's policy.
    Saves the state_dict to `checkpoint` (loadable by `puffer train`).

    NOTE: vanilla Go-Explore trajectories are dominated by RANDOM exploration
    actions, so the (obs -> action) map is only weakly learnable — expect modest
    BC accuracy. BC here is a stable warm-start; the policy is improved by the
    phase-2b PPO fine-tune (env depth/scout reward). For stronger transfer, use a
    restricted/biased action prior in phase 1 or a self-imitation / backward
    algorithm in phase 2 (see README next steps). Grad clipping keeps BC stable
    on the raw 0-255 char observations."""
    import torch
    import torch.nn.functional as F

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trajectories = select_demo_trajectories(archive, top_k)
    demos = collect_demos(trajectories, game_seed, disp)
    if not demos:
        print("[bc] no demonstrations collected; skipping.")
        return None
    n_pairs = sum(len(a) for _, a in demos)
    print(f"[bc] collected {len(demos)} demos, {n_pairs} (obs,action) pairs")

    policy, obs_size, act_sizes = _build_puffer_policy(device)

    # Pad sequences to a common length; mask padding in the loss.
    T = max(len(a) for _, a in demos)
    B = len(demos)
    X = np.zeros((B, T, obs_size), dtype=np.uint8)
    Y = np.zeros((B, T), dtype=np.int64)
    M = np.zeros((B, T), dtype=np.float32)
    for i, (o, a) in enumerate(demos):
        X[i, :len(a)] = o
        Y[i, :len(a)] = a
        M[i, :len(a)] = 1.0
    Xt = torch.as_tensor(X, device=device)
    Yt = torch.as_tensor(Y, device=device)
    Mt = torch.as_tensor(M, device=device)

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    policy.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits, _values = policy(Xt)               # logits: (B*T, n_actions)
        loss_all = F.cross_entropy(logits, Yt.reshape(-1), reduction="none")
        loss = (loss_all * Mt.reshape(-1)).sum() / Mt.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        opt.step()
    with torch.no_grad():
        pred = policy(Xt)[0].argmax(-1)
        acc = ((pred == Yt.reshape(-1)).float() * Mt.reshape(-1)).sum() / Mt.sum()
    print(f"[bc] final loss {loss.item():.4f}  train acc {acc.item():.3f}")

    torch.save(policy.state_dict(), checkpoint)
    print(f"[bc] saved BC weights -> {checkpoint}")
    return checkpoint


# --------------------------------------------------------------------------- #
# Phase 2b: PPO fine-tune via PufferLib (torch backend loads the BC state_dict)
# --------------------------------------------------------------------------- #
def ppo_finetune_native(bc_checkpoint: str | None = None, *, total_timesteps: int = 50_000_000,
                        total_agents: int = 4096, num_threads: int = 16, num_buffers: int = 1,
                        gpus: int = 1, wandb: bool = False,
                        wandb_project: str = "nethack-goexplore", extra_args=None):
    """Launch `puffer train nethack` (torch backend) initialized from BC weights.
    Returns the subprocess exit code."""
    cmd = [sys.executable, "-m", "pufferlib.pufferl", "train", "nethack", "--slowly",
           "--vec.total-agents", str(total_agents),
           "--vec.num-threads", str(num_threads),
           "--vec.num-buffers", str(num_buffers),
           "--train.total-timesteps", str(total_timesteps),
           "--train.gpus", str(gpus)]
    if bc_checkpoint:
        cmd += ["--load-model-path", os.path.abspath(bc_checkpoint)]
    if wandb:
        cmd += ["--wandb", "--wandb-project", wandb_project, "--wandb-group", "phase2-ppo"]
    if extra_args:
        cmd += list(extra_args)
    print("[ppo] launching:", " ".join(cmd))
    return subprocess.call(cmd)


def robustify_native(archive, *, game_seed: int = 0, disp: int = 0x222,
                     bc_checkpoint: str = "bc_init.pt", top_k: int = 32,
                     bc_epochs: int = 200, run_ppo: bool = False,
                     total_timesteps: int = 50_000_000, total_agents: int = 4096,
                     wandb: bool = False, wandb_project: str = "nethack-goexplore"):
    """Run phase 2a (BC) and optionally phase 2b (PufferLib PPO)."""
    ckpt = bc_pretrain_native(archive, game_seed=game_seed, disp=disp, top_k=top_k,
                              epochs=bc_epochs, checkpoint=bc_checkpoint)
    if run_ppo and ckpt:
        return ppo_finetune_native(ckpt, total_timesteps=total_timesteps,
                                   total_agents=total_agents, wandb=wandb,
                                   wandb_project=wandb_project)
    return ckpt
