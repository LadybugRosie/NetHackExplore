"""Phase 2: robustify the archive's demonstrations into a policy.

Two explicit stages, complementary not interchangeable:

  2a  bc_pretrain   -- supervised behavioral cloning of (obs -> action) pairs
                       harvested from the archive's best trajectories. This is a
                       classification problem: cross-entropy + Adam. It only
                       *imitates* the demos and compounds errors off-distribution.

  2b  ppo_finetune  -- load the BC weights as initialization and fine-tune with
                       PufferLib's NATIVE PPO on the vectorized env, using the
                       real reward. This is where the policy becomes robust to
                       stochasticity and can SURPASS the demonstrations, and where
                       the C-native vecenv throughput is used. Optionally keep a
                       self-imitation / BC auxiliary loss (as the Go-Explore paper
                       does) to retain the rare successful behaviors.

    archive --> bc_pretrain (Adam+CE, init) --> ppo_finetune (PufferLib PPO)

Everything heavy (torch / pufferlib) is imported lazily so phase 1 stays
dependency-free. On the mock env (scalar obs, no glyph conv) BC falls back to a
small MLP so the pipeline is still exercisable end-to-end offline.
"""

from __future__ import annotations

from . import nle_constants as C


# --------------------------------------------------------------------------- #
# Demonstration harvesting
# --------------------------------------------------------------------------- #
def select_demo_trajectories(archive, top_k: int = 32):
    """Deepest / highest-scoring cells make the best demonstrations."""
    entries = sorted(archive.cells.values(),
                     key=lambda e: (e.depth, e.score), reverse=True)
    return [e.traj for e in entries[:top_k] if e.traj]


def collect_demonstrations(env, trajectories, full_obs: bool):
    """Replay each trajectory and emit (obs, action) supervised pairs.

    ``full_obs=True`` keeps the raw obs dict (for the real glyph+blstats policy);
    ``full_obs=False`` extracts a scalar feature vector (mock / MLP fallback).
    """
    X, y = [], []
    for traj in trajectories:
        obs = env.reset()
        for action in traj:
            X.append(obs if full_obs else _scalar_features(obs))
            y.append(int(action))
            obs, _, done, _ = env.step(action)
            if done:
                break
    return X, y


def _scalar_features(obs):
    bl = obs["blstats"]
    idx = [C.BL_X, C.BL_Y, C.BL_DEPTH, C.BL_HP, C.BL_HPMAX, C.BL_SCORE, C.BL_TIME]
    return [float(bl[i]) for i in idx]


# --------------------------------------------------------------------------- #
# Phase 2a: behavioral cloning
# --------------------------------------------------------------------------- #
def bc_pretrain(env, archive, *, epochs: int = 300, top_k: int = 32,
                lr: float = 1e-2, use_glyph_policy: bool = False,
                checkpoint: str | None = None):
    """Behavioral cloning. Returns the trained module (or None if torch absent)."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[bc] torch not installed; skipping phase 2a. "
              "Best trajectory available via archive.best().traj")
        return None

    trajectories = select_demo_trajectories(archive, top_k)
    X, y = collect_demonstrations(env, trajectories, full_obs=use_glyph_policy)
    if not X:
        print("[bc] no demonstrations collected; skipping.")
        return None
    n_actions = env.n_actions

    if use_glyph_policy:
        # Real NLE: clone PufferLib's policy architecture so weights transfer.
        from .policy import build_policy
        net = build_policy(n_actions)
        Xb = {
            "glyphs": torch.stack([torch.as_tensor(o["glyphs"]) for o in X]),
            "blstats": torch.stack([torch.as_tensor(o["blstats"]) for o in X]),
        }
        forward = lambda: net(Xb)[0]  # logits only for BC
    else:
        # Mock / scalar fallback.
        net = nn.Sequential(
            nn.Linear(len(X[0]), 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )
        Xt = torch.tensor(X, dtype=torch.float32)
        forward = lambda: net(Xt)

    yt = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(forward(), yt)
        loss.backward()
        opt.step()

    acc = (forward().argmax(1) == yt).float().mean().item()
    print(f"[bc] cloned {len(X)} (obs,action) pairs -> train acc {acc:.2f}")
    if checkpoint:
        torch.save(net.state_dict(), checkpoint)
        print(f"[bc] saved BC weights -> {checkpoint}")
    return net


# --------------------------------------------------------------------------- #
# Phase 2b: PPO fine-tuning with PufferLib's native trainer
# --------------------------------------------------------------------------- #
def ppo_finetune(*, env_id: str = "NetHackChallenge-v0", num_envs: int = 256,
                 total_timesteps: int = 10_000_000, bc_checkpoint: str | None = None,
                 backend: str = "native", wandb: bool = False,
                 wandb_project: str = "nethack-goexplore", wandb_entity=None,
                 **train_kwargs):
    """Fine-tune with PufferLib's native PPO, initialized from BC weights.

    This is the step that uses the C-native vectorized throughput: PufferLib
    steps ``num_envs`` lanes in native code and runs PPO on the rollouts. The
    exact trainer entry point has changed across PufferLib versions; the
    version-sensitive calls are isolated here.
    """
    try:
        import torch
        import pufferlib
        import pufferlib.vector
    except ImportError as exc:
        raise RuntimeError("ppo_finetune needs torch + pufferlib installed.") from exc

    from .vecenv import PufferVecEnv
    from .policy import make_pufferlib_policy

    # Build the native vectorized env (the throughput path).
    vec = PufferVecEnv(num_envs, env_id=env_id, backend=backend)
    policy = make_pufferlib_policy(vec.vecenv)

    # Initialize from the behaviorally-cloned weights.
    if bc_checkpoint:
        state = torch.load(bc_checkpoint)
        missing, unexpected = policy.load_state_dict(state, strict=False)
        print(f"[ppo] loaded BC init from {bc_checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # Hand off to PufferLib's native PPO. Newer PufferLib exposes this as
    # ``pufferlib.pufferl`` (a.k.a. clean_pufferl). Pin the exact symbol to your
    # installed version; both common entry points are tried below.
    # PufferLib's native trainer has its OWN wandb integration; enabling it via
    # the config makes phase 2b log SPS / episodic return / losses to wandb
    # automatically (as a separate run from phase 1).
    ppo_config = _default_ppo_config(total_timesteps, num_envs, **train_kwargs)
    if wandb:
        ppo_config.update(track=True, wandb_project=wandb_project,
                          wandb_entity=wandb_entity, wandb_group="phase2-ppo")

    try:
        from pufferlib import pufferl
        trainer = pufferl.PuffeRL(config=ppo_config, vecenv=vec.vecenv, policy=policy)
        trainer.train()
        return trainer
    except (ImportError, AttributeError):
        try:
            from pufferlib import clean_pufferl
            return clean_pufferl.train(
                vec.vecenv, policy,
                total_timesteps=total_timesteps, **train_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Could not locate PufferLib's PPO trainer. Set the entry point "
                "for your installed pufferlib version in ppo_finetune()."
            ) from exc


def _default_ppo_config(total_timesteps, num_envs, **overrides):
    cfg = dict(
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        learning_rate=2.5e-4,
        gamma=0.999,        # NetHack episodes are very long
        gae_lambda=0.95,
        clip_coef=0.1,
        ent_coef=0.01,
        update_epochs=2,
        batch_size=num_envs * 128,
    )
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def robustify(env, archive, *, use_glyph_policy: bool = False,
              bc_checkpoint: str = "bc_init.pt", run_ppo: bool = False,
              env_id: str = "NetHackChallenge-v0", num_envs: int = 256,
              total_timesteps: int = 10_000_000, wandb: bool = False,
              wandb_project: str = "nethack-goexplore", wandb_entity=None):
    """Run phase 2a (BC) and optionally phase 2b (PufferLib PPO)."""
    net = bc_pretrain(env, archive, use_glyph_policy=use_glyph_policy,
                      checkpoint=bc_checkpoint if use_glyph_policy else None)
    if run_ppo:
        return ppo_finetune(env_id=env_id, num_envs=num_envs,
                            total_timesteps=total_timesteps,
                            bc_checkpoint=bc_checkpoint, wandb=wandb,
                            wandb_project=wandb_project, wandb_entity=wandb_entity)
    return net
