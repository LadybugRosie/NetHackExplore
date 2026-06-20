"""CLI entry point: run Go-Explore on NLE (or the mock).

Phase 1, single env (simple, low throughput):
    python -m goexplore.run --env mock --iterations 1500

Phase 1, VECTORIZED across N lanes (uses the C-native batched step):
    python -m goexplore.run --env mock   --vectorized --num-envs 64 --max-env-steps 200000
    python -m goexplore.run --env puffer --vectorized --num-envs 256 --max-env-steps 50000000

Phase 2: behavioral cloning, then PufferLib native PPO fine-tuning:
    python -m goexplore.run --env mock   --vectorized --robustify
    python -m goexplore.run --env puffer --vectorized --robustify --run-ppo --ppo-envs 256
"""

from __future__ import annotations

import argparse
import json

from .cells import CellConfig
from .env import DeterministicEnv, make_env
from .goexplore import GoExplore, GoExploreConfig
from .vecenv import make_vecenv
from .vectorized import VecGoExploreConfig, VectorizedGoExplore


def build_parser():
    p = argparse.ArgumentParser(description="Go-Explore for NLE / PufferLib")
    p.add_argument("--env", default="mock", choices=["mock", "nle", "puffer"])
    p.add_argument("--env-id", default="NetHackChallenge-v0")
    p.add_argument("--seed-core", type=int, default=0, help="NLE CORE game seed (shared by all lanes)")
    p.add_argument("--seed-disp", type=int, default=0, help="NLE display rng seed")
    p.add_argument("--search-seed", type=int, default=0, help="seed for the search itself")
    p.add_argument("--explore-steps", type=int, default=60)
    p.add_argument("--repeat-prob", type=float, default=0.0)
    # phase-1 mode
    p.add_argument("--vectorized", action="store_true", help="run phase 1 across a batched vecenv")
    p.add_argument("--num-envs", type=int, default=64, help="lanes for vectorized phase 1")
    p.add_argument("--max-env-steps", type=int, default=200_000, help="vectorized step budget")
    p.add_argument("--iterations", type=int, default=2000, help="rounds for single-env phase 1")
    p.add_argument("--log-every", type=int, default=None)
    # cell granularity
    p.add_argument("--cell-w", type=int, default=None, help="x-bucket (default: 1 mock / 6 nle)")
    p.add_argument("--cell-h", type=int, default=None, help="y-bucket (default: 1 mock / 3 nle)")
    p.add_argument("--map-hash", action="store_true", help="add explored-map hash to cell key")
    # phase 2
    p.add_argument("--robustify", action="store_true", help="run phase-2a BC after exploring")
    p.add_argument("--run-ppo", action="store_true", help="run phase-2b PufferLib PPO fine-tune")
    p.add_argument("--ppo-envs", type=int, default=256)
    p.add_argument("--ppo-timesteps", type=int, default=10_000_000)
    p.add_argument("--out", default=None, help="write best trajectory + stats to this JSON file")
    return p


def _run_phase1(args, cell_cfg):
    if args.vectorized:
        vec = make_vecenv(args.env, num_envs=args.num_envs, env_id=args.env_id)
        cfg = VecGoExploreConfig(
            num_envs=args.num_envs,
            max_env_steps=args.max_env_steps,
            explore_steps=args.explore_steps,
            repeat_prob=args.repeat_prob,
            search_seed=args.search_seed,
            game_seed=args.seed_core,
            log_every=args.log_every if args.log_every is not None else 20_000,
        )
        ge = VectorizedGoExplore(vec, cell_cfg, cfg)
    else:
        raw = make_env(args.env, env_id=args.env_id)
        env = DeterministicEnv(raw, seed_core=args.seed_core, seed_disp=args.seed_disp)
        cfg = GoExploreConfig(
            iterations=args.iterations,
            explore_steps=args.explore_steps,
            repeat_prob=args.repeat_prob,
            search_seed=args.search_seed,
            log_every=args.log_every if args.log_every is not None else 100,
        )
        ge = GoExplore(env, cell_cfg, cfg)
    return ge, ge.run()


def main(argv=None):
    args = build_parser().parse_args(argv)

    cell_w = args.cell_w if args.cell_w is not None else (1 if args.env == "mock" else 6)
    cell_h = args.cell_h if args.cell_h is not None else (1 if args.env == "mock" else 3)
    cell_cfg = CellConfig(cell_w=cell_w, cell_h=cell_h, use_map_hash=args.map_hash)

    ge, stats = _run_phase1(args, cell_cfg)
    best = ge.archive.best()
    print(
        f"\nDONE  cells={len(ge.archive)}  max_depth={stats.max_depth}  "
        f"best_score={best.score}  best_traj_len={len(best.traj)}  "
        f"env_steps={stats.env_steps}"
    )

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({
                "stats": vars(stats),
                "best_trajectory": list(best.traj),
                "best_score": best.score,
                "best_depth": best.depth,
            }, fh)
        print(f"wrote {args.out}")

    if args.robustify:
        from .robustify import robustify
        # BC demos are replayed on a single deterministic env, seeded to match
        # the search so the stored trajectories reproduce exactly.
        demo_env = DeterministicEnv(make_env(args.env, env_id=args.env_id),
                                    seed_core=args.seed_core, seed_disp=args.seed_disp)
        robustify(
            demo_env, ge.archive,
            use_glyph_policy=(args.env != "mock"),
            run_ppo=args.run_ppo,
            env_id=args.env_id,
            num_envs=args.ppo_envs,
            total_timesteps=args.ppo_timesteps,
        )

    return ge


if __name__ == "__main__":
    main()
