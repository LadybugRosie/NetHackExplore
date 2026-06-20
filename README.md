# Vanilla Go-Explore for the PufferLib NLE environment

An implementation of **Go-Explore** ("first return, then explore") targeting the
NetHack Learning Environment as built by
[this PufferLib NLE fork](https://github.com/liujonathan24/NetHack). It includes:

* **Phase 1** — both a simple single-env driver *and* a **vectorized async-lane
  driver** that drives a batched vecenv so the fork's C-native throughput is
  actually used.
* **Phase 2** — two explicit stages: **2a** behavioral cloning (init) → **2b**
  fine-tuning with **PufferLib's native PPO** on the vectorized env.

## Why this works on NLE

Go-Explore's "return to a cell" step is implemented here by **replaying the
stored action sequence from a seeded reset**. That is exact *because* the engine
is a pure deterministic function of `(core_seed, disp_seed, actions)` when seeded
with `reseed=False`:

* All gameplay randomness comes from one ISAAC64 stream (`src/src/rnd.c`,
  `CORE` rng), whose entire state lives inside the per-env `nle_ctx_t`
  (`isaac64_ctx rng_state[2]`, pointer-free).
* The periodic entropy re-injection (`reseed_random`) is a **no-op** unless
  `has_strong_rngseed` is set — which is set from the `reseed` flag you pass to
  `nle_set_seed`. Seed with `reseed=False` and the whole game is replayable.

So you do **not** need C-level state snapshotting to get vanilla Go-Explore
working. (Snapshotting `nle_ctx_t` is the *optimization* that turns each O(len)
replay into O(1) — a natural next step, see below.)

## Layout

| File | Role |
|------|------|
| `goexplore/cells.py` | cell descriptor (the key design choice) + score/depth extraction |
| `goexplore/archive.py` | the cell archive + novelty/frontier-weighted selection |
| `goexplore/env.py` | `DeterministicEnv` (single-env seed + replay return) and `make_env` |
| `goexplore/goexplore.py` | the single-env phase-1 return-then-explore loop |
| `goexplore/vecenv.py` | `MockVecEnv` (testable) + `PufferVecEnv` (native batched step) |
| `goexplore/vectorized.py` | **vectorized async-lane phase-1 driver** |
| `goexplore/policy.py` | NetHack actor-critic shared by BC and PPO (lazy torch) |
| `goexplore/robustify.py` | phase-2: `bc_pretrain` (2a) + `ppo_finetune` via PufferLib PPO (2b) |
| `goexplore/mock_nle.py` | deterministic NLE stand-in for offline testing |
| `goexplore/run.py` | CLI entry point |
| `tests/test_smoke.py` | single-env algorithm test (beats random baseline) |
| `tests/test_vectorized.py` | vectorized driver test (deep, reproducible, exact replay) |

## How the vectorized phase 1 uses the C throughput

The single-env driver (`goexplore.py`) steps one env in a Python loop — correct
but it ignores PufferLib's whole point. `vectorized.py` instead drives **N lanes
of a batched vecenv**, where each lane is independently in one of two phases:

* **REPLAY** — feeding its stored action sequence to return to a sampled cell
* **EXPLORE** — random actions, inserting visited states into the shared archive

Lanes are **asynchronous**: a short-trajectory lane flips to EXPLORE while a
long-trajectory lane is still replaying, so a single `vec.step(actions)` (native
C for the real env) never has an idle lane. When a lane finishes its explore
budget it **retires** and is re-homed to the game seed with a fresh sampled cell.
`reset_lanes()` — re-homing specific lanes to the seed on demand — is the one
operation a stock vecenv doesn't expose, so `vecenv.py` makes it first-class.

## Run it now (no heavy deps)

```bash
python3 tests/test_smoke.py
python3 tests/test_vectorized.py
# single-env phase 1
python3 -m goexplore.run --env mock --iterations 800 --explore-steps 40
# vectorized phase 1 (async lanes)
python3 -m goexplore.run --env mock --vectorized --num-envs 32 --max-env-steps 80000
```

Expected: Go-Explore digs far deeper than an equal-budget random rollout (e.g.
`max_depth=12` vs a random baseline of `~7`), and the vectorized driver matches
the single-env archive quality.

## Run it on the real environment

Requires a Python the stack supports (currently **3.9–3.11**; this box has 3.14,
which `nle`/`pufferlib` don't build against yet) and the env installed:

```bash
pip install nle pufferlib torch        # builds libnethack.so from the C fork

# Phase 1, vectorized across 256 native lanes:
python3 -m goexplore.run --env puffer --vectorized --num-envs 256 \
        --max-env-steps 50000000 --env-id NetHackChallenge-v0

# Phase 1 + Phase 2 (BC init -> PufferLib native PPO fine-tune):
python3 -m goexplore.run --env puffer --vectorized --robustify --run-ppo \
        --ppo-envs 256 --ppo-timesteps 50000000
```

Two integration points to confirm against your installed versions (both isolated
and commented): the PufferLib **vecenv construction + per-lane re-seed** in
`vecenv.py::PufferVecEnv`, and the PufferLib **PPO trainer entry point** in
`robustify.py::ppo_finetune` (tries `pufferlib.pufferl` then `clean_pufferl`).

## Tracking with Weights & Biases

```bash
pip install wandb && wandb login        # one-time

# Phase 1: logs cells / max_depth / best_score / rounds / SPS vs env-steps
python3 -m goexplore.run --env mock --vectorized --max-env-steps 200000 \
        --wandb --wandb-project nethack-goexplore --wandb-name ge-mock-1

# Phase 1 + Phase 2: phase-1 metrics logged here; PufferLib's PPO logs its OWN
# wandb run (SPS / episodic return / losses) for phase 2b.
python3 -m goexplore.run --env puffer --vectorized --max-env-steps 50000000 \
        --robustify --run-ppo --wandb --wandb-project nethack-goexplore
```

Notes:
* `--wandb` is opt-in; without it (or if `wandb` isn't installed) the run prints
  to stdout and continues — tracking never crashes a run.
* You get **two wandb runs** for a full pipeline: phase 1 (logged by
  `goexplore/tracking.py`) and phase 2b PPO (logged by PufferLib's native wandb
  integration, which `ppo_finetune` switches on via `track=True`). Group them
  with `--wandb-project`, or filter by the `phase1` / `phase2-ppo` group tags.
* Phase-1 metrics are logged against `env_steps` as the x-axis, so the
  exploration curve is comparable across `--num-envs` settings.

## Tuning knobs that matter

* `--cell-w/--cell-h` — cell granularity. Coarser = smaller archive, faster but
  blurrier. The default for real NLE is a 6×3 position bucket per dungeon level.
* `--map-hash` — add a hash of the explored glyph map to the cell key (finer,
  bigger archive; useful once position alone stops discriminating).
* `--explore-steps`, `--repeat-prob` — exploration budget per round and action
  "stickiness" (repeat last action), which helps in NLE's large action space.

## Known limitations / next steps (in priority order)

1. **Single seed.** Like all vanilla Go-Explore, this solves *one* dungeon and
   overfits it. To generalize, run phase 1 across many seeds and randomize seeds
   during phase-2 robustification.
2. **Random exploration is weak in real NLE** (~100+ actions, many no-ops).
   Replace uniform sampling in `goexplore.py::_act` with a restricted action set
   or a pretrained/BC action prior.
3. **O(len) returns.** The vectorized driver parallelizes the replay but still
   re-simulates each prefix. Add `nle_clone(ctx)`/`nle_restore(ctx, snap)` in the
   C engine (the `nle_ctx_t` + arena is the snapshot unit) to make return O(1) —
   then `reset_lanes()` becomes a `memcpy` instead of a replay.
4. **Per-lane re-seed on the real vecenv.** `reset_lanes()` is exact on
   `MockVecEnv`; on `PufferVecEnv` confirm your version re-homes a single lane to
   the chosen seed (autoreset semantics vary). Snapshot/restore (item 3) sidesteps
   this entirely.
5. **Phase 2b reward.** PPO fine-tunes on the env's native reward; for deep
   descent you'll likely want NLE reward shaping (depth/score) and possibly a
   self-imitation auxiliary loss to retain the BC demonstrations.
