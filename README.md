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
| `goexplore/vecenv.py` | `MockVecEnv` + `NativeVecEnv` (real engine) + legacy `PufferVecEnv` |
| `goexplore/vectorized.py` | **vectorized async-lane phase-1 driver** |
| `goexplore/native/ge_nethack.c` | C shim: deterministic (fixed-seed) wrapper around PufferLib's Ocean NetHack env |
| `goexplore/native/build_ge.sh` | builds `libge_nethack.so` against `libnethack.so` |
| `goexplore/native_env.py` | ctypes `NativeNLE` / `NativeVecEnv` — the real deterministic engine |
| `goexplore/robustify_native.py` | **phase-2 (real)**: BC into PufferLib's policy + `puffer train` PPO fine-tune |
| `goexplore/policy.py` | (legacy mock/standalone) actor-critic for BC |
| `goexplore/robustify.py` | (legacy) phase-2 for the mock path |
| `goexplore/mock_nle.py` | deterministic NLE stand-in for offline testing |
| `goexplore/run.py` | CLI entry point |
| `scripts/build_native.sh` | one-shot build of the whole native stack |
| `scripts/env.sh` | source for NETHACKDIR / LD_LIBRARY_PATH / venv |
| `scripts/train.sh` | wrapper for native `puffer train nethack` |
| `tests/test_smoke.py` | single-env algorithm test (beats random baseline) |
| `tests/test_vectorized.py` | vectorized driver test (deep, reproducible, exact replay) |
| `tests/test_native.py` | real-engine determinism + a real-NetHack GE run |

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

## Run it on the real environment (`--env native`)

The real stack is **PufferLib 4.0** (Ocean C-binding env) + the
[liujonathan24/NetHack](https://github.com/liujonathan24/NetHack) fork, built
into `libnethack.so` + `pufferlib/_C.so`. Go-Explore drives the engine through a
tiny deterministic C shim, `libge_nethack.so` (`goexplore/native/ge_nethack.c`),
loaded via ctypes — **not** the old `--env puffer` path (which targeted a
PufferLib API that no longer exists). The shim reuses PufferLib's Ocean env but
swaps in a **fixed-seed reset** (the stock env advances its seed every reset),
which is what makes replay-based "return" exact.

Build everything (Debian 12 + NVIDIA driver + passwordless sudo assumed):

```bash
scripts/build_native.sh          # apt deps, CUDA toolkit, clone, libnethack.so,
                                 # _C.so (float32), libge_nethack.so
source scripts/env.sh            # NETHACKDIR, LD_LIBRARY_PATH, venv
python tests/test_native.py      # determinism + a real-NetHack GE run
```

Then:

```bash
# Phase 1, vectorized Go-Explore on the real deterministic engine:
python3 -m goexplore.run --env native --vectorized --num-envs 64 \
        --max-env-steps 2000000

# Phase 1 + Phase 2 (BC warm-start -> PufferLib PPO fine-tune):
python3 -m goexplore.run --env native --vectorized --robustify --run-ppo \
        --ppo-envs 4096 --ppo-timesteps 1000000000 --wandb

# Native PPO training directly (no Go-Explore search; uses the env's built-in
# depth/scout/score reward shaping — the fast path on the H100s):
scripts/train.sh --wandb --vec.total-agents 4096 --train.gpus 1
```

Notes / current limits (see "Known limitations" below):
* The vecenv steps N lanes in one Python process. For more throughput run several
  across processes (the engine is per-env; each `ge_step` re-anchors its ctx).
* BC clones the archive's best trajectories, but vanilla Go-Explore explores
  **randomly**, so BC accuracy is modest — it's a stable warm-start; the PPO
  fine-tune does the real learning. Stronger transfer wants an action prior in
  phase 1 or a self-imitation/backward algorithm in phase 2.

## Wiki-grounded intrinsic reward (experimental, Motif-flavored)

Vanilla GE explores randomly, so its trajectories don't behaviorally-clone well
in NLE's huge action space. An alternative to BC is a **knowledge-distilled
exploration reward**: reward the agent for reaching states the game's own
encyclopedia (`dat/data.base`, the source the wiki is built from) describes —
i.e. for encountering new monsters/items/features. It's the semantic-space analog
of the env's per-tile `scout` bonus.

```bash
source scripts/env.sh
python tests/test_wiki_reward.py            # corpus + matching + novelty
python tests/test_wiki_ppo.py              # decode/patch/injection logic (no GPU)

# Train PPO with the wiki reward added (torch backend; launch when GPUs free):
scripts/train_wiki.sh --vec.total-agents 2048 --train.gpus 1 \
    --train.total-timesteps 200000000 --wandb --wandb-project nethack-goexplore
# tune: GE_WIKI_COEF (default 0.5), GE_WIKI_THRESHOLD (0.20)
```

How it works: `goexplore/wiki_reward.py` matches each step's NLE `message` to the
nearest encyclopedia concept (TF-IDF cosine, or a sentence-embedding backend) and
grants a per-episode first-encounter bonus. `goexplore/wiki_ppo.py` monkeypatches
PufferLib's torch rollout to add that bonus to the env reward before GAE (CPU,
~15 ms/step for 4096 lanes — no extra GPU load), reusing all of `puffer train`'s
wandb/dashboard/checkpoint machinery. The principled upgrade to full **Motif** is
a reward model trained from LLM/wiki *preferences* over message pairs — same
`WikiReward.score` plug point. (nethackwiki.com is Cloudflare-blocked from this
host; `data.base` is the local, license-clean stand-in and the corpus is
swappable for a wiki dump.)

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
