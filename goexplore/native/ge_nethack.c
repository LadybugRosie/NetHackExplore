/* ge_nethack.c — deterministic single-env wrapper around PufferLib's Ocean
 * NetHack env, for Go-Explore phase 1.
 *
 * WHY THIS EXISTS
 * ---------------
 * PufferLib's Ocean NetHack env (ocean/nethack/nethack.h) is fast and correct,
 * but its vecenv autoresets in C and — crucially — ADVANCES the per-env seed via
 * an LCG on every reset (nethack_slow_reset: seed_a = seed_a*M + C). That makes
 * every reset a *different* game, which breaks Go-Explore's replay-based "return"
 * (return-to-a-cell = replay its action prefix from a seeded reset; only exact if
 * reset(seed) reproduces the same game).
 *
 * This shim reuses the Ocean env's stepping (reward shaping, prompt auto-dismiss,
 * blstats/scout bookkeeping) but replaces reset with a FIXED-seed reset (no LCG
 * advance, reseed=0). With reseed=0 the engine is a pure function of
 * (seeds, actions) — see vendor/nle README and rnd.c — so replay is exact.
 *
 * The Ocean env anchors current_nle_ctx inside nle_start/nle_step (vendor/nle/
 * src/src/nle.c), so driving N of these single-env wrappers sequentially in one
 * process (one Python loop) is safe — each step re-anchors its own ctx. Run many
 * processes for parallelism (the per-env nle_ctx_t carries all mutable state).
 *
 * Built against the *same* libnethack.so PufferLib uses; see build_ge.sh.
 */

/* Observation fields we need for Go-Explore cells. blstats gives x/y/depth/score;
 * glyphs/chars let cell descriptors optionally hash the explored map. These are
 * compile-time selects consumed by nethack.h. */
#define NETHACK_USE_BLSTATS 1
#define NETHACK_USE_GLYPHS  1
#define NETHACK_USE_CHARS   1

#include "nethack.h"   /* pulls in nleobs.h, profile.h; defines Nethack, init/c_reset/c_step */
#include <stdint.h>

typedef struct GeEnv {
    Nethack env;
    unsigned long core;   /* fixed CORE seed — never advanced */
    unsigned long disp;   /* fixed display seed */
} GeEnv;

/* Deterministic reset: tear down any live game and start a fresh one with the
 * FIXED seeds and reseed=0. Mirrors nethack_slow_reset but without the LCG
 * advance, so reset() always yields the identical game. */
static void ge_det_reset(GeEnv *g) {
    Nethack *env = &g->env;
    env->pending_reset = 0;
    if (env->ctx != NULL && env->fn_end) {
        env->fn_end(env->ctx);
        env->ctx = NULL;
    }
    if (env->dl_handle != NULL) {
        nethack_unload_lib(env);
    }
    nethack_load_lib(env);
    nethack_bind_obs(env);
    env->obs.action = 0;
    env->obs.done = 0;
    env->obs.in_normal_game = 0;
    env->obs.how_done = 0;

    nle_seeds_init_t seeds;
    memset(&seeds, 0, sizeof(seeds));
    seeds.seeds[0] = g->core;
    seeds.seeds[1] = g->disp;
    seeds.reseed = 0;                 /* deterministic: no anti-TAS reseeding */
    env->ctx = env->fn_start(&env->obs, NULL, &seeds, &env->settings);

    nethack_drain_prompts_cat(env, PROF_FN_STEPS_RESET_DRAIN);
    nethack_reset_bookkeeping(env);
}

/* ---- ctypes ABI (all extern "C"-compatible plain C) ------------------------ */

GeEnv *ge_make(unsigned long core, unsigned long disp) {
    GeEnv *g = (GeEnv *)calloc(1, sizeof(GeEnv));
    if (!g) return NULL;
    Nethack *env = &g->env;
    env->num_agents = 1;
    env->observations = (unsigned char *)calloc(NETHACK_OBS_SIZE, 1);
    env->actions      = (float *)calloc(1, sizeof(float));
    env->rewards      = (float *)calloc(1, sizeof(float));
    env->terminals    = (float *)calloc(1, sizeof(float));
    env->rng = 0;
    init(env);                       /* settings (NETHACKDIR vardir), coefs, bind_obs */
    g->core = core;
    g->disp = disp;
    return g;
}

void ge_set_seed(GeEnv *g, unsigned long core, unsigned long disp) {
    g->core = core;
    g->disp = disp;
}

/* Reset to the fixed seed (the archive root). */
void ge_reset(GeEnv *g) {
    ge_det_reset(g);
}

/* Apply one action (index into NETHACK_ACTION_TABLE). Returns 1 if the episode
 * ended this step, else 0. On done we clear pending_reset so a stray follow-up
 * step can't trigger the seed-advancing autoreset; the caller re-homes via
 * ge_reset(). */
int ge_step(GeEnv *g, int action) {
    Nethack *env = &g->env;
    env->actions[0] = (float)action;
    c_step(env);
    if (env->terminals[0] > 0.5f) {
        env->pending_reset = 0;
        return 1;
    }
    return 0;
}

int            ge_num_actions(void)   { return NETHACK_NUM_ACTIONS; }
int            ge_blstats_len(void)   { return NLE_BLSTATS_SIZE; }
long          *ge_blstats(GeEnv *g)   { return g->env.blstats; }
float          ge_reward(GeEnv *g)    { return g->env.rewards[0]; }
int            ge_done(GeEnv *g)      { return g->env.terminals[0] > 0.5f; }
unsigned char *ge_chars(GeEnv *g)     { return g->env.chars; }
short         *ge_glyphs(GeEnv *g)    { return g->env.glyphs; }

void ge_free(GeEnv *g) {
    if (!g) return;
    Nethack *env = &g->env;
    if (env->ctx != NULL && env->fn_end) {
        env->fn_end(env->ctx);
        env->ctx = NULL;
    }
    nethack_rm_vardir(env->vardir);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
    free(env->terminals);
    free(g);
}
