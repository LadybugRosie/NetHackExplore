"""Inject the wiki-grounded intrinsic reward into PufferLib's PPO rollout.

PufferLib's fast native (CUDA) trainer steps + computes rewards entirely in C, so
a Python/CPU reward can't be added there. The **torch backend** (`--slowly`)
runs the rollout in Python, which is where we add the bonus. We monkeypatch
``PuffeRL.rollouts`` (rather than fork PufferLib) so all of `puffer train`'s
machinery — wandb, dashboard, checkpointing — is reused unchanged.

Requirements:
  * ``_C.so`` built with NETHACK_USE_MESSAGE=1 (the message is then the last
    256 bytes of each obs row). See scripts/train_wiki.sh.
  * float32 build (torch backend requires precision_bytes==4).

Tunables via env vars: GE_WIKI_COEF (default 0.5), GE_WIKI_THRESHOLD (0.20),
GE_WIKI_DATABASE (path to data.base). The bonus is added to the env reward
before GAE; per-lane "discovered" sets reset on episode termination.
"""

from __future__ import annotations

import os

from .wiki_reward import WikiReward

MESSAGE_LEN = 256  # NLE_MESSAGE_SIZE; message occupies the last MESSAGE_LEN obs bytes


def enable(coef: float | None = None, threshold: float | None = None,
           data_base_path: str | None = None):
    """Monkeypatch PufferLib's torch PuffeRL.rollouts to add the wiki reward.
    Idempotent. Call before pufferlib.pufferl.train()."""
    import torch
    from pufferlib import torch_pufferl as T

    if getattr(T.PuffeRL, "_wiki_patched", False):
        return

    coef = float(os.environ.get("GE_WIKI_COEF", 0.5 if coef is None else coef))
    threshold = float(os.environ.get("GE_WIKI_THRESHOLD", 0.20 if threshold is None else threshold))
    data_base_path = os.environ.get("GE_WIKI_DATABASE", data_base_path)
    subset = os.environ.get("GE_WIKI_SUBSET", "all")        # "all" | "progression"
    analyzer = os.environ.get("GE_WIKI_ANALYZER", "word")    # "word" | "char"

    Profile = T.Profile
    sample_logits = T.sample_logits
    _actions_for_vec_step = T._actions_for_vec_step

    def rollouts(self):
        # Lazy one-time setup (PuffeRL.__init__ doesn't know about us).
        if not hasattr(self, "_wiki"):
            self._wiki = WikiReward(threshold=threshold, data_base_path=data_base_path,
                                    subset=subset, analyzer=analyzer)
            self._wiki_coef = coef
            self._wiki_msg_off = int(self.vec_obs.shape[1]) - MESSAGE_LEN
            self._wiki_disc = [set() for _ in range(self.total_agents)]
            print(f"[wiki-ppo] enabled: coef={coef} threshold={threshold} "
                  f"subset={subset} analyzer={analyzer} "
                  f"concepts={len(self._wiki.corpus)} msg_off={self._wiki_msg_off}")

        prof = self.profile
        config = self.config
        device = self.device
        horizon = config['horizon']

        self.state = tuple(torch.zeros_like(s) for s in self.state) if self.state else ()
        o = self.vec_obs
        r = torch.zeros(self.total_agents, device=device)
        d = torch.zeros(self.total_agents, device=device)

        wiki_hits = 0
        wiki_sum = 0.0
        P = Profile
        prof.mark(0)
        for t in range(horizon):
            o_device = torch.as_tensor(o, device=device)

            prof.mark(1)
            with torch.no_grad():
                logits, value, state = self.policy.forward_eval(o_device, self.state)
                action, logprob, _ = sample_logits(logits)
            prof.mark(2)

            with torch.no_grad():
                self.state = state
                self.observations[t] = o_device
                self.actions[t] = action
                self.logprobs[t] = logprob
                self.rewards[t] = torch.as_tensor(r, device=device)
                self.terminals[t] = torch.as_tensor(d, device=device).float()
                self.values[t] = value.flatten()

            prof.mark(2)
            actions_flat = _actions_for_vec_step(action)
            if self.gpu:
                actions_flat = actions_flat.cuda()
                self._vec.gpu_step(actions_flat.data_ptr())
                torch.cuda.synchronize()
            else:
                self._vec.cpu_step(actions_flat.data_ptr())

            o, r, d = self.vec_obs, self.vec_rewards, self.vec_terminals

            # ---- wiki intrinsic reward injection ----
            if self._wiki_coef:
                msgs = _decode_messages(o, self._wiki_msg_off)
                bonus = self._wiki.reward_batch(msgs, self._wiki_disc)
                dd = d.detach().to("cpu").numpy()
                for i in range(self.total_agents):
                    if dd[i] > 0.5:
                        self._wiki_disc[i].clear()  # new episode -> concepts novel again
                    if bonus[i]:
                        wiki_hits += 1
                        wiki_sum += bonus[i]
                bonus_t = torch.as_tensor(bonus, device=device, dtype=r.dtype)
                r = r + self._wiki_coef * bonus_t

            prof.mark(3)
            prof.elapsed(P.EVAL_GPU, 1, 2)
            prof.elapsed(P.EVAL_ENV, 2, 3)

        prof.mark(1)
        prof.elapsed(P.ROLLOUT, 0, 1)
        self.global_step += self.total_agents * horizon
        self.env_logs = self._vec.log()
        # surface wiki metrics on the dashboard / wandb if env_logs is a dict
        if isinstance(self.env_logs, dict):
            self.env_logs["wiki_hits_per_rollout"] = float(wiki_hits)
            self.env_logs["wiki_reward_per_rollout"] = float(wiki_sum)

    T.PuffeRL.rollouts = rollouts
    T.PuffeRL._wiki_patched = True


def _decode_messages(o, msg_off):
    """Decode the message slice (last 256 bytes) of each obs row into strings."""
    sl = o[:, msg_off:msg_off + MESSAGE_LEN]
    arr = sl.detach().to("cpu").numpy().astype("uint8")
    out = []
    for row in arr:
        out.append(row.tobytes().split(b"\x00", 1)[0].decode("latin1"))
    return out
