"""Optional Weights & Biases tracking for Go-Explore phase 1.

Phase 1 is a custom search loop (not PufferLib's trainer), so it logs its own
metrics here. Phase 2b PPO is logged separately by PufferLib's *native* wandb
integration (see robustify.ppo_finetune) -- that produces its own wandb run.

The logger is a no-op unless ``use_wandb=True`` and ``wandb`` is importable, so
phase 1 keeps zero hard dependencies.
"""

from __future__ import annotations


class NullLogger:
    def log(self, metrics: dict, step: int | None = None):
        pass

    def finish(self):
        pass


class WandbLogger:
    def __init__(self, project, entity=None, name=None, group=None, config=None):
        import wandb  # lazy; only imported when tracking is requested
        self._wandb = wandb
        self.run = wandb.init(project=project, entity=entity, name=name,
                              group=group, config=config or {})

    def log(self, metrics: dict, step: int | None = None):
        self._wandb.log(metrics, step=step)

    def finish(self):
        self.run.finish()


def make_logger(use_wandb: bool, *, project="nethack-goexplore", entity=None,
                name=None, group="phase1", config=None):
    """Return a WandbLogger, or a NullLogger if tracking is off/unavailable."""
    if not use_wandb:
        return NullLogger()
    try:
        return WandbLogger(project=project, entity=entity, name=name,
                           group=group, config=config)
    except Exception as exc:  # noqa: BLE001 - never let logging crash a run
        print(f"[wandb] disabled ({exc}); continuing without tracking.")
        return NullLogger()
