"""The single point where the planner resolves a method name into an adapter.

Kept in its own module, apart from ``paarbench.methods``, because ``planning/mpc.py``
imports it and the import must stay cheap and free of cycles: the planner must not
drag in the runner, the selection machinery, or any method's dependencies just to
discover that no adaptation was requested.

The planner passes whatever sits under ``planner.adapter`` in the Hydra config::

    planner:
      adapter:
        method: adajepa          # a directory name under methods/
        params:                  # adapter constructor kwargs; override method.yaml
          steps: 10
          lr: 5.0e-4

Omitting the block entirely, or setting ``method: null``, gives the frozen baseline.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def build_adapter(cfg: Optional[Mapping[str, Any]], wm, preprocessor):
    """Resolve a ``planner.adapter`` config block into a ``TestTimeAdapter``.

    Returns ``None`` for the frozen baseline, which the planner turns into a
    ``NullAdapter``.
    """
    if cfg is None:
        return None

    # OmegaConf nodes are Mappings but not dicts; normalize without importing hydra.
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    if not isinstance(cfg, Mapping):
        raise TypeError(
            f"planner.adapter must be a mapping with a 'method' key, got {type(cfg).__name__}"
        )

    name = cfg.get("method")
    if name is None or name in ("", "none", "frozen"):
        return None

    params = cfg.get("params") or {}
    if not isinstance(params, Mapping):
        raise TypeError("planner.adapter.params must be a mapping")

    from paarbench import methods

    method = methods.load(str(name))
    adapter = method.build(wm=wm, preprocessor=preprocessor, **dict(params))

    missing = [
        hook
        for hook in ("on_episode_start", "on_transition", "before_plan", "metrics")
        if not callable(getattr(adapter, hook, None))
    ]
    if missing:
        raise TypeError(
            f"method {name!r}: {type(adapter).__name__} is missing required hook(s) "
            f"{missing}; see paarbench/adapter.py:TestTimeAdapter"
        )
    return adapter
