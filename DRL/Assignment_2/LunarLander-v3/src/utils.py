"""
utils.py
========
Small shared helpers: reproducible seeding, moving averages, path handling and
construction of the *fixed validation state set* used for the Q-value plot in
Task (d).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

# Project directories -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for _d in (RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    """Seed python, numpy and torch so that a run is bit-wise reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def moving_average(x, window: int) -> np.ndarray:
    """Trailing moving average of `x` over `window` samples.

    The first `window-1` entries average over however many samples exist, so the
    output has the same length as the input (no NaN padding in the plots).
    """
    x = np.asarray(x, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    return (csum[idx] - csum[lo]) / (idx - lo)


def build_validation_states(n_states: int = 512, seed: int = 12345) -> np.ndarray:
    """Collect a *fixed* set of states with a random policy on the ORIGINAL env.

    The same set is reused by all four agents and never changes during training,
    which is what makes the "average predicted Q-value" curves comparable.
    """
    from .env_wrapper import make_env  # local import avoids a circular import

    env = make_env("original")
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    while len(states) < n_states:
        states.append(np.asarray(obs, dtype=np.float32))
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(0, env.action_space.n)))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.asarray(states[:n_states], dtype=np.float32)


def get_validation_states(path: Path | None = None, **kwargs) -> np.ndarray:
    """Load the cached validation states, creating them on first use."""
    path = path or (RESULTS_DIR / "validation_states.npy")
    if path.exists():
        return np.load(path)
    states = build_validation_states(**kwargs)
    np.save(path, states)
    return states


def save_json(obj, path: Path) -> None:
    """Dump a dict to JSON, converting numpy scalars/arrays to plain python."""

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o)}")

    Path(path).write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
