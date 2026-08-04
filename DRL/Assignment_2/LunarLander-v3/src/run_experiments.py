"""
run_experiments.py
==================
Driver for Tasks (b), (c) and (d): trains the four agents

    DQN  - original      DDQN - original
    DQN  - modified      DDQN - modified

under *identical* experimental conditions (same seed, architecture, optimizer,
replay buffer, exploration schedule, hyper-parameters and number of episodes).
Only `algo` (target-Q computation) and `env_kind` (wrapper on/off) differ.

The whole 4-condition grid is optionally repeated over several random seeds so
that the comparison plots can show a mean +/- spread instead of a single noisy
run.  Seed 42 is the reference run required by the assignment; the extra seeds
only serve as a robustness check.

Runs are independent, so they are executed in parallel worker processes (each
pinned to a single torch thread).  Results go to `results/<RUN>_seed<S>.json`.

Usage
-----
    python -m src.run_experiments --episodes 800 --seeds 42
    python -m src.run_experiments --episodes 800 --seeds 42 43 44 --workers 6
    python -m src.run_experiments --episodes 800 --seeds 42 --serial
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

from .agent import AgentConfig
from .train import run_stem, train
from .utils import RESULTS_DIR, get_validation_states, save_json

# The four experimental conditions of Task (d).
CONDITIONS = [
    ("dqn", "original"),
    ("ddqn", "original"),
    ("dqn", "modified"),
    ("ddqn", "modified"),
]


def _worker(args) -> str:
    """Train a single (algo, env_kind, seed) run inside its own process."""
    algo, env_kind, episodes, seed, max_steps, cfg_dict = args
    train(
        algo=algo,
        env_kind=env_kind,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        config=AgentConfig(**cfg_dict),
        val_states=get_validation_states(),   # identical fixed set for every run
    )
    return run_stem(algo, env_kind, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DQN/DDQN on original & modified LunarLander-v3")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--serial", action="store_true", help="disable multiprocessing")
    args = ap.parse_args()

    cfg = AgentConfig()
    # Materialise the shared validation states *before* spawning workers so that
    # every process loads exactly the same file.
    val = get_validation_states()
    save_json(
        {"episodes": args.episodes, "seeds": args.seeds, "max_steps": args.max_steps,
         "config": cfg.to_dict(), "n_validation_states": int(val.shape[0]),
         "conditions": [f"{a.upper()}_{e}" for a, e in CONDITIONS]},
        RESULTS_DIR / "experiment_setup.json",
    )

    jobs = [(a, e, args.episodes, s, args.max_steps, cfg.to_dict())
            for s in args.seeds for a, e in CONDITIONS]
    t0 = time.time()
    if args.serial:
        for j in jobs:
            _worker(j)
    else:
        with mp.get_context("spawn").Pool(processes=min(args.workers, len(jobs))) as pool:
            for stem in pool.imap_unordered(_worker, jobs):
                print(f"finished: {stem}", flush=True)
    print(f"\n{len(jobs)} runs complete in {time.time() - t0:.0f} s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
