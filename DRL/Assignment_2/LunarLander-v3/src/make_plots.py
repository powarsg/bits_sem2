"""
make_plots.py
=============
Task (d): comparison of the four trained agents.

Produces (and saves to `figures/`) the four required curves:
  1. episode reward vs training episode
  2. average predicted Q-value on the FIXED validation state set vs episode
  3. successful-landing rate vs episode (100-episode moving average)
  4. average thruster activations per episode vs episode

When several seeds are available every curve shows the across-seed mean with a
+/- 1 standard-deviation band; with a single seed the raw per-episode signal is
drawn faintly behind the moving average instead.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .train import load_result, run_name, run_stem
from .utils import FIGURES_DIR, RESULTS_DIR, moving_average

CONDITIONS = [("dqn", "original"), ("ddqn", "original"),
              ("dqn", "modified"), ("ddqn", "modified")]

# Colour encodes the algorithm, line style encodes the environment.
STYLE = {
    "DQN_original":  dict(color="#1f77b4", ls="-",  label="DQN - Original"),
    "DDQN_original": dict(color="#2ca02c", ls="-",  label="DDQN - Original"),
    "DQN_modified":  dict(color="#d62728", ls="--", label="DQN - Modified"),
    "DDQN_modified": dict(color="#9467bd", ls="--", label="DDQN - Modified"),
}


def available_seeds() -> list[int]:
    """Seeds for which *all four* conditions have been trained."""
    seeds = set()
    for p in RESULTS_DIR.glob("DQN_original_seed*.json"):
        seeds.add(int(p.stem.split("seed")[-1]))
    return sorted(s for s in seeds
                  if all((RESULTS_DIR / f"{run_stem(a, e, s)}.json").exists() for a, e in CONDITIONS))


def load_all(seeds: list[int] | None = None) -> dict[str, list[dict]]:
    """Load every saved run, grouped by condition name."""
    seeds = seeds or available_seeds()
    if not seeds:
        raise FileNotFoundError("no training results found - run `python -m src.run_experiments`")
    return {run_name(a, e): [load_result(a, e, s) for s in seeds] for a, e in CONDITIONS}


def _metric(hist: dict, key: str) -> np.ndarray:
    """Per-episode metric, including two derived ones.

    `thruster_rate` (fraction of steps on which a thruster was selected) removes
    the confound between "fires less" and "episode is shorter", which matters for
    the fuel-economy analysis in Task (e).
    """
    if key == "thruster_rate":
        return np.asarray(hist["thrusters"], float) / np.maximum(np.asarray(hist["steps"], float), 1)
    if key == "reward_per_step":
        return np.asarray(hist["reward"], float) / np.maximum(np.asarray(hist["steps"], float), 1)
    return np.asarray(hist[key], float)


def _curves(runs: list[dict], key: str, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed curve per seed -> (mean, std, raw of the first seed)."""
    raw = np.asarray([_metric(r["history"], key) for r in runs], dtype=np.float64)
    smoothed = np.asarray([moving_average(row, window) if window > 1 else row for row in raw])
    return smoothed.mean(axis=0), smoothed.std(axis=0), raw[0]


def _panel(ax, results, key, window, title, ylabel, scale=1.0):
    """Draw one metric for all four conditions on a single axis."""
    single_seed = len(next(iter(results.values()))) == 1
    for name, runs in results.items():
        mean, std, raw = _curves(runs, key, window)
        x = np.arange(1, len(mean) + 1)
        st = STYLE[name]
        if single_seed:
            ax.plot(x, scale * raw, color=st["color"], alpha=0.10, lw=0.7)
        else:
            ax.fill_between(x, scale * (mean - std), scale * (mean + std),
                            color=st["color"], alpha=0.13, lw=0)
        ax.plot(x, scale * mean, color=st["color"], ls=st["ls"], lw=1.8, label=st["label"])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_all(results: dict | None = None, window: int = 100, show: bool = False):
    """Create the four required figures individually plus a 2x2 overview."""
    results = results or load_all()
    panels = [
        ("reward", window, f"1. Episode reward ({window}-episode moving average)",
         "episode reward", 1.0),
        ("val_q", 20, "2. Average predicted Q-value on the fixed validation set",
         r"mean$_s$ max$_a$ Q(s,a)", 1.0),
        ("safe_landing", window, f"3. Successful landing rate ({window}-episode moving average)",
         "safe landings [%]", 100.0),
        ("thrusters", window, f"4. Thruster activations per episode ({window}-ep. moving average)",
         "thruster actions selected", 1.0),
    ]

    for i, (key, w, title, ylab, scale) in enumerate(panels, start=1):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        _panel(ax, results, key, w, title, ylab, scale)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"plot{i}_{key}.png", dpi=150)
        if not show:
            plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    for ax, (key, w, title, ylab, scale) in zip(axes.ravel(), panels):
        _panel(ax, results, key, w, title, ylab, scale)
    n_seeds = len(next(iter(results.values())))
    fig.suptitle("DQN vs DDQN on original and modified (stochastic-failure) LunarLander-v3 "
                 f"- {n_seeds} seed(s)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURES_DIR / "plot0_overview.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return results


def plot_supplementary(results: dict | None = None, window: int = 100, show: bool = False):
    """Extra evidence for Task (e): fuel-economy and episode-duration behaviour."""
    results = results or load_all()
    panels = [
        ("thruster_rate", window, "5a. Thruster firing RATE (fraction of steps with a thruster)",
         "P(selected action != 0)", 1.0),
        ("steps", window, "5b. Episode length", "steps per episode", 1.0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, (key, w, title, ylab, scale) in zip(axes.ravel(), panels):
        _panel(ax, results, key, w, title, ylab, scale)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "plot5_supplementary.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def summary_table(results: dict | None = None, last: int = 100) -> str:
    """Final-`last`-episode training statistics for every condition (mean over seeds)."""
    results = results or load_all()
    header = (f"{'condition':>15} {'reward':>16} {'base reward':>12} {'land rate %':>12} "
              f"{'thrusters':>10} {'fire rate':>10} {'ep len':>8} {'val Q':>8} {'misfire %':>10}")
    rows = [header, "-" * len(header)]
    sl = slice(-last, None)
    for name, runs in results.items():
        agg = lambda key, s=sl: np.array([np.mean(_metric(r["history"], key)[s]) for r in runs])
        thr = np.array([np.sum(r["history"]["thrusters"][sl]) for r in runs])
        mis = np.array([np.sum(r["history"]["misfires"][sl]) for r in runs])
        rew = agg("reward")
        rows.append(
            f"{name:>15} {rew.mean():>9.1f}+/-{rew.std():>5.1f} {agg('base_reward').mean():>12.1f} "
            f"{100*agg('safe_landing').mean():>12.1f} {agg('thrusters').mean():>10.1f} "
            f"{agg('thruster_rate').mean():>10.3f} {agg('steps').mean():>8.0f} "
            f"{agg('val_q', slice(-10, None)).mean():>8.2f} "
            f"{100*mis.sum()/thr.sum() if thr.sum() else 0.0:>10.2f}"
        )
        rows[-1] = rows[-1]
    rows.append(f"\n(mean over the last {last} training episodes; +/- is the spread across seeds)")
    return "\n".join(rows)


def q_gap_table(results: dict | None = None, window: int = 20) -> str:
    """|Q_DQN - Q_DDQN| on the fixed validation set, absolute and relative - Task (e) Q1."""
    results = results or load_all()
    lines = [f"{'episode':>9} {'ORIGINAL: DQN':>14} {'DDQN':>8} {'gap':>8} {'rel.gap':>9}"
             f" | {'MODIFIED: DQN':>14} {'DDQN':>8} {'gap':>8} {'rel.gap':>9}"]
    lines.append("-" * len(lines[0]))
    q = {n: _curves(r, "val_q", window)[0] for n, r in results.items()}
    n_ep = len(next(iter(q.values())))
    for ep in list(range(200, n_ep, 200)) + [n_ep]:
        i = ep - 1
        cells = []
        for kind in ("original", "modified"):
            a, b = q[f"DQN_{kind}"][i], q[f"DDQN_{kind}"][i]
            scale = max(abs(a), abs(b), 1e-9)
            cells.append(f"{a:>14.2f} {b:>8.2f} {a-b:>+8.2f} {100*(a-b)/scale:>8.1f}%")
        lines.append(f"{ep:>9} {cells[0]} | {cells[1]}")
    lines.append("\nrel.gap = (Q_DQN - Q_DDQN) / max(|Q_DQN|,|Q_DDQN|)")
    return "\n".join(lines)


def main() -> None:
    seeds = available_seeds()
    print(f"aggregating seeds: {seeds}")
    results = plot_all(load_all(seeds))
    plot_supplementary(results)
    print(summary_table(results))
    print()
    print(q_gap_table(results))
    print(f"\nfigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
