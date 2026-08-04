"""
evaluate.py
===========
Post-training greedy evaluation used in Tasks (d) and (e).

For each of the four trained agents it reports, over `n_episodes` greedy
(epsilon = 0) episodes on its *own* training environment:

  * mean episode reward and mean unshaped LunarLander reward
  * safe-landing rate
  * mean thruster activations per episode
  * **value-estimation bias**:  V_pred(s0) = max_a Q(s0, a) predicted at the
    initial state, versus the *actual* discounted return G0 obtained from that
    same state.  `bias = V_pred - G0 > 0` means the agent over-estimates its
    own value, which is exactly the pathology Double DQN is designed to reduce.
"""

from __future__ import annotations

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .env_wrapper import StochasticActionFailureWrapper, is_safe_landing, make_env
from .train import load_result, run_name, run_stem
from .utils import RESULTS_DIR, save_json


def load_agent(algo: str, env_kind: str, seed: int = 42) -> tuple[DQNAgent, dict]:
    """Rebuild an agent from its saved config and load the trained weights."""
    res = load_result(algo, env_kind, seed)
    cfg = AgentConfig(**{**res["config"], "hidden_sizes": tuple(res["config"]["hidden_sizes"])})
    agent = DQNAgent(8, 4, cfg, algo=algo, seed=res["seed"])
    agent.q_online.load_state_dict(torch.load(RESULTS_DIR / f"{run_stem(algo, env_kind, seed)}.pt"))
    agent.q_online.eval()
    return agent, res


def evaluate(algo: str, env_kind: str, train_seed: int = 42, n_episodes: int = 100,
             seed: int = 10_000, max_steps: int = 1000) -> dict:
    """Run a greedy policy and collect performance + value-bias statistics."""
    agent, _res = load_agent(algo, env_kind, train_seed)
    gamma = agent.cfg.gamma
    env = make_env(env_kind, seed=seed)
    is_mod = isinstance(env, StochasticActionFailureWrapper)

    rewards, base_rewards, landings, thrusters, v_pred, g0, misfire_rate = [], [], [], [], [], [], []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)          # same start states for all four agents
        v_pred.append(agent.mean_validation_q(np.asarray([obs], dtype=np.float32)))

        ret = disc = base = 0.0
        thr = mis = steps = 0
        terminated = truncated = False
        for t in range(max_steps):
            steps += 1
            a = agent.act(obs, eps=0.0)             # greedy
            obs, r, terminated, truncated, _ = env.step(a)
            ret += r
            disc += (gamma ** t) * r
            thr += int(a != 0)
            if is_mod:
                base += env.last_base_reward
                mis += int(env.last_misfired)
            else:
                base += r
            if terminated or truncated:
                break

        rewards.append(ret)
        base_rewards.append(base)
        landings.append(int(is_safe_landing(obs, terminated, truncated)))
        thrusters.append(thr)
        lengths.append(steps)
        g0.append(disc)
        misfire_rate.append(mis / thr if thr else 0.0)
    env.close()

    v_pred, g0 = np.asarray(v_pred), np.asarray(g0)
    return {
        "name": run_name(algo, env_kind),
        "algo": algo,
        "env_kind": env_kind,
        "train_seed": train_seed,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_base_reward": float(np.mean(base_rewards)),
        "landing_rate": float(np.mean(landings)),
        "mean_thrusters": float(np.mean(thrusters)),
        "mean_episode_len": float(np.mean(lengths)),
        "thruster_rate": float(np.sum(thrusters) / np.sum(lengths)),
        "mean_V_pred_s0": float(v_pred.mean()),
        "mean_actual_G0": float(g0.mean()),
        "value_bias": float((v_pred - g0).mean()),
        "observed_misfire_rate": float(np.mean(misfire_rate)),
    }


def evaluate_all(n_episodes: int = 100, seeds: list[int] | None = None) -> list[dict]:
    """Evaluate all four conditions (averaged over training seeds) and persist the report."""
    from .make_plots import CONDITIONS, available_seeds

    seeds = seeds or available_seeds()
    rows = []
    for algo, env_kind in CONDITIONS:
        per_seed = [evaluate(algo, env_kind, train_seed=s, n_episodes=n_episodes) for s in seeds]
        merged = {k: v for k, v in per_seed[0].items() if isinstance(v, str)}
        merged["n_episodes"] = n_episodes
        merged["seeds"] = seeds
        for k in per_seed[0]:
            if isinstance(per_seed[0][k], float):
                merged[k] = float(np.mean([p[k] for p in per_seed]))
        merged["reward_across_seeds_std"] = float(np.std([p["mean_reward"] for p in per_seed]))
        rows.append(merged)
    save_json(rows, RESULTS_DIR / "greedy_evaluation.json")
    return rows


def report(rows: list[dict] | None = None) -> str:
    """Format the greedy-evaluation results as a text table."""
    rows = rows or evaluate_all()
    head = (f"{'condition':>15} {'reward':>18} {'base rew':>10} {'land %':>8} {'thrust':>8} "
            f"{'fire rate':>10} {'ep len':>7} {'V_pred(s0)':>11} {'actual G0':>10} {'bias':>8}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['name']:>15} {r['mean_reward']:>9.1f} +/-{r['std_reward']:>6.1f} "
            f"{r['mean_base_reward']:>10.1f} {100*r['landing_rate']:>8.1f} "
            f"{r['mean_thrusters']:>8.1f} {r['thruster_rate']:>10.3f} "
            f"{r['mean_episode_len']:>7.0f} {r['mean_V_pred_s0']:>11.2f} "
            f"{r['mean_actual_G0']:>10.2f} {r['value_bias']:>+8.2f}"
        )
    lines.append("\n+/- is the across-episode std of the greedy return.")
    lines.append("bias = V_pred(s0) - actual discounted return G0   (positive => over-estimation)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
