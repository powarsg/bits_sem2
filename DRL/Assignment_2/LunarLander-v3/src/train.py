"""
train.py
========
Training loop shared by all four experimental conditions

    {DQN, DDQN} x {original LunarLander-v3, modified LunarLander-v3}

Using one loop for every condition guarantees that the random seed, network
architecture, optimizer, replay buffer, exploration schedule, hyper-parameters
and training duration are identical across runs - only `algo` and `env_kind`
change.

Per-episode metrics recorded for Task (d):
    reward              total reward returned by the environment being trained on
    base_reward         total *unshaped* LunarLander reward (comparable across envs)
    safe_landing        1 if the episode ended in a safe landing, else 0
    thrusters           number of steps on which the agent *selected* a in {1,2,3}
    misfires            number of thruster commands silently dropped (modified env)
    val_q               mean_s max_a Q(s,a) over the fixed validation state set
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .env_wrapper import StochasticActionFailureWrapper, is_safe_landing, make_env
from .utils import RESULTS_DIR, get_validation_states, save_json, set_global_seeds


def run_name(algo: str, env_kind: str) -> str:
    """Canonical identifier of an experimental condition (used in plot legends)."""
    return f"{algo.upper()}_{env_kind}"


def run_stem(algo: str, env_kind: str, seed: int) -> str:
    """File stem of a single training run: one condition trained with one seed."""
    return f"{run_name(algo, env_kind)}_seed{seed}"


def train(
    algo: str = "dqn",
    env_kind: str = "original",
    episodes: int = 700,
    max_steps: int = 1000,
    seed: int = 42,
    config: AgentConfig | None = None,
    val_states: np.ndarray | None = None,
    device: str = "cpu",
    log_every: int = 50,
    verbose: bool = True,
) -> dict:
    """Train one agent and return a history dict of per-episode metrics."""
    config = config or AgentConfig()
    set_global_seeds(seed)                       # python / numpy / torch
    torch.set_num_threads(1)                     # keeps parallel runs from oversubscribing

    env = make_env(env_kind, seed=seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    is_modified = isinstance(env, StochasticActionFailureWrapper)

    agent = DQNAgent(state_dim, action_dim, config, algo=algo, seed=seed, device=device)
    if val_states is None:
        val_states = get_validation_states()

    hist = {k: [] for k in
            ("reward", "base_reward", "safe_landing", "thrusters", "misfires",
             "steps", "epsilon", "val_q")}

    eps = config.eps_start
    t0 = time.time()

    for ep in range(1, episodes + 1):
        # Seeding each episode with `seed + ep` gives every condition the *same*
        # sequence of initial states, so differences come from the algorithm/env only.
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = ep_base = 0.0
        thrusters = misfires = ep_steps = 0
        terminated = truncated = False

        for _ in range(max_steps):
            ep_steps += 1
            action = agent.act(obs, eps)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(obs, action, reward, next_obs, terminated)  # bootstrap through truncation
            obs = next_obs

            ep_reward += reward
            thrusters += int(action != 0)
            if is_modified:
                ep_base += env.last_base_reward
                misfires += int(env.last_misfired)
            else:
                ep_base += reward

            if done:
                break

        eps = max(config.eps_end, eps * config.eps_decay)   # decay once per episode

        hist["reward"].append(ep_reward)
        hist["base_reward"].append(ep_base)
        hist["safe_landing"].append(int(is_safe_landing(obs, terminated, truncated)))
        hist["thrusters"].append(thrusters)
        hist["misfires"].append(misfires)
        hist["steps"].append(ep_steps)
        hist["epsilon"].append(eps)
        hist["val_q"].append(agent.mean_validation_q(val_states))

        if verbose and (ep % log_every == 0 or ep == 1):
            w = slice(max(0, ep - 100), ep)
            print(
                f"[{run_stem(algo, env_kind, seed):>22}] ep {ep:4d}/{episodes} "
                f"| R(100) {np.mean(hist['reward'][w]):8.2f} "
                f"| land% {100*np.mean(hist['safe_landing'][w]):5.1f} "
                f"| Qval {hist['val_q'][-1]:7.2f} "
                f"| thr {np.mean(hist['thrusters'][w]):6.1f} "
                f"| eps {eps:.3f} | {time.time()-t0:6.0f}s",
                flush=True,
            )

    env.close()

    result = {
        "name": run_name(algo, env_kind),
        "algo": algo,
        "env_kind": env_kind,
        "episodes": episodes,
        "seed": seed,
        "config": config.to_dict(),
        "wall_time_s": time.time() - t0,
        "history": hist,
    }
    stem = run_stem(algo, env_kind, seed)
    save_json(result, RESULTS_DIR / f"{stem}.json")
    agent.save(RESULTS_DIR / f"{stem}.pt")
    return result


def load_result(algo: str, env_kind: str, seed: int = 42, results_dir: Path | None = None) -> dict:
    """Read back a previously saved training run."""
    import json

    p = (results_dir or RESULTS_DIR) / f"{run_stem(algo, env_kind, seed)}.json"
    return json.loads(p.read_text(encoding="utf-8"))
