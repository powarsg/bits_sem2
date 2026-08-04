"""
env_wrapper.py
==============
Task (a): custom `gymnasium.Wrapper` around **LunarLander-v3** that injects
stochastic actuator (engine) failures and a modified reward function.

Contract implemented (exactly as specified in the assignment):

    Step 1  store the action `a` selected by the agent
    Step 2  if a in {1,2,3}: draw r ~ U[0,1); if r < 0.15 -> a_exec = 0 else a_exec = a
            if a == 0      : a_exec = 0 (unchanged)
    Step 3  execute `a_exec` in the base environment
    Step 4  R = R_base - 0.3 * 1(a != 0) + B        (penalty uses the *selected* action)
    Step 5  B = 50 only for a *safe landing* (all criteria simultaneously), else 0
    Step 6  return (observation, R, terminated, truncated, info) with an **untouched** info dict

Everything else (observation space, action space, transition dynamics,
termination / truncation rules) is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

# ----------------------------------------------------------------------------- constants
ENV_ID = "LunarLander-v3"

FAILURE_PROB = 0.15    # P(thruster command is silently dropped)
FUEL_PENALTY = 0.3     # subtracted whenever the agent *selects* a thruster action
LANDING_BONUS = 50.0   # awarded once, only on a safe landing
SAFE_TOL = 0.10        # tolerance on |vx|, |vy| and |theta|

# Indices of the 8-dimensional LunarLander observation vector that matter here.
IDX_VX, IDX_VY, IDX_ANGLE, IDX_LEG_L, IDX_LEG_R = 2, 3, 4, 6, 7


# ----------------------------------------------------------------------------- helpers
def is_safe_landing(obs, terminated: bool, truncated: bool, tol: float = SAFE_TOL) -> bool:
    """Return True iff the episode ended in a *safe landing* (Step 5 criteria).

    All of the following must hold simultaneously:
      terminated and not truncated, both legs in contact,
      |vx| < tol, |vy| < tol, |angle| < tol.
    """
    if not terminated or truncated:
        return False
    obs = np.asarray(obs, dtype=np.float64)
    return bool(
        obs[IDX_LEG_L] == 1.0
        and obs[IDX_LEG_R] == 1.0
        and abs(obs[IDX_VX]) < tol
        and abs(obs[IDX_VY]) < tol
        and abs(obs[IDX_ANGLE]) < tol
    )


@dataclass
class FailureStats:
    """Book-keeping counters used *only* for the Task (a) verification report.

    These live on the wrapper object; they are never written into `info` and are
    never visible to the learning agent.
    """

    steps: int = 0
    thruster_attempts: int = 0          # steps on which the agent selected a in {1,2,3}
    misfires: int = 0                   # thruster attempts silently replaced by 0
    penalised_steps: int = 0            # steps on which the 0.3 fuel penalty was applied
    bonuses_awarded: int = 0            # number of +50 landing bonuses given
    per_action_attempts: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})
    per_action_misfires: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})

    def misfire_rate(self) -> float:
        return self.misfires / self.thruster_attempts if self.thruster_attempts else float("nan")


class StochasticActionFailureWrapper(gym.Wrapper):
    """LunarLander-v3 with intermittent engine failure + modified reward.

    Parameters
    ----------
    env : gym.Env
        A `LunarLander-v3` (or compatible discrete-action) environment.
    failure_prob : float
        Probability that a *thruster* command (a in {1,2,3}) is replaced by 0.
    fuel_penalty : float
        Constant penalty applied for every *selected* thruster action.
    landing_bonus : float
        Bonus added on the terminal step of a safe landing.
    seed : int | None
        Seed of the *independent* RNG used for the failure draws, so that
        failure sequences are reproducible without disturbing the physics RNG.
    record_log : bool
        When True, every step is appended to `self.log` as a dict.  Used by the
        verification script; disabled during training for speed/memory.
    """

    def __init__(
        self,
        env: gym.Env,
        failure_prob: float = FAILURE_PROB,
        fuel_penalty: float = FUEL_PENALTY,
        landing_bonus: float = LANDING_BONUS,
        seed: int | None = None,
        record_log: bool = False,
    ) -> None:
        super().__init__(env)
        self.failure_prob = float(failure_prob)
        self.fuel_penalty = float(fuel_penalty)
        self.landing_bonus = float(landing_bonus)

        # Dedicated RNG: keeps actuator noise independent of the Box2D physics RNG.
        self._rng = np.random.default_rng(seed)

        self.record_log = record_log
        self.log: list[dict] = []
        self.stats = FailureStats()

        # Diagnostics of the most recent step (read by the training loop for logging).
        self.last_selected_action: int | None = None
        self.last_executed_action: int | None = None
        self.last_base_reward: float = 0.0
        self.last_bonus: float = 0.0
        self.last_misfired: bool = False

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the base env; re-seed the failure RNG when an explicit seed is given."""
        if seed is not None:
            # Offset keeps the actuator-noise stream distinct from the physics stream.
            self._rng = np.random.default_rng(seed + 10_000_019)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Apply Steps 1-6 of the specification for a single environment step."""
        # --- Step 1: store the action selected by the agent -------------------
        a = int(action)

        # --- Step 2: simulate intermittent engine failure ---------------------
        a_exec, r_draw, misfired = a, None, False
        if a != 0:
            r_draw = float(self._rng.random())          # r ~ U[0, 1)
            if r_draw < self.failure_prob:
                a_exec, misfired = 0, True              # engine silently does nothing

        # --- Step 3: execute a_exec in the *original* environment -------------
        obs, base_reward, terminated, truncated, info = self.env.step(a_exec)
        base_reward = float(base_reward)

        # --- Step 5: safe-landing bonus (evaluated on the returned observation)
        safe = is_safe_landing(obs, terminated, truncated)
        bonus = self.landing_bonus if safe else 0.0

        # --- Step 4: modified reward (penalty keyed on the *selected* action) --
        penalty = self.fuel_penalty if a != 0 else 0.0
        reward = base_reward - penalty + bonus

        self._update_diagnostics(a, a_exec, r_draw, misfired, base_reward, penalty,
                                 bonus, reward, terminated, truncated, safe, obs)

        # --- Step 6: return unchanged info dict -------------------------------
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ internals
    def _update_diagnostics(self, a, a_exec, r_draw, misfired, base_reward, penalty,
                            bonus, reward, terminated, truncated, safe, obs) -> None:
        """Maintain verification counters / optional per-step log (never exposed)."""
        s = self.stats
        s.steps += 1
        if a != 0:
            s.thruster_attempts += 1
            s.per_action_attempts[a] += 1
            s.penalised_steps += 1
            if misfired:
                s.misfires += 1
                s.per_action_misfires[a] += 1
        if bonus > 0:
            s.bonuses_awarded += 1

        self.last_selected_action = a
        self.last_executed_action = a_exec
        self.last_base_reward = base_reward
        self.last_bonus = bonus
        self.last_misfired = misfired

        if self.record_log:
            self.log.append(
                {
                    "a": a,
                    "r": r_draw,
                    "a_exec": a_exec,
                    "misfired": misfired,
                    "base_reward": base_reward,
                    "penalty": penalty,
                    "bonus": bonus,
                    "reward": reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "safe_landing": bool(safe),
                    "obs_tail": np.asarray(obs, dtype=np.float64)[[2, 3, 4, 6, 7]].tolist(),
                }
            )

    def reset_stats(self) -> None:
        """Clear verification counters and the per-step log."""
        self.stats = FailureStats()
        self.log = []


# ----------------------------------------------------------------------------- factory
def make_env(kind: str = "modified", seed: int | None = None, record_log: bool = False,
             render_mode: str | None = None) -> gym.Env:
    """Build either the `original` or the `modified` LunarLander-v3 environment.

    A single factory guarantees that both experimental conditions differ *only*
    by the presence of the failure/reward wrapper.
    """
    if kind not in ("original", "modified"):
        raise ValueError(f"kind must be 'original' or 'modified', got {kind!r}")
    env = gym.make(ENV_ID, render_mode=render_mode)
    if kind == "modified":
        env = StochasticActionFailureWrapper(env, seed=seed, record_log=record_log)
    return env
