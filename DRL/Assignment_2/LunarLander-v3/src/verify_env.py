"""
verify_env.py
=============
Task (a): experimental verification that the modified environment behaves
*exactly* as specified.

Four independent checks are performed and printed as a report:

  V0  structural check   - observation space, action space and episode lengths
                           are unchanged w.r.t. the original environment.
  V1  misfire rate       - approximately 15% of attempted thruster actions are
                           replaced by "Do Nothing"  (binomial 95% CI + z-test,
                           reported globally and per thruster action).
  V2  fuel penalty       - for *every* step,  R - R_base - B  equals exactly
                           -0.3 when the *selected* action is a thruster and 0.0
                           otherwise, regardless of whether the engine misfired.
  V3  landing bonus      - +50 is awarded if and only if the safe-landing
                           criterion holds.  Random rollouts supply the negative
                           cases; the built-in LunarLander heuristic controller
                           supplies the positive cases (a random policy
                           essentially never lands safely).  A deterministic
                           unit test on a stub environment covers the exact
                           boundary conditions of the criterion.
"""

from __future__ import annotations

import math
from collections import Counter

import gymnasium as gym
import numpy as np

from .env_wrapper import (
    FAILURE_PROB,
    FUEL_PENALTY,
    LANDING_BONUS,
    StochasticActionFailureWrapper,
    is_safe_landing,
    make_env,
)

TOL = 1e-9   # float comparison tolerance for exact reward decomposition


# --------------------------------------------------------------------------- helpers
def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def collect_random_rollouts(n_episodes: int = 300, seed: int = 7, max_steps: int = 1000):
    """Run `n_episodes` of a uniform-random policy on the modified env, logging every step."""
    env = make_env("modified", seed=seed, record_log=True)
    rng = np.random.default_rng(seed)
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        start = len(env.log)
        terminated = truncated = False
        for _ in range(max_steps):
            obs, _r, terminated, truncated, _ = env.step(int(rng.integers(0, 4)))
            if terminated or truncated:
                break
        episodes.append(
            {
                "slice": (start, len(env.log)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_obs": np.asarray(obs, dtype=np.float64),
            }
        )
    log, stats = list(env.log), env.stats
    env.close()
    return log, stats, episodes


def collect_heuristic_rollouts(n_episodes: int = 40, seed: int = 21, max_steps: int = 1000):
    """Same as above but driven by Gymnasium's built-in `heuristic` LunarLander controller.

    This is only used to *produce successful landings*, so that the +50 bonus
    logic can be verified on positive as well as negative examples.
    """
    from gymnasium.envs.box2d.lunar_lander import heuristic

    env = make_env("modified", seed=seed, record_log=True)
    episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        start = len(env.log)
        terminated = truncated = False
        for _ in range(max_steps):
            action = heuristic(env.unwrapped, obs)     # deterministic PD controller
            obs, _r, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        episodes.append(
            {
                "slice": (start, len(env.log)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "final_obs": np.asarray(obs, dtype=np.float64),
            }
        )
    log = list(env.log)
    env.close()
    return log, episodes


# --------------------------------------------------------------------------- V0
def verify_structure() -> bool:
    """V0: spaces, dynamics and termination rules must be untouched."""
    _rule("V0  STRUCTURAL EQUIVALENCE WITH THE ORIGINAL ENVIRONMENT")
    orig, mod = make_env("original"), make_env("modified", seed=0)

    same_obs = orig.observation_space == mod.observation_space
    same_act = orig.action_space == mod.action_space
    print(f"observation space  original={orig.observation_space}")
    print(f"observation space  modified={mod.observation_space}   -> identical: {same_obs}")
    print(f"action space       original={orig.action_space}  modified={mod.action_space}"
          f"   -> identical: {same_act}")

    # Transition dynamics: with a = 0 the wrapper can never modify the action, so
    # feeding the same 'do nothing' sequence to both envs must give identical states.
    o1, _ = orig.reset(seed=99)
    o2, _ = mod.reset(seed=99)
    max_dev = float(np.max(np.abs(np.asarray(o1) - np.asarray(o2))))
    for _ in range(120):
        o1, r1, t1, tr1, _ = orig.step(0)
        o2, r2, t2, tr2, _ = mod.step(0)
        max_dev = max(max_dev, float(np.max(np.abs(np.asarray(o1) - np.asarray(o2)))))
        assert (t1, tr1) == (t2, tr2), "termination flags diverged"
        if t1 or tr1:
            break
    dynamics_ok = max_dev < 1e-12
    print(f"identical trajectories under a=0 (no failure possible): max |deviation| = "
          f"{max_dev:.2e}  -> unchanged dynamics/termination: {dynamics_ok}")
    orig.close(); mod.close()
    ok = same_obs and same_act and dynamics_ok
    print(f"\nV0 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- V1
def verify_misfire_rate(log, stats) -> bool:
    """V1: ~15% of attempted thruster actions must be replaced by action 0."""
    _rule("V1  ENGINE-FAILURE RATE  (target = 15% of attempted thruster actions)")
    n, k = stats.thruster_attempts, stats.misfires
    p_hat = k / n
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    lo, hi = p_hat - 1.96 * se, p_hat + 1.96 * se
    z = (p_hat - FAILURE_PROB) / math.sqrt(FAILURE_PROB * (1 - FAILURE_PROB) / n)

    print(f"total steps logged             : {stats.steps}")
    print(f"thruster actions attempted (n) : {n}")
    print(f"replaced by 'Do Nothing'   (k) : {k}")
    print(f"empirical misfire rate         : {p_hat:.4f}   (target {FAILURE_PROB:.2f})")
    print(f"95% binomial CI                : [{lo:.4f}, {hi:.4f}]")
    print(f"z-statistic vs p=0.15          : {z:+.3f}  (|z| < 1.96 -> consistent with 15%)")

    print("\nper-action breakdown:")
    print(f"  {'action':>7} {'attempts':>10} {'misfires':>10} {'rate':>8}")
    for a in (1, 2, 3):
        na, ka = stats.per_action_attempts[a], stats.per_action_misfires[a]
        print(f"  {a:>7} {na:>10} {ka:>10} {ka/na if na else float('nan'):>8.4f}")

    # Action 0 must NEVER be modified, and a misfire must always map to a_exec == 0.
    a0_modified = sum(1 for e in log if e["a"] == 0 and e["a_exec"] != 0)
    bad_exec = sum(1 for e in log if e["misfired"] and e["a_exec"] != 0)
    kept = sum(1 for e in log if e["a"] != 0 and not e["misfired"] and e["a_exec"] != e["a"])
    # The replacement must be driven by r < 0.15 exactly.
    rule_violations = sum(
        1 for e in log if e["a"] != 0 and e["misfired"] != (e["r"] < FAILURE_PROB)
    )
    print(f"\n'Do Nothing' (a=0) ever modified            : {a0_modified}  (must be 0)")
    print(f"misfires not mapped to a_exec=0             : {bad_exec}  (must be 0)")
    print(f"successful firings with a_exec != a         : {kept}  (must be 0)")
    print(f"steps where misfire != (r < 0.15)           : {rule_violations}  (must be 0)")

    print("\nsample of the step log (matches the example table in the problem statement):")
    print(f"  {'a':>3} {'r':>10} {'a_exec':>7}")
    shown = Counter()
    for e in log:
        key = (e["a"], e["misfired"])
        if shown[key] < 1 and shown.total() < 7:
            shown[key] += 1
            r_txt = "  -" if e["r"] is None else f"{e['r']:.4f}"
            print(f"  {e['a']:>3} {r_txt:>10} {e['a_exec']:>7}")

    ok = (abs(z) < 1.96) and a0_modified == 0 and bad_exec == 0 and kept == 0 and rule_violations == 0
    print(f"\nV1 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- V2
def verify_fuel_penalty(log) -> bool:
    """V2: penalty depends on the SELECTED action only, never on the executed one."""
    _rule("V2  FUEL PENALTY APPLIED PER *SELECTED* THRUSTER ACTION (-0.3)")
    groups = {"a=0 (no-op)": [], "a!=0 fired": [], "a!=0 misfired": []}
    for e in log:
        implied = e["reward"] - e["base_reward"] - e["bonus"]      # must equal -penalty
        key = "a=0 (no-op)" if e["a"] == 0 else ("a!=0 misfired" if e["misfired"] else "a!=0 fired")
        groups[key].append(implied)

    ok = True
    print(f"  {'case':>16} {'steps':>8} {'mean implied penalty':>22} {'max abs error':>15}")
    for key, vals in groups.items():
        expected = 0.0 if key.startswith("a=0") else -FUEL_PENALTY
        v = np.asarray(vals)
        err = float(np.max(np.abs(v - expected))) if len(v) else 0.0
        ok &= err < 1e-6
        print(f"  {key:>16} {len(v):>8} {v.mean() if len(v) else 0.0:>22.6f} {err:>15.2e}")

    print(f"\nexpected: 0.000000 for a=0, {-FUEL_PENALTY:.6f} for BOTH fired and misfired thrusters")
    print("-> the penalty is charged for every *attempted* thruster action, "
          "independently of whether the engine actually fired.")
    print(f"\nV2 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- V3
def _bonus_consistency(log) -> tuple[int, int, int]:
    """Count bonus rows and any mismatch between awarded bonus and the criterion."""
    awarded = mismatches = 0
    for e in log:
        vx, vy, ang, l_leg, r_leg = e["obs_tail"]
        criterion = (
            e["terminated"] and not e["truncated"]
            and l_leg == 1.0 and r_leg == 1.0
            and abs(vx) < 0.10 and abs(vy) < 0.10 and abs(ang) < 0.10
        )
        got = e["bonus"] == LANDING_BONUS
        awarded += int(got)
        mismatches += int(got != criterion) + int(e["bonus"] not in (0.0, LANDING_BONUS))
    return awarded, mismatches, len(log)


def verify_landing_bonus(random_log, random_eps, heur_log, heur_eps) -> bool:
    """V3: +50 is awarded exactly when the safe-landing criterion is satisfied."""
    _rule("V3  SAFE-LANDING BONUS (+50) AWARDED ONLY UNDER THE FULL CRITERION")

    for label, log, eps in (("random policy", random_log, random_eps),
                            ("heuristic policy", heur_log, heur_eps)):
        awarded, mismatch, n = _bonus_consistency(log)
        succ = sum(1 for e in eps if is_safe_landing(e["final_obs"], e["terminated"], e["truncated"]))
        print(f"\n[{label}]  episodes={len(eps)}  steps={n}")
        print(f"  safe landings by criterion : {succ}")
        print(f"  +50 bonuses awarded        : {awarded}   (must equal the line above)")
        print(f"  bonus/criterion mismatches : {mismatch}  (must be 0)")
        if awarded != succ or mismatch:
            print("  -> FAIL")
            return False

    # Show why the negative cases were rejected: which sub-condition failed.
    print("\nreason a terminal step did NOT receive the bonus (random-policy episodes):")
    reasons = Counter()
    for e in random_eps:
        obs, term, trunc = e["final_obs"], e["terminated"], e["truncated"]
        if is_safe_landing(obs, term, trunc):
            reasons["safe landing (+50 awarded)"] += 1
            continue
        if trunc:
            reasons["truncated episode"] += 1
        elif not (obs[6] == 1.0 and obs[7] == 1.0):
            reasons["legs not both in contact (crash / airborne)"] += 1
        elif abs(obs[2]) >= 0.10:
            reasons["|horizontal velocity| >= 0.10"] += 1
        elif abs(obs[3]) >= 0.10:
            reasons["|vertical velocity| >= 0.10"] += 1
        else:
            reasons["|orientation angle| >= 0.10"] += 1
    for k, v in reasons.most_common():
        print(f"  {v:>5}  {k}")

    ok = _unit_test_bonus_boundaries()
    print(f"\nV3 RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


class _StubEnv(gym.Env):
    """Deterministic 1-step stub returning a scripted (obs, reward, terminated, truncated).

    It lets us exercise every boundary condition of the safe-landing criterion
    without relying on the physics simulator ever producing those exact states.
    """

    def __init__(self, obs, terminated, truncated, base_reward=1.0):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (8,), np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self._obs = np.asarray(obs, dtype=np.float32)
        self._t, self._tr, self._r = terminated, truncated, base_reward

    def reset(self, *, seed=None, options=None):
        return self._obs.copy(), {}

    def step(self, action):
        return self._obs.copy(), self._r, self._t, self._tr, {}


def _unit_test_bonus_boundaries() -> bool:
    """Deterministic table-driven test of the Step-5 criterion (boundary cases)."""
    def obs(vx=0.0, vy=0.0, ang=0.0, l=1.0, r=1.0):
        return [0.0, 0.0, vx, vy, ang, 0.0, l, r]

    cases = [
        ("all criteria satisfied",              obs(),                      True,  False, True),
        ("truncated instead of terminated",     obs(),                      False, True,  False),
        ("terminated AND truncated",            obs(),                      True,  True,  False),
        ("left leg not in contact",             obs(l=0.0),                 True,  False, False),
        ("right leg not in contact",            obs(r=0.0),                 True,  False, False),
        ("|vx| exactly at the 0.10 boundary",   obs(vx=0.10),               True,  False, False),
        ("|vy| exactly at the 0.10 boundary",   obs(vy=-0.10),              True,  False, False),
        ("|angle| exactly at 0.10 rad",         obs(ang=0.10),              True,  False, False),
        ("just inside every tolerance",         obs(vx=0.099, vy=-0.099,
                                                    ang=0.099),            True,  False, True),
        ("not terminated at all",               obs(),                      False, False, False),
    ]
    print("\ndeterministic boundary unit test (stub environment, selected action a=2):")
    print(f"  {'case':>36} {'expected B':>11} {'actual B':>9} {'reward':>9} {'ok':>4}")
    all_ok = True
    for name, o, term, trunc, expect_bonus in cases:
        env = StochasticActionFailureWrapper(_StubEnv(o, term, trunc, base_reward=1.0), seed=0)
        env.failure_prob = 0.0                    # force the engine to fire, isolating the bonus
        env.reset()
        _, reward, _, _, _ = env.step(2)
        expected_reward = 1.0 - FUEL_PENALTY + (LANDING_BONUS if expect_bonus else 0.0)
        actual_bonus = env.last_bonus
        ok = (actual_bonus == (LANDING_BONUS if expect_bonus else 0.0)) and abs(reward - expected_reward) < TOL
        all_ok &= ok
        print(f"  {name:>36} {LANDING_BONUS if expect_bonus else 0.0:>11.1f} "
              f"{actual_bonus:>9.1f} {reward:>9.2f} {'OK' if ok else 'FAIL':>4}")
    return all_ok


# --------------------------------------------------------------------------- driver
def main(n_random_episodes: int = 300, n_heuristic_episodes: int = 40, seed: int = 7) -> bool:
    """Run all verification checks and print a consolidated report."""
    print("TASK (a) - VERIFICATION OF THE MODIFIED LunarLander-v3 ENVIRONMENT")
    print(f"random-policy episodes = {n_random_episodes}, "
          f"heuristic-policy episodes = {n_heuristic_episodes}, seed = {seed}")

    v0 = verify_structure()
    rnd_log, rnd_stats, rnd_eps = collect_random_rollouts(n_random_episodes, seed=seed)
    heur_log, heur_eps = collect_heuristic_rollouts(n_heuristic_episodes, seed=seed + 100)

    v1 = verify_misfire_rate(rnd_log, rnd_stats)
    v2 = verify_fuel_penalty(rnd_log + heur_log)
    v3 = verify_landing_bonus(rnd_log, rnd_eps, heur_log, heur_eps)

    _rule("SUMMARY")
    for name, ok in (("V0 structural equivalence", v0), ("V1 15% misfire rate", v1),
                     ("V2 fuel penalty on selected action", v2), ("V3 +50 safe-landing bonus", v3)):
        print(f"  {name:<38} {'PASS' if ok else 'FAIL'}")
    all_ok = v0 and v1 and v2 and v3
    print(f"\nOVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return all_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
