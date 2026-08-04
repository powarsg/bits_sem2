"""
agent.py
========
Tasks (b) and (c).

Contains the three learning components shared by **both** algorithms

  * `QNetwork`      - the value network (identical architecture for DQN / DDQN)
  * `ReplayBuffer`  - uniform experience replay
  * `DQNAgent`      - epsilon-greedy control + target network + learning rule

The DQN and the DDQN agents are the *same* class instantiated with a different
`algo` flag.  The flag changes exactly one expression - the computation of the
bootstrapped target Q-value in `_learn` - which is the only difference the
assignment allows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- config
@dataclass
class AgentConfig:
    """All hyper-parameters in one place so DQN and DDQN provably share them."""

    hidden_sizes: tuple = (128, 128)
    lr: float = 5e-4                 # Adam learning rate
    gamma: float = 0.99              # discount factor
    buffer_size: int = 100_000       # replay capacity
    batch_size: int = 64             # minibatch size
    learn_every: int = 4             # env steps between gradient updates
    updates_per_learn: int = 1       # gradient steps per learning event
    tau: float = 1e-3                # soft target-network update coefficient
    warmup: int = 1_000              # transitions collected before learning starts
    grad_clip: float = 10.0          # gradient-norm clipping (stability)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995         # multiplicative, applied once per episode

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- network
class QNetwork(nn.Module):
    """Feed-forward Q-network: state (8) -> hidden -> hidden -> Q(s, .) (4)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers, in_dim = [], state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))   # linear head -> raw Q-values
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------------------------------------------------------- replay
class ReplayBuffer:
    """Fixed-capacity circular buffer with uniform random sampling."""

    def __init__(self, capacity: int, state_dim: int, seed: int = 0):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done) -> None:
        """Store one transition, overwriting the oldest entry when full."""
        i = self.ptr
        self.states[i], self.actions[i], self.rewards[i] = s, a, r
        self.next_states[i], self.dones[i] = s2, float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        """Draw a uniform minibatch and move it to `device` as torch tensors."""
        idx = self._rng.integers(0, self.size, size=batch_size)
        to = lambda arr, dtype: torch.as_tensor(arr[idx], dtype=dtype, device=device)
        return (
            to(self.states, torch.float32),
            to(self.actions, torch.int64).unsqueeze(1),
            to(self.rewards, torch.float32).unsqueeze(1),
            to(self.next_states, torch.float32),
            to(self.dones, torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return self.size


# ----------------------------------------------------------------------------- agent
class DQNAgent:
    """Value-based agent implementing both DQN and Double DQN.

    `algo="dqn"`  -> target uses max_a' Q_target(s', a')                (Mnih et al., 2015)
    `algo="ddqn"` -> target uses Q_target(s', argmax_a' Q_online(s',a')) (van Hasselt, 2016)
    """

    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig,
                 algo: str = "dqn", seed: int = 0, device: str | torch.device = "cpu"):
        if algo not in ("dqn", "ddqn"):
            raise ValueError("algo must be 'dqn' or 'ddqn'")
        self.algo = algo
        self.cfg = config
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Both networks start from the *same* weights; the seed makes the initial
        # parameters identical for DQN and DDQN.
        torch.manual_seed(seed)
        self.q_online = QNetwork(state_dim, action_dim, config.hidden_sizes).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, config.hidden_sizes).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.buffer_size, state_dim, seed=seed)

        self._rng = random.Random(seed)     # epsilon-greedy exploration RNG
        self._step_count = 0
        self.last_loss: float | None = None

    # -------------------------------------------------------------- acting
    def act(self, state, eps: float = 0.0) -> int:
        """epsilon-greedy action selection w.r.t. the online network."""
        if self._rng.random() < eps:
            return self._rng.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_online(s).argmax(dim=1).item())

    # -------------------------------------------------------------- learning
    def step(self, s, a, r, s2, done) -> None:
        """Store a transition and periodically run a gradient update."""
        self.memory.add(s, a, r, s2, done)
        self._step_count += 1
        if len(self.memory) < max(self.cfg.warmup, self.cfg.batch_size):
            return
        if self._step_count % self.cfg.learn_every == 0:
            for _ in range(self.cfg.updates_per_learn):
                self._learn()

    def _target_q(self, next_states: torch.Tensor) -> torch.Tensor:
        """Bootstrapped next-state value - the ONLY difference between DQN and DDQN."""
        if self.algo == "dqn":
            # Vanilla DQN: the target network both selects and evaluates the action.
            return self.q_target(next_states).max(dim=1, keepdim=True)[0]
        # Double DQN: the ONLINE network selects, the TARGET network evaluates,
        # which decorrelates selection from evaluation and reduces overestimation.
        next_actions = self.q_online(next_states).argmax(dim=1, keepdim=True)
        return self.q_target(next_states).gather(1, next_actions)

    def _learn(self) -> None:
        """One Adam step on the temporal-difference (Huber) loss."""
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.cfg.batch_size, self.device
        )
        with torch.no_grad():
            q_next = self._target_q(next_states)
            targets = rewards + self.cfg.gamma * q_next * (1.0 - dones)

        q_pred = self.q_online(states).gather(1, actions)
        loss = F.smooth_l1_loss(q_pred, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.last_loss = float(loss.detach())

        self._soft_update()

    def _soft_update(self) -> None:
        """Polyak averaging: theta_target <- tau*theta_online + (1-tau)*theta_target."""
        tau = self.cfg.tau
        with torch.no_grad():
            for p_t, p_o in zip(self.q_target.parameters(), self.q_online.parameters()):
                p_t.mul_(1.0 - tau).add_(tau * p_o)

    # -------------------------------------------------------------- diagnostics
    @torch.no_grad()
    def mean_validation_q(self, val_states: np.ndarray) -> float:
        """Mean over the fixed validation set of max_a Q(s, a) - Task (d), plot 2."""
        s = torch.as_tensor(val_states, dtype=torch.float32, device=self.device)
        return float(self.q_online(s).max(dim=1)[0].mean())

    def save(self, path) -> None:
        torch.save(self.q_online.state_dict(), path)
