# dsr_agent.py
# Deep Successor Representations
#
# Architecture:
#   state (5) → Encoder → φ (32)
#   φ → reward head:  r̂ = φ^T w
#   φ → SR head:      ψ(s,a) = u_α(φ)[a]   shape (B, A, F)
#   Q(s,a) = ψ(s,a)^T w
#
# Why the encoder matters:
#   r = φ^T w only holds if φ is a learned representation.
#   With raw state coordinates, reward is non-linear (e.g. collision penalty)
#   and w can never fit it. The encoder gives the network freedom to find
#   a representation where reward IS approximately linear.
#
# Split optimizers:
#   opt_enc_rw : encoder + w       (reward regression)
#   opt_enc_sr : encoder + SR head (SR Bellman)
#   Note: encoder is shared but updated by both losses independently.

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────
class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: deque = deque(maxlen=int(cap))

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        self.buf.append((
            np.asarray(s,  dtype=np.float32),
            int(a),
            float(r),
            np.asarray(s2, dtype=np.float32),
            bool(done),
        ))

    def sample(self, n: int, device: str) -> Dict[str, torch.Tensor]:
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        return {
            "s":  torch.as_tensor(np.stack(s),  dtype=torch.float32, device=device),
            "s2": torch.as_tensor(np.stack(s2), dtype=torch.float32, device=device),
            "a":  torch.as_tensor(a,             dtype=torch.long,    device=device),
            "r":  torch.as_tensor(r,             dtype=torch.float32, device=device),
            "d":  torch.as_tensor(d,             dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return len(self.buf)


# ──────────────────────────────────────────────
# Encoder  f_θ
# state_dim → feat_dim
# Small but non-trivial: gives freedom for r = φ^T w to hold
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, state_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)   # (B, feat_dim)


# ──────────────────────────────────────────────
# SR Head  u_α
# feat_dim → sr_hidden → n_actions * feat_dim
# ──────────────────────────────────────────────
class SRHead(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.n_actions = int(n_actions)
        self.feat_dim  = int(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_actions * self.feat_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi).view(-1, self.n_actions, self.feat_dim)  # (B, A, F)


# ──────────────────────────────────────────────
# DSR
# ──────────────────────────────────────────────
class DSR(nn.Module):
    def __init__(self, state_dim: int, feat_dim: int, n_actions: int, sr_hidden: int = 128):
        super().__init__()
        self.encoder = Encoder(state_dim, feat_dim)
        self.sr      = SRHead(feat_dim, n_actions, hidden=sr_hidden)
        self.w       = nn.Parameter(torch.empty(feat_dim))
        nn.init.normal_(self.w, mean=0.0, std=1.0 / feat_dim ** 0.5)

    def phi(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def r_hat(self, phi: torch.Tensor) -> torch.Tensor:
        return phi.matmul(self.w)

    def q_all(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.encoder(s)
        psi = self.sr(phi)              # (B, A, F)
        q   = psi.matmul(self.w)       # (B, A)
        return q, psi, phi


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
@dataclass
class DSRConfig:
    state_dim:   int   = 5    # raw input size (fixed by env)
    feat_dim:    int   = 32   # encoder output size — φ lives here
    action_size: int   = 7
    enc_hidden:  int   = 128

    map_size:    float = 10.0
    sr_hidden:   int   = 128  # increased from 64

    buffer_size:  int   = 100_000
    batch_size:   int   = 32
    start_steps:  int   = 5_000    # start learning early (was 50k — caused epsilon/warmup mismatch)
    train_freq:   int   = 4

    gamma:         float = 0.95   # was 0.99: max |psi| ~ phi/(1-gamma) = 100x -> 20x
    learning_rate: float = 1e-4

    # linear epsilon decay
    epsilon_start:       float = 1.0
    epsilon_end:         float = 0.05
    epsilon_decay_steps: int   = 200_000

    # hard target update
    target_update_freq:  int   = 5_000   # was 10_000: fresher targets = less drift

    reward_clip: Optional[float] = None
    grad_clip:   float           = 0.5   # was 1.0: tighter clipping on SR gradients

    lambda_r:  float = 1.0
    lambda_sr: float = 0.5              # was 1.0: slower SR growth, less amplification

    device: Optional[str] = None


# ──────────────────────────────────────────────
# DSRAgent
# ──────────────────────────────────────────────
class DSRAgent:
    def __init__(self, config: DSRConfig):
        self.cfg    = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = int(config.action_size)
        self.feat_dim  = int(config.feat_dim)
        self.state_dim = int(config.state_dim)
        self.map_size  = float(config.map_size)

        self.net = DSR(self.state_dim, self.feat_dim, self.n_actions, self.cfg.sr_hidden).to(self.device)
        self.tgt = DSR(self.state_dim, self.feat_dim, self.n_actions, self.cfg.sr_hidden).to(self.device)
        self.tgt.load_state_dict(self.net.state_dict())
        for p in self.tgt.parameters():
            p.requires_grad_(False)

        # Split optimizers:
        #   opt_rw: encoder + w       → reward regression shapes φ to be reward-predictive
        #   opt_sr: encoder + SR head → SR Bellman shapes φ to be successor-predictive
        # Encoder is shared — both losses contribute to shaping φ
        self.opt_rw = torch.optim.Adam(
            list(self.net.encoder.parameters()) + [self.net.w],
            lr=config.learning_rate,
        )
        self.opt_sr = torch.optim.Adam(
            list(self.net.encoder.parameters()) + list(self.net.sr.parameters()),
            lr=config.learning_rate,
        )

        self.rb = Replay(config.buffer_size)
        self.total_steps = 0

    # ── linear epsilon schedule ───────────────────────────────────────────────
    @property
    def epsilon(self) -> float:
        t     = min(self.total_steps, self.cfg.epsilon_decay_steps)
        ratio = t / max(self.cfg.epsilon_decay_steps, 1)
        return self.cfg.epsilon_start + ratio * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    # ── obs → normalized state vector ────────────────────────────────────────
    def get_state_vec(self, obs: Any) -> np.ndarray:
        """
        Parses ExpertObservations dict:
          obs["robot"]  = [goal_x, goal_y, theta]
          obs["humans"] = [hx, hy]
        Normalizes to [-1, 1] and returns shape (5,).
        """
        if isinstance(obs, (tuple, list)) and len(obs) >= 1 and isinstance(obs[0], dict):
            obs = obs[0]

        if not isinstance(obs, dict):
            return np.zeros((self.state_dim,), dtype=np.float32)

        robot = obs.get("robot", None)
        if robot is not None:
            r  = np.asarray(robot, dtype=np.float32).flatten()
            gx = float(r[0]) if r.size > 0 else 0.0
            gy = float(r[1]) if r.size > 1 else 0.0
            th = float(r[2]) if r.size > 2 else 0.0
        else:
            gx, gy, th = 0.0, 0.0, 0.0

        humans = obs.get("humans", None)
        if humans is not None:
            h  = np.asarray(humans, dtype=np.float32).flatten()
            hx = float(h[0]) if h.size > 0 else 0.0
            hy = float(h[1]) if h.size > 1 else 0.0
        else:
            hx, hy = 0.0, 0.0

        return np.array([
            gx / self.map_size,
            gy / self.map_size,
            th / np.pi,
            hx / self.map_size,
            hy / self.map_size,
        ], dtype=np.float32)

    # ── action selection ──────────────────────────────────────────────────────
    @torch.no_grad()
    def sample_action(self, s_np: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and (random.random() < self.epsilon):
            return random.randrange(self.n_actions)
        s    = torch.as_tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        q, _, _ = self.net.q_all(s)
        return int(q.argmax(dim=1).item())

    # ── learning step ─────────────────────────────────────────────────────────
    def update_from_replay(self) -> Optional[Tuple[float, float]]:
        if len(self.rb) < max(self.cfg.batch_size, self.cfg.start_steps):
            return None

        batch = self.rb.sample(self.cfg.batch_size, device=self.device)
        s    = batch["s"]
        s2   = batch["s2"]
        a    = batch["a"].long()
        r    = batch["r"]
        d    = batch["d"]

        # ── Step 1: reward regression ─────────────────────────────────────────
        # Trains encoder + w so that φ^T w ≈ reward
        self.opt_rw.zero_grad()
        phi  = self.net.phi(s)
        loss_r = self.cfg.lambda_r * F.mse_loss(self.net.r_hat(phi), r)
        loss_r.backward()
        nn.utils.clip_grad_norm_(
            list(self.net.encoder.parameters()) + [self.net.w],
            self.cfg.grad_clip,
        )
        self.opt_rw.step()

        # ── Step 2: SR Bellman ────────────────────────────────────────────────
        # Trains encoder + SR head so that ψ(s,a) ≈ φ(s) + γ ψ(s',a*)
        self.opt_sr.zero_grad()

        _, psi_all, phi_s = self.net.q_all(s)
        psi_sa = psi_all.gather(
            1, a.view(-1, 1, 1).expand(-1, 1, psi_all.size(-1))
        ).squeeze(1)                                            # (B, F)

        with torch.no_grad():
            # Double-DQN: select action from online network (uses online w)
            q2_online, _, _ = self.net.q_all(s2)
            a_star = q2_online.argmax(1, keepdim=True)

            # Bootstrap ψ from target SR + target encoder
            _, psi2_all, _ = self.tgt.q_all(s2)
            psi2 = psi2_all.gather(
                1, a_star.view(-1, 1, 1).expand(-1, 1, psi2_all.size(-1))
            ).squeeze(1)

            # BUG 2 FIX: detach φ(s) so SR loss cannot backprop through
            # the encoder via the identity term.
            # φ comes from target encoder for stability (frozen copy).
            phi_s_target = self.tgt.phi(s).detach()
            target_psi   = phi_s_target + (1.0 - d).unsqueeze(-1) * self.cfg.gamma * psi2

        loss_sr = self.cfg.lambda_sr * F.mse_loss(psi_sa, target_psi)
        loss_sr.backward()
        nn.utils.clip_grad_norm_(
            list(self.net.encoder.parameters()) + list(self.net.sr.parameters()),
            self.cfg.grad_clip,
        )
        self.opt_sr.step()

        # ── BUG 1 FIX: only copy encoder + SR to target, NOT w ───────────────
        # w changes fast during reward learning — copying it to the target
        # causes noisy a* selection and unstable SR TD targets.
        if self.total_steps % self.cfg.target_update_freq == 0:
            self.tgt.encoder.load_state_dict(self.net.encoder.state_dict())
            self.tgt.sr.load_state_dict(self.net.sr.state_dict())

        return float(loss_sr.item()), float(loss_r.item())

    # ── store transition + learn ───────────────────────────────────────────────
    def observe_and_learn(
        self,
        obs:        Any,
        action:     int,
        reward:     float,
        next_obs:   Any,
        terminated: bool,
        truncated:  bool,
    ) -> Optional[Tuple[float, float]]:
        s    = self.get_state_vec(obs)
        s2   = self.get_state_vec(next_obs)
        done = bool(terminated or truncated)

        r = float(reward)
        if self.cfg.reward_clip is not None:
            r = float(np.clip(r, -self.cfg.reward_clip, self.cfg.reward_clip))

        self.rb.push(s, action, r, s2, done)
        self.total_steps += 1

        if self.total_steps % self.cfg.train_freq == 0:
            return self.update_from_replay()
        return None

    # ── full episode ───────────────────────────────────────────────────────────
    def run_episode(
        self,
        env,
        eval_mode: bool = False,
    ) -> Tuple[int, float]:
        obs, _info = env.reset()
        ep_ret = 0.0
        ep_len = 0

        while True:
            s = self.get_state_vec(obs)
            a = self.sample_action(s, eval_mode=eval_mode)

            next_obs, reward, terminated, truncated, _info = env.step(a)

            if not eval_mode:
                self.observe_and_learn(obs, a, reward, next_obs, terminated, truncated)

            ep_ret += float(reward)
            ep_len += 1
            obs     = next_obs

            if terminated or truncated:
                break

        return ep_len, ep_ret