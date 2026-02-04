# agents/dsr_agent.py
# Drop-in DSR agent consistent with your repo patterns (Gymnasium terminated/truncated,
# SocNavGym dict obs, and robot obs layout used by your StateDiscretizer).

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------
# Replay Buffer
# --------------------
class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf = deque(maxlen=int(cap))

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        self.buf.append((np.asarray(s, dtype=np.float32), int(a), float(r), np.asarray(s2, dtype=np.float32), bool(done)))

    def sample(self, n: int, device: str):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        S = torch.as_tensor(np.stack(s), dtype=torch.float32, device=device)
        S2 = torch.as_tensor(np.stack(s2), dtype=torch.float32, device=device)
        A = torch.as_tensor(a, dtype=torch.long, device=device)
        R = torch.as_tensor(r, dtype=torch.float32, device=device)
        D = torch.as_tensor(d, dtype=torch.float32, device=device)
        return {"s": S, "a": A, "r": R, "s2": S2, "d": D}

    def __len__(self):
        return len(self.buf)


# --------------------
# SR Head: ψ(s,a)
# --------------------
class SRHead(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.n_actions = int(n_actions)
        self.feat_dim = int(feat_dim)

        self.net = nn.Sequential(
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_actions * self.feat_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        out = self.net(phi)  # (B, A*F)
        return out.view(-1, self.n_actions, self.feat_dim)  # (B, A, F)


# --------------------
# DSR model
# --------------------
class DSR(nn.Module):
    def __init__(self, n_actions: int, feat_dim: int, hidden: int = 256, reward_hidden: int = 64):
        super().__init__()
        self.n_actions = int(n_actions)
        self.feat_dim = int(feat_dim)

        self.sr = SRHead(self.feat_dim, self.n_actions, hidden=hidden)

        # Q(s,a) = ψ(s,a) · q_vec
        self.q_vec = nn.Parameter(torch.zeros(self.feat_dim))
        nn.init.normal_(self.q_vec, mean=0.0, std=0.1)

        # r_hat(s) = MLP(φ(s))
        self.r_head = nn.Sequential(
            nn.Linear(self.feat_dim, reward_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(reward_hidden, 1),
        )
        for m in self.r_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def q_all(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        psi = self.sr(phi)            # (B, A, F)
        q = psi.matmul(self.q_vec)    # (B, A)
        return q, psi

    def r_hat(self, phi: torch.Tensor) -> torch.Tensor:
        return self.r_head(phi).squeeze(-1)  # (B,)


# --------------------
# Loss + target update
# --------------------
def dsr_loss_step(net: DSR, tgt: DSR, batch: Dict[str, torch.Tensor], gamma: float = 0.99, q_coef: float = 1.0):
    """
    Total loss = SR Bellman error + reward regression + q_coef * Q-learning TD loss.
    batch:
      s,s2: (B,F)   a: (B,)   r: (B,)   d: (B,)  (d is float 0/1)
    """
    phi = batch["s"]
    phi2 = batch["s2"]
    a = batch["a"].long()
    r = batch["r"]
    d = batch["d"]

    # Online forward
    q, psi = net.q_all(phi)  # q: (B,A), psi: (B,A,F)

    # ψ(s,a)
    psi_sa = psi.gather(1, a.view(-1, 1, 1).expand(-1, 1, psi.size(-1))).squeeze(1)  # (B,F)

    # SR target (Double-style: a* from online, ψ from target)
    with torch.no_grad():
        q_next_online, _ = net.q_all(phi2)
        a_star = q_next_online.argmax(1, keepdim=True)  # (B,1)

        psi_next_all = tgt.sr(phi2)  # (B,A,F)
        psi_next = psi_next_all.gather(1, a_star.view(-1, 1, 1).expand(-1, 1, psi_next_all.size(-1))).squeeze(1)  # (B,F)

        target_psi = phi + (1.0 - d).unsqueeze(-1) * gamma * psi_next  # (B,F)

    loss_sr = F.smooth_l1_loss(psi_sa, target_psi)

    # Reward regression
    r_hat = net.r_hat(phi)
    loss_r = F.smooth_l1_loss(r_hat, r)

    # Q TD target using target network
    with torch.no_grad():
        psi_next_tgt = tgt.sr(phi2)                    # (B,A,F)
        q_next_tgt = psi_next_tgt.matmul(tgt.q_vec)    # (B,A)
        y_q = r + gamma * (1.0 - d) * q_next_tgt.max(dim=1).values

    q_sa = q.gather(1, a.view(-1, 1)).squeeze(1)
    loss_q = F.smooth_l1_loss(q_sa, y_q)

    return loss_sr + loss_r + q_coef * loss_q


def soft_update(net: nn.Module, tgt: nn.Module, tau: float = 0.005):
    with torch.no_grad():
        for p, pt in zip(net.parameters(), tgt.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)


# --------------------
# DSRAgent (repo-friendly interface)
# --------------------
@dataclass
class DSRConfig:
    action_size: int = 7
    buffer_size: int = 50_000
    batch_size: int = 64
    start_steps: int = 1_000
    train_freq: int = 1
    gamma: float = 0.99
    learning_rate: float = 6e-3
    epsilon: float = 0.05
    q_coef: float = 1.0
    tau: float = 0.005
    reward_clip: float = 1.0
    grad_clip: float = 5.0
    device: str | None = None


class DSRAgent:
    """
    Minimal DSR agent for SocNavGym dict observations.

    Uses φ(s) = [goal_dx, goal_dy] extracted from obs["robot"] following your
    StateDiscretizer convention:
      robot obs = [one_hot(D), goal_dx, goal_dy, robot_radius]  => dx=r[-3], dy=r[-2]
    """

    def __init__(self, config: DSRConfig):
        self.cfg = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = int(config.action_size)
        self.max_humans = 1  # choose K
        self.feat_dim = 3 + 2 * self.max_humans
        self.net = DSR(n_actions=self.n_actions, feat_dim=self.feat_dim).to(self.device)
        self.tgt = DSR(n_actions=self.n_actions, feat_dim=self.feat_dim).to(self.device)
        self.tgt.load_state_dict(self.net.state_dict())

        self.opt = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate)
        self.rb = Replay(config.buffer_size)

        self.total_steps = 0

    def _extract_goal_theta_from_robot(self, robot_arr: np.ndarray) -> tuple[float, float, float]:
        r = np.asarray(robot_arr, dtype=np.float32).flatten()
        if r.size == 0:
            return 0.0, 0.0, 0.0

        # obs["robot"] = [goal_x, goal_y, theta, ...]
        if r.size >= 3:
            gx, gy, th = float(r[0]), float(r[1]), float(r[2])
            return gx, gy, th

        return 0.0, 0.0, 0.0

    def _extract_humans_xy(self, humans) -> np.ndarray:
        """
        Return (N,2) array of human (x,y) in robot frame.
        Robust to common shapes/encodings.
        """
        if humans is None:
            return np.zeros((0, 2), dtype=np.float32)

        h = np.asarray(humans, dtype=np.float32)

        # (N,2) or (N,>=2)
        if h.ndim == 2 and h.shape[1] >= 2:
            return h[:, :2].astype(np.float32, copy=False)

        h = h.flatten()
        if h.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # single (x,y)
        if h.size == 2:
            return np.asarray([[h[0], h[1]]], dtype=np.float32)

        # Try entity-block style (default one_hot_len=6, block=14): x,y at (6,7)
        one_hot_len = 6
        block = 14

        # if divisible, decode blocks
        if h.size % block == 0:
            pts = []
            for i in range(0, h.size, block):
                base = i + one_hot_len
                if base + 1 < h.size:
                    pts.append((float(h[base]), float(h[base + 1])))
            return np.asarray(pts, dtype=np.float32)

        # fallback: if at least 8, assume (6,7)
        if h.size >= 8:
            return np.asarray([[float(h[6]), float(h[7])]], dtype=np.float32)

        # fallback: first two
        if h.size >= 2:
            return np.asarray([[float(h[0]), float(h[1])]], dtype=np.float32)

        return np.zeros((0, 2), dtype=np.float32)

    def get_features(self, obs) -> np.ndarray:
        """
        φ(s) = [goal_dx, goal_dy, theta, h1x, h1y, ..., hKx, hKy]
        padded with zeros if <K humans present.
        """
        if isinstance(obs, (tuple, list)) and len(obs) == 2 and isinstance(obs[0], dict):
            obs = obs[0]
        if not isinstance(obs, dict):
            return np.zeros((self.feat_dim,), dtype=np.float32)

        # robot
        robot = obs.get("robot", None)
        if robot is None:
            gx, gy, th = 0.0, 0.0, 0.0
        else:
            gx, gy, th = self._extract_goal_theta_from_robot(robot)

        # humans
        humans_xy = self._extract_humans_xy(obs.get("humans", None))  # (N,2)

        # keep first K humans (or sort by distance if you want)
        K = self.max_humans
        if humans_xy.shape[0] > K:
            humans_xy = humans_xy[:K]

        # pad to K
        pad = np.zeros((K, 2), dtype=np.float32)
        if humans_xy.shape[0] > 0:
            pad[: humans_xy.shape[0], :] = humans_xy

        phi = np.concatenate([np.asarray([gx, gy, th], dtype=np.float32), pad.reshape(-1)], axis=0)
        return phi.astype(np.float32, copy=False)


    @torch.no_grad()
    def sample_action(self, phi_np: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and (random.random() < self.cfg.epsilon):
            return random.randrange(self.n_actions)

        phi = torch.as_tensor(phi_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,2)
        q, _ = self.net.q_all(phi)
        return int(q.argmax(dim=1).item())

    def update_from_replay(self) -> float | None:
        if len(self.rb) < max(self.cfg.batch_size, self.cfg.start_steps):
            return None

        batch = self.rb.sample(self.cfg.batch_size, device=self.device)
        self.opt.zero_grad(set_to_none=True)
        loss = dsr_loss_step(self.net, self.tgt, batch, gamma=self.cfg.gamma, q_coef=self.cfg.q_coef)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.opt.step()
        soft_update(self.net, self.tgt, tau=self.cfg.tau)
        return float(loss.item())

    # ---------- step-style API (fits typical repo training loops) ----------
    def observe_and_learn(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        terminated: bool,
        truncated: bool,
    ) -> float | None:
        phi = self.get_features(obs)
        phi2 = self.get_features(next_obs)
        done = bool(terminated or truncated)

        r = float(reward)
        if self.cfg.reward_clip is not None:
            c = float(self.cfg.reward_clip)
            r = float(np.clip(r, -c, c))

        self.rb.push(phi, action, r, phi2, done)
        self.total_steps += 1

        if self.total_steps % self.cfg.train_freq == 0:
            return self.update_from_replay()
        return None

    # ---------- optional: run one episode ----------
    def run_episode(self, env, eval_mode: bool = False, max_steps: int = 10_000):
        obs, info = env.reset()
        ep_ret = 0.0
        ep_len = 0

        while True:
            phi = self.get_features(obs)
            a = self.sample_action(phi, eval_mode=eval_mode)

            next_obs, reward, terminated, truncated, info = env.step(a)

            if not eval_mode:
                self.observe_and_learn(obs, a, reward, next_obs, terminated, truncated)

            ep_ret += float(reward)
            ep_len += 1
            obs = next_obs

            if terminated or truncated or ep_len >= max_steps:
                break

        return ep_len, ep_ret
