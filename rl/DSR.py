# agents/dsr_agent.py
# DSR architecture matching the picture: Encoder -> phi, SR(psi) per action, linear reward phi^T w,
# Decoder reconstructs s. Loss = SR Bellman + reward regression + reconstruction.

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
        self.buf.append(
            (
                np.asarray(s, dtype=np.float32),
                int(a),
                float(r),
                np.asarray(s2, dtype=np.float32),
                bool(done),
            )
        )

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
# Encoder / Decoder (as in picture)
# --------------------
# --------------------
# Encoder / Decoder (MLP, matching paper layer sizes without CNN)
# --------------------
class Encoder(nn.Module):
    """
    Paper: 3 conv layers + FC(512).
    No-CNN equivalent: 3 FC layers ending at feat_dim=512.
    """
    def __init__(self, state_dim: int, feat_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim),  # FC 512 (paper)
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)  # (B, feat_dim)


class Decoder(nn.Module):
    """
    Paper decoder deconv feature sizes: {512, 256, 128, 64, 4}.
    No-CNN equivalent: MLP 512->256->128->64->state_dim.
    (Replace "4" by state_dim for vector state reconstruction.)
    """
    def __init__(self, feat_dim: int = 512, state_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, state_dim),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi)  # (B, state_dim)


# --------------------
# SR Head: θψ with two fully-connected layers (as in caption)
# --------------------
class SRHead(nn.Module):
    """
    Paper: θψ contains two fully-connected layers.
    Implement: FC -> ReLU -> FC -> ReLU -> output (N*F)
    """
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.n_actions = int(n_actions)
        self.feat_dim = int(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),  # FC1
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),         # FC2
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_actions * self.feat_dim),  # output
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        out = self.net(phi)  # (B, A*F)
        return out.view(-1, self.n_actions, self.feat_dim)  # (B, A, F)


# --------------------
# Full DSR module (matches picture)
# --------------------
class DSR(nn.Module):
    """
    - phi(s) = Encoder(s)
    - psi(phi, a) = SRHead(phi)[a]
    - r_hat(s) = phi(s)^T w   (linear reward, as in paper figure)
    - s_hat = Decoder(phi)
    - Q(s,a) = psi(s,a)^T w   (common in DSR)
    """
    def __init__(self, state_dim: int, feat_dim: int, n_actions: int, sr_hidden: int = 256):
        super().__init__()
        self.enc = Encoder(state_dim, feat_dim)
        self.dec = Decoder(feat_dim, state_dim)
        self.sr = SRHead(feat_dim, n_actions, hidden=sr_hidden)
        self.w = nn.Parameter(torch.zeros(feat_dim))
        nn.init.normal_(self.w, mean=0.0, std=0.1)


    def phi(self, s: torch.Tensor) -> torch.Tensor:
        return self.enc(s)

    def s_hat(self, phi: torch.Tensor) -> torch.Tensor:
        return self.dec(phi)

    def r_hat(self, phi: torch.Tensor) -> torch.Tensor:
        return phi.matmul(self.w)  # (B,)

    def q_all(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        psi = self.sr(phi)          # (B,A,F)
        q = psi.matmul(self.w)      # (B,A)
        return q, psi


# --------------------
# Loss step (matches picture + SR Bellman)
# --------------------
def dsr_loss_step(
    net: DSR,
    tgt: DSR,
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.99,
    lambda_r: float = 1.0,
    lambda_d: float = 1.0,
    lambda_sr: float = 1.0,
):
    """
    Picture loss:
      (r - phi^T w)^2 + ||s - d(phi)||^2
    plus SR Bellman loss:
      || psi(s,a) - (phi(s) + gamma * psi(s',a*)) ||^2

    batch:
      s,s2: (B,state_dim)   a: (B,)   r: (B,)   d: (B,)  (d is float 0/1)
    """
    s = batch["s"]
    s2 = batch["s2"]
    a = batch["a"].long()
    r = batch["r"]
    d = batch["d"]

    # Encode
    phi = net.phi(s)         # (B,F)
    phi2 = net.phi(s2)       # (B,F)

    # ---------- reward regression (phi^T w) ----------
    r_hat = net.r_hat(phi)   # (B,)
    loss_r = F.mse_loss(r_hat, r)

    # ---------- reconstruction (decoder) ----------
    s_hat = net.s_hat(phi)   # (B,state_dim)
    loss_d = F.mse_loss(s_hat, s)

    # ---------- SR Bellman ----------
    q, psi = net.q_all(phi)  # psi: (B,A,F)

    psi_sa = psi.gather(1, a.view(-1, 1, 1).expand(-1, 1, psi.size(-1))).squeeze(1)  # (B,F)

    with torch.no_grad():
        # a* from online Q (double-style); psi from target
        q2_online, _ = net.q_all(phi2)
        a_star = q2_online.argmax(1, keepdim=True)  # (B,1)

        psi2_all = tgt.sr(phi2)  # (B,A,F)
        psi2 = psi2_all.gather(1, a_star.view(-1, 1, 1).expand(-1, 1, psi2_all.size(-1))).squeeze(1)  # (B,F)

        target_psi = phi + (1.0 - d).unsqueeze(-1) * gamma * psi2  # (B,F)

    loss_sr = F.mse_loss(psi_sa, target_psi)

    return lambda_sr * loss_sr + lambda_r * loss_r + lambda_d * loss_d


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
    state_dim: int = 5

    feat_dim: int = 512      # paper FC size
    sr_hidden: int = 256     # SR head hidden (2 FC layers)
    # (Encoder/Decoder sizes are fixed to match paper values above)

    buffer_size: int = 50_000
    batch_size: int = 128
    start_steps: int = 5_000
    train_freq: int = 1

    gamma: float = 0.99
    learning_rate: float = 3e-4  # recommended; 6e-3 is usually unstable here
    epsilon: float = 0.3

    tau: float = 0.005
    reward_clip: float | None = None
    grad_clip: float = 5.0

    lambda_r: float = 1.0
    lambda_d: float = 0.1
    lambda_sr: float = 1.0

    device: str | None = None



class DSRAgent:
    """
    DSR matching the paper figure: learns phi(s), reconstructs s, learns linear reward, learns psi.

    IMPORTANT: you must define get_state_vec(obs) -> np.ndarray of shape (state_dim,)
    that is consistent across episodes and resets.
    """

    def __init__(self, config: DSRConfig):
        self.cfg = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.n_actions = int(config.action_size)
        self.state_dim = int(config.state_dim)
        self.feat_dim = int(config.feat_dim)

        self.net = DSR(self.state_dim, self.feat_dim, self.n_actions, sr_hidden=self.cfg.sr_hidden).to(self.device)
        self.tgt = DSR(self.state_dim, self.feat_dim, self.n_actions, sr_hidden=self.cfg.sr_hidden).to(self.device)


        self.tgt.load_state_dict(self.net.state_dict())

        self.opt_sr = torch.optim.Adam(self.net.sr.parameters(), lr=config.learning_rate)

        self.opt_repr = torch.optim.Adam(
            list(self.net.enc.parameters()) +
            list(self.net.dec.parameters()) +
            [self.net.w],
            lr=config.learning_rate,
        )
        self.rb = Replay(config.buffer_size)
        self.total_steps = 0

        # --------- observation to state vector settings ----------
        self.max_humans = 1  # choose K for fixed-size encoding

    def _extract_goal_theta_from_robot(self, robot_arr: np.ndarray) -> tuple[float, float, float]:
        r = np.asarray(robot_arr, dtype=np.float32).flatten()
        if r.size >= 3:
            return float(r[0]), float(r[1]), float(r[2])
        return 0.0, 0.0, 0.0

    def _extract_humans_xy(self, humans) -> np.ndarray:
        if humans is None:
            return np.zeros((0, 2), dtype=np.float32)
        h = np.asarray(humans, dtype=np.float32)
        if h.ndim == 2 and h.shape[1] >= 2:
            return h[:, :2].astype(np.float32, copy=False)
        h = h.flatten()
        if h.size == 2:
            return np.asarray([[h[0], h[1]]], dtype=np.float32)
        if h.size >= 2:
            return np.asarray([[float(h[0]), float(h[1])]], dtype=np.float32)
        return np.zeros((0, 2), dtype=np.float32)

    def get_state_vec(self, obs: Any) -> np.ndarray:
        """
        Fixed-size raw state s (input to encoder). Keep it simple and stable.
        Example here:
          s = [goal_x, goal_y, theta, h1x, h1y, ..., hKx, hKy]
        So state_dim must be 3 + 2*K.
        """
        if isinstance(obs, (tuple, list)) and len(obs) == 2 and isinstance(obs[0], dict):
            obs = obs[0]
        if not isinstance(obs, dict):
            return np.zeros((self.state_dim,), dtype=np.float32)

        gx, gy, th = self._extract_goal_theta_from_robot(obs.get("robot", None))

        humans_xy = self._extract_humans_xy(obs.get("humans", None))
        K = self.max_humans
        if humans_xy.shape[0] > K:
            humans_xy = humans_xy[:K]
        pad = np.zeros((K, 2), dtype=np.float32)
        if humans_xy.shape[0] > 0:
            pad[: humans_xy.shape[0], :] = humans_xy

        s = np.concatenate([np.asarray([gx, gy, th], dtype=np.float32), pad.reshape(-1)], axis=0)

        # safety: enforce shape = (state_dim,)
        if s.size != self.state_dim:
            out = np.zeros((self.state_dim,), dtype=np.float32)
            m = min(out.size, s.size)
            out[:m] = s[:m]
            s = out

        return s.astype(np.float32, copy=False)

    @torch.no_grad()
    def sample_action(self, s_np: np.ndarray, eval_mode: bool = False) -> int:
        if (not eval_mode) and (random.random() < self.cfg.epsilon):
            return random.randrange(self.n_actions)

        s = torch.as_tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,state_dim)
        phi = self.net.phi(s)
        q, _ = self.net.q_all(phi)
        return int(q.argmax(dim=1).item())

    def update_from_replay(self):
        if len(self.rb) < max(self.cfg.batch_size, self.cfg.start_steps):
            return None

        batch = self.rb.sample(self.cfg.batch_size, device=self.device)

        # =========================
        # Step 1: SR update (θ_ψ)
        # =========================
        self.opt_sr.zero_grad()

        s = batch["s"]
        s2 = batch["s2"]
        a = batch["a"].long()
        d = batch["d"]

        with torch.no_grad():
            phi = self.net.phi(s)        # detach encoder
            phi2 = self.net.phi(s2)

        psi = self.net.sr(phi)
        psi_sa = psi.gather(1, a.view(-1,1,1).expand(-1,1,psi.size(-1))).squeeze(1)

        with torch.no_grad():
            q2, _ = self.net.q_all(phi2)
            a_star = q2.argmax(1, keepdim=True)
            psi2_all = self.tgt.sr(phi2)
            psi2 = psi2_all.gather(1, a_star.view(-1,1,1).expand(-1,1,psi2_all.size(-1))).squeeze(1)
            target_psi = phi + (1-d).unsqueeze(-1)*self.cfg.gamma*psi2

        loss_sr = F.mse_loss(psi_sa, target_psi)
        loss_sr.backward()
        self.opt_sr.step()

        # =========================
        # Step 2: Representation update (θ_φ, θ_d, w)
        # =========================
        self.opt_repr.zero_grad()

        phi = self.net.phi(s)
        r_hat = phi.matmul(self.net.w)
        loss_r = F.mse_loss(r_hat, batch["r"])

        s_hat = self.net.dec(phi)
        loss_d = F.mse_loss(s_hat, s)

        loss_repr = loss_r + loss_d
        loss_repr.backward()
        self.opt_repr.step()

        soft_update(self.net, self.tgt, tau=self.cfg.tau)

        return float(loss_sr.item() + loss_repr.item())

    def observe_and_learn(
        self,
        obs: dict,
        action: int,
        reward: float,
        next_obs: dict,
        terminated: bool,
        truncated: bool,
    ) -> float | None:
        s = self.get_state_vec(obs)
        s2 = self.get_state_vec(next_obs)
        done = bool(terminated or truncated)

        r = float(reward)
        if self.cfg.reward_clip is not None:
            c = float(self.cfg.reward_clip)
            r = float(np.clip(r, -c, c))

        self.rb.push(s, action, r, s2, done)
        self.total_steps += 1

        if self.total_steps % self.cfg.train_freq == 0:
            return self.update_from_replay()
        return None

    def run_episode(self, env, eval_mode: bool = False, max_steps: int = 10_000):
        obs, info = env.reset()
        ep_ret = 0.0
        ep_len = 0

        while True:
            s = self.get_state_vec(obs)
            a = self.sample_action(s, eval_mode=eval_mode)

            next_obs, reward, terminated, truncated, info = env.step(a)

            if not eval_mode:
                self.observe_and_learn(obs, a, reward, next_obs, terminated, truncated)

            ep_ret += float(reward)
            ep_len += 1
            obs = next_obs

            if terminated or truncated or ep_len >= max_steps:
                break

        return ep_len, ep_ret
