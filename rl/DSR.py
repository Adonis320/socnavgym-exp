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
# Configuration
# ──────────────────────────────────────────────
@dataclass
class DSRConfig:
    state_dim: int = 5          # For SocNavGym
    n_actions: int = 7          # For SocNavGym
    feat_dim: int = 64          # Feature dimension
    gamma: float = 0.99         # Discount factor
    lambda_r: float = 2.0       # Reward regression weight (was 0.5 — encoder must be reward-predictive)
    lambda_sr: float = 0.5      # SR Bellman weight
    lambda_recon: float = 0.1   # Reconstruction weight (was 0.5 — recon is auxiliary, don't let it dominate)
    lambda_var: float = 0.1     # Phi variance loss weight (collapse prevention)
    lr_rw: float = 5e-4         # Learning rate for reward + reconstruction
    lr_sr: float = 1e-4         # Learning rate for SR
    grad_clip: float = 0.5      # Gradient clipping
    target_update_freq: int = 1000  # Target network update frequency
    tau: float = 0.005          # Soft update coefficient
    train_freq: int = 4         # Training frequency
    batch_size: int = 128       # Batch size
    replay_cap: int = 100_000   # Replay buffer capacity
    start_steps: int = 10_000   # Number of steps before training starts
    epsilon_start: float = 1.0  # Starting epsilon for exploration
    epsilon_end: float = 0.05   # Ending epsilon for exploration
    epsilon_decay_steps: int = 50_000  # Steps to decay epsilon
    reward_clip: Optional[float] = 0.5  # Reward clipping
    device: Optional[str] = None

# ──────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────
class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: deque = deque(maxlen=int(cap))

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
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
# Encoder
# ──────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, state_dim: int, feat_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
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
        return self.net(s)  # (B, feat_dim)

# ──────────────────────────────────────────────
# Decoder (Reconstruction Head)
# ──────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, feat_dim: int, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi)  # (B, state_dim)

# ──────────────────────────────────────────────
# SR Head with LayerNorm
# ──────────────────────────────────────────────
class SRHead(nn.Module):
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.n_actions = int(n_actions)
        self.feat_dim  = int(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions * feat_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        psi = self.net(phi)                                     # (B, n_actions * feat_dim)
        return psi.view(-1, self.n_actions, self.feat_dim)      # (B, A, F)

# ──────────────────────────────────────────────
# DSR Network
# ──────────────────────────────────────────────
class DSRNet(nn.Module):
    def __init__(self, cfg: DSRConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = Encoder(cfg.state_dim, cfg.feat_dim)
        self.sr      = SRHead(cfg.feat_dim, cfg.n_actions)
        self.w       = nn.Linear(cfg.feat_dim, 1, bias=False)  # Reward weights
        self.decoder = Decoder(cfg.feat_dim, cfg.state_dim)

        nn.init.normal_(self.w.weight, mean=0.0, std=0.1)

    def phi(self, s: torch.Tensor) -> torch.Tensor:
        return self.encoder(s)

    def r_hat(self, phi: torch.Tensor) -> torch.Tensor:
        return self.w(phi).squeeze(-1)

    def psi(self, phi: torch.Tensor) -> torch.Tensor:
        return self.sr(phi)

    def q_all(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.phi(s)
        psi = self.psi(phi)
        q   = torch.einsum("baf,f->ba", psi, self.w.weight.squeeze())
        return q, psi, phi

    def reconstruct(self, phi: torch.Tensor) -> torch.Tensor:
        """Decode phi back to state space."""
        return self.decoder(phi)

    def phi_var_loss(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Penalise collapsed (near-constant) phi representations.
        Pushes each feature dimension to have std >= 1 across the batch.
        Returns 0 when all dims already have std >= 1.
        """
        phi_c = phi - phi.mean(dim=0, keepdim=True)     # centre per-feature
        std   = phi_c.pow(2).mean(dim=0).sqrt()         # (feat_dim,)
        return torch.relu(1.0 - std).mean()

# ──────────────────────────────────────────────
# DSR Agent
# ──────────────────────────────────────────────
class DSRAgent:
    def __init__(self, cfg: DSRConfig):
        self.cfg    = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.net = DSRNet(cfg).to(self.device)
        self.tgt = DSRNet(cfg).to(self.device)
        self.tgt.load_state_dict(self.net.state_dict())

        # FIX: Encoder is trained ONLY by the reward/recon objective.
        # The SR head is trained separately with a stop-gradient through the encoder,
        # preventing the SR Bellman loss from corrupting the feature representation.
        self.opt_rw = torch.optim.Adam(
            list(self.net.encoder.parameters()) +
            list(self.net.w.parameters())       +
            list(self.net.decoder.parameters()),
            lr=cfg.lr_rw,
        )
        # SR head only — encoder weights are NOT updated by this optimizer.
        self.opt_sr = torch.optim.Adam(
            self.net.sr.parameters(),
            lr=cfg.lr_sr,
        )

        self.rb          = Replay(cfg.replay_cap)
        self.epsilon     = cfg.epsilon_start
        self.total_steps = 0

    # ──────────────────────────────────────────
    def get_state_vec(self, obs: Any) -> np.ndarray:
        if isinstance(obs, (tuple, list)) and len(obs) >= 1 and isinstance(obs[0], dict):
            obs = obs[0]

        if not isinstance(obs, dict):
            return np.zeros((self.cfg.state_dim,), dtype=np.float32)

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

        return np.array([gx, gy, th, hx, hy], dtype=np.float32)

    # ──────────────────────────────────────────
    def sample_action(self, s: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.cfg.n_actions - 1)

        with torch.no_grad():
            s_t     = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            q, _, _ = self.net.q_all(s_t)
            return int(q.argmax().item())

    # ──────────────────────────────────────────
    def update_from_replay(self) -> Tuple[float, float, float]:
        if len(self.rb) < self.cfg.batch_size:
            return 0.0, 0.0, 0.0

        batch = self.rb.sample(self.cfg.batch_size, device=self.device)
        s, a, r, s2, d = (
            batch["s"], batch["a"], batch["r"], batch["s2"], batch["d"]
        )

        # ── Step 1: Encoder → reward regression + reconstruction + var loss ──
        self.opt_rw.zero_grad()

        phi        = self.net.phi(s)
        loss_r     = self.cfg.lambda_r     * F.mse_loss(self.net.r_hat(phi), r)
        loss_recon = self.cfg.lambda_recon * F.mse_loss(self.net.reconstruct(phi), s)
        loss_var   = self.cfg.lambda_var   * self.net.phi_var_loss(phi)  # collapse prevention

        (loss_r + loss_recon + loss_var).backward()
        nn.utils.clip_grad_norm_(
            list(self.net.encoder.parameters()) +
            list(self.net.w.parameters())       +
            list(self.net.decoder.parameters()),
            self.cfg.grad_clip,
        )
        self.opt_rw.step()

        # ── Step 2: SR Bellman (stop-gradient through encoder) ───────────────
        self.opt_sr.zero_grad()

        # Detach encoder so SR loss does not flow back into encoder weights.
        with torch.no_grad():
            phi_s_sg  = self.net.phi(s)
            phi_s2_sg = self.net.phi(s2)

        psi_all = self.net.sr(phi_s_sg)
        psi_sa  = psi_all.gather(
            1, a.view(-1, 1, 1).expand(-1, 1, psi_all.size(-1))
        ).squeeze(1)  # (B, F)

        with torch.no_grad():
            # Double-DQN: select action from online SR head
            q2_online = torch.einsum(
                "baf,f->ba",
                self.net.sr(phi_s2_sg),
                self.net.w.weight.squeeze(),
            )
            a_star = q2_online.argmax(1, keepdim=True)

            # Bootstrap psi from target network
            psi2_all = self.tgt.sr(self.tgt.phi(s2))
            psi2     = psi2_all.gather(
                1, a_star.view(-1, 1, 1).expand(-1, 1, psi2_all.size(-1))
            ).squeeze(1)

            target_psi = phi_s_sg + (1.0 - d).unsqueeze(-1) * self.cfg.gamma * psi2
            target_psi = target_psi.clamp(-10.0, 10.0)

        loss_sr = self.cfg.lambda_sr * F.mse_loss(psi_sa, target_psi)
        loss_sr.backward()
        nn.utils.clip_grad_norm_(self.net.sr.parameters(), self.cfg.grad_clip)
        self.opt_sr.step()

        # ── Step 3: Soft-update target network ───────────────────────────────
        for tgt_p, p in zip(self.tgt.parameters(), self.net.parameters()):
            tgt_p.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * tgt_p.data)

        # ── Epsilon decay ─────────────────────────────────────────────────────
        self.epsilon = max(
            self.cfg.epsilon_end,
            self.cfg.epsilon_start
            - (self.total_steps / self.cfg.epsilon_decay_steps)
            * (self.cfg.epsilon_start - self.cfg.epsilon_end),
        )

        return float(loss_sr.item()), float(loss_r.item()), float(loss_recon.item())

    # ──────────────────────────────────────────
    def observe_and_learn(
        self,
        obs:        Any,
        action:     int,
        reward:     float,
        next_obs:   Any,
        terminated: bool,
        truncated:  bool,
    ) -> Optional[Tuple[float, float, float]]:
        s    = self.get_state_vec(obs)
        s2   = self.get_state_vec(next_obs)
        done = bool(terminated or truncated)

        r = float(reward)
        if self.cfg.reward_clip is not None:
            r = float(np.clip(r, -self.cfg.reward_clip, self.cfg.reward_clip))

        self.rb.push(s, action, r, s2, done)
        self.total_steps += 1

        if len(self.rb) >= self.cfg.start_steps and self.total_steps % self.cfg.train_freq == 0:
            return self.update_from_replay()
        return None

    # ──────────────────────────────────────────
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