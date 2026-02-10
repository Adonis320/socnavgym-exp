# agents/modular_dsr_agent.py
# Modular Deep Successor Representation (MDSR)
# - No reconstruction
# - Optional encoders (identity/handcrafted or learned MLP)
# - Two modules by default: topo + social
# - Q(s,a) = sum_m  ψ_m(s,a) · w_m
# - Loss: sum_m SR Bellman loss (+ optional reward regression)

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

    def push(self, obs: Any, a: int, r: float, next_obs: Any, done: bool):
        # store raw obs dict (or already-vector), keep lightweight
        self.buf.append((obs, int(a), float(r), next_obs, bool(done)))

    def sample(self, n: int):
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


# --------------------
# Small MLP blocks
# --------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden_layers: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(int(n_hidden_layers)):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SRHead(nn.Module):
    """
    Takes module features phi_m (B,Fm) and outputs psi_m for all actions: (B,A,Fm)
    Two-FC-layer version (as in common DSR descriptions):
      Fm -> hidden -> hidden -> A*Fm
    """
    def __init__(self, feat_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.n_actions = int(n_actions)
        self.feat_dim = int(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_actions * self.feat_dim),
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
# Modular DSR Network
# --------------------
class ModularDSR(nn.Module):
    """
    Modules m in {topo, social, ...}
    Each module can have:
      - optional encoder E_m: o_vec -> phi_m
      - SR head Psi_m: phi_m -> psi_m(s, a) for all actions
      - reward weights w_m: phi_m · w_m (optional reward regression)
    """
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        module_feat_dims: Dict[str, int],
        use_learned_encoders: bool = False,
        enc_hidden: int = 128,
        sr_hidden: int = 256,
    ):
        super().__init__()
        self.n_actions = int(n_actions)
        self.obs_dim = int(obs_dim)

        self.module_names = list(module_feat_dims.keys())
        self.module_feat_dims = {k: int(v) for k, v in module_feat_dims.items()}

        # Optional encoders
        self.use_learned_encoders = bool(use_learned_encoders)
        if self.use_learned_encoders:
            self.encoders = nn.ModuleDict(
                {
                    m: MLP(self.obs_dim, hidden=enc_hidden, out_dim=self.module_feat_dims[m], n_hidden_layers=2)
                    for m in self.module_names
                }
            )
        else:
            self.encoders = None  # features are provided externally

        # SR heads per module
        self.sr_heads = nn.ModuleDict(
            {m: SRHead(self.module_feat_dims[m], n_actions=self.n_actions, hidden=sr_hidden) for m in self.module_names}
        )

        # Reward weights per module (linear in phi)
        self.w = nn.ParameterDict(
            {m: nn.Parameter(torch.empty(self.module_feat_dims[m])) for m in self.module_names}
        )
        for m in self.module_names:
            nn.init.normal_(self.w[m], mean=0.0, std=0.1)

    def encode(self, o_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dict of module features phi_m
        """
        if not self.use_learned_encoders:
            raise RuntimeError("encode() called but use_learned_encoders=False; provide phi externally.")
        return {m: self.encoders[m](o_vec) for m in self.module_names}

    def psi_all(self, phi_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        For each module m: returns psi_m(s, :) shape (B,A,Fm)
        """
        return {m: self.sr_heads[m](phi_dict[m]) for m in self.module_names}

    def q_all(self, phi_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Q(s,a) = sum_m <psi_m(s,a), w_m>
        Returns:
          q: (B,A)
          psi_dict: dict m -> (B,A,Fm)
        """
        psi_dict = self.psi_all(phi_dict)
        q = None
        for m in self.module_names:
            # (B,A,Fm) · (Fm,) -> (B,A)
            qm = psi_dict[m].matmul(self.w[m])
            q = qm if q is None else (q + qm)
        return q, psi_dict

    def r_hat(self, phi_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        r_hat(s) = sum_m <phi_m(s), w_m>
        """
        r = None
        for m in self.module_names:
            rm = phi_dict[m].matmul(self.w[m])  # (B,)
            r = rm if r is None else (r + rm)
        return r


# --------------------
# Feature extractors (handcrafted / identity)
# --------------------
def default_obs_to_vec(obs: Any, max_humans: int = 1) -> np.ndarray:
    """
    Produces a stable raw vector representation from SocNavGym-like dict obs.
    Default: [goal_x, goal_y, theta, h1x, h1y, ..., hKx, hKy]
    """
    if isinstance(obs, (tuple, list)) and len(obs) == 2 and isinstance(obs[0], dict):
        obs = obs[0]
    if not isinstance(obs, dict):
        return np.zeros((3 + 2 * max_humans,), dtype=np.float32)

    robot = np.asarray(obs.get("robot", [0.0, 0.0, 0.0]), dtype=np.float32).flatten()
    gx, gy, th = (float(robot[0]), float(robot[1]), float(robot[2])) if robot.size >= 3 else (0.0, 0.0, 0.0)

    humans = obs.get("humans", None)
    if humans is None:
        humans_xy = np.zeros((0, 2), dtype=np.float32)
    else:
        h = np.asarray(humans, dtype=np.float32)
        if h.ndim == 2 and h.shape[1] >= 2:
            humans_xy = h[:, :2]
        else:
            h = h.flatten()
            humans_xy = np.asarray([[float(h[0]), float(h[1])]], dtype=np.float32) if h.size >= 2 else np.zeros((0, 2), dtype=np.float32)

    K = int(max_humans)
    if humans_xy.shape[0] > K:
        humans_xy = humans_xy[:K]
    pad = np.zeros((K, 2), dtype=np.float32)
    if humans_xy.shape[0] > 0:
        pad[: humans_xy.shape[0], :] = humans_xy

    s = np.concatenate([np.asarray([gx, gy, th], dtype=np.float32), pad.reshape(-1)], axis=0)
    return s.astype(np.float32, copy=False)


def split_vec_into_modules(o_vec: np.ndarray, max_humans: int = 1) -> Dict[str, np.ndarray]:
    """
    Handcrafted split for two modules:
      topo: [goal_x, goal_y, theta]  -> dim 3
      social: [h1x,h1y,...]          -> dim 2*K
    """
    o = np.asarray(o_vec, dtype=np.float32).flatten()
    topo = o[:3]
    social = o[3 : 3 + 2 * int(max_humans)]
    return {"topo": topo, "social": social}


# --------------------
# Loss (no reconstruction)
# --------------------
def mdsr_loss_step(
    net: ModularDSR,
    tgt: ModularDSR,
    batch_samples: List[Tuple[Any, int, float, Any, bool]],
    device: str,
    gamma: float,
    max_humans: int,
    use_learned_encoders: bool,
    lambda_sr: Dict[str, float],
    lambda_r: float = 0.0,
):
    """
    Total loss:
      sum_m lambda_sr[m] * || psi_m(s,a) - (phi_m(s) + gamma * psi_m(s',a*)) ||^2
      + lambda_r * (r - r_hat(s))^2   [optional]

    Notes:
      - a* selected using target Q for stability.
      - Terminal masking applied: (1-done).
    """
    obs, a, r, next_obs, done = zip(*batch_samples)
    A = torch.as_tensor(a, dtype=torch.long, device=device)          # (B,)
    R = torch.as_tensor(r, dtype=torch.float32, device=device)       # (B,)
    D = torch.as_tensor(done, dtype=torch.float32, device=device)    # (B,)

    # Build raw vectors
    o = np.stack([default_obs_to_vec(x, max_humans=max_humans) for x in obs]).astype(np.float32)
    o2 = np.stack([default_obs_to_vec(x, max_humans=max_humans) for x in next_obs]).astype(np.float32)

    O = torch.as_tensor(o, dtype=torch.float32, device=device)   # (B, obs_dim)
    O2 = torch.as_tensor(o2, dtype=torch.float32, device=device) # (B, obs_dim)

    # Features per module
    if use_learned_encoders:
        phi = net.encode(O)
        with torch.no_grad():
            phi2 = tgt.encode(O2)
    else:
        # handcrafted per-module features (no gradient path)
        phi_np = [split_vec_into_modules(x, max_humans=max_humans) for x in o]
        phi2_np = [split_vec_into_modules(x, max_humans=max_humans) for x in o2]

        phi = {m: torch.as_tensor(np.stack([d[m] for d in phi_np]), dtype=torch.float32, device=device) for m in net.module_names}
        phi2 = {m: torch.as_tensor(np.stack([d[m] for d in phi2_np]), dtype=torch.float32, device=device) for m in net.module_names}

    # Online Q for current state
    q, psi_dict = net.q_all(phi)  # q: (B,A)

    # Choose a* using TARGET Q at next state for stability
    with torch.no_grad():
        q2_tgt, _ = tgt.q_all(phi2)         # (B,A)
        a_star = q2_tgt.argmax(1, keepdim=True)  # (B,1)

    # SR losses per module
    total_sr = 0.0
    per_module_sr: Dict[str, torch.Tensor] = {}

    for m in net.module_names:
        psi_all_m = psi_dict[m]  # (B,A,Fm)
        psi_sa_m = psi_all_m.gather(1, A.view(-1, 1, 1).expand(-1, 1, psi_all_m.size(-1))).squeeze(1)  # (B,Fm)

        with torch.no_grad():
            psi2_all_m = tgt.sr_heads[m](phi2[m])  # (B,A,Fm)
            psi2_star_m = psi2_all_m.gather(
                1, a_star.view(-1, 1, 1).expand(-1, 1, psi2_all_m.size(-1))
            ).squeeze(1)  # (B,Fm)
            target_psi_m = phi[m] + (1.0 - D).unsqueeze(-1) * gamma * psi2_star_m

        loss_m = F.mse_loss(psi_sa_m, target_psi_m)
        per_module_sr[m] = loss_m
        total_sr = total_sr + float(lambda_sr.get(m, 1.0)) * loss_m

    # Optional reward regression (often stabilizes w)
    if lambda_r > 0.0:
        r_hat = net.r_hat(phi)  # (B,)
        loss_r = F.mse_loss(r_hat, R)
    else:
        loss_r = torch.zeros((), device=device)

    total = total_sr + float(lambda_r) * loss_r
    return total, per_module_sr, loss_r


def soft_update(net: nn.Module, tgt: nn.Module, tau: float = 0.005):
    with torch.no_grad():
        for p, pt in zip(net.parameters(), tgt.parameters()):
            pt.data.mul_(1.0 - tau).add_(tau * p.data)


# --------------------
# Agent
# --------------------
@dataclass
class MDSRConfig:
    action_size: int = 7
    max_humans: int = 1

    # raw vector obs dimension used by default_obs_to_vec()
    obs_dim: int = 5  # = 3 + 2*max_humans

    # modules + feature dims (handcrafted uses same dims; learned encoders map obs_dim -> these)
    feat_topo: int = 3
    feat_social: int = 2  # per human * K => you likely want 2*max_humans here if handcrafted

    # learned encoders (optional)
    use_learned_encoders: bool = False
    enc_hidden: int = 128

    # SR heads
    sr_hidden: int = 256

    # optimization
    buffer_size: int = 100_000
    batch_size: int = 128
    start_steps: int = 5_000
    train_freq: int = 1
    gamma: float = 0.99
    learning_rate: float = 3e-4
    epsilon: float = 0.3
    tau: float = 0.005
    grad_clip: float = 5.0

    # losses
    lambda_sr_topo: float = 1.0
    lambda_sr_social: float = 1.0
    lambda_r: float = 1.0  # set 0.0 to disable reward regression
    reward_clip: float | None = None

    device: str | None = None


class MDSRAgent:
    """
    Drop-in agent:
      - sample_action(obs, eval_mode=False) -> int
      - observe_and_learn(obs, a, r, next_obs, terminated, truncated) -> float|None
    """
    def __init__(self, cfg: MDSRConfig):
        self.cfg = cfg
        self.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = int(cfg.action_size)
        self.max_humans = int(cfg.max_humans)

        # module dims
        # If handcrafted split, topo=3, social=2*K. If learned, you can choose anything.
        if cfg.use_learned_encoders:
            module_feat_dims = {"topo": int(cfg.feat_topo), "social": int(cfg.feat_social)}
        else:
            module_feat_dims = {"topo": 3, "social": 2 * self.max_humans}

        self.net = ModularDSR(
            n_actions=self.n_actions,
            obs_dim=int(cfg.obs_dim),
            module_feat_dims=module_feat_dims,
            use_learned_encoders=bool(cfg.use_learned_encoders),
            enc_hidden=int(cfg.enc_hidden),
            sr_hidden=int(cfg.sr_hidden),
        ).to(self.device)

        self.tgt = ModularDSR(
            n_actions=self.n_actions,
            obs_dim=int(cfg.obs_dim),
            module_feat_dims=module_feat_dims,
            use_learned_encoders=bool(cfg.use_learned_encoders),
            enc_hidden=int(cfg.enc_hidden),
            sr_hidden=int(cfg.sr_hidden),
        ).to(self.device)

        self.tgt.load_state_dict(self.net.state_dict())

        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(cfg.learning_rate))
        self.rb = Replay(cfg.buffer_size)
        self.total_steps = 0

        self.lambda_sr = {
            "topo": float(cfg.lambda_sr_topo),
            "social": float(cfg.lambda_sr_social),
        }

    def _obs_to_raw_vec(self, obs: Any) -> np.ndarray:
        return default_obs_to_vec(obs, max_humans=self.max_humans)

    @torch.no_grad()
    def sample_action(self, obs: Any, eval_mode: bool = False) -> int:
        eps = 0.0 if eval_mode else float(self.cfg.epsilon)
        if random.random() < eps:
            return random.randrange(self.n_actions)

        o = self._obs_to_raw_vec(obs)
        O = torch.as_tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.cfg.use_learned_encoders:
            phi = self.net.encode(O)
        else:
            parts = split_vec_into_modules(o, max_humans=self.max_humans)
            phi = {m: torch.as_tensor(parts[m], dtype=torch.float32, device=self.device).unsqueeze(0) for m in self.net.module_names}

        q, _ = self.net.q_all(phi)
        return int(q.argmax(dim=1).item())

    def update_from_replay(self) -> float | None:
        if len(self.rb) < max(int(self.cfg.batch_size), int(self.cfg.start_steps)):
            return None

        batch_samples = self.rb.sample(int(self.cfg.batch_size))

        self.opt.zero_grad(set_to_none=True)
        total, per_m, loss_r = mdsr_loss_step(
            self.net,
            self.tgt,
            batch_samples=batch_samples,
            device=self.device,
            gamma=float(self.cfg.gamma),
            max_humans=self.max_humans,
            use_learned_encoders=bool(self.cfg.use_learned_encoders),
            lambda_sr=self.lambda_sr,
            lambda_r=float(self.cfg.lambda_r),
        )
        total.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), float(self.cfg.grad_clip))
        self.opt.step()
        soft_update(self.net, self.tgt, tau=float(self.cfg.tau))

        return float(total.item())

    def observe_and_learn(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        terminated: bool,
        truncated: bool,
    ) -> float | None:
        done = bool(terminated or truncated)

        r = float(reward)
        if self.cfg.reward_clip is not None:
            c = float(self.cfg.reward_clip)
            r = float(np.clip(r, -c, c))

        self.rb.push(obs, action, r, next_obs, done)
        self.total_steps += 1

        if self.total_steps % int(self.cfg.train_freq) == 0:
            return self.update_from_replay()
        return None

    def run_episode(self, env, eval_mode: bool = False, max_steps: int = 10_000):
        obs, info = env.reset()
        ep_ret, ep_len = 0.0, 0

        while True:
            a = self.sample_action(obs, eval_mode=eval_mode)
            next_obs, reward, terminated, truncated, info = env.step(a)

            if not eval_mode:
                self.observe_and_learn(obs, a, reward, next_obs, terminated, truncated)

            ep_ret += float(reward)
            ep_len += 1
            obs = next_obs

            if terminated or truncated or ep_len >= max_steps:
                break

        return ep_len, ep_ret
