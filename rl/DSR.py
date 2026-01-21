import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import torch.nn.functional as F

# --------------------
# Replay Buffer
# --------------------
class Replay:
    def __init__(self, cap=100_000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.buf.append((np.array(s), int(a), float(r), np.array(s2), bool(d)))

    def sample(self, n, device="cuda"):
        s, a, r, s2, d = zip(*random.sample(self.buf, n))
        S  = torch.tensor(np.stack(s),  dtype=torch.float32, device=device)
        S2 = torch.tensor(np.stack(s2), dtype=torch.float32, device=device)
        A  = torch.tensor(a, dtype=torch.long,  device=device)
        R  = torch.tensor(r, dtype=torch.float32, device=device)
        D  = torch.tensor(d, dtype=torch.float32, device=device)
        return {"s": S, "a": A, "r": R, "s2": S2, "d": D}

    def __len__(self):
        return len(self.buf)


# --------------------
# SR Head: ψ(s,a)
# --------------------
class SRHead(nn.Module):
    def __init__(self, feat_dim=2, n_actions=7, hidden=256):
        super().__init__()
        self.n_actions = n_actions
        self.feat_dim = feat_dim

        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 128),      nn.ReLU(inplace=True),
            nn.Linear(128, 256),     nn.ReLU(inplace=True),
            nn.Linear(256, n_actions * feat_dim)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, phi):  # phi: (B, F)
        out = self.net(phi)  # (B, A*F)
        return out.view(-1, self.n_actions, self.feat_dim)  # (B, A, F)


# --------------------
# DSR model
# --------------------
class DSR(nn.Module):
    def __init__(self, n_actions=7, feat_dim=2, hidden=256, reward_hidden=64):
        super().__init__()

        self.n_actions = n_actions
        self.feat_dim = feat_dim

        # Successor features head ψ(s,a)
        self.sr = SRHead(feat_dim, n_actions, hidden)

        # Q-vector: Q(s,a) = ψ(s,a) · q_vec
        self.q_vec = nn.Parameter(torch.zeros(feat_dim))

        # Non-linear reward head: r_hat(s) = MLP(φ(s))
        self.r_head = nn.Sequential(
            nn.Linear(feat_dim, reward_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(reward_hidden, 1)
        )

        # Init
        nn.init.normal_(self.q_vec, mean=0.0, std=0.1)
        for m in self.r_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # Q(s, a) = ψ(s,a) · q_vec
    def q_all(self, phi):
        # phi: (B, F)
        psi = self.sr(phi)               # (B, A, F)
        q   = psi.matmul(self.q_vec)     # (B, A)
        return q, psi, phi

    # r̂(s)
    def r_hat(self, phi):                # phi: (B, F)
        return self.r_head(phi).squeeze(-1)


# --------------------
# Loss and utils
# --------------------
def dsr_loss_step(net, tgt, batch, gamma=0.99, q_coef=1.0):
    """
    Total loss = SR Bellman error + immediate reward regression + q_coef * Q-learning TD loss.
    Shapes:
      s,s2: (B,F)   a: (B,)   r: (B,)   d: (B,)
    """
    phi  = batch["s"]          # (B,F)
    phi2 = batch["s2"]         # (B,F)
    a    = batch["a"].long()   # (B,)
    r    = batch["r"]          # (B,)
    d    = batch["d"]          # (B,)

    # ---- Forward on current state (online net) ----
    q, psi, _ = net.q_all(phi)                       # q: (B,A), psi: (B,A,F)

    # ψ(s,a)
    psi_sa = psi.gather(
        1, a.view(-1, 1, 1).expand(-1, 1, psi.size(-1))
    ).squeeze(1)                                     # (B,F)

    # ---- SR target (use target ψ, online a* for Double-style) ----
    with torch.no_grad():
        q_next_online, _, _ = net.q_all(phi2)        # (B,A)
        a_star = q_next_online.argmax(1, keepdim=True)  # (B,1)

        psi_next_all = tgt.sr(phi2)                  # (B,A,F)
        psi_next = psi_next_all.gather(
            1, a_star.view(-1, 1, 1).expand(-1, 1, psi_next_all.size(-1))
        ).squeeze(1)                                 # (B,F)

        target_psi = phi + (1.0 - d).unsqueeze(-1) * gamma * psi_next  # (B,F)

    loss_sr = F.smooth_l1_loss(psi_sa, target_psi)

    # ---- Reward regression (immediate): r_hat = non-linear MLP(φ) ----
    r_hat  = net.r_hat(phi)                          # (B,)
    loss_r = F.smooth_l1_loss(r_hat, r)

    # ---- Q-learning TD loss (uses target ψ and target q_vec) ----
    with torch.no_grad():
        psi_next_tgt = tgt.sr(phi2)                  # (B,A,F)
        q_next_tgt   = psi_next_tgt.matmul(tgt.q_vec)  # (B,A)
        y_q          = r + gamma * (1.0 - d) * q_next_tgt.max(dim=1).values  # (B,)

    q_sa   = q.gather(1, a.view(-1, 1)).squeeze(1)   # (B,)
    loss_q = F.smooth_l1_loss(q_sa, y_q)

    return loss_sr + loss_r + q_coef * loss_q


def soft_update(net, tgt, tau=0.005):
    with torch.no_grad():
        for p, pt in zip(net.parameters(), tgt.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)


# --------------------
# DSRAgent
# --------------------
class DSRAgent:
    def __init__(self,
                 action_size=7,
                 buffer_size=50000,
                 batch_size=64,
                 start_steps=1_000,
                 train_freq=1,
                 gamma=0.99,
                 learning_rate=6e-3,
                 epsilon=0.05,
                 lambda_rec=1.0,
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Feature dimension (here: 2 values from robot obs)
        self.F = 2

        # Networks
        self.net = DSR(action_size, feat_dim=self.F).to(self.device)
        self.tgt = DSR(action_size, feat_dim=self.F).to(self.device)
        self.tgt.load_state_dict(self.net.state_dict())

        self.opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.rb  = Replay(buffer_size)

        self.n_actions   = action_size
        self.batch_size  = batch_size
        self.start_steps = start_steps
        self.train_freq  = train_freq
        self.gamma       = gamma
        self.epsilon     = epsilon
        self.lambda_rec  = lambda_rec
        self.total_steps = 0

    def get_features(self, obs):
        """
        Extract a 2D feature vector from obs.
        Here: last two values of obs["robot"] (e.g., dx, dy or x, y).
        """
        if isinstance(obs, (list, tuple)) and len(obs) > 0 and isinstance(obs[0], dict):
            obs = obs[0]
        robot = obs.get("robot", None)
        x = robot[-2:]  # (2,)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def sample_action(self, state_tensor):
        state_tensor = state_tensor[None, :]  # (1,F)
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        q, _, _ = self.net.q_all(state_tensor)
        return int(q.argmax(1).item())

    def update(self, batch):
        self.opt.zero_grad(set_to_none=True)
        loss = dsr_loss_step(self.net, self.tgt, batch, gamma=self.gamma)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        soft_update(self.net, self.tgt, tau=0.005)
        return loss.item()

    def act(self, env, obs, eval_mode=False, max_steps=10_000):
        """
        Run one episode with online DSR training.
        """
        state = self.get_features(obs)
        done = False
        total_reward = 0.0
        episode_length = 0

        while True:
            if eval_mode:
                eps_backup = self.epsilon
                self.epsilon = 0.0
            action = self.sample_action(state)
            if eval_mode:
                self.epsilon = eps_backup

            next_obs, reward, done, truncated, info = env.step(action)
            next_state = self.get_features(next_obs)

            episode_length += 1
            total_reward += reward

            # Store transition
            self.rb.push(
                state.cpu().numpy(),
                action,
                np.clip(reward, -1, 1),
                next_state.cpu().numpy(),
                done or truncated
            )
            self.total_steps += 1

            # Train
            if len(self.rb) > 1000:
                batch = self.rb.sample(self.batch_size, device=self.device)
                self.update(batch)

            if done or truncated or episode_length >= max_steps:
                break

            state = next_state

        return episode_length, total_reward
