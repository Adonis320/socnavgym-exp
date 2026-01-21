import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import deque

# ======== CUDA setup ========
def get_device(force_cuda=True):
    if force_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Install CUDA or set force_cuda=False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =============== Q-network (raw one-hot features + 1 hidden layer) ===============
class QNetRaw(nn.Module):
    """
    MLP with 1 hidden layer on raw flat one-hot features.
    Input:  φ(s) ∈ R^{3*H*W}  (concat [goal, walls, agent])
    Output: Q(s, ·) ∈ R^{n_actions}
    """
    def __init__(self, obs_dim, n_actions, hidden=256):
        super().__init__()
        #self.net = nn.Sequential(
        #    nn.Linear(obs_dim, 64), nn.ReLU(inplace=True),
        #    nn.Linear(64, 128), nn.ReLU(inplace=True),
        #    nn.Linear(128, 256), nn.ReLU(inplace=True),
        #    nn.Linear(256, n_actions)
        #)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, n_actions)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, 3*H*W)
        return self.net(x)

# =============== Replay ===============
class Replay:
    def __init__(self, cap=100000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.buf.append((np.array(s, dtype=np.float32), int(a), float(r),
                         np.array(s2, dtype=np.float32), bool(d)))

    def sample(self, n, device):
        s,a,r,s2,d = zip(*random.sample(self.buf, n))
        S  = torch.as_tensor(np.stack(s),  dtype=torch.float32, device=device)  # (B,3F)
        S2 = torch.as_tensor(np.stack(s2), dtype=torch.float32, device=device)  # (B,3F)
        A  = torch.as_tensor(a, dtype=torch.long,    device=device)             # (B,)
        R  = torch.as_tensor(r, dtype=torch.float32, device=device)             # (B,)
        D  = torch.as_tensor(d, dtype=torch.float32, device=device)             # (B,)
        return S, A, R, S2, D

    def __len__(self): return len(self.buf)

# =============== DQN (raw features only, 1 hidden layer) ===============
class DQN:
    def __init__(self, env, features=219, action_size=7,
                 gamma=0.99, learning_rate=1e-3, tau=0.005, epsilon=0.05,
                 hidden=256,
                 device=None, force_cuda=True):
        self.F = features
        self.n_actions = action_size
        self.gamma, self.tau = gamma, tau
        self.epsilon = epsilon
        self.env = env

        # device
        self.device = device if device is not None else get_device(force_cuda=force_cuda)

        # nets
        obs_dim = self.F
        self.q   = QNetRaw(obs_dim, action_size, hidden=hidden).to(self.device)
        self.tgt = QNetRaw(obs_dim, action_size, hidden=hidden).to(self.device)
        self.tgt.load_state_dict(self.q.state_dict())

        self.opt = torch.optim.Adam(self.q.parameters(), lr=learning_rate)
        self.rb = Replay(50000)

    # ----- raw one-hot features (flat) -----
    def get_features(self, obs):
        """
        Convert full observation dict into a single flat vector for DQN.
        """

        # ---------------------------
        # Unwrap Gymnasium output
        # ---------------------------
        if isinstance(obs, (tuple, list)) and len(obs) == 2:
            if isinstance(obs[0], dict):
                obs = obs[0]

        if not isinstance(obs, dict):
            return np.zeros(1, dtype=np.float32)

        # ---------------------------
        # 1) Robot features
        # ---------------------------
        robot_raw = np.asarray(obs.get("robot", []), dtype=np.float32).flatten()

        # last 3 = [dx, dy, radius]
        if robot_raw.size >= 3:
            robot_feat = robot_raw[-3:]
        else:
            robot_feat = np.zeros(3, dtype=np.float32)

        # ---------------------------
        # 2) Humans
        # ---------------------------
        humans_raw = np.asarray(obs.get("humans", []), dtype=np.float32).flatten()
        humans_feat = []

        if humans_raw.size > 0:
            block = 14
            one_hot_len = 6

            if humans_raw.size % block == 0:
                for i in range(0, humans_raw.size, block):
                    cont = humans_raw[i + one_hot_len : i + block]
                    humans_feat.extend(cont.tolist())
            else:
                if humans_raw.size >= 8:
                    humans_feat.append(float(humans_raw[6]))
                    humans_feat.append(float(humans_raw[7]))

        humans_feat = np.asarray(humans_feat, dtype=np.float32)

        # ---------------------------
        # 3) Walls
        # ---------------------------
        walls_raw = np.asarray(obs.get("walls", []), dtype=np.float32).flatten()
        walls_feat = []

        if walls_raw.size > 0:
            # detect repeating block length
            candidates = [12, 14, 16]
            block = None
            for bc in candidates:
                if walls_raw.size % bc == 0:
                    block = bc
                    break

            if block is not None:
                one_hot_len = 6
                useful_len = min(5, block - one_hot_len)
                for i in range(0, walls_raw.size, block):
                    useful = walls_raw[i + one_hot_len : i + one_hot_len + useful_len]
                    walls_feat.extend(useful.tolist())
            else:
                if walls_raw.size >= 8:
                    walls_feat.append(float(walls_raw[6]))
                    walls_feat.append(float(walls_raw[7]))

        walls_feat = np.asarray(walls_feat, dtype=np.float32)

        # ---------------------------
        # Final flat feature vector
        # ---------------------------
        return np.concatenate([robot_feat, humans_feat, walls_feat]).astype(np.float32)

    @torch.no_grad()
    def sample_action(self, s_flat):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        x = torch.as_tensor(s_flat, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,3F)
        q = self.q(x)
        return int(q.argmax(1).item())

    def update(self, batch):
        s, a, r, s2, d = batch  # tensors on self.device

        # Q(s,a)
        q_sa = self.q(s).gather(1, a.view(-1,1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            a_star = self.q(s2).argmax(1, keepdim=True)               # online argmax
            q_next = self.tgt(s2).gather(1, a_star).squeeze(1)        # target eval
            target = r + (1.0 - d) * self.gamma * q_next

        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        # Soft update target
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.tgt.parameters()):
                pt.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return float(loss.item())

    def act(self, env, obs, eval=False, reverse=False, upd_social=None):

        state = self.get_features(obs)  # (3F,)

        total_reward = 0
        episode_length = 0

        while True:

            action = self.sample_action(state)
            next_state, reward, done, truncated, info = env.step(action)

            phi_next = self.get_features(next_state)
            total_reward += reward
            episode_length += 1
            # collision info (if provided)
            
            # push transition
            self.rb.push(state, action, reward, phi_next, done or truncated)

            # train
            if len(self.rb) > 1000:
                batch = self.rb.sample(64, device=self.device)
                self.update(batch)

            if done or truncated:
                break

            state = phi_next

        return episode_length, total_reward
