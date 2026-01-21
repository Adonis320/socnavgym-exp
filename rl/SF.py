import numpy as np
from collections import defaultdict

class SF:
    """
    Action-conditional Successor Features with linear rewards.
    Q(s,a) = psi(s,a)^T w,  r(s) ≈ phi(s)^T w
    phi(s) is a 1-hot over discretized states (dynamic dictionary).
    """

    def __init__(
        self,
        action_size: int,
        epsilon: float = 0.05,
        gamma: float = 0.99,
        sf_lr: float = 0.01,        # learning rate for successor features
        w_lr: float = 0.01,         # learning rate for reward weights
        xy_bins: int = 20,
        xy_max_abs: float = 10.0,
        xy_edges=None,
        seed: int | None = None,
    ):
        self.action_size = int(action_size)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.sf_lr = float(sf_lr)
        self.w_lr = float(w_lr)

        # discretization
        self.xy_bins = max(2, int(xy_bins))
        self.xy_max_abs = float(xy_max_abs)
        self.xy_edges = None if xy_edges is None else _prep_edges(xy_edges)

        # feature map: dynamic 1-hot over seen states
        self.state_index = {}     # state_key -> feature index
        self.d = 0               # feature dimension
        self.w = np.zeros(0, dtype=np.float64)

        # psi storage: dict[(s,a)] -> np.ndarray shape (d,)
        self.psi = defaultdict(lambda: defaultdict(self._zeros_feat))

        if seed is not None:
            np.random.seed(int(seed))

    # ---------- core RL ----------

    def sample_action(self, state_key, eval: bool = False) -> int:
        if not eval and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        phi_s = self._phi(state_key)  # ensures dimension
        Q = np.zeros(self.action_size, dtype=np.float64)
        for a in range(self.action_size):
            psi_sa = self._psi(state_key, a)
            Q[a] = float(np.dot(psi_sa, self.w))
        maxq = Q.max()
        idx = np.flatnonzero(np.isclose(Q, maxq))
        return int(np.random.choice(idx))

    def update_sf(self, s, a, sp, done: bool):
        """
        TD(0) for successor features:
        target = phi(sp) + gamma * psi(sp, a')       if not done
               = phi(sp)                              if done
        """
        phi_sp = self._phi(sp)
        if not done:
            ap = self.sample_action(sp)  # on-policy bootstrap
            psi_sp_ap = self._psi(sp, ap)
            target = phi_sp + self.gamma * psi_sp_ap
        else:
            target = phi_sp

        psi_sa = self._psi(s, a)
        # handle possible dimension growth
        if psi_sa.shape[0] != self.d:
            psi_sa = self._pad_to_d(psi_sa)
            self.psi[s][a] = psi_sa

        self.psi[s][a] = psi_sa + self.sf_lr * (target - psi_sa)

    def update_reward_weights(self, s, r: float):
        """
        SGD for linear reward model: r ≈ w^T phi(s)
        w <- w + w_lr * (r - w^T phi(s)) * phi(s)
        """
        phi_s = self._phi(s)
        err = float(r) - float(np.dot(self.w, phi_s))
        self.w = self.w + self.w_lr * err * phi_s

    def act(self, env, obs, eval: bool = False):
        """
        Run one episode with online learning.
        Returns (episode_length, episode_return).
        """
        ep_ret, ep_len = 0.0, 0
        s = self.get_state_key(obs)

        while True:
            a = self.sample_action(s, eval=eval)
            next_obs, r, done, truncated, info = env.step(a)
            sp = self.get_state_key(next_obs)

            ep_ret += float(r)
            ep_len += 1

            self.update_sf(s, a, sp, done or truncated)
            self.update_reward_weights(s, float(r))

            s = sp
            if done or truncated:
                break

        return ep_len, ep_ret

    # ---------- state encoding (goal only) ----------

    def get_state_key(self, obs):
        state = self._state_tuple(obs)
        return tuple(int(x) for x in state)

    def _state_tuple(self, obs):
        """
        Tuple = (goal_bin_x, goal_bin_y).
        If obs is not a dict, fall back to stable bytes key.
        """
        # unwrap (obs, info)
        if isinstance(obs, (list, tuple)) and obs and isinstance(obs[0], dict):
            obs = obs[0]

        if not isinstance(obs, dict):
            arr = np.asarray(obs, dtype=np.float32).flatten()
            return ("raw", arr.tobytes())

        # robot: [one_hot(D), goal_dx, goal_dy, robot_radius]
        r = np.asarray(obs.get("robot", []), dtype=np.float32).flatten()
        D = max(0, r.size - 3)
        dx = float(r[D]) if r.size > D else 0.0
        dy = float(r[D + 1]) if r.size > D + 1 else 0.0

        gx = _bin_scalar(dx, self.xy_edges, self.xy_bins, self.xy_max_abs)
        gy = _bin_scalar(dy, self.xy_edges, self.xy_bins, self.xy_max_abs)
        return (gx, gy)

    # ---------- features / storage ----------

    def _phi(self, state_key):
        """
        1-hot feature for state_key. Grows dimension on first sight.
        """
        if state_key not in self.state_index:
            self.state_index[state_key] = self.d
            self.d += 1
            # grow w
            self.w = self._pad_to_d(self.w)
        feat = np.zeros(self.d, dtype=np.float64)
        feat[self.state_index[state_key]] = 1.0
        return feat

    def _psi(self, s, a):
        """
        Returns psi(s,a) with current dimension; pads if needed.
        """
        v = self.psi[s][a]
        if v.shape[0] != self.d:
            v = self._pad_to_d(v)
            self.psi[s][a] = v
        return v

    def _zeros_feat(self):
        return np.zeros(self.d, dtype=np.float64)

    def _pad_to_d(self, arr):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape[0] == self.d:
            return arr
        if arr.shape[0] < self.d:
            pad = np.zeros(self.d - arr.shape[0], dtype=np.float64)
            return np.concatenate([arr, pad], axis=0)
        # if larger (should not happen), truncate
        return arr[: self.d]

# ---------- helpers ----------

def _prep_edges(edges):
    arr = np.asarray(edges, dtype=np.float32).flatten()
    arr = arr[np.isfinite(arr)]
    arr = np.unique(arr)
    return arr

def _bin_scalar(x, edges, n_bins, max_abs):
    if edges is None:
        thresholds = np.linspace(-max_abs, max_abs, num=max(n_bins - 1, 1), dtype=np.float32)
        xv = float(np.clip(x, -max_abs, max_abs))
    else:
        thresholds = edges
        xv = float(x)
    for i, e in enumerate(thresholds):
        if xv < e:
            return i
    return len(thresholds)
