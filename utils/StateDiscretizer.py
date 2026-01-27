import numpy as np
from utils.StateEncoder import StateEncoder


class StateDiscretizer(StateEncoder):
    """
    Handles discretization / binning of continuous state info:
    - Robot (goal_dx, goal_dy) from obs["robot"]
    - Closest human (x, y) from obs["humans"]
    - Robot yaw angle theta from env.robot.orientation
    """

    def __init__(
        self,
        xy_bins=30,
        xy_max_abs=10.0,
        xy_edges=None,
        human_xy_bins=30,
        human_xy_max_abs=10.0,
        human_xy_edges=None,
        theta_bins=8,
        env=None,
    ):
        # Goal / robot position discretization
        self.xy_bins = int(xy_bins) if int(xy_bins) >= 2 else 2
        self.xy_max_abs = float(xy_max_abs)

        if xy_edges is not None:
            edges = np.asarray(xy_edges, dtype=np.float32).flatten()
            edges = edges[np.isfinite(edges)]
            edges = np.unique(edges)  # sort & dedupe
            self.xy_edges = edges
        else:
            self.xy_edges = None

        # Humans discretization config
        self.human_xy_bins = int(human_xy_bins) if int(human_xy_bins) >= 2 else 2
        self.human_xy_max_abs = float(human_xy_max_abs)

        if human_xy_edges is not None:
            hedges = np.asarray(human_xy_edges, dtype=np.float32).flatten()
            hedges = hedges[np.isfinite(hedges)]
            hedges = np.unique(hedges)
            self.human_xy_edges = hedges
        else:
            self.human_xy_edges = None

        # Theta discretization
        self.theta_bins = int(theta_bins) if int(theta_bins) >= 2 else 2
        # length B-1 for B bins
        self.theta_edges = np.linspace(-np.pi, np.pi, num=self.theta_bins - 1, dtype=np.float32)

        self.env = env

    @staticmethod
    def _bin_scalar(x, edges: np.ndarray) -> int:
        """
        Bin scalar x using ascending edges (open on right).
        edges: array of boundaries, length B-1 for B bins
        Returns index in [0, B-1].
        """
        for i, e in enumerate(edges):
            if x < e:
                return i
        return len(edges)

    def _discretize_robot_xy(self, obs) -> tuple[int, int]:
        """
        Discretize robot goal-relative position (goal_dx, goal_dy) using xy_*.
        Expects obs["robot"] with format [one_hot(D), goal_dx, goal_dy, robot_radius].
        Returns (bx, by). If missing, returns (-1, -1).
        """
        robot = obs.get("robot", None)
        if robot is None:
            return -1, -1

        r = np.asarray(robot, dtype=np.float32).flatten()
        # Robot obs format: [one_hot(D), goal_dx, goal_dy, robot_radius] with total length D+3
        D = max(0, r.size - 3)
        dx = float(r[D]) if r.size > D else 0.0
        dy = float(r[D + 1]) if r.size > D + 1 else 0.0

        edges = (
            self.xy_edges
            if self.xy_edges is not None
            else np.linspace(
                -self.xy_max_abs,
                self.xy_max_abs,
                num=max(self.xy_bins - 1, 1),
                dtype=np.float32,
            )
        )

        bx = self._bin_scalar(np.clip(dx, -self.xy_max_abs, self.xy_max_abs), edges)
        by = self._bin_scalar(np.clip(dy, -self.xy_max_abs, self.xy_max_abs), edges)
        return int(bx), int(by)

    def _discretize_theta(self, theta: float) -> int:
        """
        Discretize robot yaw angle θ into [0, theta_bins-1].
        θ is expected in radians.
        """
        if theta is None:
            return -1

        # Wrap angle to [-π, π]
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        for i, e in enumerate(self.theta_edges):
            if theta < e:
                return int(i)
        return int(len(self.theta_edges))

    def _decode_closest_human_xy(self, humans_array: np.ndarray) -> tuple[float, float] | None:
        """
        Decode humans array and return (hx, hy) for the closest human to the robot (0,0),
        based on min hx^2 + hy^2.

        Supports the same assumptions as your original code:
        - default one_hot_len=6, block=14
        - inference attempt if size mismatch
        - minimal fallback: assume single human with (x,y) at indices (6,7)

        Returns None if cannot decode or no humans.
        """
        h = np.asarray(humans_array, dtype=np.float32).flatten()
        if h.size == 0:
            return None

        one_hot_len = 6
        block = 14

        # Try to infer encoding if size doesn't match
        if h.size % block != 0:
            inferred = False
            for k in range(3, 16):
                bs = k + 8
                if bs <= 0 or h.size % bs != 0:
                    continue
                oh = h[:k]
                # crude one-hot check
                if np.all((oh == 0) | (oh == 1)) and np.isclose(np.sum(oh), 1.0):
                    one_hot_len = k
                    block = bs
                    inferred = True
                    break

            if not inferred:
                # Minimal fallback: assume single (x, y) at indices (6,7)
                if h.size >= 8:
                    return float(h[6]), float(h[7])
                return None

        # Normal case: h.size is multiple of block
        if h.size % block != 0:
            return None

        best_d2 = float("inf")
        best_xy = None

        for i in range(0, h.size, block):
            base = i + one_hot_len  # x at base, y at base + 1
            if base + 1 >= h.size:
                continue
            hx = float(h[base])
            hy = float(h[base + 1])
            d2 = hx * hx + hy * hy
            if d2 < best_d2:
                best_d2 = d2
                best_xy = (hx, hy)

        return best_xy

    def _discretize_closest_human_xy(self, humans_array: np.ndarray) -> tuple[int, int]:
        """
        Return (hx_bin, hy_bin) for the closest human, or (-1, -1) if none.
        """
        xy = self._decode_closest_human_xy(humans_array)
        if xy is None:
            return -1, -1

        hx, hy = xy

        hedges = (
            self.human_xy_edges
            if self.human_xy_edges is not None
            else np.linspace(
                -self.human_xy_max_abs,
                self.human_xy_max_abs,
                num=max(self.human_xy_bins - 1, 1),
                dtype=np.float32,
            )
        )

        hx_bin = self._bin_scalar(np.clip(hx, -self.human_xy_max_abs, self.human_xy_max_abs), hedges)
        hy_bin = self._bin_scalar(np.clip(hy, -self.human_xy_max_abs, self.human_xy_max_abs), hedges)
        return int(hx_bin), int(hy_bin)

    def encode(self, obs) -> tuple:
        """
        Public entry point:
        - obs: environment observation (dict with "humans" and "robot" keys)
        Returns a flat, hashable state tuple of integers:
            (gx_bin, gy_bin, theta_bin, closest_hx_bin, closest_hy_bin)
        """

        # Normalize input: obs may be (obs, info) or other wrappers
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            # Gymnasium: (obs, info)
            if len(obs) == 2 and isinstance(obs[0], dict):
                obs = obs[0]
            # Gym: (obs, reward, done, info, ...) variants
            elif isinstance(obs[0], dict):
                obs = obs[0]

        if not isinstance(obs, dict):
            return (-1, -1, -1, -1, -1)

        # 1) Robot bins (goal_dx, goal_dy)
        gx, gy = self._discretize_robot_xy(obs)

        # 2) Closest human bins only
        hx_bin, hy_bin = -1, -1
        humans = obs.get("humans", None)
        if humans is not None:
            hx_bin, hy_bin = self._discretize_closest_human_xy(np.asarray(humans, dtype=np.float32))

        # 3) Robot direction (theta)
        theta_raw = None
        if self.env is not None:
            # keep your original access pattern
            try:
                theta_raw = self.env.env.env.env.robot.orientation
            except Exception:
                # fallback attempts
                try:
                    theta_raw = self.env.robot.orientation
                except Exception:
                    theta_raw = None

        theta_bin = self._discretize_theta(theta_raw)

        return (gx, gy, theta_bin, hx_bin, hy_bin)
