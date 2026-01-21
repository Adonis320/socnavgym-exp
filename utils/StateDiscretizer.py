import numpy as np
from utils.StateEncoder import StateEncoder

class StateDiscretizer(StateEncoder):
    """
    Handles discretization / binning of continuous state info:
    - Robot (x, y) in world or robot frame (here: env.robot.x, env.robot.y)
    - Humans (x, y) from obs["humans"]
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
        env = None,
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

        self.theta_bins = theta_bins              # or any number ≥2
        self.theta_edges = np.linspace(
            -np.pi, np.pi, num=self.theta_bins - 1, dtype=np.float32
        )
        self.env = env

    @staticmethod
    def _bin_scalar(x, edges: np.ndarray) -> int:
        """
        Bin scalar x using ascending edges (open on right).
        edges: array of boundaries, length B-1 for B bins
        Returns index in [0, B-1] if inside, or B if x >= last edge.
        """
        for i, e in enumerate(edges):
            if x < e:
                return i
        return len(edges)

    def _discretize_robot_xy(self, obs) -> tuple[int, int]:
        """
        Discretize robot position (x, y) using xy_*.
        Expects obs["robot"] with format [one_hot(D), goal_dx, goal_dy, robot_radius].
        Always returns a pair of integers (bx, by). If robot info is missing,
        returns a fixed sentinel (-1, -1).
        """
        robot = obs.get("robot", None)
        if robot is None:
            # Sentinel for "no robot info" but still a flat tuple of ints
            return -1, -1

        r = np.asarray(robot, dtype=np.float32).flatten()
        # Robot obs format: [one_hot(D), goal_dx, goal_dy, robot_radius] with total length D+3
        D = max(0, r.size - 3)
        dx = float(r[D]) if r.size > D else 0.0
        dy = float(r[D + 1]) if r.size > D + 1 else 0.0

        # Build edges if not provided
        if self.xy_edges is not None:
            edges = self.xy_edges
        else:
            edges = np.linspace(
                -self.xy_max_abs,
                self.xy_max_abs,
                num=max(self.xy_bins - 1, 1),
                dtype=np.float32,
            )

        bx = self._bin_scalar(np.clip(dx, -self.xy_max_abs, self.xy_max_abs), edges)
        by = self._bin_scalar(np.clip(dy, -self.xy_max_abs, self.xy_max_abs), edges)
        return int(bx), int(by)

    def _discretize_humans_xy(self, humans_array: np.ndarray) -> list[int]:
        """
        Discretize all humans (x, y) positions from a flat array.
        Returns [hx_bin_0, hy_bin_0, hx_bin_1, hy_bin_1, ...]
        """
        humans_bins = []
        h = humans_array.flatten().astype(np.float32)
        if h.size == 0:
            return humans_bins

        # Default assumptions about human encoding
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

            if not inferred and h.size >= 8:
                # Minimal fallback: assume single (x, y) at indices (6,7)
                hx = float(h[6])
                hy = float(h[7])
                if self.human_xy_edges is not None:
                    hedges = self.human_xy_edges
                else:
                    hedges = np.linspace(
                        -self.human_xy_max_abs,
                        self.human_xy_max_abs,
                        num=max(self.human_xy_bins - 1, 1),
                        dtype=np.float32,
                    )
                humans_bins.append(
                    self._bin_scalar(
                        np.clip(hx, -self.human_xy_max_abs, self.human_xy_max_abs),
                        hedges,
                    )
                )
                humans_bins.append(
                    self._bin_scalar(
                        np.clip(hy, -self.human_xy_max_abs, self.human_xy_max_abs),
                        hedges,
                    )
                )
                return [int(b) for b in humans_bins]

        # Normal case: h.size is multiple of block
        if h.size % block != 0:
            # cannot decode, return empty bins
            return humans_bins

        if self.human_xy_edges is not None:
            hedges = self.human_xy_edges
        else:
            hedges = np.linspace(
                -self.human_xy_max_abs,
                self.human_xy_max_abs,
                num=max(self.human_xy_bins - 1, 1),
                dtype=np.float32,
            )

        for i in range(0, h.size, block):
            base = i + one_hot_len  # x at base, y at base + 1
            if base + 1 < h.size:
                hx = float(h[base])
                hy = float(h[base + 1])
                bx = self._bin_scalar(
                    np.clip(hx, -self.human_xy_max_abs, self.human_xy_max_abs),
                    hedges,
                )
                by = self._bin_scalar(
                    np.clip(hy, -self.human_xy_max_abs, self.human_xy_max_abs),
                    hedges,
                )
                humans_bins.append(int(bx))
                humans_bins.append(int(by))

        return humans_bins
    
    def _discretize_theta(self, theta: float) -> int:
        """
        Discretize robot yaw angle θ into [0, theta_bins].
        θ is expected in radians, range [-π, π] or any real.
        """
        if theta is None:
            return -1

        # Wrap angle to [-π, π]
        theta = np.arctan2(np.sin(theta), np.cos(theta))

        # Bin using theta_edges
        for i, e in enumerate(self.theta_edges):
            if theta < e:
                return int(i)

        return int(len(self.theta_edges))

    def encode(self, obs) -> tuple:
        """
        Public entry point:
        - obs: environment observation (dict with "humans" and "robot" keys)
        Returns a flat, hashable state tuple of integers.
        """

        # Normalize input: obs may be (obs, info) or other wrappers
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            # Gymnasium: (obs, info)
            if len(obs) == 2 and isinstance(obs[0], dict):
                obs = obs[0]
            # Gym: (obs, reward, done, info, ...) variants
            elif isinstance(obs[0], dict):
                obs = obs[0]

        # If still not dict, return a fixed fallback tuple (always a tuple)
        if not isinstance(obs, dict):
            # Sentinel state: always a tuple of ints
            return (-1, -1)

        # 1) Robot bins (always a pair of ints)
        gx, gy = self._discretize_robot_xy(obs)

        # 2) Humans bins (list of ints, maybe empty)
        humans_bins = []
        humans = obs.get("humans", None)
        if humans is not None:
            humans_bins = self._discretize_humans_xy(
                np.asarray(humans, dtype=np.float32)
            )

        # 3) Robot direction (theta)
        theta_raw = None
        
            # attempt nested extraction
        theta_raw = self.env.env.env.env.robot.orientation

        theta_bin = self._discretize_theta(theta_raw)

        # final
        state = (gx, gy, theta_bin, *humans_bins)
        return state
