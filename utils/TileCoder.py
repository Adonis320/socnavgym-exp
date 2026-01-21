import numpy as np
from utils.StateEncoder import StateEncoder

class TileCoder(StateEncoder):
    """
    Tile coder.

    Encodes:
    - Robot (goal_dx, goal_dy) from obs["robot"]
    - Humans (x, y) from obs["humans"]

    Returns a binary feature vector with multiple overlapping tilings.
    """

    def __init__(
        self,
        num_tilings=8,
        tiles_per_dim=10,
        robot_xy_max_abs=10.0,
        human_xy_max_abs=10.0,
        max_humans=3,
    ):
        """
        num_tilings: number of overlapping tilings
        tiles_per_dim: number of tiles per dimension (x, y) per tiling
        robot_xy_max_abs: range [-max_abs, max_abs] for robot dx, dy
        human_xy_max_abs: range [-max_abs, max_abs] for humans x, y
        max_humans: maximum number of humans to encode (extra are ignored)
        """
        self.num_tilings = int(num_tilings)
        self.tiles_per_dim = int(tiles_per_dim)
        self.robot_xy_max_abs = float(robot_xy_max_abs)
        self.human_xy_max_abs = float(human_xy_max_abs)
        self.max_humans = int(max_humans)

        # tiles per (x,y) grid per tiling
        self._tiles_per_grid = self.tiles_per_dim * self.tiles_per_dim

        # one "bank" per entity (robot + each human)
        self._bank_size = self.num_tilings * self._tiles_per_grid

        # total feature dimension
        self.feature_dim = (1 + self.max_humans) * self._bank_size

        # tile widths for robot and humans
        self._robot_tile_width = (2.0 * self.robot_xy_max_abs) / self.tiles_per_dim
        self._human_tile_width = (2.0 * self.human_xy_max_abs) / self.tiles_per_dim

    # ------------------------------------------------------------------
    # Basic parsers – same assumptions as StateDiscretizer
    # ------------------------------------------------------------------
    def _extract_robot_dx_dy(self, obs):
        """
        Extract (dx, dy) from obs["robot"].
        Robot obs format: [one_hot(D), goal_dx, goal_dy, robot_radius]
        """
        robot = obs.get("robot", None)
        if robot is None:
            return 0.0, 0.0

        r = np.asarray(robot, dtype=np.float32).flatten()
        D = max(0, r.size - 3)
        dx = float(r[D]) if r.size > D else 0.0
        dy = float(r[D + 1]) if r.size > D + 1 else 0.0
        return dx, dy

    def _extract_humans_xy(self, humans_array: np.ndarray):
        """
        Extract list of (x, y) for all humans from flat array.

        Mirrors the heuristic block-structure logic of StateDiscretizer,
        but returns raw (x, y) floats instead of discretized bins.
        """
        res = []
        h = humans_array.flatten().astype(np.float32)
        if h.size == 0:
            return res

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
                res.append((hx, hy))
                return res

        # Normal case: h.size is multiple of block
        if h.size % block != 0:
            # cannot decode, return empty list
            return res

        for i in range(0, h.size, block):
            base = i + one_hot_len  # x at base, y at base + 1
            if base + 1 < h.size:
                hx = float(h[base])
                hy = float(h[base + 1])
                res.append((hx, hy))

        return res

    # ------------------------------------------------------------------
    # Core tile coding
    # ------------------------------------------------------------------
    def _coords_to_tiles(self, x, y, max_abs, tile_width):
        """
        For a single (x, y) pair, return the list of tile indices
        across all tilings for one bank (size = num_tilings * tiles_per_dim^2).
        """
        # Clamp
        x = float(np.clip(x, -max_abs, max_abs))
        y = float(np.clip(y, -max_abs, max_abs))

        indices = []
        for t in range(self.num_tilings):
            # Offset each tiling by a fraction of tile width
            offset_x = (t / self.num_tilings) * tile_width
            offset_y = (t / self.num_tilings) * tile_width

            # Shift to [0, 2*max_abs]
            x_shift = x + max_abs + offset_x
            y_shift = y + max_abs + offset_y

            ix = int(x_shift / tile_width)
            iy = int(y_shift / tile_width)

            # Keep indices in range
            ix = min(max(ix, 0), self.tiles_per_dim - 1)
            iy = min(max(iy, 0), self.tiles_per_dim - 1)

            # index within this tiling's grid
            idx_in_tiling = ix * self.tiles_per_dim + iy
            idx = t * self._tiles_per_grid + idx_in_tiling
            indices.append(idx)

        return indices

    def encode(self, obs) -> tuple:
        """
        Hashable tile-coded state representation, analogous to StateDiscretizer:
            (robot_code, h0_code, h1_code, ...)
        where each code is a deterministic integer computed from activated tiles.
        """

        # Normalize obs like in StateDiscretizer
        if isinstance(obs, (list, tuple)) and len(obs) > 0:
            if isinstance(obs[0], dict):
                obs = obs[0]

        if not isinstance(obs, dict):
            try:
                arr = np.asarray(obs, dtype=np.float32)
                return ("raw", hash(arr.tobytes()))
            except Exception:
                return ("raw", str(obs))

        state = []

        # ----------------------------------------------------------
        # 1) Robot tile code
        # ----------------------------------------------------------
        dx, dy = self._extract_robot_dx_dy(obs)

        robot_tiles = self._coords_to_tiles(
            dx, dy,
            max_abs=self.robot_xy_max_abs,
            tile_width=self._robot_tile_width
        )

        # collapse all tilings into a deterministic integer
        robot_code = int(sum(robot_tiles))
        state.append(robot_code)

        # ----------------------------------------------------------
        # 2) Humans tile codes
        # ----------------------------------------------------------
        humans = obs.get("humans", None)
        if humans is not None:
            humans_xy = self._extract_humans_xy(np.asarray(humans, dtype=np.float32))
        else:
            humans_xy = []

        # encode up to max_humans
        for (hx, hy) in humans_xy[: self.max_humans]:
            human_tiles = self._coords_to_tiles(
                hx, hy,
                max_abs=self.human_xy_max_abs,
                tile_width=self._human_tile_width
            )
            human_code = int(sum(human_tiles))
            state.append(human_code)

        # pad fewer humans with zero codes
        missing = self.max_humans - len(humans_xy)
        if missing > 0:
            state.extend([0] * missing)

        return tuple(state)