import numpy as np
from collections import defaultdict
import math


class MSR:
    """
    Modular SR with external discretizer.

    State key is a tuple:
        state_key = (topo_key, social_key)

    Topographic branch:
        SR_topo[topo][a][topo'] ≈ discounted expected visitation of topo'
        R_topo[topo] = expected immediate reward component for topo

    Social branch:
        SR_social[soc][a][soc'] ≈ discounted expected visitation of soc'
        R_social[soc] = expected immediate reward component for social

    Combined Q-value:
        Q(s,a) = Q_topo(s,a) + Q_social(s,a)
    """

    def __init__(
        self,
        action_size,
        epsilon=0.05,
        gamma=0.99,
        learning_rate_topo=0.01,
        learning_rate_social=0.01,
        r_learning_rate_topo=0.01,
        r_learning_rate_social=0.01,
        discretizer=None,
    ):
        self.action_size = action_size

        # SR[state][action][next_state]
        self.SR_topo = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.SR_social = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Reward functions R[state] = expected immediate reward (state-only, as in your original MSR)
        self.R_topo =  defaultdict(lambda: defaultdict(float))
        self.R_social =  defaultdict(lambda: defaultdict(float))

        self.epsilon = epsilon
        self.gamma = gamma

        self.learning_rate_topo = learning_rate_topo
        self.learning_rate_social = learning_rate_social
        self.r_learning_rate_topo = r_learning_rate_topo
        self.r_learning_rate_social = r_learning_rate_social

        # External discretizer (must implement encode(obs))
        self.discretizer = discretizer

    # --------------
    # State handling
    # --------------

    def get_state(self, obs):
        """
        Use external discretizer to obtain a modular state:
            state_key = (topo_key, social_key)

        Here:
            - topo_key  = first 3 discretized values
            - social_key = remaining discretized values
        """
        if self.discretizer is None:
            raise ValueError("MSR: discretizer is None. Provide a discretizer with encode(obs).")

        enc = self.discretizer.encode(obs)   # expected to return a flat array/tuple/list

        # Convert to tuple
        if isinstance(enc, np.ndarray):
            enc = enc.tolist()
        enc = tuple(enc)

        # Split: first 3 → topo, rest → social
        if len(enc) < 3:
            # Not enough dims → topo padded, social empty
            topo_key = enc + (0,) * (3 - len(enc))
            social_key = ()
        else:
            topo_key = enc[:3]
            social_key = enc[3:]

        return (tuple(topo_key), tuple(social_key))

    def get_state_key(self, obs):
        """Convert observation to stable, hashable key (same API as SRDec)."""
        return self.get_state(obs)

    # --------------
    # Q-values
    # --------------

    def _q_values(self, state_key):
        topo_key, social_key = state_key
        Q = np.zeros(self.action_size, dtype=np.float32)

        for a in range(self.action_size):
            q = 0.0

            # topo contribution: use R_topo[t_prime][a]
            for t_prime, m_t in self.SR_topo[topo_key][a].items():
                q += m_t * self.R_topo[t_prime][a]

            # social contribution: use R_social[s_prime][a]
            for s_prime, m_s in self.SR_social[social_key][a].items():
                q += m_s * self.R_social[s_prime][a]

            Q[a] = q

        return Q

    def sample_action(self, state_key, eval=False):
        eps = 0.0 if eval else self.epsilon
        if np.random.rand() < eps:
            return np.random.randint(self.action_size)

        Q = self._q_values(state_key)
        # small noise for tie-breaking
        Q = Q + 1e-6 * np.random.randn(self.action_size)
        return int(np.argmax(Q))

    # --------------
    # SR updates
    # --------------

    def update_sr(self, state_key, action, next_state_key, done, upd_social=True):
        """
        TD update for topographic and (optionally) social SRs.

        For each branch b ∈ {topo, social}:

          M_b(s_b, a, ·) ← M_b(s_b, a, ·) + α_b[(e_{s_b} + γ M_b(s'_b, a*, ·)) − M_b(s_b, a, ·)]

        where:
        - s_b is the current branch state (topo or social part),
        - s'_b is the next branch state,
        - e_{s_b} is 1 at CURRENT branch state, 0 elsewhere,
        - a* is the greedy action in the full state s' (off-policy control).
        """
        topo_key, social_key = state_key
        next_topo_key, next_social_key = next_state_key

        # Choose greedy next action based on full next_state_key
        if not done:
            next_action = self.sample_action(next_state_key, eval=True)
        else:
            next_action = None

        # ---- topo branch ----
        if not done and next_action is not None:
            rel_topo = (
                set(self.SR_topo[topo_key][action].keys())
                | set(self.SR_topo[next_topo_key][next_action].keys())
                | {topo_key, next_topo_key}
            )
        else:
            # also include next_topo_key so M(s,a,next_state) can be non-zero
            rel_topo = set(self.SR_topo[topo_key][action].keys()) | {topo_key, next_topo_key}

        for t_prime in tuple(rel_topo):
            indicator = 1.0 if t_prime == topo_key else 0.0
            if not done and next_action is not None:
                future_sr = self.SR_topo[next_topo_key][next_action][t_prime]
            else:
                future_sr = 0.0

            target = indicator + self.gamma * future_sr
            current_sr = self.SR_topo[topo_key][action][t_prime]
            td_error = target - current_sr

            self.SR_topo[topo_key][action][t_prime] = current_sr + self.learning_rate_topo * td_error

        # ---- social branch ----
        if upd_social:
            if not done and next_action is not None:
                rel_social = (
                    set(self.SR_social[social_key][action].keys())
                    | set(self.SR_social[next_social_key][next_action].keys())
                    | {social_key, next_social_key}
                )
            else:
                rel_social = (
                    set(self.SR_social[social_key][action].keys())
                    | {social_key, next_social_key}
                )

            for s_prime in tuple(rel_social):
                indicator = 1.0 if s_prime == social_key else 0.0
                if not done and next_action is not None:
                    future_sr = self.SR_social[next_social_key][next_action][s_prime]
                else:
                    future_sr = 0.0

                target = indicator + self.gamma * future_sr
                current_sr = self.SR_social[social_key][action][s_prime]
                td_error = target - current_sr

                self.SR_social[social_key][action][s_prime] = (
                    current_sr + self.learning_rate_social * td_error
                )

    # --------------
    # Reward updates (state-only, per branch)
    # --------------

    def update_reward(self, state_key, action, reward, upd_social=True):
        topo_key, social_key = state_key

        # topo reward for (topo_key, action)
        rt = self.R_topo[topo_key][action]
        self.R_topo[topo_key][action] = rt + self.r_learning_rate_topo * (reward - rt)

        # social reward for (social_key, action)
        if upd_social:
            rs = self.R_social[social_key][action]
            self.R_social[social_key][action] = rs + self.r_learning_rate_social * (reward - rs)


    # --------------
    # Interaction loop
    # --------------

    def act(self, env, obs, eval=False, upd_social=True):
        """
        Run one episode using modular SR control, updating SR and R online.

        Parameters
        ----------
        env : Gymnasium-like environment
        obs : initial observation
        eval : if True, use greedy actions (no exploration)
        upd_social : if False, only topo SR/R are updated (social frozen)
        """
        episode_reward = 0.0
        episode_length = 0

        # Initial state
        state_key = self.get_state_key(obs)

        while True:
            # Select action
            action = self.sample_action(state_key, eval=eval)

            # Environment step
            next_obs, reward, done, truncated, info = env.step(action)

            # Next state
            next_state_key = self.get_state_key(next_obs)

            # Accumulate reward
            episode_reward += reward

            # Update SR and branch rewards
            self.update_sr(state_key, action, next_state_key, done or truncated, upd_social=upd_social)
            self.update_reward(state_key, action, reward, upd_social=upd_social)

            # Move to next state
            state_key = next_state_key
            episode_length += 1

            if done or truncated:
                break

        return episode_length, episode_reward