import numpy as np
from collections import defaultdict

class SR:
    """
    Tabular SR on a discretized continuous state space, using StateDiscretizer.

    - SR[s][a][s']  ≈ discounted expected visitation of s' when starting from (s,a).
    - R[s][a]       = expected immediate reward for taking action a in state s.
    - Q(s,a)        = sum_{s'} SR[s][a][s'] * R[s'][a]
    """

    def __init__(
        self,
        action_size,
        epsilon=0.05,
        gamma=0.99,
        learning_rate=0.01,
        r_learning_rate=0.01,
        discretizer=None,
    ):
        self.action_size = action_size

        # SR[state][action][next_state]
        self.SR = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

        # Reward function R[state][action] = expected immediate reward
        self.R = defaultdict(lambda: defaultdict(float))

        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.r_learning_rate = r_learning_rate

        self.discretizer = discretizer

    # --------------
    # State handling
    # --------------
    def get_state(self, obs):
        """Discretize observation into a hashable state representation."""
        return self.discretizer.encode(obs)

    def get_state_key(self, obs):
        """Convert discretized state to a stable, hashable key."""
        state = self.get_state(obs)
        return state

    # --------------
    # Q-values
    # --------------
    def _q_values(self, state_key):
        """
        Q(s,a) = sum_{s'} SR[s][a][s'] * R[s'][a]
        Reward depends on (s', a).
        """
        Q = np.zeros(self.action_size, dtype=np.float32)
        for a in range(self.action_size):
            q = 0.0
            for s_prime, m_sas in self.SR[state_key][a].items():
                q += m_sas * self.R[s_prime][a]
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
    # SR update
    # --------------
    def update_sr(self, state_key, action, next_state_key, done):
        """
        TD update for SR:

          M(s,a,·) ← M(s,a,·) + α[(e_s + γ M(s',a*,·)) − M(s,a,·)]

        where:
        - e_s is 1 at the CURRENT state, 0 elsewhere,
        - a* is the greedy action in s' (off-policy control).
        """
        if not done:
            # Greedy next action for control (off-policy)
            next_action = self.sample_action(next_state_key, eval=True)
            relevant_s_primes = (
                set(self.SR[state_key][action].keys())
                | set(self.SR[next_state_key][next_action].keys())
                | {state_key, next_state_key}
            )
        else:
            next_action = None
            # also include next_state_key so M(s,a,next_state) can be non-zero
            relevant_s_primes = (
                set(self.SR[state_key][action].keys()) | {state_key, next_state_key}
            )

        for s_prime in tuple(relevant_s_primes):
            # e_s: 1 on CURRENT state, 0 otherwise
            indicator = 1.0 if s_prime == state_key else 0.0

            if not done and next_action is not None:
                future_sr = self.SR[next_state_key][next_action][s_prime]
            else:
                future_sr = 0.0

            target = indicator + self.gamma * future_sr
            current_sr = self.SR[state_key][action][s_prime]
            td_error = target - current_sr

            self.SR[state_key][action][s_prime] += self.learning_rate * td_error

    # --------------
    # Reward update R(s,a)
    # --------------
    def update_reward(self, state_key, action, reward):
        """
        Update R(s,a) from the observed immediate reward r_t
        obtained after taking action a in state s.
        """
        r_old = self.R[state_key][action]
        self.R[state_key][action] = r_old + self.r_learning_rate * (reward - r_old)

    # --------------
    # Interaction loop
    # --------------
    def act(self, env, obs, eval=False):
        """
        Run one episode using SR control, updating SR and R(s,a) online.
        Assumes `env` follows Gymnasium API or similar.
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

            # Update SR and state-action reward
            self.update_sr(state_key, action, next_state_key, done or truncated)
            self.update_reward(next_state_key, action, reward)

            # Move to next state
            state_key = next_state_key
            episode_length += 1

            if done or truncated:
                break

        return episode_length, episode_reward
