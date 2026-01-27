import numpy as np
from collections import defaultdict
import math

class MQL():
    def __init__(self, action_size, epsilon=0.05, gamma=0.99, learning_rate=0.01, discretizer=None):
        self.action_size = action_size
        # Two separate Q-tables for topographic and social features
        self.q_values_topo = defaultdict(lambda: np.zeros(action_size))
        self.q_values_social = defaultdict(lambda: np.zeros(action_size))
        self.gamma = 0.99
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discretizer = discretizer

    def sample_action(self, state, eval=False):
        features_topo, features_social = state
        key_topo = str(features_topo)
        key_social = str(features_social)
        q_topo = self.q_values_topo[key_topo]
        q_social = self.q_values_social[key_social]
        q_sum = q_topo + q_social

        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = int(np.argmax(q_sum))
        return action
    
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

    def update(
        self,
        state,
        action: int,
        reward: float,
        terminated: bool,
        next_state
    ):
        # Updates the Q-values for both topographic and social features.
        features_topo, features_social = state
        next_topo, next_social = next_state

        key_topo = str(features_topo)
        key_social = str(features_social)
        key_topo_next = str(next_topo)
        key_social_next = str(next_social)

        # Sum Q-values for next state for action selection
        q_topo_next = self.q_values_topo[key_topo_next]
        q_social_next = self.q_values_social[key_social_next]
        q_sum_next = q_topo_next + q_social_next

        future_q_value = (not terminated) * np.max(q_sum_next)
        q_topo = self.q_values_topo[key_topo][action]
        q_social = self.q_values_social[key_social][action]
        q_sum = q_topo + q_social

        temporal_difference = (
            reward + self.gamma * future_q_value - q_sum
        )

        # Update both Q-tables equally
        self.q_values_topo[key_topo][action] += self.learning_rate * temporal_difference / 2
        self.q_values_social[key_social][action] += self.learning_rate * temporal_difference / 2

    def act(self, env, obs, eval=False, upd_social=None):

        state = self.get_state_key(obs)
        total_reward = 0

        episode_length = 0

        while True:
            # Get Humanoid actions            # Epsilon-greedy action selection
            action = self.sample_action(state, eval)
            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            next_state =  self.get_state_key(next_state)
            
            total_reward += reward
            
            self.update(state, action, reward, done, next_state)
            if done or truncated:
                break
            episode_length += 1
            state = next_state

        return  episode_length, total_reward
