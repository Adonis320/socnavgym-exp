import numpy as np
from collections import defaultdict
import math
from utils.StateDiscretizer import *

class QL():
    def __init__(self, action_size=7, epsilon=0.05, gamma=0.99, learning_rate=0.01, discretizer=None):
        self.action_size = action_size
        self.q_values = defaultdict(lambda: np.zeros(action_size))
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        
        self.discretizer = discretizer

    def sample_action(self, state, eval=False):
        # Samples action using epsilon-greedy
        if eval:
            epsilon = 0
        else:
            epsilon = self.epsilon
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = int(np.argmax(self.q_values[state]))
        return action
    
    def get_state(self, obs):
        return self.discretizer.encode(obs)
    
    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs
    ):
        # Update the Q-value of an action
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.gamma * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.learning_rate * temporal_difference
        )
        return temporal_difference

    def act(self, env, obs):
        state = self.get_state(obs)
        total_reward = 0
        episode_length = 0

        while True:
            # Epsilon-greedy action selection
            action = self.sample_action(state)
            # Take action
            obs, reward, done, truncated, info = env.step(action)

            next_state =  self.get_state(obs)
            
            total_reward += reward
            
            self.update(state, action, reward, done or truncated, next_state)
            if done or truncated:
                break
            episode_length += 1
            state = next_state

        return episode_length, total_reward
