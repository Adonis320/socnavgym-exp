from abc import ABC, abstractmethod
import numpy as np

class StateEncoder(ABC):
    """
    Abstract interface for all state encoders.
    A state encoder converts an observation dict → encoded state representation.

    Subclasses must implement:
        encode(obs) → encoded_state
    """

    @abstractmethod
    def encode(self, obs):
        """
        Encode observation to any representation (tuple, hash, feature vector, etc).
        """
        pass
