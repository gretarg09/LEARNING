
from abc import ABC, abstractmethod
import numpy as np


class Cost(ABC):

    @abstractmethod
    def fn(a: np.ndarray,
           y: np.ndarray):
        pass

    @abstractmethod
    def delta(z: np.ndarray,
              a: np.ndarray,
              y: np.ndarray):
        pass
