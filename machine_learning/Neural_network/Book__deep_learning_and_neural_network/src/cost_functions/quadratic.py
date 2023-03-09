from src.activation_functions import Sigmoid
import numpy as np

from src.cost_functions.interface import Cost

class Quadratic(Cost):
    @staticmethod
    def fn(a: np.ndarray,
           y: np.ndarray):
        """Return the cost associated with an output a and desired output y.
        """
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z: np.ndarray,
              a: np.ndarray,
              y: np.ndarray):
        """Return the error delta from the output layer."""
        return (a - y) * Sigmoid().prime(z)
