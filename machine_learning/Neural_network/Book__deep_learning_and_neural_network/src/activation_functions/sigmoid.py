import numpy as np

class Sigmoid:
    def fn(self,
           z: np.ndarray) -> np.ndarray:
        """the sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def prime(self,
              z: np.ndarray) -> np.ndarray:
        """Derivative of the sigmoid function."""
        return (z) * (1 - self.fn(z))
