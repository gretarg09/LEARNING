import numpy as np

from src.cost_functions.interface import Cost



class CrossEntropy(Cost):
    @staticmethod
    def fn(a: np.ndarray,
           y: np.ndarray):
        """Return the cost associated with an output a and desired output y.

        Note
        -----
        np.nan_to_num is used to ensure numerical stability.
        In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a) returns nan. 
        The np.nan_to_num ensures that that is converted to the correct value (0.0).
        """

        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z: np.ndarray,
              a: np.ndarray,
              y: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer.

        Note
        ----
        parameter z is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """

        return (a - y)
