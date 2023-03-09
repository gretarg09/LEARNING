import numpy as np
from src.initializer.initializer_interface import Initializer


class NaiveInitializer(Initializer):
    """ Initialize each weight using a Gaussian distribution with mena 0 and standard deviation 1.
    Initialize the biases using a Gaussian distribution with mean 0 and standard deviation 1. 
    """
    
    @staticmethod
    def initialize_biases(sizes):
        return [np.random.randn(y, 1) for y in sizes[1:]]

    @staticmethod
    def initialize_weights(sizes):
        return [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


