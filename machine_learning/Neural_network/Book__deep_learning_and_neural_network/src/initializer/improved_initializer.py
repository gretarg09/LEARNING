import numpy as np
from src.initializer.initializer_interface import Initializer

class ImprovedInitializer(Initializer):
    """Initialize each weight using a Gaussian distribution with mean 0 and standard deviation 1
    over the square root of the number of weights connecting to the same neuron.  Initialize
    the biases using a Gaussian distribution with mean 0 and standard deviation 1. Note that
    the first layer is assumed to be an input layer, and by convention we won't set any biases
    for those neurons, since biases are only ever used in computing the outputs from later
    layers.
    """

    @staticmethod
    def initialize_biases(sizes):
        return [np.random.randn(y, 1) for y in sizes[1:]]

    @staticmethod
    def initialize_weights(sizes):
        return [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
