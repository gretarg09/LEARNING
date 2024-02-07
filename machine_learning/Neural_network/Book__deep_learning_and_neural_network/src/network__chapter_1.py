"""
network.py
~~~~~~~~~~
IT WORKS

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

from typing import List, Tuple 
import random
import numpy as np


class Initializer():
    @staticmethod
    def initialize_biases(sizes):
        """ Initilizes the biases randomly using a Gaussian distribution with mean 0, and
        variance 1. 
        """
        return [np.random.randn(y, 1) for y in sizes[1:]]

    @staticmethod
    def initialize_weights(sizes):
        """ Initilizes the weights randomly using a Gaussian distribution with mean 0, and
        variance 1. 
        """
        return [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


class Network(Initializer):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = self.initialize_biases(sizes)
        self.weights = self.initialize_weights(sizes)

    def feedforward(self,
                    a: np.ndarray):
        """Return the output of the network if ``a`` is input."""

        for b, w in zip(self.biases, self.weights):
            a = Sigmoid.normal(np.dot(w, a)+b)
        return a

    def stochastic_gradient_descent(self,
                                    training_data: List[Tuple[np.ndarray, np.ndarray]],
                                    epochs: int,
                                    mini_batch_size: int,
                                    eta: int,
                                    test_data: List[Tuple[np.ndarray, np.ndarray]]=None):
        """Train the neural network using mini-batch stochastic gradient descent.

        parameters:
        -----------         
        training_data:    The training data is a list of tuples (x,y) representing the training
                          inputs and the desire output.
        epochs:           The number of epoch to train for. One epoch is one run through the whole
                          training data.
        mini_batch_size:  In one epoch the whole training data is divided into multiple mini batches.
                          The size of each mini-batches can be chosen with mini_batch_size.
        eta:              The learning rate.
        test_data:        If provided then the network will be evaluated against the test training_data
                          after each epoch, and a partial progress printed out. This is useful for tracking
                          progress, but slows things down substantially.
        """

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)

            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_weights_biases(mini_batch, eta)

            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_weights_biases(self,
                              data_batch: List[Tuple[np.ndarray]],
                              eta: int):
        """ Update the network's weights and biases by applying gradient descent using
        backpropagation to a single batch of data."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # The partial derivative dC/dw and dC/db are calculated for each datapoint. They are then collected as a sum
        # within the nabla_b and nabla_w data structure. 
        # In the end the average is calculated by dividing by the len(data_batch)
        for x, y in data_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # get the gradient estimation
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update the biases and weights.
        self.biases = [b - (eta / len(data_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta / len(data_batch)) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self,
                 x: List[Tuple[np.ndarray]],
                 y: List[Tuple[np.ndarray]]):
        """
        Executes the backpropagation algorithm to find the partial derivatives dC/dw and dC/db. 

        Returns
        -------
        Tuple[np.ndarray]
            Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x.
            nabla_b and nabla_w are layer-by-layer lists of numpy arrays, similar to self.biases and self.weights."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Sigmoid.normal(z)
            activations.append(activation)
        # backward pass

        delta = self.cost_derivative(activations[-1], y) * Sigmoid.prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = Sigmoid.prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self,
                 test_data: List[Tuple[np.ndarray]]):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x \partial a for the output activations."""
        return (output_activations - y)


class Sigmoid:
    @staticmethod
    def normal(z: np.ndarray):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def prime(z: np.ndarray):
        """Derivative of the sigmoid function."""
        return Sigmoid.normal(z) * (1 - Sigmoid.normal(z))
