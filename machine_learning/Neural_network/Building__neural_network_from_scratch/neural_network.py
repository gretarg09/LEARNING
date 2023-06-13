import numpy as np
from data_loader import load_data


def init_params():
    '''initialize parameters of the neural network'''

    w_1 = np.random.rand(10, 784)
    b_1 = np.random.rand(10, 1)
    w_2 = np.random.rand(10, 10)
    b_2 = np.random.rand(10, 1)

    return w_1, b_1, w_2, b_2

def ReLU(z):
    '''ReLU activation function'''
    return np.maximum(0, z)

def softmax(z):
    '''softmax activation function'''
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def forward_propagation(x):
    '''forward propagation of the neural network'''

    # get the parameters
    w_1, b_1, w_2, b_2 = init_params()

    # calculate the output of the first layer
    z_1 = w_1 @ x.T + b_1
    a_1 = ReLU(z_1)

    z_2 = w_2 @ a_1 + b_2
    a_2 = softmax(z_2)

    print('calculating forward propagation')

    return a_2
