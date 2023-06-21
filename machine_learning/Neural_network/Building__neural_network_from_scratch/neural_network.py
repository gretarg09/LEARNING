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

def derivative_ReLU(z):
    '''derivative of the ReLU activation function'''
    return z > 0

def softmax(z):
    '''softmax activation function'''
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def forward_propagation(x, w_1, b_1, w_2, b_2):
    '''forward propagation of the neural network'''


    # calculate the output of the first layer
    z_1 = w_1 @ x.T + b_1
    a_1 = ReLU(z_1)

    z_2 = w_2 @ a_1 + b_2
    a_2 = softmax(z_2)

    print('calculating forward propagation')

    return z_1, a_1, z_2, a_2


def one_hot_encoding(y):
    '''one hot encoding of the labels'''
    y_one_hot = np.zeros((y.size, y.max()+1))
    y_one_hot[np.arange(y.size), y] = 1

    return y_one_hot.T

def backpropagation(z_1,
                    a_1,
                    a_2,
                    w_2,
                    x,
                    y):
    '''backpropagation of the neural network'''
    y = one_hot_encoding(y)
    m = y.shape[1] # number of samples

    # Calculating W_2 
    d_z_2 = a_2 - y
    d_w_2 = (1 / m) * d_z_2 @ a_1.T
    d_b_2 = (1 / m) * np.sum(d_z_2, axis=1, keepdims=True)

    # Calculating W_1
    d_z_1 = w_2.T @ d_z_2 * derivative_ReLU(z_1)
    d_w_1 = (1 / m) * d_z_1 @ x
    d_b_1 = (1 / m) * np.sum(d_z_1, axis=1, keepdims=True)

    return (d_w_1,
            d_b_1,
            d_w_2,
            d_b_2)


def update_parameters(w_1,
                      b_1,
                      w_2, 
                      b_2,
                      d_w_1,
                      d_b_1,
                      d_w_2,
                      d_b_2,
                      learning_rate):
    '''gradient descent algorithm'''
    w_1_updated = w_1 - learning_rate * d_w_1
    b_1_updated = b_1 - learning_rate * d_b_1
    w_2_updated = w_2 - learning_rate * d_w_2
    b_2_updated = b_2 - learning_rate * d_b_2

    return (w_1_updated,
            b_1_updated,
            w_2_updated,
            b_2_updated)


def get_predictions(a_2):
    '''get predictions of the neural network'''
    return np.argmax(a_2, axis=0)

def get_accuracy(predictions, Y):
    '''get accuracy of the neural network'''
    return np.sum(predictions == Y) / Y.size

def gradient_descent(x, y, learning_rate, iterations):
    '''gradient descent algorithm'''

    # get the parameters
    w_1, b_1, w_2, b_2 = init_params()

    for i in range(iterations):
        z_1, a_1, z_2, a_2 = forward_propagation(x, w_1, b_1, w_2, b_2)
        d_w_1, d_b_1, d_w_2, d_b_2 = backpropagation(z_1, a_1, a_2, w_2, x, y)
        w_1, b_1, w_2, b_2 = update_parameters(w_1,
                                               b_1,
                                               w_2,
                                               b_2,
                                               d_w_1,
                                               d_b_1,
                                               d_w_2,
                                               d_b_2,
                                               learning_rate)

        if i % 50 == 0:
            print('\niteration: ', i)
            print('accuracy: ', get_accuracy(get_predictions(a_2), y))

    return w_1, b_1, w_2, b_2


if __name__ == '__main__':
    # Get data
    data = load_data()
    X_train = data['training_data']['x'] 
    Y_train = data['training_data']['y'] 
    X_test = data['test_data']['x'] 
    Y_test = data['test_data']['y'] 

    # Train model 
    w_1, b_1, w_2, b_2 = gradient_descent(X_train, Y_train, 0.1, 1000)

    # Test model
    z_1, a_1, z_2, a_2 = forward_propagation(X_test, w_1, b_1, w_2, b_2)
    print('\nFINAL ACCURACY: ', get_accuracy(get_predictions(a_2), Y_test))

