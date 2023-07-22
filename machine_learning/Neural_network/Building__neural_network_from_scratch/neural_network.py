import numpy as np
from data_loader import load_data
from matplotlib import pyplot as plt


def init_params():
    '''initialize parameters of the neural network

    Our initial strategy is to intialize the weights and biases randomly between 
    -0.5 and 0.5
    '''


    w_1 = np.random.rand(10, 784) - 0.5
    b_1 = np.random.rand(10, 1) - 0.5
    w_2 = np.random.rand(10, 10) - 0.5
    b_2 = np.random.rand(10, 1) - 0.5

    return w_1, b_1, w_2, b_2

def ReLU(z):
    '''ReLU activation function'''
    return np.maximum(0, z)

def derivative_ReLU(z):
    '''derivative of the ReLU activation function'''
    return z >  0

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
    d_b_2 = (1 / m) * np.sum(d_z_2)

    # Calculating W_1
    d_z_1 = w_2.T @ d_z_2 * derivative_ReLU(z_1)
    d_w_1 = (1 / m) * d_z_1 @ x
    d_b_1 = (1 / m) * np.sum(d_z_1)

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


def make_predictions(X, w_1, b_1, w_2, b_2):
    _, _, _, A2 = forward_propagation(X, w_1, b_1, w_2, b_2)
    predictions = get_predictions(A2)
    return predictions


    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# TODO create a result image with 9 subplots
# 1. image should contain 9 random images from the test set.
# 2. image should contain 9 random images, from the test set, that are correctly classified by the model
# 2. image should contain 9 random images, from the test set, that are not correctly classified by the model

def create_result_image(X, Y, pred):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))  # Create a grid of 3x3 subplots

    for i in range(3):
        for j in range(3):
            axs[i, j].imshow(X[i+j].reshape((28, 28)) * 255)  # Replace with your data
            axs[i, j].set_title(f'Label: {Y[i+j]}, Prediction: {pred[i+j]}')

    plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure.
    plt.show()


if __name__ == '__main__':
    # Get data
    data = load_data()
    X_train = data['training_data']['x'] 
    Y_train = data['training_data']['y'] 
    X_test = data['test_data']['x'] 
    Y_test = data['test_data']['y'] 

    # Train model 
    w_1, b_1, w_2, b_2 = gradient_descent(X_train, Y_train, 0.1, 100)

    # Test model on test data
    z_1, a_1, z_2, a_2 = forward_propagation(X_test, w_1, b_1, w_2, b_2)
    print('\nFINAL ACCURACY: ', get_accuracy(get_predictions(a_2), Y_test))


    # Create result images
    # randomly select 9 images from the test set
    indices = np.random.choice(Y_test.shape[0], size=9, replace=False)

    selected_images = X_test[indices]
    selected_labels = Y_test[indices]
    selected_predictions = make_predictions(selected_images, w_1, b_1, w_2, b_2)

    create_result_image(selected_images, selected_labels, selected_predictions)


