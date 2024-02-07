
from src import mnist_loader
from src import network__chapter_1

def execute_network__chapter_1():
    '''Executing network from chapter 1'''

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    training_data = list(training_data)
    validation_data = list(training_data)
    test_data = list(test_data)

    net = network__chapter_1.Network([784, 100, 10])
    net.stochastic_gradient_descent(training_data=training_data[:1000],
                                    epochs=30,
                                    mini_batch_size=10,
                                    eta=100,
                                    test_data=validation_data[:100])


if __name__ == '__main__':
    execute_network__chapter_1()
