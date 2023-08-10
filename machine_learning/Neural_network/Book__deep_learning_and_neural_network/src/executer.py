from src import mnist_loader
from src import network__chapter_1  as network


# python main function
def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 100, 10])

    EPOCH = 30
    MINI_BATCH_SIZE = 10
    ETA = 3.0

    net.stochastic_gradient_descent(training_data,
                                    EPOCH,
                                    MINI_BATCH_SIZE,
                                    ETA,
                                    test_data=test_data)

if __name__ == "__main__":
    main()
