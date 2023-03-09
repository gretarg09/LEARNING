from abc import ABC, abstractmethod


class Initializer(ABC):

    @classmethod
    def execute(cls, size):
        return (cls.initialize_weights(size),
                cls.initialize_biases(size))

    @abstractmethod
    def initialize_biases(sizes):
        pass

    @abstractmethod
    def initialize_weights(sizes): 
        pass
