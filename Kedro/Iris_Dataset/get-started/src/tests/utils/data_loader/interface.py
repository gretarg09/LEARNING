from abc import ABC, abstractmethod


class LoaderInterface(ABC):

    @abstractmethod
    def load(self, path):
        pass
