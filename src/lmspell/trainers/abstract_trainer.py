from abc import ABC, abstractmethod


class AbstractTrainer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass
