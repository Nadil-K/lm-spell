from abc import ABC, abstractmethod
from Exceptions.EvaluateModelException import EvaluateModelException

class ModelAbstract(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def correct(self, text):
        pass

    @abstractmethod
    def correctFromFile(self, src):
        pass
    
    @abstractmethod
    def evaluate(self, src, target):
        pass
    
    @abstractmethod
    def evaluateFromFile(self, src_file, target_file):
        pass
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def get_model_name(self):
        pass
    
    @abstractmethod
    def load_tokenizer(self):
        pass
    
    @abstractmethod
    def save_tokenizer(self):
        pass
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
        