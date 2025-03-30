from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

class Trainer(ABC):
    def __init__(
        self, 
        model: AutoModel, 
        tokenizer: AutoTokenizer,
        optimizer,
        scheduler,
        early_stopping,
        criterion,
        train_dataset,
        test_dataset,
        val_dataset,
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        max_length: int = 512,
        
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save_model(self, save_path: str):
        pass
    
    @abstractmethod
    def save_tokenizer(self, save_path: str):
        pass
    
    @abstractmethod
    def save_optimizer(self, save_path: str):
        pass
    
    @abstractmethod
    def save_scheduler(self, save_path: str):
        pass
    
    