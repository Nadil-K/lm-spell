from ModelEnum import ModelEnum
from Models.Mt5.Mt5AbstractModel import Mt5AbstractModel
from transformers import MT5ForConditionalGeneration, T5TokenizerFast
import torch

class Mt5LargeModel(Mt5AbstractModel):
    def __init__(self):
        super().__init__()
        self.model_label = ModelEnum.MT5_LARGE
        # Replace
        self.exp_dir = 'C:/Users/Tharusha/Documents/Desktop Removals/neural-spell-checker-library'

        model_path = self.model_label.get_model_path()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.model.to(self.device)
    
    def evaluate(self, src, target):
        pass
    
    def evaluateFromFile(self, src_file, target_file):
        pass
    
    def load_model(self):
        pass
    
    def save_model(self):
        pass
    
    def get_model_name(self):
        pass
    
    def load_tokenizer(self):
        pass
    
    def save_tokenizer(self):
        pass
