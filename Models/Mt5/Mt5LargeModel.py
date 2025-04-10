from ModelEnum import ModelEnum
from Models.Mt5.Mt5AbstractModel import Mt5AbstractModel
from transformers import MT5ForConditionalGeneration, T5TokenizerFast

class Mt5LargeModel(Mt5AbstractModel):
    def __init__(self):
        self.model_label = ModelEnum.MT5_LARGE

        model_path = self.model_label.get_model_path()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        
    def correct(self, text):
        pass

    def correctFromFile(self, src):
        pass
    
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
