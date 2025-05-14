from Models.EncoderDecoderAbstract import EncoderDecoderAbstract
from ModelEnum import ModelEnum

class MbartAbstractModel(EncoderDecoderAbstract):
    def __init__(self):
        self.model = ModelEnum.MBART50
        self.tokenizer = "Tokenizer for MBART50"
        print('MBART50 Large Model')

        # set lang when initializing the tokenizer
        

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