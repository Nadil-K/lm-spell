from Models.ModelAbstract import ModelAbstract

class Mt5AbstractModel(ModelAbstract):
    def __init__(self):
        super().__init__()
        
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
        