from Models.ModelAbstract import ModelAbstract
from ModelEnum import ModelEnum
class Llama31BaseModel(ModelAbstract):
    def __init__(self, model_name=ModelEnum.LLAMA_31_8B):
        self.model_name = model_name
        pass
        

    def check(self, text):
        pass