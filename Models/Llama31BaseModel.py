from Models.ModelAbstract import ModelAbstract
class Llama31BaseModel(ModelAbstract):
    def __init__(self, model_name=get_model_names("llama3.1-8b")):
        self.model_name = model_name
        pass
        

    def check(self, text):
        pass