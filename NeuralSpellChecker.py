from ModelEnum import ModelEnum
from Exceptions.InvalidModelException import InvalidModelException
from Exceptions.EvaluateModelException import EvaluateModelException

class NeuralSpellChecker:

    def __init__(self, model):
        
        try: 
            modelEnum = ModelEnum(model)
        except ValueError:
            raise InvalidModelException(f"Invalid model: {model}") from None

        modelClass = modelEnum.getModelClass()
        self.modelClassInstance = modelClassInstance = modelClass()

        self.model = modelClassInstance.model
        self.tokenizer = modelClassInstance.tokenizer

    def evaluate(self, text=None, src=None):

        if text is None and src is None:
            raise EvaluateModelException("Either text or src is required") from None
        else:
            self.modelClassInstance.evaluate(text, src)



    ### Helper functions
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    def get_model_name(self):
        return self.modelClassInstance.get_model_name()