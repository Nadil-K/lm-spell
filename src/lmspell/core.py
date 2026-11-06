from lmspell.model_enum import ModelEnum
from lmspell.exceptions import InvalidModelException
from lmspell.exceptions import EvaluateModelException

class LMSpell:

    def __init__(self, model):
        
        try: 
            modelEnum = ModelEnum(model)
        except ValueError:
            raise InvalidModelException(f"Invalid model: {model}") from None

        modelClass = modelEnum.getModelClass()
        self._model = modelClass()

    def __getattr__(self, name):
        return getattr(self._model, name)
