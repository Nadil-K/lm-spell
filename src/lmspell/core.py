from lmspell.exceptions import InvalidModelException
from lmspell.model_enum import ModelEnum


class LMSpell:

    def __init__(self, model):

        try:
            model_enum = ModelEnum(model)
        except ValueError:
            raise InvalidModelException(f"Invalid model: {model}") from None

        model_class = model_enum.getModelClass()
        self._model = model_class()

    def __getattr__(self, name):
        return getattr(self._model, name)
