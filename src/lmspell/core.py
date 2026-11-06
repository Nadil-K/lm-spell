from lmspell.exceptions import InvalidModelException
from lmspell.model_enum import ModelEnum


class LMSpell:

    def __init__(self, model, language="Sinhala"):
        """
        Create a spell corrector for a supported Hugging Face model id (see ModelEnum).
        `language` is the language name used in the LLM prompt and for the mBART language code.
        """

        try:
            model_enum = ModelEnum(model)
        except ValueError:
            raise InvalidModelException(f"Invalid model: {model}") from None

        model_class = model_enum.getModelClass()
        self._model = model_class(model_label=model_enum, language=language)

    def __getattr__(self, name):
        # Only called for attributes LMSpell does not have; guard against recursion
        # when _model itself is missing (e.g. during unpickling or a failed __init__).
        if name == "_model":
            raise AttributeError(name)
        return getattr(self._model, name)
