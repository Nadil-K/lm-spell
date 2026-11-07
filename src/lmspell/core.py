from lmspell.model_resolver import resolve_model


class LMSpell:

    def __init__(self, model, language="Sinhala"):
        """
        Create a spell corrector. `model` is a preset model id (see ModelEnum), any Hugging Face
        checkpoint with a supported architecture (mT5, mBART, XLM-RoBERTa, RoBERTa, Llama, Gemma),
        or a local checkpoint folder such as one saved by a trainer.
        `language` is the language name used in the LLM prompt and for the mBART language code.
        """
        model_label, model_class = resolve_model(model)
        self._model = model_class(model_label=model_label, language=language)

    def __getattr__(self, name):
        # Only called for attributes LMSpell does not have; guard against recursion
        # when _model itself is missing (e.g. during unpickling or a failed __init__).
        if name == "_model":
            raise AttributeError(name)
        return getattr(self._model, name)
