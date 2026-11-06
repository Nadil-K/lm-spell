from lmspell.models.decoder_abstract import DecoderAbstract


class LlamaAbstractModel(DecoderAbstract):
    def __init__(self, model_label=None, language="Sinhala"):
        # TODO
        super().__init__(model_label, language)

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
