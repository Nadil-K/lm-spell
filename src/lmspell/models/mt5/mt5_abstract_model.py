from transformers import MT5ForConditionalGeneration, T5TokenizerFast

from lmspell.models.encoder_decoder_abstract import EncoderDecoderAbstract


class Mt5AbstractModel(EncoderDecoderAbstract):

    def __init__(self, model_label=None, language="Sinhala"):
        super().__init__(model_label, language)
        model_path = self.model_label.get_model_path()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)
        self.model.to(self.device)

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

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
