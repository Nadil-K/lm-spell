import torch
from transformers import MBart50TokenizerFast, MBartConfig, MBartForConditionalGeneration

from lmspell.models.encoder_decoder_abstract import EncoderDecoderAbstract
from lmspell.models.mbart.language_codes import MBART50_LANG_CODE


class MbartAbstractModel(EncoderDecoderAbstract):

    def __init__(self, model_label=None, language="Sinhala"):
        super().__init__(model_label, language)
        model_path = self.model_label.get_model_path()
        self.config = MBartConfig.from_pretrained(model_path)
        self.model = MBartForConditionalGeneration.from_pretrained(model_path, config=self.config, device_map='auto',
                                                                   torch_dtype=torch.bfloat16).to(self.device)
        lang_code = MBART50_LANG_CODE.get(self.language.lower(), MBART50_LANG_CODE["english"])
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path, src_lang=lang_code, tgt_lang=lang_code)
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
