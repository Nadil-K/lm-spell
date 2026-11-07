from accelerate import Accelerator
from transformers import AutoConfig, AutoTokenizer, XLMRobertaForMaskedLM

from lmspell.models.encoder_abstract import EncoderAbstract


class XlmrAbstractModel(EncoderAbstract):
    def __init__(self, model_label=None, language="Sinhala"):
        super().__init__(model_label, language)
        model_path = self.model_label.get_model_path()
        config = AutoConfig.from_pretrained(model_path)
        self.model = XLMRobertaForMaskedLM.from_pretrained(model_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

        self.seperator_token = "</s>"
        # As in the original XLM-R setup, <ZWJ> is always a token (and the embeddings grow to fit it);
        # whether U+200D is converted to it is controlled by fix_zwj.xlmr.
        self.add_zwj_token()
        self.enable_zwj_fix("xlmr")
        self.accelerator = Accelerator()
