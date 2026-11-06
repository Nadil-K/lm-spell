from accelerate import Accelerator
from transformers import AutoModelForMaskedLM, AutoTokenizer

from lmspell.models.encoder_abstract import EncoderAbstract


class SinBertAbstractModel(EncoderAbstract):
    """SinBERT (a RoBERTa masked LM for Sinhala), used the same way as XLM-R."""

    def __init__(self, model_label=None, language="Sinhala"):
        super().__init__(model_label, language)
        model_path = self.model_label.get_model_path()
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

        self.accelerator = Accelerator()
