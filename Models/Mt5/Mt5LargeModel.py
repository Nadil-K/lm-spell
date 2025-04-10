from ModelEnum import ModelEnum
from Models.Mt5.Mt5AbstractModel import Mt5AbstractModel
from transformers import MT5ForConditionalGeneration, T5TokenizerFast

class Mt5LargeModel(Mt5AbstractModel):
    def __init__(self):
        self.model_label = ModelEnum.MT5_LARGE

        model_path = self.model_label.get_model_path()
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_path)

        