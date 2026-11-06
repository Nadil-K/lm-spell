from lmspell.model_enum import ModelEnum
from lmspell.models.xlmr.xlmr_abstract_model import XlmrAbstractModel


class XlmrLargeModel(XlmrAbstractModel):

    def __init__(self, model_label=ModelEnum.XLMR_LARGE, language="Sinhala"):
        super().__init__(model_label, language)
