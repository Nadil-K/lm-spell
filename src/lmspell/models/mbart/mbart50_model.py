from lmspell.model_enum import ModelEnum
from lmspell.models.mbart.mbart_abstract_model import MbartAbstractModel


class Mbart50Model(MbartAbstractModel):

    def __init__(self, model_label=ModelEnum.MBART50, language="Sinhala"):
        super().__init__(model_label, language)
