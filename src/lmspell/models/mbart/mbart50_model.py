from lmspell.model_enum import ModelEnum
from lmspell.models.mbart.mbart_abstract_model import MbartAbstractModel


class Mbart50Model(MbartAbstractModel):

    def __init__(self):
        self.model_label = ModelEnum.MBART50
        super().__init__()
