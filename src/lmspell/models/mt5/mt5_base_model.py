from lmspell.model_enum import ModelEnum
from lmspell.models.mt5.mt5_abstract_model import Mt5AbstractModel


class Mt5BaseModel(Mt5AbstractModel):

    def __init__(self):
        self.model_label = ModelEnum.MT5_BASE
        super().__init__()
