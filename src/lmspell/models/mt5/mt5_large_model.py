from lmspell.model_enum import ModelEnum
from lmspell.models.mt5.mt5_abstract_model import Mt5AbstractModel

class Mt5LargeModel(Mt5AbstractModel):
    
    def __init__(self):
        self.model_label = ModelEnum.MT5_LARGE
        super().__init__()
