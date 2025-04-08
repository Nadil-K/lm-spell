from enum import Enum
from typing import Dict, Type

from Models.Llama31BaseModel import Llama31BaseModel
from Models.Mbart50Model import Mbart50LargeModel
from Models.Mt5BaseModel import Mt5BaseModel
from Models.SinBertLargeModel import SinBertLargeModel
from Models.XlmrBaseModel import XlmrBaseModel
# Import other model classes as needed

class ModelEnum(Enum):
    """
    Enum class for all the models supported by the 
    Neural Spell Checker Library
    """
    # Fine-tuned models
    MT5_FT_SI = "surge-masks/mt5-base-ft-ssc"
    MT5_XL_FT_SI = "surge-masks/mt5-xl-ft-ssc"
    MBART_FT_SI = "surge-masks/mbart50-ft-ssc"
    SINBERT_FT_SI = "surge-masks/sinbert-large-ft-ssc"
    XLMR_FT_SI = "surge-masks/xlmr-base-ft-ssc"
    LLAMA31_FT_SI = "surge-masks/llama-3.1-8b-ft-ssc"
    GEMMA2_FT_SI = "surge-masks/gemma-2-9b-ft-ssc"
    
    # MT5 models
    MT5_SMALL = "google/mt5-small"
    MT5_BASE = "google/mt5-base"
    MT5_LARGE = "google/mt5-large"
    MT5_XL = "google/mt5-xl"
    MT5_XXL = "google/mt5-xxl"
    
    # MBART models
    MBART50 = "facebook/mbart-large-50"
    
    # SinBERT models
    SINBERT_LARGE = "NLPC-UOM/SinBERT-large"
    SINBERT_SMALL = "NLPC-UOM/SinBERT-small"
    
    # XLMR models
    XLMR_BASE = "FacebookAI/xlm-roberta-base"
    XLMR_LARGE = "FacebookAI/xlm-roberta-large"
    
    # Llama 3.1 models
    LLAMA_31_8B = "meta-llama/Llama-3.1-8B"
    LLAMA_31_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Llama 3.2 models
    LLAMA_32_1B = "meta-llama/Llama-3.2-1B"
    LLAMA_32_1B_INSTRUCT = "meta-llama/Llama-3.2-1B-Instruct"
    LLAMA_32_3B = "meta-llama/Llama-3.2-3B"
    LLAMA_32_3B_INSTRUCT = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Gemma 2 models
    GEMMA_2_2B = "google/gemma-2-2b"
    GEMMA_2_2B_INSTRUCT = "google/gemma-2-2b-Instruct"
    GEMMA_2_9B = "google/gemma-2-9b"
    GEMMA_2_9B_INSTRUCT = "google/gemma-2-9b-Instruct"

    @staticmethod
    def getModelClass(modelEnum):
        modelClasses: Dict[ModelEnum, Type] = {
            ModelEnum.LLAMA_31_8B: Llama31BaseModel,
            ModelEnum.MBART50: Mbart50LargeModel,
            ModelEnum.MT5_BASE: Mt5BaseModel,
            ModelEnum.XLMR_BASE: XlmrBaseModel,
            ModelEnum.SINBERT_LARGE: SinBertLargeModel,
            # Add other model mappings here
            # You'll need to create appropriate model classes for each new model type
        }
        return modelClasses.get(modelEnum)
    
    @staticmethod
    def get_model_path(name: str):
        """Get the model path from the model name"""
        for model in ModelEnum:
            return model.value
        raise ValueError(f"Model {name} not found")
    
    @staticmethod
    def get_supported_models():
        """Return a list of all supported model names"""
        return [model.name.lower() for model in ModelEnum]