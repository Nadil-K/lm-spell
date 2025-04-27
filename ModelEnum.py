from enum import Enum
from Models.ModelAbstract import ModelAbstract
from typing import Dict, Type


class ModelEnum(Enum):
    """
    Enum class for all the models supported by the 
    Neural Spell Checker Library
    """
    # Fine-tuned models
    LLAMA31_FT_SI = "surge-masks/llama-3.1-8b-ft-ssc"
    GEMMA2_FT_SI = "surge-masks/gemma-2-9b-ft-ssc"
    MT5_FT_SI = "surge-masks/mt5-base-ft-ssc"
    MT5_XL_FT_SI = "surge-masks/mt5-xl-ft-ssc"
    MBART_FT_SI = "surge-masks/mbart50-ft-ssc"
    SINBERT_FT_SI = "surge-masks/sinbert-large-ft-ssc"
    XLMR_FT_SI = "surge-masks/xlmr-base-ft-ssc"

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

    # MT5 models
    MT5_SMALL = "google/mt5-small"
    MT5_BASE = "google/mt5-base"
    MT5_LARGE = "google/mt5-large"
    MT5_XL = "google/mt5-xl"
    MT5_XXL = "google/mt5-xxl"
    
    # MBART models
    MBART50 = "facebook/mbart-large-50"
    
    # SinBERT models
    SINBERT_SMALL = "NLPC-UOM/SinBERT-small"
    SINBERT_LARGE = "NLPC-UOM/SinBERT-large"
    
    # XLMR models
    XLMR_BASE = "FacebookAI/xlm-roberta-base"
    XLMR_LARGE = "FacebookAI/xlm-roberta-large"

    def getModelClass(self) -> ModelAbstract:
        """Return the model class associated with the enum value"""

        from Models.Llama.LlamaAbstractModel import LlamaAbstractModel
        from Models.Llama.Llama318BModel import Llama318BModel

        from Models.Gemma.GemmaAbstractModel import GemmaAbstractModel
        from Models.Gemma.Gemma29BModel import Gemma29BModel

        from Models.Mt5.Mt5AbstractModel import Mt5AbstractModel
        from Models.Mt5.Mt5LargeModel import Mt5LargeModel

        from Models.Mbart.MbartAbstractModel import MbartAbstractModel
        from Models.SinBert.SinBertAbstractModel import SinBertAbstractModel
        from Models.Xlmr.XlmrAbstractModel import XlmrAbstractModel
        # import other model classes as needed

        modelClasses: Dict[ModelEnum, ModelAbstract] = {
            # Fine-tuned models
            ModelEnum.LLAMA31_FT_SI: LlamaAbstractModel,
            ModelEnum.GEMMA2_FT_SI: GemmaAbstractModel,
            ModelEnum.MT5_FT_SI: Mt5AbstractModel,
            ModelEnum.MT5_XL_FT_SI: Mt5AbstractModel,
            ModelEnum.MBART_FT_SI: MbartAbstractModel,
            ModelEnum.SINBERT_FT_SI: SinBertAbstractModel,
            ModelEnum.XLMR_FT_SI: XlmrAbstractModel,

            # Llama 3.1 models
            ModelEnum.LLAMA_31_8B: Llama318BModel,
            ModelEnum.LLAMA_31_8B_INSTRUCT: LlamaAbstractModel,

            # Llama 3.2 models
            ModelEnum.LLAMA_32_1B: LlamaAbstractModel,
            ModelEnum.LLAMA_32_1B_INSTRUCT: LlamaAbstractModel,
            ModelEnum.LLAMA_32_3B: LlamaAbstractModel,
            ModelEnum.LLAMA_32_3B_INSTRUCT: LlamaAbstractModel,

            # Gemma 2 models
            ModelEnum.GEMMA_2_2B: GemmaAbstractModel,
            ModelEnum.GEMMA_2_2B_INSTRUCT: GemmaAbstractModel,
            ModelEnum.GEMMA_2_9B: Gemma29BModel,
            ModelEnum.GEMMA_2_9B_INSTRUCT: GemmaAbstractModel,

            # mT5 models
            ModelEnum.MT5_SMALL: Mt5AbstractModel,
            ModelEnum.MT5_BASE: Mt5AbstractModel,
            ModelEnum.MT5_LARGE: Mt5LargeModel,
            ModelEnum.MT5_XL: Mt5AbstractModel,
            ModelEnum.MT5_XXL: Mt5AbstractModel,

            # mBART models
            ModelEnum.MBART50: MbartAbstractModel,

            # SinBERT models
            ModelEnum.SINBERT_SMALL: SinBertAbstractModel,
            ModelEnum.SINBERT_LARGE: SinBertAbstractModel,
            
            # XLMR models
            ModelEnum.XLMR_BASE: XlmrAbstractModel,
            ModelEnum.XLMR_LARGE: XlmrAbstractModel,

            # Add other model-class mappings here
        }
        return modelClasses.get(self)
    
    def get_model_path(self) -> str:
        """Return the model path"""
        return self.value
    
    @staticmethod
    def get_supported_models():
        """Return a list of all supported model names"""
        return [model.value for model in ModelEnum]