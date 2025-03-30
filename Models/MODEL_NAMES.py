MODEL_NAMES = {
    "mt5-ft": "surge-masks/mt5-base-ft-ssc",
    "mt5-xl-ft": "surge-masks/mt5-xl-ft-ssc",
    "mbart-ft": "surge-masks/mbart50-ft-ssc",
    "sinbert-ft": "surge-masks/sinbert-large-ft-ssc",
    "xlmr-ft": "surge-masks/xlmr-base-ft-ssc", 
    "llama3.1-ft": "surge-masks/llama-3.1-8b-ft-ssc",
    "gemma2-ft": "surge-masks/gemma-2-9b-ft-ssc",
    
    "mt5-small": "google/mt5-small",
    "mt5-base": "google/mt5-base",
    "mt5-large": "google/mt5-large",
    "mt5-xl": "google/mt5-xl",
    "mt5-xxl": "google/mt5-xxl",
    
    "mbart50": "facebook/mbart-large-50",
    
    "sinbert-large": "NLPC-UOM/SinBERT-large",
    "sinbert-small": "NLPC-UOM/SinBERT-small",
    
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.1-8b-i" : "meta-llama/Llama-3.1-8B-Instruct",
    
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama-3.2-1b-i": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.2-3b-i": "meta-llama/Llama-3.2-3B-Instruct",
    
    "gemma-2-2b": "google/gemma-2-2b",
    "gemma-2-2b-i": "google/gemma-2-2b-Instruct",
    "gemma-2-9b": "google/gemma-2-9b",
    "gemma-2-9b-i": "google/gemma-2-9b-Instruct",
}

def get_model_names():
    return MODEL_NAMES