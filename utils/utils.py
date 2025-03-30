def clean_memory():
    import gc
    import torch
    
    # Collect garbage
    gc.collect()

    # Empty the CUDA cache
    torch.cuda.empty_cache()
