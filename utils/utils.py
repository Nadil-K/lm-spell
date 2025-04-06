from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def clean_memory():
    import gc
    import torch
    
    # Collect garbage
    gc.collect()

    # Empty the CUDA cache
    torch.cuda.empty_cache()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_in_tab_seperated_format(df):
    filtered_df = df[df['Without Replacement'].astype(str).str.contains(r"\.")]
    row_data = "\t".join(
        f"{float(without_replacement):.2f}" if float(without_replacement) == float(with_replacement) 
        else f"{float(without_replacement):.2f}({float(with_replacement):.2f})"
        for without_replacement, with_replacement 
        in zip(filtered_df['Without Replacement'], filtered_df['With Replacement'])
    )

    print(row_data)

import argparse
def str2bool(v):
    """The type=bool argument take any non-empty string is considered True. Hence a custom function is used to parse boolean arguments.

    Args:
        v (string/Bool): The string to be converted to a boolean.

    Raises:
        argparse.ArgumentTypeError: If the string is not a valid boolean value.

    Returns:
        bool: The boolean value of the string.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
