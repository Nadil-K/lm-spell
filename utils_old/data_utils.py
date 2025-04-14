def read_and_clean_data(args=None, dataset_size=1, data_dict=None):
    datase_size = dataset_size
    if args is None:
        class Args:
            train_path = 'train.csv'
            val_path = 'val.csv'
            test_path = 'test.csv'
            dataset_size = datase_size
            seed = 42
            ZWJ_Fix = False
        args = Args()
    
    print(data_dict is None)
    download() if data_dict is None else download(data_dict)
    if 0 < args.dataset_size <= 1:
        # train = pd.read_csv(args.train_path, skiprows=range(1, 25000), nrows=25000).dropna()
        train = pd.read_csv(args.train_path).dropna()
        val = pd.read_csv(args.val_path).dropna()
        test = pd.read_csv(args.test_path).dropna()
    else:
        print(f"Dataset Size: {dataset_size}. Secret mode activated. Testing using 100 samples for train, 50 for val and 20 for test.") 
        train = pd.read_csv(args.train_path, nrows=100).dropna()
        val = pd.read_csv(args.val_path, nrows=50).dropna()
        test = pd.read_csv(args.test_path, nrows=20).dropna()
        
    required_columns = ['text', 'expected']
    for col in required_columns:
        if col not in train.columns or col not in val.columns or col not in test.columns:
            raise ValueError(f"Column '{col}' is missing in one of the datasets.")
    
    if args.dataset_size < 1:
        print(f"Reducing dataset size to {args.dataset_size*100}%")
        train = train.sample(frac=args.dataset_size, random_state=args.seed).reset_index(drop=True)
        # val = val.sample(frac=max(0.1, dataset_size)).reset_index(drop=True)
        # test = test.sample(frac=max(0.5, dataset_size)).reset_index(drop=True)
    print(f"Train Size: {train.shape}")
    print(f"Val Size: {val.shape}")
    print(f"Test Size: {test.shape}")
    if args.ZWJ_Fix:
        train['text'] = train['text'].str.replace('\u200D', '<ZWJ>')
        train['expected'] = train['expected'].str.replace('\u200D', '<ZWJ>')

        val['text'] = val['text'].str.replace('\u200D', '<ZWJ>')
        val['expected'] = val['expected'].str.replace('\u200D', '<ZWJ>')

        test['text'] = test['text'].str.replace('\u200D', '<ZWJ>')
        test['expected'] = test['expected'].str.replace('\u200D', '<ZWJ>')
    
    return train, val, test

def load_data_from_hf(file_path, dataset_size=1):
    from google_utils import download
    if isinstance(file_path, str):
        file_path = [file_path]
    for path in file_path:
        pass

import torch
def remove_special_tokens(tokens, all_special_ids, special_token_id_to_keep_tensor):
    # print(f"Removing Special Tokens Except for {special_token_id_to_keep_tensor}")
    tokens_tensor = torch.tensor(tokens.clone().detach(), dtype=torch.int64)
    mask = (tokens_tensor == special_token_id_to_keep_tensor) | (~torch.isin(tokens_tensor, all_special_ids))
    filtered_tokens = tokens_tensor[mask]
    return filtered_tokens.tolist()