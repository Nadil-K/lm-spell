from torch.utils.data import Dataset
from Utils.ConfigUtils import ConfigUtils

class SeqSeqDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        inputs = self.tokenizer(item[self.get_column_name('inputs', 'text')], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        targets = self.tokenizer(item[self.get_column_name('targets', 'expected')], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        inputs['labels'] = targets['input_ids']
        return {key: val.squeeze() for key, val in inputs.items()}
    
    def get(self):
        return len(self.dataframe)

    @staticmethod
    def get_column_name(column: str, default: str = None):
        config = ConfigUtils()
        return config.get(f"dataset.seq2seq.{column}", default)
        
