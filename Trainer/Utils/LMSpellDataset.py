from torch.utils.data import Dataset

class LMSpellDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        inputs = self.tokenizer(item['text'], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        targets = self.tokenizer(item['expected'], return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        inputs['labels'] = targets['input_ids']
        return {key: val.squeeze() for key, val in inputs.items()}
    
    def get(self):
        return len(self.dataframe)
