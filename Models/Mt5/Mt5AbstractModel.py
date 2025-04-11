from Models.EncoderDecoderAbstract import EncoderDecoderAbstract
import torch
from utils.data_utils import remove_special_tokens
import pandas as pd
from DataSets import Seq2SeqDataset
from torch.utils.data import DataLoader
import tqdm

class Mt5AbstractModel(EncoderDecoderAbstract):
    def predict(self, input_set: list[str] | str, max_length: int, batch_size: int, shuffle: bool):
        if isinstance(input_set, str):
            input_set = [input_set]
            batch_size = 1

        dataset = Seq2SeqDataset(input_set, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.model.eval()
        originals, predictions, labels = [], [], []
        with tqdm(dataloader, leave=True) as pbar:
            for batch in pbar:
                with torch.no_grad():
                    outputs = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
                original = batch['input_ids']
                prediction = torch.argmax(outputs.logits, dim=-1)
                label = batch.get('labels') #this is only present when we are testing. its not there when we are predicting better to have a mechanism to prevent this
                if batch.get('labels') is not None:
                    labels.extend(label)
                originals.extend(original),  predictions.extend(prediction)


        self.decode(originals, predictions, labels)
    
    def decode(self, originals, predictions, labels):
        """
        Decode a tensor of tensors and save it to a file. Multiple examples
        """
        # May be we can use it to decode single example as well.
        special_token_to_keep = "<ZWJ>"
        skip_special_tokens = False
        special_token_id_to_keep = self.tokenizer.convert_tokens_to_ids(special_token_to_keep) # This is different in mbart model. Same for xlmr, sinbert and mt5. We don't need to do this in large models. 
        all_special_ids = torch.tensor(self.tokenizer.all_special_ids, dtype=torch.int64).to(self.device) # Need to add self.device to the model abstract
        special_token_id_to_keep_tensor = torch.tensor([special_token_id_to_keep], dtype=torch.int64).to(self.device)

        originals = [remove_special_tokens(tokens.to(self.device), all_special_ids, special_token_id_to_keep_tensor) for tokens in originals]
        predictions = [remove_special_tokens(tokens.to(self.device), all_special_ids, special_token_id_to_keep_tensor) for tokens in predictions]
        labels = [remove_special_tokens(tokens.to(self.device), all_special_ids, special_token_id_to_keep_tensor) for tokens in labels]
        
        originals_decoded = [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '').strip() for ids in originals]
        predictions_decoded = [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '').strip() for ids in predictions] 
        labels_decoded = [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '').strip() for ids in labels]

        results_data = {
            'Original': originals_decoded,
            'Corrected': predictions_decoded,
            'Expected': labels_decoded,
        }
        # Write CSV inside the main process condition
        results_df = pd.DataFrame(results_data)
        
        # When Using Bert models these character were present in the output. They throws errors when saving in excel format
        results_df = results_df.applymap(lambda x: x.replace('.\uffff', '').replace('\x11', '') if isinstance(x, str) else x)

        print("Replacing ZWJ token")
        # for whatever reason the ZWJ token is decoded with a additional space in the end.
        results_df['Original'] = results_df['Original'].replace('<ZWJ> ', '\u200D', regex=True)
        results_df['Corrected'] = results_df['Corrected'].replace('<ZWJ> ', '\u200D', regex=True)
        results_df['Expected'] = results_df['Expected'].replace('<ZWJ> ', '\u200D', regex=True)
        results_df['Original'] = results_df['Original'].replace('<ZWJ>', '\u200D', regex=True)
        results_df['Corrected'] = results_df['Corrected'].replace('<ZWJ>', '\u200D', regex=True)
        results_df['Expected'] = results_df['Expected'].replace('<ZWJ>', '\u200D', regex=True)

        print(results_df.head(3))
        results_df.to_excel(f'{self.exp_dir}/test_results.xlsx', index=False)
        results_df.to_csv(f'{self.exp_dir}/test_results.csv', index=False)