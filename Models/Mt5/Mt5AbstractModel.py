import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from Utils.ConfigUtils import ConfigUtils
from Utils.EvaluateUtils import EvaluateUtils
from Datasets.Seq2SeqDataset import Seq2SeqDataset
from Utils.PrePostProcessingUtils import PrePostProcessingUtils
from Models.EncoderDecoderAbstract import EncoderDecoderAbstract
from NeuralSpellCheckerException import NeuralSpellCheckerException

class Mt5AbstractModel(EncoderDecoderAbstract):

    def process_input(self, input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str | pd.DataFrame = None):
        """
        Process the input set and target set. The input set should be a list of strings or a DataFrame.
        The target set should be a list of strings or a DataFrame.
        """
        evaluate_flag = True
        if isinstance(input_set, str):
            if target_set is not None and isinstance(target_set, str):
                input_set = pd.DataFrame([{"text": input_set, "expected": target_set}])
            else:  
                input_set = pd.DataFrame([{"text": input_set, "expected": ""}])
                evaluate_flag = False
        elif isinstance(input_set, list) and isinstance(input_set[0], str):
            if target_set is not None and isinstance(target_set, list) and len(target_set) == len(input_set):
                input_set = pd.DataFrame([{"text": s, "expected": t} for s, t in zip(input_set, target_set)])
            else:
                input_set = pd.DataFrame([{"text": s, "expected": ""} for s in input_set])
                evaluate_flag = False

        return input_set, evaluate_flag

    def correct(self, max_length: int, batch_size: int, shuffle: bool, input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str | pd.DataFrame = None):
        input_set, evaluate_flag = self.process_input(input_set, target_set)

        dataset = Seq2SeqDataset(input_set, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        self.model.eval()
        originals, predictions, labels = [], [], []

        with tqdm(dataloader, leave=True) as pbar:
            for batch in pbar:
                with torch.no_grad():
                    outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                original = batch["input_ids"]
                prediction = torch.argmax(outputs.logits, dim=-1)
                label = batch.get("labels") #this is only present when we are testing. its not there when we are predicting better to have a mechanism to prevent this
                if batch.get("labels") is not None:
                    labels.extend(label)
                originals.extend(original),  predictions.extend(prediction)

        results_df = self.decode(originals, predictions, labels)
        if evaluate_flag:
            EvaluateUtils.evaluate_from_dataframe(results_df, self.exp_dir)
        return None


    def decode(self, originals, predictions, labels):
        """
        Decode a tensor of tensors and save it to a file. Multiple examples
        """
        # May be we can use it to decode single example as well.

        skip_special_tokens = False

        # This is different in mbart model. Same for xlmr, sinbert and mt5. We don't need to do this in large models.
        special_token_id_to_keep = self.tokenizer.convert_tokens_to_ids(PrePostProcessingUtils.ZWJ)
        all_special_ids = torch.tensor(self.tokenizer.all_special_ids, dtype=torch.int64).to(self.device)

        special_token_tensor = torch.tensor([special_token_id_to_keep], dtype=torch.int64).to(self.device)

        originals, predictions, labels = [
            [PrePostProcessingUtils.remove_special_tokens(tokens.to(self.device), all_special_ids, special_token_tensor) for tokens in data]
            for data in (originals, predictions, labels)
        ]
        
        originals_decoded, predictions_decoded, labels_decoded = [
            [self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens).replace('\n', '').strip() for ids in data]
            for data in (originals, predictions, labels)
        ]

        config = ConfigUtils()
        result_col_names = [
            config.get('results.original', 'Original'),
            config.get('results.predicted', 'Corrected'),
            config.get('results.expected', 'Expected')
        ]

        results_data = {
            result_col_names[0]: originals_decoded,
            result_col_names[1]: predictions_decoded,
            result_col_names[2]: labels_decoded,
        }

        # Write CSV inside the main process condition
        results_df = pd.DataFrame(results_data)

        # When using BERT models, these character were present in the output. They throws errors when saving in excel format
        results_df = results_df.apply(lambda col: col.map(lambda x: x.replace('.\uffff', '').replace('\x11', '') if isinstance(x, str) else x))

        print("Replacing ZWJ token")
        for column in result_col_names:
            results_df[column] = PrePostProcessingUtils.clean_zwj(results_df[column])

        # TO-DO: Evaluate the model and save the results

        PrePostProcessingUtils.save_dataframe(results_df, self.exp_dir)

        return results_df

    def correctFromFile(self, max_length: int, batch_size: int, shuffle: bool, src: str, target: str = None):
        """
        Corrects the text from a file. The file should be in the format of
        """
        if src.endswith('.csv'):
            input_set = pd.read_csv(src)
        elif src.endswith('.txt'):
            with open(src, 'r', encoding='utf-8') as file:
                src_lines = file.readlines()
            if target is not None:
                with open(target, 'r', encoding='utf-8') as file:
                    target_lines = file.readlines()
                if len(src_lines) != len(target_lines):
                    raise NeuralSpellCheckerException("Source and target files must have the same number of lines.") from None
                input_set = pd.DataFrame([{"text": src_line.strip(), "expected": target_line.strip()} for src_line, target_line in zip(src_lines, target_lines)])
            else:
                input_set = pd.DataFrame([{"text": src_line.strip(), "expected": ""} for src_line in src_lines])
        else:
            raise NeuralSpellCheckerException("Unsupported file format. Only .csv and .txt are supported.") from None
                
        return self.correct(max_length, batch_size, shuffle, input_set)