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

    def process_input(self, input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str = None):
        """
        Process the input set and target set. The input set should be a string, a list of strings or a DataFrame.
        The target set should be a string or a list of strings.
        """

        dataset_col_names = ConfigUtils.get_dataset_columns()

        if isinstance(input_set, pd.DataFrame):
            return input_set, True
            
        evaluate_flag = False
        data = []
        
        if isinstance(input_set, str):

            text = input_set
            evaluate_flag = target_set is not None and isinstance(target_set, str)
            expected = target_set if evaluate_flag else ""
            data = [{dataset_col_names[0]: text, dataset_col_names[1]: expected}]
            
        elif isinstance(input_set, list) and isinstance(input_set[0], str):

            evaluate_flag = (target_set is not None and 
                               isinstance(target_set, list) and 
                               len(target_set) == len(input_set))
            
            data = [{dataset_col_names[0]: s.strip(), dataset_col_names[1]: t.strip()} for s, t in zip(input_set, target_set)] if evaluate_flag else [{dataset_col_names[0]: s.strip(), dataset_col_names[1]: ""} for s in input_set]

        input_set = pd.DataFrame(data)
        return input_set, evaluate_flag


    def correct(self, max_length: int, batch_size: int, shuffle: bool, input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str | pd.DataFrame = None):
        """
        Corrects the text. The input set should be a string, a list of strings or a DataFrame.
        Perform the evaluation if the target set is provided.
        """

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
            print("Evaluating the outputs...")
            EvaluateUtils.evaluate_from_dataframe(results_df, self.exp_dir)


    def decode(self, originals, predictions, labels):
        """
        Decode a tensor of tensors and save it to a file.
        """

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

        result_col_names = ConfigUtils.get_results_columns()

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

        PrePostProcessingUtils.save_dataframe(results_df, self.exp_dir)

        return results_df

    def correctFromFile(self, max_length: int, batch_size: int, shuffle: bool, src: str, target: str = None):
        """
        Corrects the text from a file. The file should be in the format of
        """

        if src.endswith('.csv'):
            src = pd.read_csv(src)

        elif src.endswith('.txt'):
            with open(src, 'r', encoding='utf-8') as file:
                src = file.readlines()
            if target is not None:
                with open(target, 'r', encoding='utf-8') as file:
                    target = file.readlines()
        else:
            raise NeuralSpellCheckerException("Unsupported file format. Only .csv and .txt are supported.") from None
                
        return self.correct(max_length, batch_size, shuffle, src, target)