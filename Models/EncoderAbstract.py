from Models.ModelAbstract import ModelAbstract
from transformers import DataCollatorForSeq2Seq
from Trainer.Utils.Metrics import Metrics
from Datasets.LMSpellDataset import LMSpellDataset
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
import torch
from Utils.GeneralUtils import GeneralUtils

class EncoderAbstract(ModelAbstract):

    def correct(self, input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str | pd.DataFrame = None, max_length: int = 128, batch_size: int = 8):
        if max_length > 514:
            print("Warning: max_length is set to 514, as it is the maximum length for XLMR models.")
            max_length = 514
        input_set, evaluate_flag = self.process_input(input_set, target_set)

        dataset = LMSpellDataset(input_set, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.model = self.accelerator.prepare_model(self.model, evaluation_mode=True)
        dataloader = self.accelerator.prepare_data_loader(dataloader)
        self.model.resize_token_embeddings(len(self.tokenizer))

        GeneralUtils.clean_memory()
        self.model.eval()

        if self.accelerator.is_main_process: self.metrics = Metrics(dataloader, self.exp_dir, model=self.model)
        eval_loss = 0

        with tqdm(dataloader, leave=True, disable=not self.accelerator.is_local_main_process) as pbar:
            for batch in pbar:
                with torch.no_grad():
                    outputs = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
                predictions = torch.argmax(outputs.logits, dim=-1)
                loss = outputs.loss.mean().item()
                eval_loss += loss
                
                original = batch['input_ids']
                labels = batch['labels']
                gathered_original, gathered_predictions, gathered_labels = self.accelerator.gather_for_metrics((original, predictions, labels))    
                self.accelerator.wait_for_everyone() 
                if evaluate_flag and self.accelerator.is_main_process:
                    self.metrics.evaluate(gathered_original, gathered_predictions, gathered_labels, self.tokenizer, self.accelerator.device, "Test")
                    
        final_loss = self.accumulate_loss(eval_loss, dataloader)
        if evaluate_flag and self.accelerator.is_main_process: self.metrics.log_test(final_loss, self.tokenizer)
    
