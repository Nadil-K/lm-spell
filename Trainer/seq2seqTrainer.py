import torch
from accelerate import Accelerator
from Utils.GeneralUtils import GeneralUtils
from Utils.DatasetUtils import DatasetUtils
from Trainer.Utils.EarlyStopping import EarlyStopping
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from tqdm import tqdm
from Trainer.Utils.LMSpellDataset import LMSpellDataset
from torch.utils.data import DataLoader
import os
from ModelEnum import ModelEnum
from Trainer.Utils.Metrics import Metrics

class Seq2SeqTrainer:

    def __init__(
            self,
            model_instance,
            train_path,
            val_path,
            test_path,
            batch_size = 8,
            epochs = 20,
            lr = 1e-5,
            accelerator = Accelerator(),
            dataset = LMSpellDataset,
            special_tokens_to_add = None,
            dataset_size = 1,
            num_warmup_steps = 1000,
            train_batch_size = 16,
            test_batch_size = 16,
            train_max_length = 128,
            test_max_length = 128,
            zero_stage = 2,
            gradient_accumulation_steps = 1,
            lr_and_opt_path = None,
            resume_training_from = None,
        ):
        self.model_instance = model_instance
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.accelerator = accelerator
        self.dataset = dataset
        self.special_tokens_to_add = special_tokens_to_add
        self.dataset_size = dataset_size
        self.num_warmup_steps = num_warmup_steps
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_max_length = train_max_length
        self.test_max_length = test_max_length
        self.zero_stage = zero_stage
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_and_opt_path = lr_and_opt_path
        self.resume_training_from = resume_training_from
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.step_counter = 0

    def train(self):
        GeneralUtils.clean_memory()

        self.initialize_model_and_tokenizer()
        starting_epoch = 0
        
        if self.resume_training:
            starting_epoch = self.resume_training_from
            self.accelerator.print(f"Resuming Training from Epoch {starting_epoch}")
        
        for epoch in range(starting_epoch, self.epochs + starting_epoch):
            self.epoch = epoch
            self.model_instance.model.train()
            total_loss = 0
            with tqdm(self.train_dataloader, leave=True, disable=not self.accelerator.is_local_main_process, mininterval=4) as pbar_train:
                for inputs in pbar_train:
                    total_loss += self.train_step(inputs)
                    self.step_counter += 1
                final_total_loss = self.accumulate_loss(total_loss, self.train_dataloader)        

            # Validation
            self.metrics = Metrics(self.val_dataloader, self.model_instance.exp_dir, ZWJ_Fix=(self.special_tokens_to_add is not None), model=self.model_instance.model_label)
            final_val_loss, last_prediction = self.validate()
            model_dir = self.save_model(epoch)
            
            # End of Validation Cycle
            if self.accelerator.is_main_process:
                training_time_tqdm = pbar_train.format_dict["elapsed"]
                self.metrics.end_epoch(epoch, final_val_loss, final_total_loss, training_time_tqdm//60)
                if self.early_stopping.step(final_total_loss, final_val_loss, model_dir):
                    self.early_stopping.summary()
                    self.accelerator.set_trigger()
            
            self.accelerator.wait_for_everyone() 
            if self.accelerator.check_trigger():
                    break

    def validate(self):
        self.accelerator.print("Validation Started")
        GeneralUtils.lean_memory()
        self.model_instance.model.eval()
        total_val_loss = 0
        with tqdm(self.val_dataloader, leave=True, disable=not self.accelerator.is_local_main_process) as pbar:
            for inputs in pbar:
                loss, prediction = self.validation_step(inputs)
                total_val_loss += loss
        
        final_val_loss = self.accumulate_loss(total_val_loss, self.val_dataloader)
        return final_val_loss, prediction
            
    def initialize_model_and_tokenizer(self):
        model = self.model_instance.model            
        tokenizer = self.model_instance.tokenizer

        if self.special_tokens_to_add is not None:
                tokenizer.add_special_tokens({'additional_special_tokens': self.special_tokens_to_add})
                model.resize_token_embeddings(len(tokenizer))
                self.accelerator.print("Added following special tokens:")
                for tok in self.special_tokens_to_add:
                     self.accelerator.print(f"Token: {tok}, ID: {tokenizer.convert_tokens_to_ids(tok)}")
            

        self.early_stopping = EarlyStopping(self.model_instance.exp_dir, self.epochs) #require to load the best model from the checkpoint
        self.criterion = CrossEntropyLoss()
        self.initialize_dataloader()
        self.optimizer = AdamW(model.parameters(), lr=self.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * self.epochs) // self.gradient_accumulation_steps
        )

        if self.lr_and_opt_path:
            self.accelerator.print(f"Loading Optimizer and Scheduler from {self.lr_and_opt_path}")
            
            optimizer_state = torch.load(os.path.join(self.model_instance.exp_dir, 'optimizer.pt'))
            scheduler_state = torch.load(os.path.join(self.model_instance.exp_dir, 'scheduler.pt'))
            self.lr_scheduler.load_state_dict(scheduler_state)
            self.optimizer.load_state_dict(optimizer_state)
        else:
            self.accelerator.print("Default Optimizer(Adaw) and Scheduler(Linear Scheduler with Warmup) is used")
        
        self.accelerator.register_for_checkpointing(self.lr_scheduler)
        
        if self.model_instance.model_label in [ModelEnum.MT5_BASE, ModelEnum.MT5_XL]:
            '''
            When launching an trainijng through the CLI and error occurs when the model is passed accellerator.prepare. 
            The error is fixed by making the layers contiguous. This error is not observed when the training is 
            launch through notebook launcher.
            '''
            for _, param in model.named_parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader, self.test_dataloader = self.accelerator.prepare(
            model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.val_dataloader, self.test_dataloader
        )

    def initialize_dataloader(self):
        tokenizer = self.model_instance.tokenizer
        dataset = DatasetUtils(train_path=self.train_path, val_path=self.val_path, test_path=self.test_path, dataset_size=self.dataset_size)

        self.accelerator.print("Initializing Dataloaders")
        self.accelerator.print("Train dataset size:", dataset.train_dataset.shape)
        self.accelerator.print("Validation dataset size:", dataset.val_dataset.shape)
        self.accelerator.print("Test dataset size:", dataset.test_dataset.shape)
        
        train_dataset = self.dataset(dataset.train_dataset, tokenizer, self.train_max_length)
        val_dataset = self.dataset(dataset.val_dataset, tokenizer, self.train_max_length)

        # causes an issue with decoding OverflowError: out of range integral type conversion attempted
        # data_collator_train = DataCollatorForSeq2Seq(
        #     tokenizer=tokenizer,
        #     padding ="longest",
        #     max_length = self.train_max_length,
        #     pad_to_multiple_of=8,
        #     label_pad_token_id = tokenizer.pad_token_id
        # )
        # val_collator_test = DataCollatorForSeq2Seq(
        #     tokenizer=tokenizer,
        #     padding ="max_length",
        #     max_length = self.train_max_length,
        #     pad_to_multiple_of=8,
        #     label_pad_token_id = tokenizer.pad_token_id
        # )      
        data_collator_test = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding = "longest", # TODO: Chnage to max_length
            max_length = self.test_max_length,
            pad_to_multiple_of=8,
            label_pad_token_id = tokenizer.pad_token_id
        )

        train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=False, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.train_batch_size, shuffle=False, pin_memory=True)

        test_dataset = self.dataset(dataset.test_dataset, tokenizer, self.test_max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=self.test_batch_size, collate_fn=data_collator_test, shuffle=False, pin_memory=True)
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def train_step(self, inputs, past_key_values = None):
        with self.accelerator.accumulate(self.model):
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
            loss = outputs.loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            return loss.detach().item()

    def validation_step(self, inputs, post_processing = None):
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        original = inputs['input_ids']
        labels = inputs['labels']
        loss = outputs.loss
        gathered_original, gathered_predictions, gathered_labels = self.accelerator.gather_for_metrics((original, predictions, labels))
        self.accelerator.wait_for_everyone() 
        if self.accelerator.is_main_process:
            if post_processing:
                gathered_original, gathered_predictions, gathered_labels = post_processing(gathered_original, gathered_predictions, gathered_labels)
            self.metrics.evaluate(gathered_original, gathered_predictions, gathered_labels, self.model_instance.tokenizer, self.accelerator.device, self.epoch)
        return loss.detach().item(), predictions
    
    def accumulate_loss(self, total_loss, dataloader):
        total_loss_t = torch.tensor(total_loss / len(dataloader)).to(self.accelerator.device)
        gathered_loss = self.accelerator.gather(total_loss_t)
        self.accelerator.wait_for_everyone()
        final_total_loss = gathered_loss.mean()
        self.accelerator.print(f'loss:{final_total_loss:.6f}')
        return final_total_loss
    
    def save_model(self, epoch):
        from Utils.ConfigUtils import ConfigUtils

        self.accelerator.print("Saving Model")
        model_dir = f'{self.model_instance.exp_dir}/epoch_{epoch + 1}'
        
        os.makedirs(model_dir, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        config = ConfigUtils()
        hf_token = config.get("huggingface.TOKEN")

        if self.zero_stage <= 2:
            unwrapped_model.save_pretrained(
                model_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                token = hf_token,
            )
        elif self.zero_stage == 3:
            unwrapped_model.save_pretrained(
                model_dir,
                is_main_process=self.accelerator.is_main_process,
                save_function=self.accelerator.save,
                state_dict=self.accelerator.get_state_dict(self.model),
                token = hf_token,
            )
        else:
            if self.accelerator.is_main_process:
                raise AssertionError(f"Unsupported zero stage: {self.zero_stage}")
        
        self.accelerator.save(self.optimizer.state_dict(), os.path.join(self.model_instance.exp_dir, f'optimizer.pt'))
        self.accelerator.save(self.lr_scheduler.state_dict(), os.path.join(self.model_instance.exp_dir, f'scheduler.pt'))
        self.accelerator.print("Model Saved along with Optimizer and Scheduler")
        if self.accelerator.scaler is not None:
            self.accelerator.print("Saving Scaler State")
            self.accelerator.save(self.accelerator.scaler.state_dict(), os.path.join(self.model_instance.exp_dir, f'scaler.pt'))
        return model_dir if self.accelerator.is_main_process else None 
