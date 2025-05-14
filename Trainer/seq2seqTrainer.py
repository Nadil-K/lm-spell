import torch
from accelerate import Accelerator
from Utils.GeneralUtils import GeneralUtils
from Utils.DatasetUtils import DatasetUtils
from EarlyStopping import EarlyStopping
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from tqdm import tqdm
from LMSpellDataset import LMSpellDataset
from torch.utils.data import DataLoader
import os
from ModelEnum import ModelEnum

class Seq2SeqTrainer:

    def __init__(
            self,
            model_instance,
            train_path,
            val_path,
            test_path,
            batch_size = 8,
            epochs = 20,
            accelerator = Accelerator(),
            dataset = LMSpellDataset,
            special_tokens_to_add = None,
            dataset_size = 1,
            train_batch_size = 16,
            test_batch_size = 16,
            train_max_length = 128,
            test_max_length = 128,
            gradient_accumulation_steps = 1,
            lr_and_opt_path = None,
        ):
        self.model_instance = model_instance
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.accelerator = accelerator
        self.dataset = dataset
        self.special_tokens_to_add = special_tokens_to_add
        self.dataset_size = dataset_size
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_max_length = train_max_length
        self.test_max_length = test_max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_and_opt_path = lr_and_opt_path

    def train(self):
        GeneralUtils.clean_memory()

        self.initialize_model_and_tokenizer()
        starting_epoch = 0
        
        if self.args.resume_training:
            starting_epoch = self.args.resume_training_from
            self.accelerator.print(f"Resuming Training from Epoch {starting_epoch}")
        
        for epoch in range(starting_epoch, self.args.epochs + starting_epoch):
            self.epoch = epoch
            self.model.train()
            total_loss = 0
            with tqdm(self.train_dataloader, leave=True, disable=not self.accelerator.is_local_main_process, mininterval=4) as pbar_train:
                for inputs in pbar_train:
                    total_loss += self.train_step(inputs)
                    if self.accelerator.is_main_process and self.step_counter % 100 == 0: 
                    self.step_counter += 1
                final_total_loss = self.accumulate_loss(total_loss, self.train_dataloader)        

            # Validation
            self.metrics = Metrics(self.val_dataloader, self.args.exp_dir, ZWJ_Fix=(self.special_tokens_to_add is not None), model=self.args.model)
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
        self.optimizer = AdamW(model.parameters(), lr=self.args.lr)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
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

        test_dataset = self.dataset(self.test_set, self.tokenizer, self.test_max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=self.test_batch_size, collate_fn=data_collator_test, shuffle=False, pin_memory=True)
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader