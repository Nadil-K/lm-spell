from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from Trainer.trainer import Trainer
from utils.utils import clean_memory
from tqdm import tqdm

class Seq2SeqTrainer(Trainer):
    def __init__(
        self, 
        model: AutoModel, 
        tokenizer: AutoTokenizer,
        optimizer,
        scheduler,
        early_stopping,
        criterion,
        train_dataset,
        test_dataset,
        val_dataset,
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        max_length: int = 512,
    ):
        super().__init__(model, 
                         tokenizer,
                         optimizer,
                         scheduler, 
                        early_stopping,
                        criterion,
                        train_dataset,
                        test_dataset,
                        val_dataset,
                         batch_size, 
                         epochs, 
                         learning_rate, 
                         max_length
                         )
    
    def _train_step(self, inputs):
        with self.accelerator.accumulate(self.model):
            outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
            # loss = self.loss_fn(outputs.logits, inputs['labels'])
            loss = outputs.loss
            self.accelerator.backward(loss)
            # accelerator.clip_grad_norm_(model.parameters(), 1.0) # Adding Clip Grad Norm
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            return loss.detach().item()
    
    
    def _accumulate_loss(self, total_loss, dataloader):
        total_loss_t = torch.tensor(total_loss / len(dataloader)).to(self.accelerator.device)
        gathered_loss = self.accelerator.gather(total_loss_t)
        self.accelerator.wait_for_everyone()
        final_total_loss = gathered_loss.mean()
        self.accelerator.print(f'loss:{final_total_loss:.6f}')
        return final_total_loss
    
    def _training_function(self):
        from accelerate import Accelerator
        clean_memory()
        self.accelerator = Accelerator()
        self.accelerator.print("Trainable Parameters:", f"{self.model.num_parameters(only_trainable = True)//1e6} Million" )
        self.accelerator.print("Model Memory Footprint:", f"{self.model.get_memory_footprint()/(1024*1024*1024) :.2f} GB")
        self.accelerator.print("Memory Used:", f"{torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_loader)
        self.model.train()
        for epoch in range(self.epochs):
            loss = 0
            with tqdm(self.train_dataloader, leave=True, disable=not self.accelerator.is_local_main_process, mininterval=4) as pbar_train:
                for inputs in pbar_train:
                    loss += self._train_step(inputs)
                total_loss = self._accumulate_loss(total_loss, self.train_dataloader)        
            print(f"Epoch {epoch + 1}/{self.epochs} completed.")

    def train(self, num_processes=1):
        from accelerate import notebook_launcher
        notebook_launcher(self._training_function, num_processes=num_processes)

    def test(self):
        self.model.eval()
        test_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = self.tokenizer(batch['input_text'], return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                labels = self.tokenizer(batch['target_text'], return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
                outputs = self.model(**inputs, labels=labels['input_ids'])
                loss = outputs.loss
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        print(f"Test loss: {avg_loss}")

    def _save_model(self, save_path: str):
        self.accelerator.print("Saving Model")
        model_dir = f'outputs/epoch_{self.epoch + 1}'

    def save_tokenizer(self, save_path: str):
        self.tokenizer.save_pretrained(save_path)
        print(f"Tokenizer saved to {save_path}")

    def save_optimizer(self, save_path: str):
        torch.save(self.optimizer.state_dict(), save_path)
        print(f"Optimizer state saved to {save_path}")

    def save_scheduler(self, save_path: str):
        torch.save(self.scheduler.state_dict(), save_path)
        print(f"Scheduler state saved to {save_path}")