from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, get_linear_schedule_with_warmup
import os
import logging
from tqdm import tqdm
from trainer import Trainer

class CLMTrainer(Trainer):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        train_dataset,
        test_dataset,
        val_dataset,
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        max_length: int = 512,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(len(train_dataset) // batch_size // gradient_accumulation_steps) * epochs
        )
        criterion = torch.nn.CrossEntropyLoss()
        early_stopping = None  # Can be implemented if needed
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopping=early_stopping,
            criterion=criterion,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            max_length=max_length
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.model.to(self.device)
        self.best_val_loss = float('inf')
        self.logger = logging.getLogger(__name__)
    
    def _prepare_dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True
        )
    
    def _process_batch(self, batch):
        inputs = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Shift labels for language modeling (input becomes the target for next token prediction)
        labels = inputs.clone()
        
        outputs = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss
    
    def train(self):
        self.model.train()
        train_dataloader = self._prepare_dataloader(self.train_dataset)
        val_dataloader = self._prepare_dataloader(self.val_dataset, shuffle=False)
        
        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            train_steps = 0
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs} [Training]")
            for step, batch in enumerate(pbar):
                loss = self._process_batch(batch)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                total_train_loss += loss.item()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    train_steps += 1
                
                pbar.set_postfix({"loss": total_train_loss / (train_steps + 1e-10)})
            
            avg_train_loss = total_train_loss / (train_steps + 1e-10)
            self.logger.info(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
            
            # Validate after each epoch
            val_loss = self._validate(val_dataloader)
            self.logger.info(f"Epoch {epoch+1} - Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.logger.info(f"New best validation loss: {val_loss:.4f}")
                # Save best model if save path is provided
    
    def _validate(self, dataloader):
        self.model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for batch in pbar:
                loss = self._process_batch(batch)
                total_val_loss += loss.item()
                val_steps += 1
                pbar.set_postfix({"loss": total_val_loss / val_steps})
        
        return total_val_loss / val_steps
    
    def test(self):
        test_dataloader = self._prepare_dataloader(self.test_dataset, shuffle=False)
        self.model.eval()
        total_test_loss = 0
        test_steps = 0
        
        with torch.no_grad():
            pbar = tqdm(test_dataloader, desc="Testing")
            for batch in pbar:
                loss = self._process_batch(batch)
                total_test_loss += loss.item()
                test_steps += 1
                pbar.set_postfix({"loss": total_test_loss / test_steps})
        
        avg_test_loss = total_test_loss / test_steps
        self.logger.info(f"Test loss: {avg_test_loss:.4f}")
        return avg_test_loss
    
    def save_model(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def save_tokenizer(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info(f"Tokenizer saved to {save_path}")
    
    def save_optimizer(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        self.logger.info(f"Optimizer saved to {save_path}")
    
    def save_scheduler(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
        self.logger.info(f"Scheduler saved to {save_path}")
    
    def generate(self, prompt, max_new_tokens=50, **kwargs):
        """Generate text from a prompt"""
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)