
import os
from Data.LMSpellDataset import LMSpellDataset
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    print("Spawn method already set, continuing...") 
    pass

def _train(*args):
    from Trainer.Seq2SeqTrainer import Seq2SeqTrainer
    # print(type(args), args)
    trainer = Seq2SeqTrainer(*args)
    trainer.train()

class MultiGPUTrainer:
    def __init__(
            self,
            model_instance,
            train_path,
            val_path,
            exp_name,
            batch_size = 8,
            epochs = 20,
            lr = 1e-5,
            dataset = LMSpellDataset,
            special_tokens_to_add = None,
            dataset_size = 1,
            num_warmup_steps = 1000,
            train_batch_size = 16,
            train_max_length = 128,
            zero_stage = 2,
            gradient_accumulation_steps = 1,
            patience = 3,
            lr_and_opt_path = None,
            resume_training_from = None,
            resume_training = False,
        ):
        self.model_instance = model_instance
        self.train_path = train_path
        self.val_path = val_path
        self.exp_name = exp_name
        self.exp_dir = os.path.join(os.getcwd(), self.exp_name)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dataset = dataset
        self.special_tokens_to_add = special_tokens_to_add
        self.dataset_size = dataset_size
        self.num_warmup_steps = num_warmup_steps
        self.train_batch_size = train_batch_size
        self.train_max_length = train_max_length
        self.zero_stage = zero_stage
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr_and_opt_path = lr_and_opt_path
        self.resume_training_from = resume_training_from
        self.resume_training = resume_training
        self.train_dataloader = None
        self.val_dataloader = None
        self.step_counter = 0
        self.patience = patience
    
    def train(self):
        from accelerate import notebook_launcher
        
        # notebook_launcher(self._train, num_processes=ConfigUtils().get("accelerator.NUM_PROCESSES", 1))
        notebook_launcher(
            _train, 
            args = (
                self.model_instance,
                self.train_path,
                self.val_path,
                self.exp_name,
                self.batch_size,
                self.epochs,
                self.lr,
                self.dataset,
                self.special_tokens_to_add,
                self.dataset_size,
                self.num_warmup_steps,
                self.train_batch_size,
                self.train_max_length,
                self.zero_stage,
                self.gradient_accumulation_steps,
                self.patience,
                self.lr_and_opt_path,
                self.resume_training_from,
                self.resume_training
                ), 
            num_processes=2, 
            mixed_precision = 'fp16')    
    
