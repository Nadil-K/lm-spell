import numpy as np
import mlflow
import os
from utils.file_utils import find_latest_epoch_folder, delete_folder

# TODO - IMPLEMET TORCH EARLY STOPPING
class EarlyStopping():
    def __init__(self, args, best_loss=np.inf):
        self.args = args
        self.exp_dir = args.exp_dir
        self.best_loss = best_loss
        self.best_model = None
        self.total_epochs = args.epochs
        self.epoch = 1
        self.best_epoch = 1
        self.no_improvement = 0
        self.patience = args.patience
        self.history = {'train_loss': [], 'val_loss': []}
        self.exp_no = args.exp_no

    def step(self, train_loss, val_loss, model):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.epoch
            self.best_model = model
            self.no_improvement = 0
        else:
            self.no_improvement += 1
            
        self.cleanup()
        if self.no_improvement >= self.patience or self.epoch >= self.total_epochs:
            return True  # Stop training
        else:
            self.epoch += 1
            return False
    
    def cleanup(self):
        if (self.epoch - self.patience - 1) > 0:
            pre_epoch_folder = os.path.join(self.exp_dir, f'epoch_{self.epoch - self.patience - 1}')
            delete_folder(pre_epoch_folder)
        
    
    def summary(self):
        for epoch in range(1, self.epoch + 1):
            if epoch != self.best_epoch:
                epoch_folder = os.path.join(self.exp_dir, f'epoch_{epoch}')
                delete_folder(epoch_folder)
        
        summary = "Stopped at epoch {}, best epoch was {}, best model was {} ".format(self.epoch, self.best_epoch, self.best_model)
        
        mlflow.log_param("summary", summary)
        print(summary)
        
    def get_best_model(self):
        if self.best_model is not None: 
            print(f"Latest epoch folder path found: {self.best_model}")
            return self.best_model        
        return find_latest_epoch_folder(f'exp_{self.exp_no}')