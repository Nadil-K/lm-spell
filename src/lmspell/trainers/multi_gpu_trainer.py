"""Multi-GPU training from a notebook (no `accelerate launch` command needed).

Accelerate's `notebook_launcher` forks one process per GPU. It refuses to start once CUDA
has been used in the notebook, so the model is given by name and each process loads its own
copy on its own GPU.
"""
import os

from lmspell.data.lm_spell_dataset import LMSpellDataset
from lmspell.exceptions import LMSpellException
from lmspell.model_resolver import resolve_model

# import multiprocessing as mp
# try:
#     mp.set_start_method("spawn", force=True)
# except RuntimeError:
#     print("Spawn method already set, continuing...")
#     pass


def _train(model_name, language, trainer_kwargs):
    """Runs in each GPU process: load the model on this process's GPU, then train."""
    import torch

    from lmspell.core import LMSpell
    from lmspell.trainers.seq2seq_trainer import Seq2SeqTrainer

    # notebook_launcher sets LOCAL_RANK; make "cuda" mean this process's GPU before loading.
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    model_instance = LMSpell(model_name, language=language)
    trainer = Seq2SeqTrainer(model_instance=model_instance, **trainer_kwargs)
    trainer.train()


class MultiGPUTrainer:
    def __init__(
            self,
            model,
            train_path,
            val_path,
            exp_name,
            language = "Sinhala",
            num_processes = None,
            mixed_precision = "no",
            use_deepspeed = False,
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
        """
        model:           Hugging Face model id, e.g. "google/mt5-base" (not an LMSpell instance:
                         creating one in the notebook puts it on the GPU, which blocks the launch).
        num_processes:   number of GPUs to use; defaults to all visible GPUs.
        mixed_precision: "no", "fp16" or "bf16". mT5 is unstable in fp16, and T4 GPUs have no bf16.
        use_deepspeed:   use DeepSpeed ZeRO (stage `zero_stage`); needs `pip install "lmspell[multigpu]"`.
        The remaining arguments are passed to Seq2SeqTrainer.
        """
        if not isinstance(model, str):
            raise LMSpellException(
                'MultiGPUTrainer needs the model id (e.g. "google/mt5-base"), not a model instance: '
                "each GPU process loads its own copy.")
        resolve_model(model) 

        self.model = model
        self.language = language
        self.num_processes = num_processes
        self.mixed_precision = mixed_precision
        self.exp_dir = os.path.join(os.getcwd(), exp_name)
        self.trainer_kwargs = {
            "train_path": train_path,
            "val_path": val_path,
            "exp_name": exp_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "dataset": dataset,
            "special_tokens_to_add": special_tokens_to_add,
            "dataset_size": dataset_size,
            "num_warmup_steps": num_warmup_steps,
            "train_batch_size": train_batch_size,
            "train_max_length": train_max_length,
            "zero_stage": zero_stage,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "patience": patience,
            "lr_and_opt_path": lr_and_opt_path,
            "resume_training_from": resume_training_from,
            "resume_training": resume_training,
            "use_deepspeed": use_deepspeed,
        }

    def train(self):
        import torch
        from accelerate import notebook_launcher

        # device_count() does not initialise CUDA, so it is safe to call before launching.
        num_processes = self.num_processes or torch.cuda.device_count()
        if num_processes < 1:
            raise LMSpellException("MultiGPUTrainer needs at least one GPU, but none is visible.")

        notebook_launcher(
            _train,
            args=(self.model, self.language, self.trainer_kwargs),
            num_processes=num_processes,
            mixed_precision=self.mixed_precision,
        )
