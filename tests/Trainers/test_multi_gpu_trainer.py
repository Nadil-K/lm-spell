from unittest.mock import patch

import pytest

from lmspell.exceptions import LMSpellException
from lmspell.trainers import MultiGPUTrainer
from lmspell.trainers import multi_gpu_trainer


def make_trainer(**kwargs):
    return MultiGPUTrainer(model="google/mt5-base", train_path="train.csv", val_path="val.csv", exp_name="exp", **kwargs)


def test_train_launches_one_process_per_gpu():
    with patch("accelerate.notebook_launcher") as launcher, patch("torch.cuda.device_count", return_value=2):
        make_trainer().train()

    function, = launcher.call_args.args
    model, language, trainer_kwargs = launcher.call_args.kwargs["args"]
    assert function is multi_gpu_trainer._train  # pylint: disable=protected-access
    assert (model, language) == ("google/mt5-base", "Sinhala")
    assert trainer_kwargs["train_path"] == "train.csv" and trainer_kwargs["use_deepspeed"] is False
    assert launcher.call_args.kwargs["num_processes"] == 2
    assert launcher.call_args.kwargs["mixed_precision"] == "no"


def test_options_are_passed_to_the_launcher():
    with patch("accelerate.notebook_launcher") as launcher:
        make_trainer(num_processes=4, mixed_precision="bf16", use_deepspeed=True).train()

    assert launcher.call_args.kwargs["num_processes"] == 4
    assert launcher.call_args.kwargs["mixed_precision"] == "bf16"
    assert launcher.call_args.kwargs["args"][2]["use_deepspeed"] is True


def test_each_process_loads_its_own_model_and_trains():
    with patch("lmspell.core.LMSpell") as lm_spell, patch("lmspell.trainers.seq2seq_trainer.Seq2SeqTrainer") as trainer_class:
        multi_gpu_trainer._train("google/mt5-base", "Sinhala", {"exp_name": "exp"})  # pylint: disable=protected-access

    lm_spell.assert_called_once_with("google/mt5-base", language="Sinhala")
    trainer_class.assert_called_once_with(model_instance=lm_spell.return_value, exp_name="exp")
    trainer_class.return_value.train.assert_called_once()


def test_model_instance_is_rejected():
    with pytest.raises(LMSpellException, match="model id"):
        MultiGPUTrainer(model=object(), train_path="t", val_path="v", exp_name="e")


def test_unknown_model_id_is_rejected():
    with pytest.raises(LMSpellException, match="Invalid model"):
        MultiGPUTrainer(model="google/mt5-huge", train_path="t", val_path="v", exp_name="e")


def test_no_gpu_raises():
    with patch("torch.cuda.device_count", return_value=0), pytest.raises(LMSpellException, match="GPU"):
        make_trainer().train()
