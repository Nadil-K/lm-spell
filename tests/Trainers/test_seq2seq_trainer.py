"""Runs one epoch of the real training loop on a tiny fake model (CPU, no download)."""
from types import SimpleNamespace

import pytest

from lmspell.exceptions import LMSpellException
from lmspell.model_enum import ModelEnum
from lmspell.trainers import Seq2SeqTrainer


def test_one_epoch_saves_a_checkpoint(fake_tokenizer, fake_seq2seq_model, write_csv, isolated_cwd):
    model_instance = SimpleNamespace(model=fake_seq2seq_model, tokenizer=fake_tokenizer, model_label=ModelEnum.MT5_LARGE)
    train_path = write_csv("train.csv", ["a quik fox", "the cat", "teh dog", "a bird"], ["a quick fox", "the cat", "the dog", "a bird"])
    val_path = write_csv("val.csv", ["a quik fox", "the cat"], ["a quick fox", "the cat"])

    trainer = Seq2SeqTrainer(model_instance=model_instance, train_path=train_path, val_path=val_path,
                             exp_name="exp", epochs=1, train_batch_size=2, train_max_length=8)
    trainer.train()

    assert (isolated_cwd / "exp" / "epoch_1" / "model.bin").exists()
    assert (isolated_cwd / "exp" / "optimizer.pt").exists()
    assert (isolated_cwd / "exp" / "scheduler.pt").exists()
    assert (isolated_cwd / "exp" / "epoch_1" / "tokenizer.json").exists()  # checkpoint can be loaded with LMSpell


def test_resume_without_start_epoch_raises(fake_tokenizer, fake_seq2seq_model, write_csv):
    model_instance = SimpleNamespace(model=fake_seq2seq_model, tokenizer=fake_tokenizer, model_label=ModelEnum.MT5_LARGE)
    path = write_csv("train.csv", ["a"], ["a"])
    trainer = Seq2SeqTrainer(model_instance=model_instance, train_path=path, val_path=path, exp_name="exp", resume_training=True)
    with pytest.raises(LMSpellException):
        trainer.train()


def test_training_data_can_be_dataframes(fake_tokenizer, fake_seq2seq_model, isolated_cwd):
    import pandas as pd

    model_instance = SimpleNamespace(model=fake_seq2seq_model, tokenizer=fake_tokenizer, model_label=ModelEnum.MT5_LARGE)
    frame = pd.DataFrame({"text": ["a quik fox", "the cat"], "expected": ["a quick fox", "the cat"]})

    trainer = Seq2SeqTrainer(model_instance=model_instance, train_path=frame, val_path=frame,
                             exp_name="exp", epochs=1, train_batch_size=2, train_max_length=8)
    trainer.train()

    assert (isolated_cwd / "exp" / "epoch_1" / "model.bin").exists()
