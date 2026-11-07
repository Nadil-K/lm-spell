"""Shared test fixtures.

Provides a tiny fake tokenizer and model so the real LMSpell code paths run on CPU
without downloading any checkpoint.
"""
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """Run every test in an empty directory: LMSpell reads LMSpellConfig.json from, and
    writes outputs/ and experiment folders to, the current directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class FakeTokenizer:
    """Whitespace tokenizer: each new word gets the next id. Ids 0-2 are special tokens."""

    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0, "<unk>": 1, "</s>": 2}
        self.all_special_ids = [0, 1, 2]

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, return_tensors=None, padding=None, truncation=False, max_length=128):  # pylint: disable=unused-argument
        ids = [self.vocab.setdefault(word, len(self.vocab)) for word in str(text).split()][: max_length - 1] + [2]
        if padding == "max_length":
            ids += [0] * (max_length - len(ids))
        input_ids = torch.tensor([ids])
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}

    def add_special_tokens(self, tokens, **_kwargs):
        for token in tokens["additional_special_tokens"]:
            self.all_special_ids.append(self.vocab.setdefault(token, len(self.vocab)))

    def save_pretrained(self, path):
        with open(f"{path}/tokenizer.json", "w", encoding="utf-8") as file:
            file.write(str(self.vocab))

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 1)

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        words = {i: w for w, i in self.vocab.items()}
        return " ".join(words[i] for i in ids if not (skip_special_tokens and i in self.all_special_ids))


class FakeSeq2SeqModel(nn.Module):
    """Always predicts its labels, so `correct()` returns the targets; still trainable."""

    def __init__(self, vocab_size=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.proj = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids=None, attention_mask=None, labels=None):  # pylint: disable=unused-argument
        # The +10 on the label always wins the argmax over the small random embedding values.
        logits = nn.functional.one_hot(labels, self.vocab_size).float() * 10 + self.proj(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        return SimpleNamespace(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return self.proj

    def resize_token_embeddings(self, new_size):  # pylint: disable=unused-argument
        return self.proj

    def save_pretrained(self, path, **kwargs):  # pylint: disable=unused-argument
        torch.save(self.state_dict(), f"{path}/model.bin")


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def fake_seq2seq_model():
    torch.manual_seed(0)
    return FakeSeq2SeqModel()


@pytest.fixture
def write_csv(tmp_path):
    """Write a text/expected CSV into the test directory and return its path."""
    def _write(name, texts, expected):
        path = tmp_path / name
        pd.DataFrame({"text": texts, "expected": expected}).to_csv(path, index=False)
        return str(path)
    return _write
