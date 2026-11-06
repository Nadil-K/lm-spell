# pylint: disable=redefined-outer-name  # pytest fixtures
from unittest.mock import patch

import pytest

from lmspell.model_enum import ModelEnum
from lmspell.models.xlmr.xlmr_large_model import XlmrLargeModel

XLMR = "lmspell.models.xlmr.xlmr_abstract_model"


@pytest.fixture
def fake_xlmr(fake_tokenizer, fake_seq2seq_model):
    with patch(f"{XLMR}.AutoConfig.from_pretrained"), \
         patch(f"{XLMR}.XLMRobertaForMaskedLM.from_pretrained", return_value=fake_seq2seq_model), \
         patch(f"{XLMR}.AutoTokenizer.from_pretrained", return_value=fake_tokenizer):
        yield XlmrLargeModel()


def test_initialize_model(fake_xlmr, fake_tokenizer, fake_seq2seq_model):
    assert fake_xlmr.model_label is ModelEnum.XLMR_LARGE
    assert fake_xlmr.get_model() is fake_seq2seq_model
    assert fake_xlmr.get_tokenizer() is fake_tokenizer


def test_correct_with_targets(fake_xlmr, isolated_cwd):
    results = fake_xlmr.correct(["the quik fox", "a cat"], ["the quick fox", "a cat"], max_length=8, batch_size=2)

    assert results["Corrected"].tolist() == ["the quick fox", "a cat"]
    assert (isolated_cwd / "outputs" / "evaluation_results.txt").exists()
