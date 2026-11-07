# pylint: disable=redefined-outer-name  # pytest fixtures
from unittest.mock import MagicMock, patch

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


def test_zwj_token_is_always_registered(fake_xlmr):
    # As in the original XLM-R setup: <ZWJ> is a token even though fix_zwj.xlmr is off by default.
    assert not fake_xlmr.zwj_fix
    assert fake_xlmr.tokenizer.convert_tokens_to_ids("<ZWJ>") in fake_xlmr.tokenizer.all_special_ids


def test_embeddings_grow_to_fit_the_zwj_token(fake_tokenizer):
    model = MagicMock()
    model.get_input_embeddings.return_value.num_embeddings = len(fake_tokenizer)  # no room for <ZWJ>
    with patch(f"{XLMR}.AutoConfig.from_pretrained"), \
         patch(f"{XLMR}.XLMRobertaForMaskedLM.from_pretrained", return_value=model), \
         patch(f"{XLMR}.AutoTokenizer.from_pretrained", return_value=fake_tokenizer):
        XlmrLargeModel()

    model.resize_token_embeddings.assert_called_once_with(len(fake_tokenizer))
