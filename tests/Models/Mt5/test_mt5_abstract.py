# pylint: disable=redefined-outer-name  # pytest fixtures
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

from lmspell.exceptions import LMSpellException
from lmspell.models.mt5.mt5_large_model import Mt5LargeModel

MT5 = "lmspell.models.mt5.mt5_abstract_model"


@pytest.fixture
def mock_model():
    """mT5-large with the Hugging Face loaders mocked out (no download)."""
    with patch(f"{MT5}.MT5ForConditionalGeneration.from_pretrained") as mock_model_loader, \
         patch(f"{MT5}.T5TokenizerFast.from_pretrained") as mock_tokenizer_loader:

        mock_model_loader.return_value = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.convert_tokens_to_ids.return_value = 123
        mock_tokenizer.all_special_ids = [0, 1, 2]
        mock_tokenizer.decode.side_effect = lambda ids, skip_special_tokens: f"decoded_{ids.tolist()}"
        mock_tokenizer_loader.return_value = mock_tokenizer

        model = Mt5LargeModel()
        model.device = "cpu"
        yield model


@pytest.fixture
def fake_mt5(fake_tokenizer, fake_seq2seq_model):
    """mT5-large wired to the fake tokenizer/model from conftest, so `correct()` really runs."""
    with patch(f"{MT5}.MT5ForConditionalGeneration.from_pretrained", return_value=fake_seq2seq_model), \
         patch(f"{MT5}.T5TokenizerFast.from_pretrained", return_value=fake_tokenizer):
        yield Mt5LargeModel()


@patch("lmspell.models.encoder_decoder_abstract.tqdm")
def test_predict_single_input(_mock_tqdm, mock_model):

    mock_model.decode = MagicMock()
    mock_model.correct(max_length=10, batch_size=1, shuffle=False, input_set="input example")
    mock_model.decode.assert_called_once()


@patch("lmspell.models.encoder_decoder_abstract.tqdm")
def test_predict_multiple_inputs(_mock_tqdm, mock_model):

    mock_model.decode = MagicMock()
    mock_model.correct(max_length=10, batch_size=2, shuffle=False, input_set=["Text A", "Text B"])
    mock_model.decode.assert_called_once()


def test_decode_function(mock_model):
    originals = [torch.tensor([1, 2, 3])]
    predictions = [torch.tensor([4, 5, 6])]
    labels = [torch.tensor([4, 5, 6])]

    with patch("lmspell.utils.pre_post_processing_utils.PrePostProcessingUtils.remove_special_tokens",
               side_effect=lambda tokens, all_ids, keep_ids: tokens), \
         patch("lmspell.utils.pre_post_processing_utils.PrePostProcessingUtils.clean_zwj", side_effect=lambda x: x), \
         patch("lmspell.utils.pre_post_processing_utils.PrePostProcessingUtils.save_dataframe") as mock_save_dataframe:

        mock_model.decode(originals, predictions, labels, "test")

        mock_save_dataframe.assert_called_once()
        df_arg = mock_save_dataframe.call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert not df_arg.empty
        assert all(col in df_arg.columns for col in ["Original", "Corrected", "Expected"])


def test_correct_with_targets_saves_results_and_metrics(fake_mt5, isolated_cwd):
    sources = ["මම පාසල් යණවා", "ඔහු ගෙදර ගියා"]
    targets = ["මම පාසල් යනවා", "ඔහු ගෙදර ගියා"]

    results = fake_mt5.correct(sources, targets, max_length=16, batch_size=2)

    assert results["Original"].tolist() == sources
    assert results["Corrected"].tolist() == targets  # the fake model predicts its labels
    assert (isolated_cwd / "outputs" / "test_results.csv").exists()
    assert (isolated_cwd / "outputs" / "evaluation_results.txt").exists()


def test_correct_from_csv(fake_mt5, write_csv):
    path = write_csv("test.csv", ["a b", "c d"], ["a b", "c e"])
    with patch.object(fake_mt5, "correct") as correct:
        fake_mt5.correctFromFile(path)
    assert correct.call_args[0][:2] == (["a b", "c d"], ["a b", "c e"])


def test_correct_from_unsupported_file_raises(fake_mt5):
    with pytest.raises(LMSpellException):
        fake_mt5.correctFromFile("data.json")


def test_correct_dataframe_with_target_column(fake_mt5, isolated_cwd):
    frame = pd.DataFrame({"text": ["a quik fox"], "expected": ["a quick fox"]})
    results = fake_mt5.correct(frame, max_length=8)
    assert results["Corrected"].tolist() == ["a quick fox"]
    assert (isolated_cwd / "outputs" / "evaluation_results.txt").exists()


def test_correct_from_csv_without_target_column(fake_mt5, isolated_cwd):
    pd.DataFrame({"text": ["a b"]}).to_csv(isolated_cwd / "only_text.csv", index=False)
    with patch.object(fake_mt5, "correct") as correct:
        fake_mt5.correctFromFile(str(isolated_cwd / "only_text.csv"))
    assert correct.call_args[0][:2] == (["a b"], None)


ZWJ_WORD = "ශ්‍රී"


def test_zwj_is_tokenized_as_a_token_and_restored(fake_mt5, isolated_cwd):  # pylint: disable=unused-argument
    results = fake_mt5.correct([ZWJ_WORD], [ZWJ_WORD], max_length=8)

    assert fake_mt5.zwj_fix
    assert "<ZWJ>" in fake_mt5.tokenizer.vocab              # registered as a token
    assert "ශ්<ZWJ>රී" in fake_mt5.tokenizer.vocab           # input was converted before tokenizing
    assert results["Corrected"].tolist() == [ZWJ_WORD]       # and turned back into U+200D


def test_zwj_fix_can_be_turned_off(fake_tokenizer, fake_seq2seq_model, isolated_cwd):
    (isolated_cwd / "LMSpellConfig.json").write_text('{"fix_zwj": {"mt5": false}}', encoding="utf-8")
    with patch(f"{MT5}.MT5ForConditionalGeneration.from_pretrained", return_value=fake_seq2seq_model),          patch(f"{MT5}.T5TokenizerFast.from_pretrained", return_value=fake_tokenizer):
        model = Mt5LargeModel()

    model.correct([ZWJ_WORD], max_length=8)
    assert not model.zwj_fix
    assert "<ZWJ>" not in fake_tokenizer.vocab
