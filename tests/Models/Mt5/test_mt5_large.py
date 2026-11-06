from unittest.mock import MagicMock, patch

from lmspell.model_enum import ModelEnum
from lmspell.models.mt5.mt5_large_model import Mt5LargeModel


@patch("lmspell.models.mt5.mt5_abstract_model.MT5ForConditionalGeneration.from_pretrained")
@patch("lmspell.models.mt5.mt5_abstract_model.T5TokenizerFast.from_pretrained")
def test_initialize_model(mock_tokenizer, mock_model):

    mock_model_instance = MagicMock(name="mT5-large-mock")
    mock_model_instance.to.return_value = mock_model_instance
    mock_model.return_value = mock_model_instance
    mock_tokenizer.return_value = MagicMock(name="mT5-large-tokenizer-mock")

    instance = Mt5LargeModel()

    assert instance.model_label is ModelEnum.MT5_LARGE
    assert instance.model is mock_model.return_value
    assert instance.tokenizer is mock_tokenizer.return_value
