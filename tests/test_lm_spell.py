import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from lmspell.core import LMSpell
from lmspell.exceptions import InvalidModelException
from lmspell.model_enum import ModelEnum

MBART = "lmspell.models.mbart.mbart_abstract_model"


class LMSpellTest(unittest.TestCase):

    @patch(f"{MBART}.MBart50TokenizerFast.from_pretrained")
    @patch(f"{MBART}.MBartForConditionalGeneration.from_pretrained")
    @patch(f"{MBART}.MBartConfig.from_pretrained")
    def test_create_valid_model_instance(self, *_mocks):
        lm_spell = LMSpell("facebook/mbart-large-50")
        self.assertEqual(lm_spell.model_label, ModelEnum.MBART50)

    def test_validate_model_name(self):
        invalid_model_name = "facebook/mbart-large-60"
        with self.assertRaises(InvalidModelException) as context:
            LMSpell(invalid_model_name)
        self.assertEqual(str(context.exception), f"Invalid model: {invalid_model_name}")

    def test_calls_are_passed_to_the_model(self):
        fake_model = MagicMock()
        fake_model.correct.return_value = "corrected"

        with patch.object(ModelEnum, "getModelClass", return_value=MagicMock(return_value=fake_model)):
            lm_spell = LMSpell("google/mt5-large")

        self.assertEqual(lm_spell.correct("text"), "corrected")

    @patch(f"{MBART}.MBart50TokenizerFast.from_pretrained")
    @patch(f"{MBART}.MBartForConditionalGeneration.from_pretrained")
    @patch(f"{MBART}.MBartConfig.from_pretrained")
    def test_language_is_passed_to_the_model(self, _config, _model, mock_tokenizer):
        lm_spell = LMSpell("facebook/mbart-large-50", language="Hindi")
        self.assertEqual(lm_spell.language, "Hindi")
        self.assertEqual(mock_tokenizer.call_args.kwargs["src_lang"], "hi_IN")

    def test_missing_model_attribute_raises_attribute_error(self):
        lm_spell = LMSpell.__new__(LMSpell)  # no _model yet
        with self.assertRaises(AttributeError):
            lm_spell.correct  # pylint: disable=pointless-statement


# Every encoder / encoder-decoder model id can be created (several used to crash: model_label was never set).
SEQ_MODELS = [m for m in ModelEnum if not m.value.startswith("unsloth/")]
HF_LOADERS = [
    "lmspell.models.mt5.mt5_abstract_model.MT5ForConditionalGeneration.from_pretrained",
    "lmspell.models.mt5.mt5_abstract_model.T5TokenizerFast.from_pretrained",
    f"{MBART}.MBartConfig.from_pretrained",
    f"{MBART}.MBartForConditionalGeneration.from_pretrained",
    f"{MBART}.MBart50TokenizerFast.from_pretrained",
    "lmspell.models.xlmr.xlmr_abstract_model.AutoConfig.from_pretrained",
    "lmspell.models.xlmr.xlmr_abstract_model.XLMRobertaForMaskedLM.from_pretrained",
    "lmspell.models.xlmr.xlmr_abstract_model.AutoTokenizer.from_pretrained",
    "lmspell.models.sinbert.sinbert_abstract_model.AutoModelForMaskedLM.from_pretrained",
    "lmspell.models.sinbert.sinbert_abstract_model.AutoTokenizer.from_pretrained",
]


@pytest.mark.parametrize("model", SEQ_MODELS, ids=lambda m: m.name)
def test_every_seq_model_can_be_created(model):
    with ExitStack() as stack:
        for target in HF_LOADERS:
            stack.enter_context(patch(target))
        lm_spell = LMSpell(model.value)
    assert lm_spell.model_label is model


if __name__ == '__main__':
    unittest.main()
