# pylint: disable=redefined-outer-name  # pytest fixtures
"""Decoder (LLM) models with Unsloth replaced by a fake module, so they run without a GPU."""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from lmspell.model_enum import ModelEnum
from lmspell.models.llama.llama31_8b_model import Llama318BModel


class FakeEncoding(dict):
    def to(self, device):  # pylint: disable=unused-argument
        return self


@pytest.fixture
def llm(monkeypatch):
    """Llama 3.1 8B whose model always "generates" a fixed answer after the ### Output: marker."""
    model = MagicMock(name="llm")
    model.generate.return_value = ["generated-ids"]
    tokenizer = MagicMock(name="llm-tokenizer")
    tokenizer.side_effect = lambda texts, return_tensors=None: FakeEncoding()
    tokenizer.decode.return_value = "### Text:\n        input\n\n        ### Output:\n        corrected sentence\n"

    fast = MagicMock(name="FastLanguageModel")
    fast.from_pretrained.return_value = (model, tokenizer)
    fast.get_peft_model.side_effect = lambda m, **kwargs: m

    monkeypatch.setitem(sys.modules, "unsloth", types.SimpleNamespace(FastLanguageModel=fast))
    with patch("torch.cuda.get_device_name", return_value="Tesla T4"):
        yield Llama318BModel()


def test_initialize_model(llm):
    assert llm.model_label is ModelEnum.LLAMA_31_8B
    assert "### Text:" in llm.PROMPT and "### Output:" in llm.PROMPT


def test_correct_returns_text_after_output_marker(llm):
    results = llm.correct(["first input", "second input"])

    assert results["Original"].tolist() == ["first input", "second input"]
    assert results["Corrected"].tolist() == ["corrected sentence", "corrected sentence"]


def test_correct_with_targets_evaluates(llm, isolated_cwd):
    llm.correct(["first input"], ["corrected sentence"])
    assert (isolated_cwd / "outputs" / "evaluation_results.txt").exists()


def test_llama_32_model_can_be_created_and_used(llm):  # pylint: disable=unused-argument
    # Llama 3.2 maps to LlamaAbstractModel, which used to crash (no model_label / do_sample).
    from lmspell.core import LMSpell

    lm_spell = LMSpell("unsloth/Llama-3.2-1B-bnb-4bit")
    assert lm_spell.model_label is ModelEnum.LLAMA_32_1B
    assert lm_spell.correct("input")["Corrected"].tolist() == ["corrected sentence"]
