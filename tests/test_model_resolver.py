import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lmspell.core import LMSpell
from lmspell.exceptions import InvalidModelException
from lmspell.model_enum import ModelEnum
from lmspell.model_resolver import CustomModel, resolve_model
from lmspell.models.llama.llama_abstract_model import LlamaAbstractModel
from lmspell.models.mt5.mt5_abstract_model import Mt5AbstractModel
from lmspell.models.mt5.mt5_large_model import Mt5LargeModel
from lmspell.models.xlmr.xlmr_abstract_model import XlmrAbstractModel

AUTO_CONFIG = "transformers.AutoConfig.from_pretrained"


def hub_config(model_type):
    return patch(AUTO_CONFIG, return_value=SimpleNamespace(model_type=model_type))


@pytest.fixture(autouse=True)
def no_hub_adapters():
    """Hub repos in these tests are not LoRA adapters (and no network is used)."""
    with patch("huggingface_hub.hf_hub_download", side_effect=OSError("no adapter_config.json")):
        yield


def test_presets_are_matched_without_reading_any_config():
    with patch(AUTO_CONFIG) as auto_config:
        label, model_class = resolve_model("google/mt5-large")
    assert label is ModelEnum.MT5_LARGE
    assert model_class is Mt5LargeModel
    auto_config.assert_not_called()


def test_hub_checkpoint_is_loaded_by_its_architecture():
    with hub_config("mt5"):
        label, model_class = resolve_model("my-org/mt5-sinhala-finetuned")
    assert isinstance(label, CustomModel)
    assert label.get_model_path() == "my-org/mt5-sinhala-finetuned"
    assert model_class is Mt5AbstractModel


def test_local_checkpoint_folder(isolated_cwd):
    folder = isolated_cwd / "exp" / "epoch_1"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"model_type": "xlm-roberta"}), encoding="utf-8")

    label, model_class = resolve_model(str(folder))

    assert label.architecture == "xlm-roberta"
    assert model_class is XlmrAbstractModel


def test_local_lora_adapter_uses_its_base_model(isolated_cwd):
    adapter = isolated_cwd / "llama_run"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B-bnb-4bit"}), encoding="utf-8")

    with hub_config("llama") as auto_config:
        label, model_class = resolve_model(str(adapter))

    assert auto_config.call_args.args[0] == "unsloth/Llama-3.2-1B-bnb-4bit"  # base model's config is read
    assert label.get_model_path() == str(adapter)  # Unsloth loads base model + adapter from here
    assert model_class is LlamaAbstractModel


def test_unsupported_architecture_is_rejected():
    with hub_config("bert"), pytest.raises(InvalidModelException, match="unsupported architecture 'bert'"):
        resolve_model("bert-base-uncased")


def test_unknown_model_is_rejected():
    with patch(AUTO_CONFIG, side_effect=OSError("not found")), \
         pytest.raises(InvalidModelException, match="^Invalid model: no/such-model$"):
        resolve_model("no/such-model")


def test_lmspell_loads_a_custom_checkpoint(fake_tokenizer, fake_seq2seq_model):
    with hub_config("mt5"), \
         patch("lmspell.models.mt5.mt5_abstract_model.MT5ForConditionalGeneration.from_pretrained",
               return_value=fake_seq2seq_model) as load_model, \
         patch("lmspell.models.mt5.mt5_abstract_model.T5TokenizerFast.from_pretrained", return_value=fake_tokenizer):
        lm_spell = LMSpell("my-org/mt5-sinhala-finetuned")

    load_model.assert_called_once_with("my-org/mt5-sinhala-finetuned")
    assert lm_spell.correct(["a quik fox"], ["a quick fox"], max_length=8)["Corrected"].tolist() == ["a quick fox"]


def test_path_objects_are_accepted(isolated_cwd):
    (isolated_cwd / "config.json").write_text(json.dumps({"model_type": "mt5"}), encoding="utf-8")
    _, model_class = resolve_model(isolated_cwd)  # a pathlib.Path
    assert model_class is Mt5AbstractModel


@pytest.mark.parametrize("bad_name", ["", None, 42])
def test_empty_or_non_string_names_are_rejected(bad_name):
    with pytest.raises(InvalidModelException):
        resolve_model(bad_name)
