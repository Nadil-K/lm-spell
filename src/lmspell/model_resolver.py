"""Find the model class for a model name: a ModelEnum preset, any compatible Hugging Face
checkpoint, or a local checkpoint folder (including LoRA adapters saved by UnslothTrainer)."""
import json
import os

from lmspell.exceptions import InvalidModelException
from lmspell.model_enum import ModelEnum
from lmspell.utils.config_utils import ConfigUtils

# config.json "model_type" -> family of LMSpell model class that can load it.
SUPPORTED_ARCHITECTURES = {
    "mt5": "mt5",
    "mbart": "mbart",
    "xlm-roberta": "xlmr",
    "roberta": "sinbert",
    "llama": "llama",
    "gemma": "gemma",
    "gemma2": "gemma",
}


class CustomModel:
    """A model that is not a ModelEnum preset: a Hugging Face model id or a local folder."""

    def __init__(self, path, architecture):
        self.value = path
        self.name = path
        self.architecture = architecture

    def get_model_path(self):
        return self.value

    def __repr__(self):
        return f"CustomModel({self.value!r}, architecture={self.architecture!r})"


def _model_class(family):
    from lmspell.models.gemma.gemma_abstract_model import GemmaAbstractModel
    from lmspell.models.llama.llama_abstract_model import LlamaAbstractModel
    from lmspell.models.mbart.mbart_abstract_model import MbartAbstractModel
    from lmspell.models.mt5.mt5_abstract_model import Mt5AbstractModel
    from lmspell.models.sinbert.sinbert_abstract_model import SinBertAbstractModel
    from lmspell.models.xlmr.xlmr_abstract_model import XlmrAbstractModel

    return {
        "mt5": Mt5AbstractModel,
        "mbart": MbartAbstractModel,
        "xlmr": XlmrAbstractModel,
        "sinbert": SinBertAbstractModel,
        "llama": LlamaAbstractModel,
        "gemma": GemmaAbstractModel,
    }[family]


def _adapter_base_model(model, token):
    """If `model` is a LoRA adapter (local folder or Hub repo), return its base model id."""
    path = os.path.join(model, "adapter_config.json")
    if not os.path.isfile(path):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(model, "adapter_config.json", token=token)
        except (OSError, ValueError):
            return None
    with open(path, encoding="utf-8") as file:
        return json.load(file).get("base_model_name_or_path")


def _model_type(model):
    """The architecture ("model_type") from the checkpoint's config.json; no weights are downloaded.
    A LoRA adapter has no config.json of its own, so its base model's is read."""
    from transformers import AutoConfig

    token = ConfigUtils().get("huggingface.TOKEN")
    config_source = _adapter_base_model(model, token) or model
    try:
        return AutoConfig.from_pretrained(config_source, token=token).model_type
    except (OSError, ValueError) as error:
        raise InvalidModelException(f"Invalid model: {model}") from error


def resolve_model(model):
    """Return (model_label, model_class) for a model name.

    ModelEnum presets are matched directly. Anything else is loaded by architecture, read
    from its config.json: mT5, mBART, XLM-RoBERTa, RoBERTa (e.g. SinBERT), Llama or Gemma.
    """
    if isinstance(model, os.PathLike):  # e.g. pathlib.Path("demo/epoch_3")
        model = os.fspath(model)

    if not isinstance(model, str):
        raise InvalidModelException(f"Invalid model: {model} (must be a string)")

    if not model:
        raise InvalidModelException("Invalid model: empty model name")

    # Presets are matched by their Hugging Face id (the enum value), not the member name.
    presets = {preset.value: preset for preset in ModelEnum}
    if model in presets:
        preset = presets[model]
        return preset, preset.getModelClass()

    print(f"Resolving model {model} by architecture...")
    model_type = _model_type(model)
    if model_type not in SUPPORTED_ARCHITECTURES:
        raise InvalidModelException(
            f"Invalid model: {model} (unsupported architecture '{model_type}'; "
            f"supported: {', '.join(sorted(SUPPORTED_ARCHITECTURES))})")
    return CustomModel(model, model_type), _model_class(SUPPORTED_ARCHITECTURES[model_type])
