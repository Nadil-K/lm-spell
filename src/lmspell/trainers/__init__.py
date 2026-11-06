"""Trainers for fine-tuning LMSpell models.

``Seq2SeqTrainer`` / ``SeqLabelTrainer`` cover encoder-decoder and encoder
models; ``UnslothTrainer`` covers decoder (LLM) models and needs the ``llm``
extra (``pip install "lmspell[llm]"``). Imports are lazy so that importing one
trainer does not pull in the dependencies of the others.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lmspell.trainers.seq2seq_trainer import Seq2SeqTrainer
    from lmspell.trainers.seq_label_trainer import SeqLabelTrainer
    from lmspell.trainers.unsloth_trainer import UnslothTrainer

__all__ = ["Seq2SeqTrainer", "SeqLabelTrainer", "UnslothTrainer"]

_LAZY_ATTRS = {
    "Seq2SeqTrainer": "lmspell.trainers.seq2seq_trainer",
    "SeqLabelTrainer": "lmspell.trainers.seq_label_trainer",
    "UnslothTrainer": "lmspell.trainers.unsloth_trainer",
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        value = getattr(import_module(_LAZY_ATTRS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lmspell.trainers' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
