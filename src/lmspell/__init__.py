"""LMSpell: spell correction with pre-trained language models.

Heavy dependencies (torch, transformers, unsloth) are imported lazily, so
``import lmspell`` stays cheap; they load on first use of ``LMSpell`` or
``ModelEnum``.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from lmspell.exceptions import (
    EvaluateModelException,
    InvalidModelException,
    LMSpellException,
)

__version__ = "1.1.0"

if TYPE_CHECKING:
    from lmspell.core import LMSpell
    from lmspell.model_enum import ModelEnum

__all__ = [
    "LMSpell",
    "ModelEnum",
    "LMSpellException",
    "InvalidModelException",
    "EvaluateModelException",
    "__version__",
]

_LAZY_ATTRS = {
    "LMSpell": "lmspell.core",
    "ModelEnum": "lmspell.model_enum",
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        value = getattr(import_module(_LAZY_ATTRS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lmspell' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
