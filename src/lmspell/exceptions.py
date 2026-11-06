"""Exceptions raised by LMSpell."""


class LMSpellException(Exception):
    """Base class for all LMSpell errors."""


class InvalidModelException(LMSpellException):
    """Raised when an unsupported model name is requested."""


class EvaluateModelException(LMSpellException):
    """Raised when evaluation of a model fails."""
