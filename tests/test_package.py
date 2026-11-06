import os
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _run(code):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)


def test_import_is_lightweight():
    out = _run(
        "import sys, lmspell\n"
        "heavy = [m for m in ('torch', 'transformers', 'unsloth', 'pandas') if m in sys.modules]\n"
        "print(','.join(heavy))"
    )
    assert out.stdout.strip() == "", f"`import lmspell` eagerly imported: {out.stdout.strip()}"


def test_version_is_pep440():
    import lmspell

    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", lmspell.__version__)


def test_lazy_public_api():
    import lmspell
    from lmspell.core import LMSpell
    from lmspell.model_enum import ModelEnum
    from lmspell.trainers.seq2seq_trainer import Seq2SeqTrainer
    import lmspell.trainers

    assert lmspell.LMSpell is LMSpell
    assert lmspell.ModelEnum is ModelEnum
    assert lmspell.trainers.Seq2SeqTrainer is Seq2SeqTrainer


def test_exceptions_exported():
    import lmspell

    assert issubclass(lmspell.InvalidModelException, lmspell.LMSpellException)
    assert issubclass(lmspell.EvaluateModelException, lmspell.LMSpellException)
