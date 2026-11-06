import pandas as pd
import pytest

from lmspell.exceptions import LMSpellException
from lmspell.models.model_abstract import ModelAbstract

process_input = ModelAbstract.process_input


def test_string_without_target():
    df, evaluate = process_input("a sentence")
    assert not evaluate
    assert df.to_dict("records") == [{"text": "a sentence", "expected": ""}]


def test_string_with_target():
    df, evaluate = process_input("a sentense", "a sentence")
    assert evaluate
    assert df.to_dict("records") == [{"text": "a sentense", "expected": "a sentence"}]


def test_list_with_targets():
    df, evaluate = process_input([" one ", "two"], ["one!", "two!"])
    assert evaluate
    assert df["text"].tolist() == ["one", "two"]
    assert df["expected"].tolist() == ["one!", "two!"]


def test_list_with_mismatched_targets_does_not_evaluate():
    _, evaluate = process_input(["one", "two"], ["one"])
    assert not evaluate


def test_dataframe_is_used_as_is():
    frame = pd.DataFrame({"text": ["a"], "expected": ["b"]})
    df, evaluate = process_input(frame)
    assert df is frame
    assert evaluate


def test_dataframe_without_target_column_is_not_evaluated():
    df, evaluate = process_input(pd.DataFrame({"text": ["a", "b"]}))
    assert not evaluate
    assert df["expected"].tolist() == ["", ""]


def test_dataframe_without_target_column_uses_target_set():
    df, evaluate = process_input(pd.DataFrame({"text": ["a", "b"]}), ["A", "B"])
    assert evaluate
    assert df["expected"].tolist() == ["A", "B"]


@pytest.mark.parametrize("bad_input", [[], [1, 2], 42])
def test_invalid_input_raises(bad_input):
    with pytest.raises(LMSpellException):
        process_input(bad_input)
