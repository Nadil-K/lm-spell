import pandas as pd
import pytest

from lmspell.utils.evaluate_utils import EvaluateUtils

METRIC_KEYS = [
    "Detection Accuracy", "Detection Recall", "Detection Precision", "Detection F1", "Detection F0.5",
    "Correction Accuracy", "Correction Recall", "Correction Precision", "Correction F1", "Correction F0.5",
]


def test_evaluate_returns_all_metrics():
    assert list(EvaluateUtils.evaluate(["a"], ["a"], ["a"])) == METRIC_KEYS


def test_correct_sentence_left_unchanged_scores_100():
    results = EvaluateUtils.evaluate(["a b c"], ["a b c"], ["a b c"])
    assert results == {key: pytest.approx(100.0) for key in METRIC_KEYS}


def test_corrected_error_scores_100():
    results = EvaluateUtils.evaluate(["the quik fox"], ["the quick fox"], ["the quick fox"])
    assert results["Correction Recall"] == pytest.approx(100.0)
    assert results["Correction Precision"] == pytest.approx(100.0)


def test_uncorrected_error_has_zero_recall():
    results = EvaluateUtils.evaluate(["the quik fox"], ["the quik fox"], ["the quick fox"])
    assert results["Correction Recall"] == pytest.approx(0.0)


def test_sinhala_split_vowel_signs_are_normalised():
    # "කො" typed as ක + ෙ + ා must count as the single code point ො.
    results = EvaluateUtils.evaluate(["කො"], ["කො"], ["කො"])
    assert results["Correction F1"] == pytest.approx(100.0)


def test_metric_values_are_unchanged():
    """Pins the metric values, so a change to the evaluator that alters reported results is noticed."""
    srcs = ["මම පාසල් යණවා", "ඔහු ගෙදර ගියා", "the quik brown fox", "a b c d", "one two three four five"]
    preds = ["මම පාසල් යනවා", "ඔහු ගෙදර ගියා", "the quick brown fox", "a b x d e", "one too three for five"]
    refs = ["මම පාසල් යනවා", "ඔහු ගෙදර ගියා", "the quick brown fox", "a b c d", "one two three four five"]
    assert EvaluateUtils.evaluate(srcs, preds, refs) == pytest.approx(SNAPSHOT)


SNAPSHOT = {
    "Detection Accuracy": 91.59420289855072,
    "Detection Recall": 60.0,
    "Detection Precision": 60.0,
    "Detection F1": 60.0,
    "Detection F0.5": 60.0,
    "Correction Accuracy": 91.59420289855072,
    "Correction Recall": 60.0,
    "Correction Precision": 60.0,
    "Correction F1": 60.0,
    "Correction F0.5": 60.0,
}


def test_evaluate_from_dataframe_writes_results_file(isolated_cwd):
    (isolated_cwd / "out").mkdir()
    df = pd.DataFrame({"Original": ["a quik"], "Corrected": ["a quick"], "Expected": ["a quick"]})

    EvaluateUtils.evaluate_from_dataframe(df, "out")

    assert "Correction F1: 100.0" in (isolated_cwd / "out" / "evaluation_results.txt").read_text(encoding="utf-8")


def test_single_strings_count_as_one_sentence():
    assert EvaluateUtils.evaluate("a b", "a b", "a b")["Correction F1"] == pytest.approx(100.0)


def test_empty_and_missing_predictions_do_not_crash():
    results = EvaluateUtils.evaluate(["", "the quik fox"], ["", None], ["", "the quick fox"])
    assert set(results) == set(METRIC_KEYS)
