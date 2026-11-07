from unittest.mock import patch

import pandas as pd

from lmspell.utils.dataset_utils import DatasetUtils


def test_loads_splits_and_drops_missing_rows(write_csv):
    train = write_csv("train.csv", ["a", "b", None], ["a", "b", "c"])
    val = write_csv("val.csv", ["v"], ["v"])

    data = DatasetUtils(train_path=train, val_path=val)

    assert data.train_dataset.num_rows == 2
    assert data.val_dataset.num_rows == 1
    assert data.test_dataset is None


def test_dataset_size_samples_the_training_split(write_csv):
    texts = [f"s{i}" for i in range(10)]
    data = DatasetUtils(train_path=write_csv("train.csv", texts, texts), dataset_size=0.5)
    assert data.train_dataset.num_rows == 5


def test_fix_zwj_replaces_zero_width_joiner():
    df = pd.DataFrame({"text": ["ශ්‍රී"], "expected": ["ශ්‍රී"]})
    fixed = DatasetUtils.fix_zwj(df)
    assert fixed["text"].tolist() == ["ශ්<ZWJ>රී"]
    assert fixed["expected"].tolist() == ["ශ්<ZWJ>රී"]



REPO = "lm-spell/sinhala-spell-correction-dataset"


def test_load_hf_data_reads_the_file_from_the_hub():
    with patch("pandas.read_csv", return_value="frame") as read_csv:
        assert DatasetUtils.load_hf_data(REPO, "train.csv") == "frame"
    read_csv.assert_called_once_with(f"hf://datasets/{REPO}/train.csv", storage_options=None)


def test_load_hf_data_uses_the_config_token(isolated_cwd):
    (isolated_cwd / "LMSpellConfig.json").write_text('{"huggingface": {"TOKEN": "hf_secret"}}', encoding="utf-8")
    with patch("pandas.read_csv") as read_csv:
        DatasetUtils.load_hf_data(REPO, "val.csv")
    read_csv.assert_called_once_with(f"hf://datasets/{REPO}/val.csv", storage_options={"token": "hf_secret"})


def test_splits_can_be_given_as_dataframes():
    frame = pd.DataFrame({"text": ["a", "b", None], "expected": ["a", "b", "c"]})
    data = DatasetUtils(train_path=frame, val_path=frame)
    assert data.train_dataset.num_rows == 2
    assert data.val_dataset.num_rows == 2
    assert len(frame) == 3  # the caller's DataFrame is not modified
