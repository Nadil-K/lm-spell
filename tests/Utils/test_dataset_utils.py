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
