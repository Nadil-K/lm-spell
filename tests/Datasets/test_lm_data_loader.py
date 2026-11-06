from lmspell.data.lm_data_loader import LMDataLoader


def test_builds_train_and_val_loaders(fake_tokenizer, write_csv):
    path = write_csv("data.csv", ["a b", "c d", "e f"], ["a b", "c d", "e g"])

    loader = LMDataLoader(fake_tokenizer, train_path=path, val_path=path, train_max_length=6, train_batch_size=2)

    assert loader.test_dataloader is None
    assert len(loader.train_dataloader) == 2  # 3 rows, batch size 2
    batch = next(iter(loader.train_dataloader))
    assert batch["input_ids"].shape == (2, 6)
    assert batch["labels"].shape == (2, 6)
