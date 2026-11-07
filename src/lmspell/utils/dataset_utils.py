class DatasetUtils:

    def __init__(self, train_path = None, val_path = None, test_path = None, dataset_size = 1, seed = 42, zwj_fix = False):
        """Each of train_path, val_path and test_path is a CSV file path or a pandas DataFrame
        with columns 'text' and 'expected'. The dataset_size is a fraction of the training split to use.
        """
        from datasets import Dataset

        self.dataset_size = dataset_size

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        if train_path is not None:
            train = self._load(train_path).dropna()
            train = train.sample(frac=self.dataset_size, random_state=seed).reset_index(drop=True)
            if zwj_fix:
                train = self.fix_zwj(train)
            print(f"Train shape: {train.shape}")
            self.train_dataset = Dataset.from_pandas(train)

        if val_path is not None:
            val = self._load(val_path).dropna()
            if zwj_fix:
                val = self.fix_zwj(val)
            print(f"Val shape: {val.shape}")
            self.val_dataset = Dataset.from_pandas(val)

        if test_path is not None:
            test = self._load(test_path).dropna()
            if zwj_fix:
                test = self.fix_zwj(test)
            print(f"Test shape: {test.shape}")
            self.test_dataset = Dataset.from_pandas(test)

    @staticmethod
    def load_hf_data(repo_id, filename):
        """
        Load a CSV file from the latest revision of a Hugging Face dataset repository as a
        DataFrame, e.g. DatasetUtils.load_hf_data("lm-spell/sinhala-spell-correction-dataset", "train.csv").
        Gated or private datasets use huggingface.TOKEN from LMSpellConfig.json, or the
        HF_TOKEN environment variable / `huggingface-cli login`.
        """
        import pandas as pd

        from lmspell.utils.config_utils import ConfigUtils

        token = ConfigUtils().get("huggingface.TOKEN")
        return pd.read_csv(
            f"hf://datasets/{repo_id}/{filename}",
            storage_options={"token": token} if token else None
        )

    @staticmethod
    def _load(data):
        """A DataFrame is used as is; anything else is read as a CSV path."""
        import pandas as pd

        return data if isinstance(data, pd.DataFrame) else pd.read_csv(data)

    @staticmethod
    def fix_zwj(df):
        from lmspell.utils.config_utils import ConfigUtils

        for col in ConfigUtils.get_dataset_columns():
            df[col] = df[col].str.replace('\u200D', '<ZWJ>')

        return df
