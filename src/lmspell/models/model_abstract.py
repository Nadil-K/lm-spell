from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import torch

from lmspell.exceptions import LMSpellException
from lmspell.utils.config_utils import ConfigUtils


class ModelAbstract(ABC):
    model_label: Any
    model: Any
    tokenizer: Any

    @abstractmethod
    def __init__(self, model_label=None, language="Sinhala"):
        '''
        Conctrete classes should define following attributes:
        - model_label (or receive it as an argument)
        - model
        - tokenizer
        - seed
        - max_seq_length
        - language
        '''
        if model_label is not None:
            self.model_label = model_label
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language = language
        self.zwj_fix = False

    def enable_zwj_fix(self, config_key):
        """
        Turn on the ZWJ fix if fix_zwj.<config_key> in LMSpellConfig.json (or its default) says so.
        These tokenizers drop the zero-width joiner (U+200D), so it is written as a <ZWJ> token
        before tokenizing and turned back into U+200D after decoding.
        """
        from lmspell.utils.pre_post_processing_utils import PrePostProcessingUtils

        self.zwj_fix = ConfigUtils.zwj_fix_enabled(config_key)
        if not self.zwj_fix:
            return
        # Keep the existing additional special tokens (e.g. mBART's language codes).
        self.tokenizer.add_special_tokens({'additional_special_tokens': [PrePostProcessingUtils.ZWJ]}, replace_additional_special_tokens=False)
        # Grow the embeddings only if needed.
        if len(self.tokenizer) > int(self.model.get_input_embeddings().num_embeddings):
            self.model.resize_token_embeddings(len(self.tokenizer))

    @abstractmethod
    def correct(self, input_set, target_set=None):
        pass

    @abstractmethod
    def correctFromFile(self, src: str, target: str = None):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def get_model_name(self):
        pass

    @abstractmethod
    def load_tokenizer(self):
        pass

    @abstractmethod
    def save_tokenizer(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model

    @staticmethod
    def process_input(input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str = None):
        """
        Process the input set and target set. The input set should be a string, a list of strings or a DataFrame.
        The target set should be a string or a list of strings.
        """

        dataset_col_names = ConfigUtils.get_dataset_columns()

        if isinstance(input_set, pd.DataFrame):
            if dataset_col_names[1] in input_set.columns:
                return input_set, True
            # No target column: add one (the dataset needs both columns), filled from target_set if given.
            if target_set is None:
                return input_set.assign(**{dataset_col_names[1]: ""}), False
            if isinstance(target_set, pd.DataFrame):
                targets = target_set.iloc[:, 0].tolist()
            else:
                targets = [target_set] if isinstance(target_set, str) else list(target_set)
            if len(targets) != len(input_set):
                raise LMSpellException(f"Got {len(input_set)} inputs but {len(targets)} targets.")
            return input_set.assign(**{dataset_col_names[1]: targets}), True

        if isinstance(input_set, list):
            if not input_set or not all(isinstance(s, str) for s in input_set):
                raise LMSpellException("input_set must be a non-empty list of strings.")
        elif not isinstance(input_set, str):
            raise LMSpellException("input_set must be a string, a list of strings or a DataFrame.")

        evaluate_flag = False
        data = []

        if isinstance(input_set, str):

            text = input_set
            evaluate_flag = target_set is not None and isinstance(target_set, str)
            expected = target_set if evaluate_flag else ""
            data = [{dataset_col_names[0]: text, dataset_col_names[1]: expected}]

        elif isinstance(input_set, list) and isinstance(input_set[0], str):

            evaluate_flag = (target_set is not None and
                                isinstance(target_set, list) and
                                len(target_set) == len(input_set))

            data = [{dataset_col_names[0]: s.strip(), dataset_col_names[1]: t.strip()} for s, t in zip(input_set, target_set)] \
                if evaluate_flag else [{dataset_col_names[0]: s.strip(), dataset_col_names[1]: ""} for s in input_set]

        input_set = pd.DataFrame(data)
        return input_set, evaluate_flag
