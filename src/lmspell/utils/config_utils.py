import json
import os

ZWJ_FIX_DEFAULTS = {"mbart": True, "mt5": True, "mt5-xl": True}


class ConfigUtils:

    def __init__(self):
        if os.path.exists('LMSpellConfig.json'):
            with open('LMSpellConfig.json', 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default


    @staticmethod
    def get_dataset_columns():
        config = ConfigUtils()
        return [
            config.get('dataset.inputs', 'text'),
            config.get('dataset.targets', 'expected')
        ]

    @staticmethod
    def get_results_columns():
        config = ConfigUtils()
        return [
            config.get('results.original', 'Original'),
            config.get('results.predicted', 'Corrected'),
            config.get('results.expected', 'Expected')
        ]

    @staticmethod
    def zwj_fix_enabled(model_key):
        """Whether the ZWJ fix is on for a model ("mbart", "mt5", "mt5-xl", "xlmr", "sinbert")."""
        return bool(ConfigUtils().get(f'fix_zwj.{model_key}', ZWJ_FIX_DEFAULTS.get(model_key, False)))
