import json
import os

class Config:

    def __init__(self):
        if os.path.exists('NeuralSpellCheckerConfig.json'):
            with open('NeuralSpellCheckerConfig.json', 'r') as f:
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