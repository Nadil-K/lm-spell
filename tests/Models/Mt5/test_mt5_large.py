import unittest
from NeuralSpellChecker import NeuralSpellChecker
from ModelEnum import ModelEnum

class ModelEnumTest(unittest.TestCase):

    def test_initialize_model(self):
        nsc = NeuralSpellChecker("google/mt5-large")
        
        self.assertEqual(nsc.model_label, ModelEnum.MT5_LARGE)

if __name__ == '__main__':
    unittest.main()