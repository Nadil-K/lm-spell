import unittest
from NeuralSpellChecker import NeuralSpellChecker
from Models.Mbart50Model import Mbart50LargeModel
from Exceptions.InvalidModelException import InvalidModelException
from ModelEnum import ModelEnum

class NeuralSpellCheckerTest(unittest.TestCase):

    def test_create_valid_model_instance(self):
        neuralSpellChecker = NeuralSpellChecker("facebook/mbart-large-50")
        self.assertEqual(neuralSpellChecker.model, ModelEnum.MBART50)
        
    def test_validate_model_name(self):
        invalid_model_name = "facebook/mbart-large-60"
        with self.assertRaises(InvalidModelException) as context:
            NeuralSpellChecker(invalid_model_name)
        self.assertEqual(str(context.exception), f"Invalid model: {invalid_model_name}")

if __name__ == '__main__':
    unittest.main()