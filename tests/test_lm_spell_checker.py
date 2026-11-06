import unittest
from lmspell.core import LMSpell
from lmspell.exceptions import InvalidModelException
from lmspell.model_enum import ModelEnum

class LMSpellTest(unittest.TestCase):

    def test_create_valid_model_instance(self):
        # neuralSpellChecker = NeuralSpellChecker("facebook/mbart-large-50")
        # self.assertEqual(neuralSpellChecker.model, ModelEnum.MBART50)
        pass
        
    def test_validate_model_name(self):
        # invalid_model_name = "facebook/mbart-large-60"
        # with self.assertRaises(InvalidModelException) as context:
        #     NeuralSpellChecker(invalid_model_name)
        # self.assertEqual(str(context.exception), f"Invalid model: {invalid_model_name}")
        pass

if __name__ == '__main__':
    unittest.main()