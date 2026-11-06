import unittest

from lmspell.core import LMSpell
from lmspell.exceptions import InvalidModelException
from lmspell.model_enum import ModelEnum


class LMSpellTest(unittest.TestCase):

    def test_create_valid_model_instance(self):
        lm_spell = LMSpell("facebook/mbart-large-50")
        self.assertEqual(lm_spell.model_label, ModelEnum.MBART50)

    def test_validate_model_name(self):
        invalid_model_name = "facebook/mbart-large-60"
        with self.assertRaises(InvalidModelException) as context:
            LMSpell(invalid_model_name)
        self.assertEqual(str(context.exception), f"Invalid model: {invalid_model_name}")

if __name__ == '__main__':
    unittest.main()
