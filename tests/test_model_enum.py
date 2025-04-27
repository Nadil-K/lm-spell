import unittest
from ModelEnum import ModelEnum
from Models.Llama.LlamaAbstractModel import LlamaAbstractModel
from Models.Mt5.Mt5LargeModel import Mt5LargeModel

class ModelEnumTest(unittest.TestCase):

    def test_gets_model_class(self):
        self.assertIs(ModelEnum.getModelClass(ModelEnum.LLAMA31_FT_SI), LlamaAbstractModel)
        self.assertIs(ModelEnum.getModelClass(ModelEnum.MT5_LARGE), Mt5LargeModel)

    def test_gets_model_name(self):
        self.assertEqual(ModelEnum.LLAMA31_FT_SI.get_model_path(), "surge-masks/llama-3.1-8b-ft-ssc")
        self.assertEqual(ModelEnum.GEMMA_2_9B.get_model_path(), "google/gemma-2-9b")
        self.assertEqual(ModelEnum.MT5_LARGE.get_model_path(), "google/mt5-large")

    def test_gets_supported_models(self):
        self.assertIn("surge-masks/llama-3.1-8b-ft-ssc", ModelEnum.get_supported_models())
        self.assertIn("google/gemma-2-9b", ModelEnum.get_supported_models())
        self.assertIn("google/mt5-large", ModelEnum.get_supported_models())
        
if __name__ == '__main__':
    unittest.main()