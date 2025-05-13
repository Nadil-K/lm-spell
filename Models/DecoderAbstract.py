import pandas as pd
from Utils.ConfigUtils import ConfigUtils
from Utils.EvaluateUtils import EvaluateUtils
from Models.ModelAbstract import ModelAbstract
from NeuralSpellCheckerException import NeuralSpellCheckerException

class DecoderAbstract(ModelAbstract):

    def __init__(self):
        from unsloth import FastLanguageModel

        super().__init__()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_label.get_model_path(),
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            random_state = self.seed,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.language = 'Sinhala'   # get it dynamically

        self.PROMPT = f"""You are an expert {self.language} spell corrector. Below is a sentence in {self.language} language. It may or may not have a spelling mistake. Give the corrected output in {self.language}.

        ### Text:
        {{}}

        ### Output:
        {{}}"""
                
    def correct(self, input_set: list[str] | str | pd.DataFrame, target_set: list[str] | str | pd.DataFrame = None):
        import re        
        from tqdm import tqdm
        from unsloth import FastLanguageModel
        
        input_set, evaluate_flag = self.process_input(input_set, target_set)

        dataset_col_names = ConfigUtils.get_dataset_columns()
        
        FastLanguageModel.for_inference(self.model) 
        pred_texts = []
        generation_mode = self.model.generation_config.get_generation_mode()
        print(f"Generation mode: {generation_mode}")

        for src in tqdm(input_set[dataset_col_names[0]].to_list(), desc="Generating predictions"):
            inputs = self.tokenizer(
                [self.PROMPT.format(src, "")],
                return_tensors="pt"
            ).to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=self.max_seq_length, do_sample=self.do_sample, use_cache=True)

            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            pred_texts.append(pred_text)
        
        print(pred_texts)
        output_texts = []
        for pred in pred_texts:
            match = re.search(r'### Output:\s*([^\n]+)', pred)
            output_texts.append(match.group(1).strip() if match else None)

        result_col_names = ConfigUtils.get_results_columns()

        results_data = {
            result_col_names[0]: input_set[dataset_col_names[0]].to_list(),
            result_col_names[1]: output_texts
        }

        results_df = pd.DataFrame(results_data)

        if evaluate_flag:
            results_df[result_col_names[2]] = input_set[dataset_col_names[1]].to_list()
            print(results_df.columns)
            print("Evaluating the outputs...")
            EvaluateUtils.evaluate_from_dataframe(results_df, self.exp_dir)            

        return results_df

    def correctFromFile(self, src: str, target: str = None, evaluate_flag: bool = False):
        import pandas as pd
        """
        Corrects the text from a file. The file should be in the format of
        """

        if src.endswith('.csv'):
            src = pd.read_csv(src)

        elif src.endswith('.txt'):
            with open(src, 'r', encoding='utf-8') as file:
                src = file.readlines()
            if target is not None:
                with open(target, 'r', encoding='utf-8') as file:
                    target = file.readlines()
        else:
            raise NeuralSpellCheckerException("Unsupported file format. Only .csv and .txt are supported.") from None
                
        return self.correct(src, target, evaluate_flag)
