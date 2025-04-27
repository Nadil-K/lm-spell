from Models.ModelAbstract import ModelAbstract

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
                
    def correct(self, text):        
        from tqdm import tqdm
        from unsloth import FastLanguageModel
        
        if isinstance(text, str):
            text = [text]
        
        FastLanguageModel.for_inference(self.model) 
        pred_texts = []
        generation_mode = self.model.generation_config.get_generation_mode()
        print(f"Generation mode: {generation_mode}")

        for src in tqdm(text, desc="Generating predictions"):
            inputs = self.tokenizer(
                [self.PROMPT.format(src, "")],
                return_tensors="pt"
            ).to("cuda")

            outputs = self.model.generate(**inputs, max_new_tokens=self.max_seq_length, do_sample=self.do_sample, use_cache=True)

            pred_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            pred_texts.append(pred_text)
