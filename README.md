# LMSpell

**LMSpell** is a flexible and extensible library designed to facilitate the implementation of spell correction systems using a wide range of pre-trained language models (PLMs). It supports encoder-based, decoder-based (i.e. LLMs), and encoder-decoder-based models, providing broad compatibility across diverse architectures.

The interface of LMSpell, abstracts core PLM functionalities such as fine-tuning and inference. This abstraction enables users to integrate and experiment with different model types without requiring model-specific implementation knowledge. Internally, LMSpell dynamically loads the appropriate classes based on the selected model, ensuring a consistent and user-friendly development experience.

## Installation

Requires Python 3.10+.

```bash
pip install lmspell
```

Decoder models (Llama, Gemma) and `UnslothTrainer` need the `llm` extra, which installs [Unsloth](https://github.com/unslothai/unsloth) and requires an NVIDIA GPU:

```bash
pip install "lmspell[llm]"
```

pip installs the default PyTorch build. For a specific CUDA version, install PyTorch first by following the [PyTorch install guide](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install lmspell
```

### From source (development)

```bash
git clone https://github.com/Nadil-K/lm-spell.git
cd lm-spell
pip install -e ".[dev]"      # add ,llm on a GPU machine
python -m pytest
```

## Quick Start

### 1. Initialize the Model

```python
from lmspell import LMSpell

# Initialize the LMSpell instance
model_instance = LMSpell("unsloth/gemma-2-9b-bnb-4bit")
```

The language defaults to Sinhala. Set it with `language`. It's used in the LLM prompt and to pick the mBART-50 language code:

```python
model_instance = LMSpell("facebook/mbart-large-50", language="Hindi")
```

### 2. Fine-tuning with a Trainer

A trainer class can be used to fine-tune a model. Three trainers have been provided for decoder-based, encoder-decoder-based and encoder-based models.

#### a. Decoder-based

```python
from lmspell.trainers import UnslothTrainer

trainer = UnslothTrainer(
    model_instance=model_instance,
    train_path="train.csv",
    val_path="val.csv",
    exp_name="demo",
)
```
#### b. Encoder-decoder-based

```python
from lmspell.trainers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model_instance=model_instance,
    train_path="train.csv",
    val_path="val.csv",
    exp_name="demo",
)
```

#### c. Encoder-based

```python
from lmspell.trainers import SeqLabelTrainer

trainer = SeqLabelTrainer(
    model_instance=model_instance,
    train_path="train.csv",
    val_path="val.csv",
    exp_name="demo",
)
```

Training and validation files are CSVs with a `text` column (input sentence) and an `expected` column (corrected sentence). You can change the column names in `LMSpellConfig.json`; see `LMSpellConfig.json.example`.

Call the `train` method on the trainer to fine-tune the model:

```python
trainer.train()
```

### 3. Run Inference

```python
import pandas as pd

test_set = pd.read_csv("test.csv")
model_instance.correct(test_set)
```

##  Supported Models

#### Fine-tuned Checkpoints  
*Note: These checkpoints will be released upon acceptance of the paper.*

#### LLaMA 3.1
- `unsloth/Meta-Llama-3.1-8B-bnb-4bit`
- `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`

#### LLaMA 3.2
- `unsloth/Llama-3.2-1B-bnb-4bit`
- `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- `unsloth/Llama-3.2-3B-bnb-4bit`
- `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`

#### Gemma 2
- `unsloth/gemma-2-2b-bnb-4bit`
- `unsloth/gemma-2-2b-it-bnb-4bit`
- `unsloth/gemma-2-9b-bnb-4bit`
- `unsloth/gemma-2-9b-it-bnb-4bit`


#### mT5
- `google/mt5-small`
- `google/mt5-base`
- `google/mt5-large`
- `google/mt5-xl`
- `google/mt5-xxl`

#### MBART
- `facebook/mbart-large-50`

#### SinBERT (Sinhala)
- `NLPC-UOM/SinBERT-small`
- `NLPC-UOM/SinBERT-large`

#### XLM-R
- `FacebookAI/xlm-roberta-base`
- `FacebookAI/xlm-roberta-large`

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  