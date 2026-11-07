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

### 4. Use a dataset from the Hugging Face Hub

`DatasetUtils.load_hf_data` loads a CSV file from a Hugging Face dataset as a pandas DataFrame. The trainers accept a DataFrame wherever they take a CSV path (`train_path`, `val_path`), and `correct()` accepts one directly:

```python
from lmspell.trainers import Seq2SeqTrainer
from lmspell.utils.dataset_utils import DatasetUtils

repo = "lm-spell/sinhala-spell-correction-dataset"

trainer = Seq2SeqTrainer(
    model_instance=model_instance,
    train_path=DatasetUtils.load_hf_data(repo, "train.csv"),
    val_path=DatasetUtils.load_hf_data(repo, "val.csv"),
    exp_name="demo",
)

model_instance.correct(DatasetUtils.load_hf_data(repo, "test_set_41p.csv"))
```

Gated or private datasets need a Hugging Face token from an account that has been granted access. Set it as `huggingface.TOKEN` in `LMSpellConfig.json`, set the `HF_TOKEN` environment variable, or run `huggingface-cli login`. The CSV columns must match the configured column names (`text` and `expected` by default).

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

## Fine-tuned Models and Datasets

Fine-tuned models and datasets can be downloaded from [Huggingface](https://huggingface.co/lm-spell)

## Demo App

A demo app that can be used for Sinhala Spell Correction can be accessed from [URL](https://nisansads.staff.uom.lk/Projects/lm-spell-web-app/)

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  

## Research Publication

This project resulted in a research paper published at **MERCon 2026**, with the published version available through IEEE Xplore.

**[LMSpell: Neural Spell Checking for Low-Resource Languages](https://ieeexplore.ieee.org/document/11691371)**

* **Published in:** MERCon 2026
* **Publisher:** IEEE
* **IEEE Xplore:** [Read the published paper](https://ieeexplore.ieee.org/document/11691371)
* **Preprint:** [Read the full paper on arXiv](https://arxiv.org/html/2512.05414v3)

The arXiv preprint includes additional experiments that could not be included in the published paper due to length limitations.
