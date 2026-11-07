# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-07

First public release of LMSpell.

### Added
- `LMSpell("<huggingface-model-id>")` interface for spell correction with
  pre-trained language models:
  - Encoder-decoder: mT5, mBART-50
  - Encoder: XLM-R, SinBERT
  - Decoder (LLM): Llama 3.1 / 3.2, Gemma 2 (via Unsloth)
- `LMSpell(..., language="Sinhala")` option for the language used in the LLM
  prompt and the mBART-50 language code.
- `correct()` and `correctFromFile()` for inference on strings, lists,
  DataFrames, CSV and TXT files. If reference corrections are given, the
  outputs are also evaluated.
- Trainers for fine-tuning: `Seq2SeqTrainer`, `SeqLabelTrainer` (Accelerate-based,
  with early stopping and per-epoch checkpoints) and `UnslothTrainer` (TRL SFT
  for decoder models).
- `MultiGPUTrainer` for multi-GPU training from a notebook (no `accelerate launch`),
  optionally with DeepSpeed ZeRO via the `multigpu` extra.
- `DatasetUtils.load_hf_data()` to load CSV files from Hugging Face datasets; the
  trainers accept DataFrames as well as CSV paths.
- Detection and correction metrics (accuracy, precision, recall, F1, F0.5).
- Zero-width joiner (ZWJ) handling for Sinhala.
- Optional `LMSpellConfig.json` for dataset/result column names and the
  Hugging Face / Google Drive credentials.
- Packaging: `pip install lmspell`, with optional extras `llm` (Unsloth / TRL
  for Llama and Gemma), `multigpu` (DeepSpeed), `dev` and `all`. Requires Python 3.10+.
- `lmspell.__version__`, and lazy top-level exports, so `import lmspell` does not
  load torch.
