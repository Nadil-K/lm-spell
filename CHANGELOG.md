# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-11-07

### Added
- `LMSpell` loads any Hugging Face checkpoint or local folder with a supported
  architecture (mT5, mBART, XLM-RoBERTa, RoBERTa, Llama, Gemma), not only the
  presets. The architecture is read from the checkpoint's `config.json`.
  This includes checkpoints saved by `Seq2SeqTrainer` / `SeqLabelTrainer`,
  LoRA adapters saved by `UnslothTrainer`, and `pathlib.Path` arguments.
- `fix_zwj` setting in `LMSpellConfig.json` to turn the zero-width joiner (ZWJ)
  fix on or off per model (`mbart`, `mt5`, `mt5-xl`, `xlmr`, `sinbert`).

### Changed
- The ZWJ fix is now applied by default for mBART, mT5 and mT5-XL, in training
  and in `correct()`. Training results for these models differ from 1.0.0.
  For checkpoints fine-tuned without the fix, set `"fix_zwj": {"mbart": false}`
  (or `mt5`) in `LMSpellConfig.json`.
- `Seq2SeqTrainer` / `SeqLabelTrainer` save the tokenizer with each checkpoint,
  so a checkpoint folder can be loaded with `LMSpell("exp/epoch_N")`.

### Fixed
- The ZWJ fix did not work for mBART: U+200D was never converted to the `<ZWJ>`
  token, so the tokenizer dropped it and split words (ශ්‍රී -> ශ් රී).
- Adding special tokens in the trainer no longer removes mBART's language codes,
  which made `si_LK` appear in mBART outputs.
- XLM-R resizes its embeddings for the `<ZWJ>` token, which could crash training
  with an index error.
- `Metrics` now uses its `ZWJ_Fix` argument.

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
