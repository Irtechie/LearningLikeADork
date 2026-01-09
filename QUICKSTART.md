# Quickstart

## Model download
- Use `ai-eng/util/prefetch_models.py` to download models into `data/models/local/<tier>/<model-name>`.
- Example:
```bash
python ai-eng/util/prefetch_models.py
```
- Model weights are intentionally ignored by git; see `data/models/README.md`.

## Training runs
- `ai-eng/finetune/run_finetune.py`: LoRA finetune (PEFT) with outputs in `experiments/lora/`.
- `ai-eng/finetune/run_full_finetune.py`: full finetune (fp32/bf16 only) with outputs in `experiments/full_ft/`.
- `ai-eng/finetune/run_overfitting.py`: tiny overfit run to validate the pipeline end-to-end.

## Eval runs
- `ai-eng/eval/run_eval.py`: RAG eval pipeline (retriever + generator) over eval cases.
- `ai-eng/eval/run_eval_no_rag.py`: eval without retrieval (generator only).
- `ai-eng/eval/run_eval_base.py`: base-model-only eval for comparison.
- `ai-eng/eval/run_rag_inference.py`: single RAG query/inference check.

## Data layout
- Training data: `data/training/`
- Eval cases: `data/eval/cases/`
- Chroma DB: `data/chroma/db/`
- Models: `data/models/local/<tier>/<model-name>/`
- Outputs: `experiments/`
