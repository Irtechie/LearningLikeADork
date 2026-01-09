# Purpose: Run finetune.
# Created: 2026-01-07
# Author: MWR

import argparse
import sys
from pathlib import Path

print("[lora] run_finetune.py imported")

# --- Path setup (explicit, deterministic) ---
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune.fine_tuner import FineTuner, LoRASettings, TrainingSettings
from models.catalog import ModelCatalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA training.")
    parser.add_argument(
        "--tier",
        choices=sorted(ModelCatalog.names()),
        default="tiny",
        help="Model tier to fine-tune (uses adapter defaults).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional override for the model path.",
    )
    parser.add_argument(
        "--train-file",
        default=str(
            PROJECT_ROOT
            / "data"
            / "training"
            / "bitheroes"
            / "smol_ft.jsonl"
        ),
        help="Path to JSONL training data.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(
            PROJECT_ROOT
            / "experiments"
            / "lora"
            / "smol_ft"
        ),
        help="Directory to write LoRA adapter weights.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--eval-after",
        action="store_true",
        help="Run eval after fine-tuning (no RAG).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed run settings.",
    )
    parser.add_argument(
        "--cases-path",
        default=str(ROOT.parent / "data" / "eval" / "cases" / "cases.jsonl"),
        help="Path to eval cases JSONL.",
    )
    parser.add_argument(
        "--results-file",
        default=str(ROOT / "eval" / "results" / "latest_finetune.json"),
        help="Path to write eval results JSON.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for pass/fail.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    spec = ModelCatalog.get(args.tier)
    model_path = args.model_path or spec.path

    if not args.quiet:
        print("[lora] run settings:")
        print(f"[lora] tier={args.tier}")
        print(f"[lora] model_path={model_path}")
        print(f"[lora] train_file={args.train_file}")
        print(f"[lora] output_dir={args.output_dir}")
        print(
            "[lora] hyperparams: "
            f"epochs={args.epochs} batch_size={args.batch_size} "
            f"lr={args.learning_rate} max_seq_length={args.max_seq_length} "
            f"lora_r={args.lora_r} lora_alpha={args.lora_alpha} "
            f"lora_dropout={args.lora_dropout}"
        )

    tuner = FineTuner(
        model_path=model_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        lora=LoRASettings(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        ),
        training=TrainingSettings(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
        ),
    )
    tuner.train()

    if args.eval_after:
        from adapters.finetuned import FinetunedAdapter
        from eval.evaluator import EvalRunner
        print("[lora] running eval after training...")
        adapter = FinetunedAdapter(
            base_model_path=model_path,
            lora_path=args.output_dir,
        )
        runner = EvalRunner(
            cases_path=Path(args.cases_path),
            results_dir=Path(args.results_file).parent,
            results_file=Path(args.results_file),
            similarity_threshold=args.similarity_threshold,
            log_each=True,
        )
        runner.run(adapter)


if __name__ == "__main__":
    main()
