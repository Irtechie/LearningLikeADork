# Purpose: Run full finetune.
# Created: 2026-01-08
# Author: MWR

import argparse
import sys
from pathlib import Path

print("[full-finetune] run_full_finetune.py imported")

# --- Path setup (explicit, deterministic) ---
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finetune.full_fine_tuner import FullFineTuner, TrainingSettings
from models.catalog import ModelCatalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full fine-tuning (saves full model).")
    parser.add_argument(
        "--tier",
        choices=sorted(ModelCatalog.names()),
        default="full",
        help="Model tier to fine-tune (uses adapter defaults).",
    )
    parser.add_argument(
        "--model-path",
        default=str(
            PROJECT_ROOT
            / "data"
            / "models"
            / "local"
            / "full"
            / "qwen2.5-3b-instruct"
        ),
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
            / "full_ft"
            / "smol_full_ft"
        ),
        help="Directory to write full model weights.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed run settings.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    spec = ModelCatalog.get(args.tier)
    model_path = args.model_path or spec.path

    if not args.quiet:
        print("[full-finetune] run settings:")
        print(f"[full-finetune] tier={args.tier}")
        print(f"[full-finetune] model_path={model_path}")
        print(f"[full-finetune] train_file={args.train_file}")
        print(f"[full-finetune] output_dir={args.output_dir}")
        print(
            "[full-finetune] hyperparams: "
            f"epochs={args.epochs} batch_size={args.batch_size} "
            f"lr={args.learning_rate} max_seq_length={args.max_seq_length}"
        )

    tuner = FullFineTuner(
        model_path=model_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        training=TrainingSettings(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
        ),
    )
    tuner.train()


if __name__ == "__main__":
    main()
