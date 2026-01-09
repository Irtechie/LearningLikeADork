# Purpose: Run eval base.
# Created: 2026-01-07
# Author: MWR

import argparse
import sys
import torch
from pathlib import Path

print("[eval] run_eval_base.py imported")

# --- Path setup (explicit, deterministic) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.evaluator import EvalRunner
from models.catalog import ModelCatalog

CASES_PATH = ROOT.parent / "data" / "eval" / "cases" / "cases.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_FILE = RESULTS_DIR / "latest_base.json"

SIM_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation against a base local HF model.")
    parser.add_argument(
        "--tier",
        choices=sorted(ModelCatalog.names()),
        default="tiny",
        help="Model tier to evaluate (uses adapter defaults).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional override for the model path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed run settings.",
    )
    return parser.parse_args()


def main():
    print("[eval] starting evaluation run (base model)")

    print("[eval] initializing adapter")
    args = parse_args()
    spec = ModelCatalog.get(args.tier)
    print(f"[eval] initializing adapter {spec.adapter}")
    adapter = spec.build_adapter(
        max_new_tokens=args.max_new_tokens,
        model_path=args.model_path,
    )
    if not args.quiet:
        print("[eval] run settings:")
        print(f"[eval] tier={args.tier}")
        print(f"[eval] model_path={args.model_path or spec.path}")
        print(f"[eval] cases_path={CASES_PATH}")
        print(f"[eval] results_file={RESULTS_FILE}")
        print(f"[eval] max_new_tokens={args.max_new_tokens}")
    print("[eval] adapter ready:", type(adapter).__name__)

    # One-time warmup to reduce first-token latency.
    model = getattr(adapter, "model", None)
    tokenizer = getattr(adapter, "tokenizer", None)
    if model is None or tokenizer is None:
        raise AttributeError("Adapter must expose model/tokenizer for warmup.")

    device = next(model.parameters()).device
    with torch.no_grad():
        warm_inputs = tokenizer(
            "warmup",
            return_tensors="pt",
        ).to(device)
        _ = model.generate(
            **warm_inputs,
            max_new_tokens=5,
            do_sample=False,
        )

    runner = EvalRunner(
        cases_path=CASES_PATH,
        results_dir=RESULTS_DIR,
        results_file=RESULTS_FILE,
        similarity_threshold=SIM_THRESHOLD,
        log_each=True,
    )
    runner.run(adapter)


if __name__ == "__main__":
    main()
