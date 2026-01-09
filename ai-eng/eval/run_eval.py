# Purpose: Run eval.
# Created: 2026-01-05
# Author: MWR

import argparse
import sys
import torch
from pathlib import Path

print("[eval] run_eval.py imported")

# --- Path setup (explicit, deterministic) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.rag import RagAdapter
from retrievers.retriever import ChromaRetriever
from eval.evaluator import EvalRunner
from models.catalog import ModelCatalog

CASES_PATH = ROOT.parent / "data" / "eval" / "cases" / "cases.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_FILE = RESULTS_DIR / "latest.json"

SIM_THRESHOLD = 0.85


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation against a local HF model.")
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
    print("[eval] starting evaluation run")
     
    # --- Adapter initialization ---

    print("[eval] initializing adapter")
    args = parse_args()
    spec = ModelCatalog.get(args.tier)
    print(f"[eval] initializing adapter {spec.adapter}")
    base_adapter = spec.build_adapter(
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
    chroma_dir = ROOT.parent / "data" / "chroma" / "db"
    retriever = ChromaRetriever(persist_dir=str(chroma_dir))
    adapter = RagAdapter(base_adapter, retriever)
    print("[eval] adapter ready:", type(adapter).__name__)

    # One-time warmup to reduce first-token latency.
    model = getattr(adapter, "model", None)
    tokenizer = getattr(adapter, "tokenizer", None)
    if model is None or tokenizer is None:
        base_adapter = getattr(adapter, "base_adapter", None)
        if base_adapter is not None:
            model = getattr(base_adapter, "model", model)
            tokenizer = getattr(base_adapter, "tokenizer", tokenizer)
    if model is None or tokenizer is None:
        raise AttributeError(
            "Adapter must expose model/tokenizer for warmup (adapter.model or adapter.base_adapter)."
        )

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
