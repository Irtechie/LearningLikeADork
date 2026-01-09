# Purpose: Module for prefetch models.
# Created: 2026-01-05
# Author: MWR

from huggingface_hub import snapshot_download
from pathlib import Path

# ---- ROOTS ----
ROOT = Path(r"C:\Playground")
MODELS_ROOT = ROOT / "data" / "models" / "local"
FINETUNES_ROOT = ROOT / "experiments"

# Ensure top-level dirs exist
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
FINETUNES_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "fast": {
        "hf_id": "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit",
        "local_name": "llama4-scout-17b-4bit",
    },
    "reasoning": {
        "hf_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit",
        "local_name": "deepseek-r1-qwen-14b-4bit",
    },
    "performance": {
        "hf_id": "unsloth/Qwen3-14B-bnb-4bit",
        "local_name": "qwen3-14b-4bit",
    },
    "full": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "local_name": "qwen2.5-3b-instruct",
    },
    # New: Smaller models
    "small": {
        "hf_id": "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit",
        "local_name": "mistral-7b-instruct-4bit",
    },
    "tiny": {
        "hf_id": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "local_name": "qwen2.5-3b-instruct-4bit",
    },
}


def download_model(tier: str, cfg: dict):
    tier_dir = MODELS_ROOT / tier
    model_dir = tier_dir / cfg["local_name"]

    # Ensure dirs exist
    tier_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Downloading {cfg['hf_id']} ===")
    print(f"-> Local path: {model_dir}")

    snapshot_download(
        repo_id=cfg["hf_id"],
        repo_type="model",
        local_dir=model_dir,
        local_dir_use_symlinks=False,  # Windows-safe
        resume_download=True,
    )

    print(f"Done: {cfg['local_name']}")


if __name__ == "__main__":
    for tier, cfg in MODELS.items():
        download_model(tier, cfg)
