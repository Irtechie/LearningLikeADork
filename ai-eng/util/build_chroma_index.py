# Purpose: Build chroma index.
# Created: 2026-01-06
# Author: MWR

import argparse
from pathlib import Path
import sys

try:
    # When run as a module: python -m util.build_chroma_index
    from .utility import build_chroma_index_from_dir
except ImportError:
    # Allow running as a script: python ai-eng\util\build_chroma_index.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from util.utility import build_chroma_index_from_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a persistent Chroma index from all JSONL files in a directory.")
    default_rag_dir = Path(__file__).resolve().parents[2] / "data" / "training" / "rag"
    default_persist_dir = Path(__file__).resolve().parents[2] / "data" / "chroma" / "db"
    parser.add_argument(
        "--rag-dir",
        default=str(default_rag_dir),
        help="Directory containing JSONL files.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(default_persist_dir),
        help="Directory for the Chroma DB.",
    )
    parser.add_argument("--collection-name", default="rag", help="Collection name.")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the collection even if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = build_chroma_index_from_dir(
        rag_dir=args.rag_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        overwrite=args.overwrite,
    )
    print(f"Indexed {count} documents into '{args.collection_name}' at {args.persist_dir}")


if __name__ == "__main__":
    main()
