# Purpose: Run RAG inference.
# Created: 2026-01-06
# Author: MWR

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
AI_ENG_ROOT = ROOT
if str(AI_ENG_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ENG_ROOT))

from adapters.hf.local_hf_qwen25 import LocalHFAdapter
from rag.rag import RagAdapter
from retrievers.retriever import ChromaRetriever
from models.catalog import ModelCatalog, ModelNames

# 1. Load retriever (this indexes the JSONL)
chroma_dir = AI_ENG_ROOT.parent / "data" / "chroma" / "db"
retriever = ChromaRetriever(persist_dir=str(chroma_dir))

# 2. Load base model adapter
base = LocalHFAdapter(
    model_path=ModelCatalog.get(ModelNames.TINY).path,
    max_new_tokens=128,
)

# 3. Wrap with RAG
rag = RagAdapter(base, retriever)

# 4. Ask a question
question = "What abilities does Booty Booty have?"
answer = rag.generate(question)

print("\n=== QUESTION ===")
print(question)
print("\n=== ANSWER ===")
print(answer)
