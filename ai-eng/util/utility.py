# Purpose: Module for utility.
# Created: 2026-01-06
# Author: MWR

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def build_chroma_index_from_dir(
    rag_dir: str,
    persist_dir: str,
    collection_name: str = "rag",
    embedding_model: str = "all-MiniLM-L6-v2",
    overwrite: bool = False,
) -> int:
    rag_dir = Path(rag_dir)
    jsonl_files = sorted(rag_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {rag_dir}")

    model = SentenceTransformer(embedding_model)
    client = chromadb.PersistentClient(path=str(persist_dir))

    if overwrite:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=collection_name)
    if not overwrite and collection.count() > 0:
        return collection.count()

    docs = []
    metadatas = []
    ids = []

    for jsonl_path in jsonl_files:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_id = f"{jsonl_path.stem}:{record['id']}"
                ids.append(record_id)
                docs.append(record["text"])
                metadatas.append(record.get("metadata", {}))

    embeddings = model.encode(docs, convert_to_numpy=True).tolist()
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(ids)
