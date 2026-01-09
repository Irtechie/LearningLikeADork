# Purpose: Retriever implementations.
# Created: 2026-01-06
# Author: MWR

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class ChromaRetriever:
    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "rag",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def retrieve(self, query: str, k: int = 3):
        query_emb = self.model.encode([query], convert_to_numpy=True).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["documents", "metadatas"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return documents, metadatas


class SimpleRetriever:
    def __init__(self, jsonl_path: str, collection_name: str = "rag"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

        docs = []
        metadatas = []
        ids = []

        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ids.append(str(record["id"]))
                docs.append(record["text"])
                metadatas.append(record.get("metadata", {}))

        embeddings = self.model.encode(docs, convert_to_numpy=True).tolist()
        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def retrieve(self, query: str, k: int = 3):
        query_emb = self.model.encode([query], convert_to_numpy=True).tolist()
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["documents", "metadatas"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return documents, metadatas
