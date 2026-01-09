# Purpose: RAG adapter wrapper that injects retrieved context before generation.
# Created: 2026-01-06
# Author: MWR

"""RAG adapter wrapper that injects retrieved context before generation."""

class RagAdapter:
    def __init__(self, base_adapter, retriever):
        self.base_adapter = base_adapter
        self.retriever = retriever
        # Expose base adapter model/tokenizer for callers expecting them.
        self.model = base_adapter.model
        self.tokenizer = base_adapter.tokenizer

    def generate(self, input: str) -> str:
        documents, metadatas = self.retriever.retrieve(input)

        context_lines = []
        for doc, meta in zip(documents, metadatas):
            entity = meta.get("entity", "unknown")
            context_lines.append(f"[{entity}] {doc}")

        context_block = "\n".join(context_lines).strip()
        prompt = (
            "Use the following context to answer.\n"
            "Rules:\n"
            "- Do not explain your reasoning\n"
            "- Do not restate the question\n"
            "- Answer concisely using only the context\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {input}\n"
            "Answer:"
     )

        return self.base_adapter.generate(prompt)
