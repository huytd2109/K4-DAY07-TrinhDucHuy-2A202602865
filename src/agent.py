from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if not question or not question.strip():
            return "No question was provided."

        if self.store.get_collection_size() == 0:
            return "I could not find any relevant information in the knowledge base."

        retrieved = self.store.search(question, top_k=max(1, top_k))
        if not retrieved:
            return "I could not find any relevant information in the knowledge base."

        context_blocks = []
        for index, result in enumerate(retrieved, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id") or f"document_{index}"
            context_blocks.append(f"[{index}] Source: {source}\n{result.get('content', '').strip()}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            "Use only the following retrieved context to answer the user's question.\n"
            "If the answer is not present in the context, say that the information is not available in the provided context.\n\n"
            f"Question: {question}\n\nRetrieved context:\n{context_text}\n\nAnswer:"
        )
        answer = self.llm_fn(prompt)
        return str(answer) if answer is not None else "No answer generated."
