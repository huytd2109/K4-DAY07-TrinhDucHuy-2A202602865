from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sentence_pattern = r"(?<=[.!?])\s+|(?<=\n)\s*"
        raw_sentences = [part.strip() for part in re.split(sentence_pattern, text.strip()) if part and part.strip()]
        if not raw_sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        for sentence in raw_sentences:
            current.append(sentence)
            if len(current) >= self.max_sentences_per_chunk:
                chunks.append(" ".join(current).strip())
                current = []

        if current:
            chunks.append(" ".join(current).strip())

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        cleaned = text.strip()
        return self._split(cleaned, list(self.separators))

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text or not current_text.strip():
            return []

        cleaned = current_text.strip()
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        if not remaining_separators:
            fallback_chunks: list[str] = []
            for start in range(0, len(cleaned), self.chunk_size):
                fallback_chunks.append(cleaned[start : start + self.chunk_size])
            return fallback_chunks

        separator = remaining_separators[0]
        if separator == "":
            return self._split(cleaned, remaining_separators[1:]) if remaining_separators[1:] else [cleaned]

        if separator in ("\n\n", "\n"):
            split_parts = [part.strip() for part in cleaned.split(separator) if part and part.strip()]
        else:
            split_parts = [part.strip() for part in re.split(re.escape(separator), cleaned) if part and part.strip()]

        if len(split_parts) <= 1:
            return self._split(cleaned, remaining_separators[1:]) if remaining_separators[1:] else [cleaned[: self.chunk_size]]

        chunks: list[str] = []
        current = ""
        for part in split_parts:
            candidate = f"{current} {part}".strip() if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) <= self.chunk_size:
                    current = part
                else:
                    subchunks = self._split(part, remaining_separators[1:]) if remaining_separators[1:] else [part[: self.chunk_size]]
                    if current:
                        chunks.extend(subchunks)
                        current = ""
                    else:
                        chunks.extend(subchunks)
                        current = ""

        if current:
            chunks.append(current)

        if not chunks:
            return self._split(cleaned, remaining_separators[1:]) if remaining_separators[1:] else [cleaned[: self.chunk_size]]

        if len(chunks) == 1 and len(chunks[0]) > self.chunk_size:
            return self._split(chunks[0], remaining_separators[1:]) if remaining_separators[1:] else [chunks[0][: self.chunk_size]]

        # Merge tiny trailing pieces to keep chunks closer to the target size when possible.
        merged: list[str] = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue
            if len(merged[-1]) + len(chunk) <= self.chunk_size + max(1, self.chunk_size // 4):
                merged[-1] = f"{merged[-1]} {chunk}".strip()
            else:
                merged.append(chunk)

        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b:
        return 0.0

    dot_product = _dot(vec_a, vec_b)
    magnitude_a = math.sqrt(sum(value * value for value in vec_a))
    magnitude_b = math.sqrt(sum(value * value for value in vec_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10)),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }

        result: dict[str, dict] = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            count = len(chunks)
            avg_length = sum(len(chunk) for chunk in chunks) / count if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
