"""Benchmark the library corpus: python bench.py --strategy all --backend mock."""
from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

from src.chunking import FixedSizeChunker, RecursiveChunker, compute_similarity
from src.embeddings import LocalEmbedder, MockEmbedder
from src.models import Document
from src.store import EmbeddingStore


ROOT = Path(__file__).resolve().parent
QUERIES = [
    {
        "question": "Với giáo trình tại phòng 111, được mượn tối đa bao nhiêu cuốn, trong bao lâu và được gia hạn thế nào?",
        "filter": {"audience": "student"},
        "gold_doc_id": "student-textbook-borrowing",
        "gold_section": "2. Chính sách mượn giáo trình",
        "gold_answer": "Tối đa 8 cuốn, 90 ngày; gia hạn 01 lần thêm 30 ngày, tổng tối đa 120 ngày.",
    },
    {
        "question": "Chính sách mượn sách tham khảo tại phòng 102 quy định số lượng, thời hạn và gia hạn như thế nào?",
        "filter": {},
        "gold_doc_id": "student-reference-book-borrowing",
        "gold_section": "2. Chính sách mượn sách tham khảo",
        "gold_answer": "Tối đa 5 cuốn, từ 1 đến 30 ngày; gia hạn 01 lần trong 7 ngày.",
    },
    {
        "question": "Phòng học nhóm phục vụ vào thời gian nào và quy trình nhận trả chìa khóa ra sao?",
        "filter": {},
        "gold_doc_id": "group-study-room",
        "gold_section": "2. Thời gian sử dụng; 3. Quy trình đăng ký và sử dụng phòng học nhóm",
        "gold_answer": "Thứ Hai đến thứ Sáu 08h00-21h00, thứ Bảy và Chủ nhật 08h00-16h00; đăng ký rồi để lại thẻ tại P.411 để nhận chìa khóa, sau khi dùng trả chìa khóa để nhận lại thẻ.",
    },
    {
        "question": "Khi quên mật khẩu tài khoản thư viện, bạn đọc cần thực hiện các bước nào?",
        "filter": {},
        "gold_doc_id": "library-account",
        "gold_section": "1. Hướng dẫn reset mật khẩu",
        "gold_answer": "Truy cập libopac.hust.edu.vn, chọn Đăng nhập, chọn Quên mật khẩu / Quên mã PIN, nhập mã số thẻ, mở liên kết trong email rồi nhập và gửi mật khẩu mới.",
    },
    {
        "question": "Bạn đọc ngoài HUST có thể đọc toàn văn tài nguyên số không và cần điều kiện gì?",
        "filter": {},
        "gold_doc_id": "digital-resource-faq",
        "gold_section": "Câu 4; Câu 5",
        "gold_answer": "Có. Bạn đọc ngoài HUST cần đăng ký làm thẻ hoặc tài khoản thư viện; sau khi đăng nhập có thể đọc toàn văn tài liệu số và tải các tài liệu học liệu mở.",
    },
]


def read_markdown(path: Path) -> tuple[dict, str]:
    """Read the corpus's flat string-valued frontmatter (not arbitrary YAML)."""
    text = path.read_text(encoding="utf-8-sig")
    match = re.match(r"\A---\s*\n(.*?)\n---[ \t]*(?:\n|$)(.*)\Z", text, re.S)
    if not match:
        raise ValueError(f"{path}: missing or invalid frontmatter")
    metadata = {}
    for line in match[1].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or line[:1].isspace():
            raise ValueError(f"{path}: expected flat key: value metadata")
        key, value = key.strip(), value.strip()
        if key in metadata:
            raise ValueError(f"{path}: duplicate metadata key {key}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        metadata[key] = value
    required = {"doc_id", "title", "source_url", "retrieved_at", "document_version", "audience"}
    if required - metadata.keys():
        raise ValueError(f"{path}: missing metadata {sorted(required - metadata.keys())}")
    if metadata["doc_id"] != path.stem:
        raise ValueError(f"{path}: doc_id must match filename")
    content = match[2].strip()
    if not content:
        raise ValueError(f"{path}: empty document body")
    return metadata, content


def heading_chunks(text: str) -> list[str]:
    """Keep each level-2 section, including its level-3 subsections, intact."""
    sections = [part.strip() for part in re.split(r"(?m)(?=^## )", text) if part.strip()]
    if len(sections) > 1 and not sections[0].startswith("## "):
        sections[1] = sections[0] + "\n\n" + sections[1]
        sections = sections[1:]
    return sections


def semantic_chunks(text: str, embed, chunk_size: int, threshold: float) -> list[str]:
    """Split when adjacent sentence/line embeddings diverge, with a size cap."""
    units = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", text) if part.strip()]
    chunks, current, previous = [], "", None
    for unit in units:
        vector = embed(unit)
        candidate = f"{current}\n{unit}" if current else unit
        boundary = previous is not None and compute_similarity(previous, vector) < threshold
        if current and (boundary or len(candidate) > chunk_size):
            chunks.append(current)
            current = ""
        if len(unit) > chunk_size:
            pieces = FixedSizeChunker(chunk_size, 0).chunk(unit)
            chunks.extend(pieces[:-1])
            current = pieces[-1]
        else:
            current = f"{current}\n{unit}" if current else unit
        previous = vector
    if current:
        chunks.append(current)
    return chunks


def run_benchmark(args) -> dict:
    paths = sorted(args.data_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown documents in {args.data_dir}")
    corpus = [(path, *read_markdown(path)) for path in paths]
    if args.strategy == "semantic" and args.backend == "mock":
        raise ValueError("Semantic Chunking requires --backend local; mock hashes do not encode meaning")
    backend = LocalEmbedder() if args.backend == "local" else MockEmbedder()
    embed = lru_cache(maxsize=None)(backend)
    strategies = ["fixed", "recursive", "heading", "semantic"] if args.strategy == "all" else [args.strategy]
    report = {
        "backend": backend._backend_name,
        "mock": args.backend == "mock",
        "parameters": {"chunk_size": args.chunk_size, "overlap": args.overlap, "semantic_threshold": args.threshold, "top_k": 3},
        "corpus": [{"doc_id": p.stem, "characters": len(body), "metadata": meta} for p, meta, body in corpus],
        "strategies": {},
    }
    print(f"Backend: {report['backend']} | Documents: {len(corpus)}")
    if report["mock"]:
        print("MOCK: pipeline smoke test only; scores do not measure semantic relevance.")
    for strategy in strategies:
        if strategy == "semantic" and report["mock"]:
            report["strategies"][strategy] = {"status": "skipped", "reason": "Requires real embeddings (--backend local)"}
            print("\nsemantic: SKIPPED (requires real embeddings)")
            continue
        store = EmbeddingStore(collection_name=f"bench_{strategy}", embedding_fn=embed)
        stats = []
        for path, metadata, content in corpus:
            if strategy == "fixed":
                chunks = FixedSizeChunker(args.chunk_size, args.overlap).chunk(content)
            elif strategy == "recursive":
                chunks = RecursiveChunker(chunk_size=args.chunk_size).chunk(content)
            elif strategy == "heading":
                chunks = heading_chunks(content)
            else:
                chunks = semantic_chunks(content, embed, args.chunk_size, args.threshold)
            documents = [Document(
                id=f"{path.stem}#{i}", content=chunk,
                metadata={**metadata, "doc_id": path.stem, "source": metadata["source_url"], "chunk_index": i, "strategy": strategy},
            ) for i, chunk in enumerate(chunks)]
            store.add_documents(documents)
            stats.append({"doc_id": path.stem, "count": len(chunks), "avg_length": sum(map(len, chunks)) / len(chunks), "max_length": max(map(len, chunks))})
        result = {"status": "completed", "chunk_count": store.get_collection_size(), "documents": stats, "queries": []}
        print(f"\n=== {strategy}: {result['chunk_count']} chunks ===")
        for number, query in enumerate(QUERIES, 1):
            hits = store.search_with_filter(query["question"], top_k=3, metadata_filter=query["filter"])
            gold_rank = next((rank for rank, hit in enumerate(hits, 1) if hit["metadata"]["doc_id"] == query["gold_doc_id"]), None)
            result["queries"].append({**query, "top3": hits, "gold_document_rank": gold_rank})
            print(f"\nQ{number}: {query['question']}\nFilter: {query['filter']}\nGold [{query['gold_doc_id']} / {query['gold_section']}]: {query['gold_answer']}")
            for rank, hit in enumerate(hits, 1):
                print(f"  {rank}. score={hit['score']:.6f} doc_id={hit['metadata']['doc_id']} chunk_id={hit['id']}\n{hit['content']}\n")
            print(f"Gold document rank: {gold_rank} (document match only; check the chunk against gold manually)")
        report["strategies"][strategy] = result
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "library")
    parser.add_argument("--strategy", choices=["all", "fixed", "recursive", "heading", "semantic"], default="all")
    parser.add_argument("--backend", choices=["mock", "local"], default="mock")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.chunk_size <= 0 or not 0 <= args.overlap < args.chunk_size:
        parser.error("Require chunk-size > 0 and 0 <= overlap < chunk-size")
    if not -1 <= args.threshold <= 1:
        parser.error("threshold must be between -1 and 1")
    try:
        run_benchmark(args)
    except (ValueError, ImportError, OSError) as error:
        parser.exit(1, f"Benchmark failed: {error}\n")


if __name__ == "__main__":
    main()
