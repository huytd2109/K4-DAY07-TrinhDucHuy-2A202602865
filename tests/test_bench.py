from argparse import Namespace
from pathlib import Path

import pytest

from bench import heading_chunks, read_markdown, run_benchmark, semantic_chunks


def test_frontmatter_preserves_url_and_excludes_metadata(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(
        '\ufeff---\ndoc_id: sample\ntitle: "Sample"\n'
        'source_url: https://example.org/page\nretrieved_at: 2026-09-19\n'
        'document_version: not-stated\naudience: student\n---\n\n# Body\n',
        encoding="utf-8",
    )
    metadata, content = read_markdown(path)
    assert metadata["source_url"] == "https://example.org/page"
    assert metadata["title"] == "Sample"
    assert content == "# Body"


def test_missing_frontmatter_fails_with_filename(tmp_path):
    path = tmp_path / "invalid.md"
    path.write_text("# Body", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid.md"):
        read_markdown(path)


def test_heading_preserves_subsections_and_preamble():
    assert heading_chunks("# Title\n\n## First\nA\n### Sub\nB\n\n## Second\nC") == [
        "# Title\n\n## First\nA\n### Sub\nB", "## Second\nC",
    ]


def test_semantic_splits_at_topic_change():
    vectors = {"Cats.": [1, 0], "Kittens.": [1, 0], "Cars.": [0, 1]}
    assert semantic_chunks("Cats. Kittens. Cars.", vectors.__getitem__, 500, 0.5) == [
        "Cats.\nKittens.", "Cars.",
    ]


def test_semantic_preserves_long_unit():
    chunks = semantic_chunks("abcdefghijk", lambda _: [1, 0], 4, 0.5)
    assert "".join(chunks) == "abcdefghijk"
    assert all(len(chunk) <= 4 for chunk in chunks)


def test_benchmark_filters_and_exports(tmp_path, capsys):
    args = Namespace(
        data_dir=Path(__file__).resolve().parents[1] / "data" / "library",
        strategy="all", backend="mock", chunk_size=500, overlap=50,
        threshold=0.5, output=tmp_path / "results.json",
    )
    result = run_benchmark(args)
    assert len(result["corpus"]) == 8
    assert result["strategies"]["semantic"]["status"] == "skipped"
    for name in ("fixed", "recursive", "heading"):
        queries = result["strategies"][name]["queries"]
        assert len(queries) == 5
        for query in queries:
            assert len(query["top3"]) == 3
            scores = [hit["score"] for hit in query["top3"]]
            assert scores == sorted(scores, reverse=True)
        assert all(hit["metadata"]["audience"] == "student" for hit in queries[0]["top3"])
    assert args.output.is_file()
    assert "score=" in capsys.readouterr().out
