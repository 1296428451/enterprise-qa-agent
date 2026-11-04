from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.embedding import LocalEmbedder
from src.q3.models import Chunk
from src.q3.retrieval import BM25

DEFAULT_DATA = ROOT / "data" / "external" / "frames_test.tsv"
DEFAULT_MODEL = Path(r"E:/Users/ASUS/PycharmProjects/pythonProject1/MyServer/models/AI-ModelScope/all-MiniLM-L6-v2")


def canonical_url(url: str) -> str:
    return url.strip()


def title_from_url(url: str) -> str:
    value = url.strip()
    if not value.startswith("http"):
        value = "https://" + value
    parsed = urlsplit(value)
    if parsed.netloc.lower() == "en.wikipedia.org" and parsed.path == "/w/index.php":
        params = parse_qs(parsed.query)
        title = params.get("title", params.get("search", [value]))[0]
    elif parsed.netloc.lower() == "w.wiki":
        title = value
    else:
        title = unquote(parsed.path.rsplit("/", 1)[-1].split("#", 1)[0])
    return title.replace("_", " ")


def load_frames(path: str | Path) -> tuple[list[dict], dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if not rows or "Prompt" not in rows[0] or "wiki_links" not in rows[0]:
        raise ValueError("FRAMES TSV must contain Prompt and wiki_links columns")
    candidates: dict[str, str] = {}
    for row in rows:
        links = ast.literal_eval(row["wiki_links"])
        if not isinstance(links, list) or not links:
            raise ValueError("each query must have a non-empty wiki_links list")
        row["gold_ids"] = [canonical_url(link) for link in links]
        for link in links:
            candidates[canonical_url(link)] = title_from_url(link)
    return rows, candidates


def retrieval_metrics(ranked: list[list[str]], gold_sets: list[set[str]], cutoffs=(5, 10)) -> dict[str, float | int]:
    count = len(gold_sets)
    if len(ranked) != count or not count:
        raise ValueError("ranked results and gold sets must have equal non-zero length")
    output: dict[str, float | int] = {"samples": count}
    for cutoff in cutoffs:
        output[f"recall@{cutoff}"] = sum(
            len(set(result[:cutoff]) & gold) / len(gold) for result, gold in zip(ranked, gold_sets)
        ) / count
    output["mrr@10"] = sum(
        next((1.0 / rank for rank, item in enumerate(result[:10], 1) if item in gold), 0.0)
        for result, gold in zip(ranked, gold_sets)
    ) / count
    def dcg(result: list[str], gold: set[str]) -> float:
        return sum((1.0 / math.log2(rank + 1)) for rank, item in enumerate(result[:10], 1) if item in gold)
    output["ndcg@10"] = sum(
        dcg(result, gold) / sum(1.0 / math.log2(rank + 1) for rank in range(1, min(10, len(gold)) + 1))
        for result, gold in zip(ranked, gold_sets)
    ) / count
    return output


def rank_bm25(rows: list[dict], candidates: dict[str, str]) -> list[list[str]]:
    chunks = [Chunk(key, key, "title-only", "", title) for key, title in sorted(candidates.items())]
    index = BM25(chunks)
    return [[item.chunk.chunk_id for item in index.search(row["Prompt"], 10)] for row in rows]


def rank_dense(rows: list[dict], candidates: dict[str, str], model_path: str | Path, batch_size: int) -> list[list[str]]:
    ids = sorted(candidates)
    embedder = LocalEmbedder(model_path)
    vectors = embedder.encode([candidates[key] for key in ids], batch_size=batch_size)
    queries = embedder.encode([row["Prompt"] for row in rows], batch_size=batch_size)
    import numpy as np
    scores = queries @ vectors.T
    return [[ids[index] for index in np.argsort(-scores[row_index], kind="stable")[:10]] for row_index in range(len(rows))]


def reciprocal_rank_fusion(*rankings: list[list[str]], limit: int = 10, constant: int = 60) -> list[list[str]]:
    fused = []
    for per_query in zip(*rankings):
        scores: dict[str, float] = {}
        for ranking in per_query:
            for rank, item in enumerate(ranking, 1):
                scores[item] = scores.get(item, 0.0) + 1.0 / (constant + rank)
        fused.append([item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]])
    return fused


def evaluate(data_path: str | Path, model_path: str | Path, include_rrf: bool = True, batch_size: int = 32) -> dict:
    rows, candidates = load_frames(data_path)
    gold = [set(row["gold_ids"]) for row in rows]
    bm25 = rank_bm25(rows, candidates)
    dense = rank_dense(rows, candidates, model_path, batch_size)
    methods = {"bm25_title_only": bm25, "all_minilm_l6_v2_title_only": dense}
    if include_rrf:
        methods["rrf_bm25_dense_title_only"] = reciprocal_rank_fusion(bm25, dense)
    report = {
        "benchmark": "Google FRAMES test split",
        "evaluation_mode": "title-only gold-link proxy; not full-text RAG",
        "queries": len(rows),
    "candidates": len(candidates),
    "gold_links": sum(len(items) for items in gold),
    "unique_candidate_titles": len(set(candidates.values())),
        "embedding_model": str(Path(model_path).resolve()),
        "methods": {name: retrieval_metrics(ranking, gold) for name, ranking in methods.items()},
        "limitations": [
            "Wikipedia article bodies were unavailable because the article API/Hugging Face access failed due to proxy TLS expiry.",
            "Each candidate contains only a Wikipedia title; this measures title matching against provided gold links, not evidence retrieval from article text.",
            "Gold links are supporting-article sets, not necessarily unique minimal paths; metrics are document-level and binary relevance.",
            "Dense rankings use local all-MiniLM-L6-v2 embeddings and cosine similarity; RRF uses k=60.",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate FRAMES title-only gold-link retrieval")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-rrf", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "external" / "frames_eval_report.json")
    args = parser.parse_args(argv)
    report = evaluate(args.data, args.model, not args.no_rrf, args.batch_size)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
