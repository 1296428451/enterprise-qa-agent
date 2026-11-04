from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.q3.embedding import LocalEmbedder
from src.q3.models import Chunk
from src.q3.retrieval import BM25

DEFAULT_DATA = ROOT / "data/external/frames_test.tsv"
DEFAULT_REPORT = ROOT / "data/external/frames_eval_report.json"


def canonicalize(url: str) -> tuple[str | None, str | None, str | None]:
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None, None, "missing_scheme"
    host = (parsed.hostname or "").lower()
    if host != "wikipedia.org" and not host.endswith(".wikipedia.org"):
        return None, None, "non_wikipedia_host"
    path = unquote(parsed.path)
    if path == "/w/index.php":
        title = parse_qs(parsed.query).get("title", [None])[0]
        if not title or title.casefold().startswith("special:"):
            return None, None, "special_search"
        path = "/wiki/" + title.replace(" ", "_")
    if not path.startswith("/wiki/"):
        return None, None, "non_article_path"
    path = path.split("#", 1)[0]
    title = unquote(path[len("/wiki/"):]).replace("_", " ")
    title = " ".join(title.split())
    if not title or title.casefold().startswith("special:"):
        return None, None, "special_namespace"
    canonical_path = "/wiki/" + quote(title.replace(" ", "_"), safe="()!,'-._~")
    return f"{parsed.scheme.lower()}://{host}{canonical_path}", title, None


def load_frames(path: str | Path) -> tuple[list[dict], dict[str, str], dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        rows = [r for r in csv.DictReader(source, delimiter="\t") if r.get("Prompt", "").strip()]
    required = {"Prompt", "wiki_links", "reasoning_types"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("TSV must contain Prompt, wiki_links and reasoning_types columns")
    candidates: dict[str, str] = {}
    raw_gold_urls: set[str] = set()
    raw_candidate_urls: set[str] = set()
    excluded_unique: dict[str, str] = {}
    excluded_occurrences: Counter[str] = Counter()
    gold_occurrences = candidate_occurrences = used_gold_links = 0
    empty_gold_rows = 0
    for row in rows:
        gold_links = ast.literal_eval(row.get("wiki_links", "[]"))
        if not isinstance(gold_links, list):
            raise ValueError("wiki_links must parse as a list")
        row["gold_ids"] = []
        for url in gold_links:
            if not isinstance(url, str):
                raise ValueError("wiki_links entries must be strings")
            gold_occurrences += 1
            raw_gold_urls.add(url)
        column_links = [value.strip() for key, value in row.items()
                        if key.startswith("wikipedia_link_") and value.strip()]
        for url in column_links:
            if not isinstance(url, str):
                raise ValueError("candidate links must be strings")
            raw_candidate_urls.add(url)
            candidate_occurrences += 1
            key, title, error = canonicalize(url)
            if error:
                excluded_occurrences[error] += 1
                excluded_unique[url] = error
            else:
                candidates[key] = title
        for url in gold_links:
            key, _, error = canonicalize(url)
            if error:
                continue
            row["gold_ids"].append(key)
            used_gold_links += 1
        row["gold_ids"] = sorted(set(row["gold_ids"]))
        row["empty_gold"] = not bool(row["gold_ids"])
        empty_gold_rows += row["empty_gold"]
    stats = {
        "non_empty_rows": len(rows),
        "gold_link_occurrences": gold_occurrences,
        "raw_gold_url_count": len(raw_gold_urls),
        "raw_candidate_url_count": len(raw_candidate_urls),
        "candidate_link_occurrences": candidate_occurrences,
        "gold_links_absent_from_candidate_columns": len({key for row in rows for key in row["gold_ids"]} - set(candidates)),
        "used_gold_links": used_gold_links,
        "valid_candidate_url_count": len(candidates),
        "distinct_normalized_titles": len({v.casefold() for v in candidates.values()}),
        "empty_gold_count": empty_gold_rows,
        "excluded_unique_url_count": len(excluded_unique),
        "excluded_occurrences_by_reason": dict(sorted(excluded_occurrences.items())),
        "excluded_urls": excluded_unique,
    }
    return rows, candidates, stats


def retrieval_metrics(ranked: list[list[str]], gold_sets: list[set[str]]) -> dict:
    if len(ranked) != len(gold_sets) or not gold_sets:
        raise ValueError("ranked results and gold sets must have equal non-zero length")
    valid = [(r, g) for r, g in zip(ranked, gold_sets) if g]
    total_gold = sum(len(g) for _, g in valid)
    result = {"samples": len(gold_sets), "non_empty_gold_samples": len(valid),
              "empty_gold_samples_excluded": len(gold_sets) - len(valid)}
    for k in (1, 5, 10):
        hits = [len(set(r[:k]) & g) for r, g in valid]
        result[f"link_micro_recall@{k}"] = sum(hits) / total_gold if total_gold else 0.0
        result[f"question_macro_recall@{k}"] = (sum(h / len(g) for h, (_, g) in zip(hits, valid)) / len(valid) if valid else 0.0)
        result[f"all_gold_hit@{k}"] = (sum(g.issubset(set(r[:k])) for r, g in valid) / len(valid) if valid else 0.0)
    result["mrr@10"] = (sum(next((1 / rank for rank, item in enumerate(r[:10], 1) if item in g), 0.0)
                               for r, g in valid) / len(valid) if valid else 0.0)
    def dcg(r: list[str], g: set[str]) -> float:
        return sum(1 / math.log2(rank + 1) for rank, item in enumerate(r[:10], 1) if item in g)
    result["ndcg@10"] = (sum(dcg(r, g) / sum(1 / math.log2(k + 1) for k in range(1, min(10, len(g)) + 1))
                                for r, g in valid) / len(valid) if valid else 0.0)
    return result


def stratify(rows: list[dict], rankings: list[list[str]]) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        for label in set(x.strip() for x in row.get("reasoning_types", "").split("|") if x.strip()):
            groups[label].append(i)
    return {label: {**retrieval_metrics([rankings[i] for i in ids],
                                        [set(rows[i]["gold_ids"]) for i in ids]),
                    "label_semantics": "overlapping multi-label; queries may occur in multiple groups"}
            for label, ids in sorted(groups.items())}


def rank_bm25(rows: list[dict], candidates: dict[str, str]) -> list[list[str]]:
    chunks = [Chunk(key, key, "title-only", "", text) for key, text in sorted(candidates.items())]
    index = BM25(chunks)
    return [[x.chunk.chunk_id for x in index.search(row["Prompt"], 10)] for row in rows]


def rank_dense(rows: list[dict], candidates: dict[str, str], model: str, batch_size: int) -> list[list[str]]:
    import numpy as np
    ids = sorted(candidates)
    embedder = LocalEmbedder(model)
    vectors = embedder.encode([candidates[x] for x in ids], batch_size=batch_size)
    queries = embedder.encode([r["Prompt"] for r in rows], batch_size=batch_size)
    scores = queries @ vectors.T
    return [[ids[i] for i in np.argsort(-scores[q], kind="stable")[:10]] for q in range(len(rows))]


def rrf(first: list[list[str]], second: list[list[str]], k: int = 60) -> list[list[str]]:
    output = []
    for left, right in zip(first, second):
        scores: dict[str, float] = {}
        for ranking in (left, right):
            for rank, item in enumerate(ranking, 1):
                scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
        output.append([x for x, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:10]])
    return output


def evaluate(data: str | Path, method: str, model: str | None, batch_size: int = 32) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    rows, candidates, counts = load_frames(data)
    bm25 = rank_bm25(rows, candidates) if method in {"bm25", "rrf"} else None
    dense = rank_dense(rows, candidates, model, batch_size) if method in {"dense", "rrf"} else None
    rankings = {"bm25_title_only": bm25} if method == "bm25" else (
        {"minilm_dense_title_only": dense} if method == "dense" else
        {"bm25_title_only": bm25, "minilm_dense_title_only": dense, "rrf_title_only": rrf(bm25, dense)})
    gold = [set(r["gold_ids"]) for r in rows]
    methods = {name: {"overall": retrieval_metrics(rank, gold), "by_reasoning_type": stratify(rows, rank)}
               for name, rank in rankings.items()}
    cases = []
    for i, row in enumerate(rows):
        case = {"row": i + 1, "source_id": row.get("", str(i)), "prompt": row["Prompt"],
                "reasoning_types": [x.strip() for x in row.get("reasoning_types", "").split("|") if x.strip()],
                "gold": sorted(set(row["gold_ids"])), "empty_gold": not bool(row["gold_ids"])}
        for name, ranked in rankings.items():
            top = ranked[i][:10]
            case[name] = {"top10": top, "matches": [x for x in top if x in set(row["gold_ids"])]}
        cases.append(case)
    data_bytes = Path(data).read_bytes()
    return {
        "benchmark": "Google FRAMES official test split",
        "evaluation_mode": "title-only gold-link proxy; not full-text RAG or QA accuracy",
        "candidate_id": "canonical URL: lowercase scheme/host, URL-decoded path; query and fragment removed",
        "candidate_text": "title extracted from URL path or index.php title parameter, URL-decoded and underscores changed to spaces",
        "qrels": "binary URL-ID relevance from canonical wiki_links; Answer is not used; empty gold rows are excluded from hit metrics and reported separately",
        "corpus": "Wikipedia_link_* columns from FRAMES test rows, canonicalized to title-only candidates; wiki_links are relevance judgments",
        "counts": counts,
        "run": {"started_at_utc": started, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                 "method": method, "python": platform.python_version(),
                 "model": Path(model).name if model else None, "rrf_k": 60 if method == "rrf" else None},
        "source_file": {"source_url": "https://huggingface.co/datasets/google/frames-benchmark/raw/main/test.tsv",
                          "license": "Apache-2.0", "split": "official test", "bytes": len(data_bytes),
                          "sha256_local": hashlib.sha256(data_bytes).hexdigest(),
                          "hash_note": "Local supplied file hash verified; current upstream bytes could not be fetched for comparison."},
        "metrics": methods, "cases": cases,
        "limitations": ["Wikipedia article bodies were unavailable because proxy TLS validation failed.",
                          "This measures title matching only, not full-text evidence retrieval or answer accuracy.",
                          "The candidate corpus is closed-world and derived from test-split gold links.",
                          "Gold links are document-level support labels, not necessarily a minimal evidence path.",
                          "Reasoning-type groups overlap; one question may occur in multiple strata."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FRAMES title-only gold-link proxy evaluation")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--method", choices=("bm25", "dense", "rrf"), default="bm25")
    parser.add_argument("--model", help="Local all-MiniLM-L6-v2 model path (or FRAMES_EMBEDDING_MODEL)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    model = args.model or os.environ.get("FRAMES_EMBEDDING_MODEL")
    if args.method in {"dense", "rrf"} and not model:
        parser.error("dense and rrf methods require --model or FRAMES_EMBEDDING_MODEL")
    report = evaluate(args.data, args.method, model, args.batch_size)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
