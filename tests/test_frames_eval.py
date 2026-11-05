import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.evaluate_frames import canonicalize, load_frames, retrieval_metrics, rrf


def test_canonicalize_filters_invalid_and_normalizes_fragment():
    key, title, error = canonicalize("https://en.wikipedia.org/wiki/World_War_I#History")
    assert key == "https://en.wikipedia.org/wiki/World_War_I"
    assert title == "World War I"
    assert error is None
    assert canonicalize("en.wikipedia.org/wiki/Grazia_Deledda")[2] == "missing_scheme"
    assert canonicalize("https://en.wikipedia.org/w/index.php?search=x&title=Special:Search")[2] == "special_search"
    assert canonicalize("https://notwikipedia.org/wiki/A")[2] == "non_wikipedia_host"


def test_metrics_include_micro_macro_and_all_gold():
    result = retrieval_metrics([["a", "x", "b"], ["x", "y"]], [{"a", "b"}, {"z"}])
    assert result["samples"] == 2
    assert result["empty_gold_samples_excluded"] == 0
    assert result["link_micro_recall@5"] == pytest.approx(2 / 3)
    assert result["question_macro_recall@5"] == pytest.approx(0.5)
    assert result["mrr@10"] == pytest.approx(0.5)
    assert result["all_gold_hit@5"] == pytest.approx(0.5)
    assert "all_gold_hit@10" in result
    assert result["ndcg@10"] > 0


def test_rrf_is_deterministic():
    result = rrf([["a", "b"], ["b", "a"]], [["b", "c"], ["c", "b"]])
    assert result[0][0] == "b"


def test_frames_source_counts_and_exclusions():
    path = Path(__file__).resolve().parents[1] / "data/external/frames_test.tsv"
    rows, candidates, stats = load_frames(path)
    assert len(rows) == 824
    assert stats["gold_link_occurrences"] == 2636
    assert stats["raw_gold_url_count"] == 2507
    assert stats["raw_candidate_url_count"] == 2507
    assert stats["gold_links_absent_from_candidate_columns"] == 0
    assert stats["excluded_occurrences_by_reason"] == {
        "missing_scheme": 2, "non_wikipedia_host": 1, "special_search": 1,
    }
    assert len(candidates) == stats["valid_candidate_url_count"]
