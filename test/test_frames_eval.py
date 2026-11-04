import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.q3.evaluate_frames import canonicalize, load_frames, retrieval_metrics, rrf


class FramesEvaluationTests(unittest.TestCase):
    def test_canonicalize_filters_invalid_and_normalizes_fragment(self):
        key, title, error = canonicalize("https://en.wikipedia.org/wiki/World_War_I#History")
        self.assertEqual(key, "https://en.wikipedia.org/wiki/World_War_I")
        self.assertEqual(title, "World War I")
        self.assertIsNone(error)
        self.assertEqual(canonicalize("en.wikipedia.org/wiki/Grazia_Deledda")[2], "missing_scheme")
        self.assertEqual(canonicalize("https://en.wikipedia.org/w/index.php?search=x&title=Special:Search")[2], "special_search")
        self.assertEqual(canonicalize("https://notwikipedia.org/wiki/A")[2], "non_wikipedia_host")

    def test_metrics_include_micro_macro_and_all_gold(self):
        result = retrieval_metrics([["a", "x", "b"], ["x", "y"]], [{"a", "b"}, {"z"}])
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["empty_gold_samples_excluded"], 0)
        self.assertAlmostEqual(result["link_micro_recall@5"], 2 / 3)
        self.assertAlmostEqual(result["question_macro_recall@5"], 0.5)
        self.assertAlmostEqual(result["mrr@10"], 0.5)
        self.assertAlmostEqual(result["all_gold_hit@5"], 0.5)
        self.assertIn("all_gold_hit@10", result)
        self.assertGreater(result["ndcg@10"], 0)

    def test_rrf_is_deterministic(self):
        result = rrf([["a", "b"], ["b", "a"]], [["b", "c"], ["c", "b"]])
        self.assertEqual(result[0][0], "b")

    def test_frames_source_counts_and_exclusions(self):
        path = Path(__file__).resolve().parents[1] / "data/external/frames_test.tsv"
        rows, candidates, stats = load_frames(path)
        self.assertEqual(len(rows), 824)
        self.assertEqual(stats["gold_link_occurrences"], 2636)
        self.assertEqual(stats["raw_gold_url_count"], 2507)
        self.assertEqual(stats["raw_candidate_url_count"], 2507)
        self.assertEqual(stats["gold_links_absent_from_candidate_columns"], 0)
        self.assertEqual(stats["excluded_occurrences_by_reason"], {
            "missing_scheme": 2, "non_wikipedia_host": 1, "special_search": 1,
        })
        self.assertEqual(len(candidates), stats["valid_candidate_url_count"])


if __name__ == "__main__":
    unittest.main()
