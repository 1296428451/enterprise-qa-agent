import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.evaluate_frames import load_frames, retrieval_metrics, reciprocal_rank_fusion


class FramesEvaluationTests(unittest.TestCase):
    def test_metrics_use_document_level_gold_links(self):
        ranked = [["a", "x", "b"], ["x", "y"]]
        gold = [{"a", "b"}, {"z"}]
        result = retrieval_metrics(ranked, gold)
        self.assertEqual(result["samples"], 2)
        self.assertAlmostEqual(result["recall@5"], 0.5)
        self.assertAlmostEqual(result["mrr@10"], 0.5)
        self.assertGreater(result["ndcg@10"], 0)

    def test_rrf_combines_ranked_lists_deterministically(self):
        result = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], [["b", "c"], ["c", "b"]], limit=2)
        self.assertEqual(result[0][0], "b")
        self.assertEqual(result[1][0], "b")

    def test_frames_source_has_824_queries_and_2507_link_candidates(self):
        data_path = Path(__file__).resolve().parents[1] / "data/external/frames_test.tsv"
        rows, candidates = load_frames(data_path)
        self.assertEqual(len(rows), 824)
        self.assertEqual(len(candidates), 2507)
        self.assertGreater(len(set(candidates.values())), 2400)


if __name__ == "__main__":
    unittest.main()
