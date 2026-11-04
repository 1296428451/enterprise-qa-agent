import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.models import Chunk, ScoredChunk
from src.q3.query_aware import (
    QUERY_AWARE_PROMPT,
    MockQueryAwareExtractor,
    OpenAICompatibleQueryAwareExtractor,
    estimate_tokens,
    split_sentences,
)

NOISY_TEXT = (
    "本公司成立于2010年，总部位于北京。年会筹备工作由行政部负责。"
    "普通费用报销应在费用发生后 30 天内提交。适用于所有正式员工。"
    "公司食堂提供免费午餐。"
)


def _chunk(text: str, doc_id: str = "D01", chunk_id: str = "D01-v1-c01") -> Chunk:
    return Chunk(chunk_id=chunk_id, document_id=doc_id, version="v1",
                 version_date="2026-01-01", text=text)


class QueryAwareTests(unittest.TestCase):
    def test_prompt_contains_rules_example_and_json_format(self):
        self.assertIn("extract only the text spans", QUERY_AWARE_PROMPT)
        self.assertIn("extractive, not abstractive", QUERY_AWARE_PROMPT)
        self.assertIn("Example", QUERY_AWARE_PROMPT)
        self.assertIn("JSON only", QUERY_AWARE_PROMPT)
        self.assertIn("key_info", QUERY_AWARE_PROMPT)
        self.assertIn("compression_note", QUERY_AWARE_PROMPT)

    def test_split_sentences_handles_chinese_punctuation(self):
        sentences = split_sentences("第一句。第二句！第三句？")
        self.assertEqual(len(sentences), 3)

    def test_estimate_tokens_counts_cjk_and_ascii(self):
        self.assertGreater(estimate_tokens("报销"), 0)
        self.assertGreater(estimate_tokens("hello world"), 0)

    def test_mock_extractor_removes_noise_keeps_evidence(self):
        items = [ScoredChunk(_chunk(NOISY_TEXT, "D01", "D01-v1-c01"), bm25_score=0.9)]
        result = MockQueryAwareExtractor().extract("报销期限多久？", items)
        self.assertEqual(result.candidates_in, 1)
        self.assertEqual(result.candidates_with_evidence, 1)
        self.assertEqual(result.evidence[0].source_id, "D01")
        self.assertIn("30 天", result.evidence[0].quote)
        self.assertNotIn("年会筹备", result.evidence[0].quote)
        self.assertLess(result.compression_ratio, 1.0)

    def test_mock_extractor_marks_irrelevant_as_none(self):
        items = [ScoredChunk(_chunk("公司食堂提供免费午餐。", "D02", "D02-v1-c01"), bm25_score=0.5)]
        result = MockQueryAwareExtractor().extract("报销期限多久？", items)
        self.assertEqual(result.evidence[0].relevance, "none")

    def test_extraction_result_formats_answer_input(self):
        items = [
            ScoredChunk(_chunk(NOISY_TEXT, "D01", "D01-v1-c01"), bm25_score=0.9),
            ScoredChunk(_chunk("普通费用报销应在费用发生后 15 天内提交。", "D02", "D02-v1-c01"), bm25_score=0.8),
        ]
        result = MockQueryAwareExtractor().extract("报销期限多久？", items)
        answer_input = result.to_answer_input()
        self.assertIn("Question:", answer_input)
        self.assertIn("Evidence:", answer_input)
        self.assertIn("(D01)", answer_input)
        self.assertIn("(D02)", answer_input)

    def test_llm_extractor_parses_valid_response(self):
        llm = OpenAICompatibleQueryAwareExtractor(
            "https://example.invalid/v1", "test-model", "test-key")
        items = [ScoredChunk(_chunk("普通费用报销应在费用发生后 30 天内提交。", "D01", "D01-v1-c01"))]
        mock_response = {
            "evidence": [{"index": 1, "source_id": "D01", "relevance": "high",
                           "quote": "普通费用报销应在费用发生后 30 天内提交。", "key_info": "报销期限 30 天"}],
            "compression_note": "removed irrelevant clauses",
        }
        with patch.object(llm, "_call", return_value=mock_response):
            result = llm.extract("报销期限多久？", items)
        self.assertEqual(result.candidates_with_evidence, 1)
        self.assertEqual(result.evidence[0].relevance, "high")
        self.assertIn("30 天", result.evidence[0].quote)

    def test_llm_extractor_rejects_bad_index(self):
        llm = OpenAICompatibleQueryAwareExtractor(
            "https://example.invalid/v1", "test-model", "test-key")
        items = [ScoredChunk(_chunk("test", "D01", "D01-v1-c01"))]
        bad = {"evidence": [{"index": 99, "source_id": "D01", "relevance": "high", "quote": "x", "key_info": "y"}]}
        with patch.object(llm, "_call", return_value=bad):
            with self.assertRaises(ValueError):
                llm.extract("test", items)

    def test_llm_extractor_rejects_duplicate_index(self):
        llm = OpenAICompatibleQueryAwareExtractor(
            "https://example.invalid/v1", "test-model", "test-key")
        items = [ScoredChunk(_chunk("test", "D01", "D01-v1-c01"))]
        bad = {"evidence": [
            {"index": 1, "source_id": "D01", "relevance": "high", "quote": "a", "key_info": "b"},
            {"index": 1, "source_id": "D01", "relevance": "high", "quote": "c", "key_info": "d"},
        ]}
        with patch.object(llm, "_call", return_value=bad):
            with self.assertRaises(ValueError):
                llm.extract("test", items)


if __name__ == "__main__":
    unittest.main()
