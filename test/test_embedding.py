import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.embedding import LocalEmbedder


class EmbeddingTests(unittest.TestCase):
    def test_local_model_smoke(self):
        try:
            import torch
            import transformers
        except ModuleNotFoundError:
            self.skipTest("py312 embedding dependencies are not active")
        model_path = Path(r"E:/Users/ASUS/PycharmProjects/pythonProject1/MyServer/models/AI-ModelScope/all-MiniLM-L6-v2")
        if not model_path.exists():
            self.skipTest("local embedding model is not available")
        embedder = LocalEmbedder(model_path)
        vectors = embedder.encode(["报销期限多久？", "差旅报销需要什么材料？"])
        self.assertEqual(vectors.shape[0], 2)
        self.assertGreater(vectors.shape[1], 0)
        self.assertAlmostEqual(float((vectors[0] ** 2).sum()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
