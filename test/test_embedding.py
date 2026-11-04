import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.embedding import LocalEmbedder


def test_local_model_smoke():
    try:
        import torch
        import transformers
    except ModuleNotFoundError:
        pytest.skip("py312 embedding dependencies are not active")
    model_path_value = os.environ.get("FRAMES_EMBEDDING_MODEL")
    if not model_path_value:
        pytest.skip("set FRAMES_EMBEDDING_MODEL to a local embedding model directory")
    model_path = Path(model_path_value)
    if not model_path.is_dir():
        pytest.skip("configured local embedding model directory is not available")
    embedder = LocalEmbedder(model_path)
    vectors = embedder.encode(["报销期限多久？", "差旅报销需要什么材料？"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
    assert float((vectors[0] ** 2).sum()) == pytest.approx(1.0, abs=1e-4)
