import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3.embedding import LocalEmbedder, dump_chunk_ids, load_chunk_ids

ROOT = Path(__file__).resolve().parents[1]


def test_chunk_id_metadata_roundtrip(tmp_path):
    path = tmp_path / "meta.json"
    dump_chunk_ids(path, ["D01-v1-c01", "D02-v1-c01"])
    assert json.loads(path.read_text(encoding="utf-8")) == {"chunk_ids": ["D01-v1-c01", "D02-v1-c01"]}
    assert load_chunk_ids(path) == ["D01-v1-c01", "D02-v1-c01"]


def test_chunk_id_metadata_reads_legacy_and_array_layouts(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text("D01-v1-c01\r\nD02-v1-c01", encoding="utf-8")
    assert load_chunk_ids(legacy) == ["D01-v1-c01", "D02-v1-c01"]

    array = tmp_path / "array.json"
    array.write_text(json.dumps(["D01-v1-c01"]), encoding="utf-8")
    assert load_chunk_ids(array) == ["D01-v1-c01"]

    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert load_chunk_ids(empty) == []


def test_shipped_vector_metadata_is_valid_json_and_matches_documents():
    metadata = json.loads((ROOT / "data" / "q3_vector_metadata.json").read_text(encoding="utf-8"))
    chunk_ids = metadata["chunk_ids"]
    documents = json.loads((ROOT / "data" / "q3_documents.json").read_text(encoding="utf-8"))
    assert chunk_ids == [chunk["chunk_id"] for chunk in documents]


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
