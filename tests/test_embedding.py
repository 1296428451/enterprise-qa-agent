import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q3 import embedding as embedding_module
from src.q3.embedding import (
    LocalEmbedder,
    default_model_dir,
    dump_chunk_ids,
    ensure_model,
    load_chunk_ids,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_model_dir_is_repo_relative():
    assert default_model_dir() == ROOT / "models" / "all-MiniLM-L6-v2"


def test_ensure_model_uses_existing_directory_as_local_only(tmp_path):
    model = tmp_path / "local-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    source, local_only = ensure_model(model)
    assert Path(source) == model
    assert local_only is True


def test_ensure_model_resolves_relative_paths_against_repo_root(tmp_path, monkeypatch):
    model = tmp_path / "relative-model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(embedding_module, "ROOT", tmp_path)
    source, local_only = ensure_model("relative-model")
    assert Path(source) == model
    assert local_only is True


def test_ensure_model_refuses_to_download_when_disabled(monkeypatch, tmp_path):
    missing = tmp_path / "absent-model"
    monkeypatch.setattr(embedding_module, "MODEL_DIR", missing)
    with pytest.raises(FileNotFoundError):
        ensure_model(None, allow_download=False)


def test_ensure_model_downloads_into_repo_models_dir(tmp_path, monkeypatch):
    target = tmp_path / "models" / "all-MiniLM-L6-v2"
    monkeypatch.setattr(embedding_module, "MODEL_DIR", target)

    def fake_download(destination=None, model_id=embedding_module.MODEL_ID):
        path = Path(destination) if destination is not None else target
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(embedding_module, "download_model", fake_download)
    source, local_only = ensure_model(None)
    assert Path(source) == target
    assert local_only is True
    assert (target / "config.json").is_file()


def test_ensure_model_treats_unknown_value_as_hub_id(tmp_path):
    source, local_only = ensure_model("some-org/some-model")
    assert source == "some-org/some-model"
    assert local_only is False


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
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("embedding dependencies (torch/transformers) are not installed")
    configured = os.environ.get("FRAMES_EMBEDDING_MODEL")
    model_path = Path(configured) if configured else default_model_dir()
    if not model_path.is_dir():
        pytest.skip("embedding model not downloaded; run `python -m src.q3.download_model`")
    # Tests never hit the network: an absent directory must fail loudly instead.
    embedder = LocalEmbedder(model_path, allow_download=False)
    vectors = embedder.encode(["报销期限多久？", "差旅报销需要什么材料？"])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
    assert float((vectors[0] ** 2).sum()) == pytest.approx(1.0, abs=1e-4)
