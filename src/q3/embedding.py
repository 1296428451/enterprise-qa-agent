from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# Repository root and model location are derived from this file, never
# hardcoded, so the checkout works from any directory or drive.
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "all-MiniLM-L6-v2"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Only what AutoTokenizer/AutoModel need. The upstream repo also ships onnx/
# and a duplicate .bin checkpoint (~500 MB together), which would be wasted.
MODEL_PATTERNS = ("*.json", "*.txt", "*.safetensors", "1_Pooling/*")
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "modules.json",
    "1_Pooling/config.json",
)


def default_model_dir() -> Path:
    return MODEL_DIR


def model_endpoint() -> str:
    """Hub endpoint used for downloads; honours HF_ENDPOINT for regional mirrors."""
    import os
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def _download_files_directly(target: Path, model_id: str, endpoint: str) -> list[str]:
    """Fetch the required files over plain HTTP.

    ``huggingface_hub`` is the normal path, but it also queries the Hub API for
    metadata, which some restricted networks block while plain file downloads
    still work. This keeps the model fetch usable behind a mirror.
    """
    import httpx
    written: list[str] = []
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0)) as client:
        for name in MODEL_FILES:
            destination = target / name
            if destination.is_file() and destination.stat().st_size > 0:
                written.append(name)
                continue
            url = f"{endpoint}/{model_id}/resolve/main/{name}"
            response = client.get(url)
            if response.status_code != 200 or not response.content:
                raise RuntimeError(
                    f"failed to download {name} from the model endpoint "
                    f"(HTTP {response.status_code}); set HF_ENDPOINT to a reachable "
                    f"mirror or place the model under models/all-MiniLM-L6-v2"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.content)
            written.append(name)
    return written


def download_model(destination: str | Path | None = None, model_id: str = MODEL_ID) -> Path:
    """Download the embedding model into a repo-relative directory."""
    target = Path(destination) if destination is not None else MODEL_DIR
    if not target.is_absolute():
        target = ROOT / target
    target.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        _download_files_directly(target, model_id, model_endpoint())
        return target
    try:
        snapshot_download(repo_id=model_id, local_dir=str(target),
                          allow_patterns=list(MODEL_PATTERNS))
    except Exception:
        # Hub API metadata can be blocked while file downloads still work.
        _download_files_directly(target, model_id, model_endpoint())
    return target


def ensure_model(model_path: str | Path | None = None, allow_download: bool = True) -> tuple[str, bool]:
    """Resolve the embedding model source and whether it must be local-only.

    * ``model_path=None`` uses ``models/all-MiniLM-L6-v2`` inside the repository
      and downloads it there on first use.
    * A relative ``model_path`` is resolved against the repository root.
    * An existing directory is always loaded with ``local_files_only=True``.
    * A value that is not an existing directory is treated as a Hugging Face
      model id and left to ``transformers`` to fetch into its own cache.
    """
    if model_path is None:
        if MODEL_DIR.is_dir() and any(MODEL_DIR.iterdir()):
            return str(MODEL_DIR), True
        if not allow_download:
            raise FileNotFoundError(
                f"embedding model not found at models/all-MiniLM-L6-v2; "
                f"run `python -m src.q3.download_model` or set --model"
            )
        return str(download_model()), True

    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if candidate.is_dir():
        return str(candidate), True
    return str(model_path), False


def dump_chunk_ids(metadata_path: str | Path, chunk_ids: list[str]) -> None:
    """Write the vector metadata sidecar as real JSON."""
    target = Path(metadata_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"chunk_ids": list(chunk_ids)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_chunk_ids(metadata_path: str | Path) -> list[str]:
    """Read the vector metadata sidecar.

    Accepts the current ``{"chunk_ids": [...]}`` layout, a bare JSON array, and
    the newline-separated text written by earlier revisions.
    """
    raw = Path(metadata_path).read_text(encoding="utf-8")
    if not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Blank lines would shift every later id out of alignment with the
        # index, so drop them here instead of trusting the file's layout.
        return [line.strip() for line in raw.splitlines() if line.strip()]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        chunk_ids = payload.get("chunk_ids", [])
        if not isinstance(chunk_ids, list):
            raise ValueError("chunk_ids must be a list")
        return [str(item) for item in chunk_ids]
    raise ValueError("vector metadata must be a JSON object or array")


class LocalEmbedder:
    def __init__(self, model_path: str | Path | None = None, device: str = "cpu",
                 allow_download: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer
        source, local_only = ensure_model(model_path, allow_download=allow_download)
        self.model_path = source
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(source, local_files_only=local_only)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Iterable[str], batch_size: int = 8):
        import numpy as np
        import torch
        values = list(texts)
        vectors = []
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(output.size()).float()
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0), dtype="float32")


class FaissIndex:
    def __init__(self, index, chunk_ids: list[str]):
        self.index = index
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, chunks, embedder: LocalEmbedder):
        import faiss
        vectors = embedder.encode([chunk.text for chunk in chunks]).astype("float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, [chunk.chunk_id for chunk in chunks])

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        import faiss
        faiss.write_index(self.index, str(index_path))
        dump_chunk_ids(metadata_path, self.chunk_ids)

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path):
        import faiss
        index = faiss.read_index(str(index_path))
        chunk_ids = load_chunk_ids(metadata_path)
        return cls(index, chunk_ids)

    def search(self, query: str, embedder: LocalEmbedder, chunks_by_id: dict[str, object], limit: int = 50):
        from .models import ScoredChunk
        vector = embedder.encode([query]).astype("float32")
        scores, positions = self.index.search(vector, min(limit, self.index.ntotal))
        return [ScoredChunk(chunks_by_id[self.chunk_ids[pos]], vector_score=float(score))
                for score, pos in zip(scores[0], positions[0]) if pos >= 0]
