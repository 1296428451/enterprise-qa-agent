from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


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
    def __init__(self, model_path: str | Path, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_path = str(model_path)
        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
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
