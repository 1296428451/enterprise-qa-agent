from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.agent import load_chunks
from src.q3.embedding import FaissIndex, LocalEmbedder


def main():
    parser = argparse.ArgumentParser(description="Build a local semantic index")
    parser.add_argument("--model", default=None,
                        help="Embedding model path (repo-relative) or hub id; defaults to "
                             "models/all-MiniLM-L6-v2, auto-downloaded on first use")
    parser.add_argument("--documents", default="data/q3_documents.json")
    parser.add_argument("--index", default="data/q3_vectors.faiss")
    parser.add_argument("--metadata", default="data/q3_vector_metadata.json")
    args = parser.parse_args()
    chunks = load_chunks(args.documents)
    embedder = LocalEmbedder(args.model)
    index = FaissIndex.build(chunks, embedder)
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    index.save(args.index, args.metadata)
    model_ref = Path(embedder.model_path)
    try:
        # Record a repo-relative, POSIX-style model reference so the manifest
        # stays portable across operating systems.
        model_ref = model_ref.relative_to(ROOT)
    except ValueError:
        pass
    Path(args.metadata).with_suffix(".manifest.json").write_text(
        json.dumps({"model": model_ref.as_posix(), "chunks": len(chunks)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"index": args.index, "metadata": args.metadata, "chunks": len(chunks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
