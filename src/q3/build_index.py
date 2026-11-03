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
    parser.add_argument("--model", required=True, help="Local embedding model directory")
    parser.add_argument("--documents", default="data/q3_documents.json")
    parser.add_argument("--index", default="data/q3_vectors.faiss")
    parser.add_argument("--metadata", default="data/q3_vector_metadata.json")
    args = parser.parse_args()
    chunks = load_chunks(args.documents)
    embedder = LocalEmbedder(args.model)
    index = FaissIndex.build(chunks, embedder)
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
    index.save(args.index, args.metadata)
    Path(args.metadata).with_suffix(".manifest.json").write_text(
        json.dumps({"model": str(Path(args.model).resolve()), "chunks": len(chunks)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"index": args.index, "metadata": args.metadata, "chunks": len(chunks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
