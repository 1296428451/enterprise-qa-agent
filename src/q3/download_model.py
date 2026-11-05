from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.embedding import MODEL_DIR, MODEL_ID, download_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the embedding model into the repository's models/ directory")
    parser.add_argument("--destination", default=str(MODEL_DIR.relative_to(ROOT)),
                        help="Destination directory, repository-relative "
                             "(default: models/all-MiniLM-L6-v2)")
    parser.add_argument("--model-id", default=MODEL_ID,
                        help=f"Hugging Face model id (default: {MODEL_ID})")
    args = parser.parse_args(argv)
    target = Path(args.destination)
    if target.is_absolute():
        parser.error("--destination must be repository-relative")
    path = download_model(target, args.model_id)
    print(json.dumps({"model_id": args.model_id, "path": str(path.relative_to(ROOT))},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
