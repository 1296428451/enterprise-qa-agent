# models/

Embedding model weights live here as a **repo-relative** directory; they are
not committed (see `.gitignore`).

| item | value |
|---|---|
| directory | `models/all-MiniLM-L6-v2` |
| model id | `sentence-transformers/all-MiniLM-L6-v2` |
| used by | `src/q3/build_index.py`, `src/q3/evaluate_frames.py --method dense|rrf`, `tests/test_embedding.py` |

## Download

```bash
python -m src.q3.download_model
```

First use of `LocalEmbedder()` also downloads automatically when the directory
is missing, so the index build works without a separate step:

```bash
python src/q3/build_index.py
```

## Offline / custom location

Point `--model` (or `FRAMES_EMBEDDING_MODEL`) at a local directory or another
Hugging Face model id. Relative paths are resolved against the repository root,
so `--model models/my-model` means `<repo>/models/my-model`. An existing
directory is always loaded with `local_files_only=True`; anything else is
treated as a hub id and left to `transformers` to fetch.
