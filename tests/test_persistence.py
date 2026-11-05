import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Tools, persist_drafts


def test_atomic_persist_writes_draft_store():
    tools = Tools({}, {}, set(), {"r1": {"published": False}})
    with tempfile.TemporaryDirectory() as directory:
        persist_drafts(tools, directory)
        path = Path(directory) / "drafts.json"
        assert json.loads(path.read_text(encoding="utf-8")) == tools.drafts
        assert not (Path(directory) / "drafts.json.tmp").exists()
