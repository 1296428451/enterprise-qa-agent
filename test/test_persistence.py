import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Tools, persist_drafts


class PersistenceTests(unittest.TestCase):
    def test_atomic_persist_writes_draft_store(self):
        tools = Tools({}, {}, set(), {"r1": {"published": False}})
        with tempfile.TemporaryDirectory() as directory:
            persist_drafts(tools, directory)
            path = Path(directory) / "drafts.json"
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), tools.drafts)
            self.assertFalse((Path(directory) / "drafts.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
