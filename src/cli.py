import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import SchedulingAgent, load_tools, persist_drafts


def main():
    parser = argparse.ArgumentParser(description="Run the offline scheduling agent")
    parser.add_argument("request", help="Path to request JSON")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    tools = load_tools(ROOT / "data")
    result = SchedulingAgent(tools).run(request)
    if result.get("draft_saved") and not result.get("duplicate"):
        persist_drafts(tools, ROOT / "data")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
