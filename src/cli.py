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
    parser.add_argument("--fresh-run", action="store_true",
                        help="Ignore stored drafts and do not persist this demonstration run")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    tools = load_tools(ROOT / "data")
    if args.fresh_run:
        tools.drafts = {}
    result = SchedulingAgent(tools).run(request)
    if result.get("draft_saved") and not result.get("duplicate") and not args.fresh_run:
        try:
            persist_drafts(tools, ROOT / "data")
        except OSError as exc:
            result["status"] = "failed"
            result["draft_saved"] = False
            result["message"] = f"草稿持久化失败: {exc}"
            for item in result["items"]:
                item["draft_save_status"] = "failed"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
