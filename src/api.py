from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import SchedulingAgent, load_tools, persist_drafts


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/api/health":
            self.send_error(404)
            return
        self._json(200, {"status": "ok", "mode": "offline-mock"})

    def do_POST(self):
        if self.path != "/api/plan":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            tools = load_tools(ROOT / "data")
            result = SchedulingAgent(tools).run(request)
            if result.get("draft_saved") and not result.get("duplicate"):
                persist_drafts(tools, ROOT / "data")
            self._json(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"status": "invalid_request", "message": str(exc)})
        except Exception as exc:
            self._json(500, {"status": "error", "message": str(exc)})

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 18080), Handler)
    print("Offline scheduling API: http://127.0.0.1:18080")
    server.serve_forever()


if __name__ == "__main__":
    main()
