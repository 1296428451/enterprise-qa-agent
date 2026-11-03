"""Offline-first scheduling agent with explicit confirmation and idempotency."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Tools:
    assets: dict[str, dict[str, Any]]
    rules: dict[str, Any]
    occupied: set[str]
    drafts: dict[str, dict[str, Any]]
    fail_save: bool = False

    def read_asset(self, asset_id: str) -> dict[str, Any] | None:
        return self.assets.get(asset_id)

    def check_rules(self, asset: dict[str, Any]) -> list[str]:
        risks = []
        if not asset.get("title"):
            risks.append("缺少素材标题")
        if asset.get("status") == "blocked":
            risks.append("素材被审核规则阻断")
        risks.extend(asset.get("risks", []))
        return risks

    def find_slot(self, requested: str, alternatives: list[str]) -> tuple[str | None, bool]:
        if requested not in self.occupied:
            return requested, False
        for slot in alternatives:
            if slot not in self.occupied:
                return slot, True
        return None, True

    def save_draft(self, key: str, draft: dict[str, Any]) -> bool:
        if key in self.drafts:
            return True
        if self.fail_save:
            return False
        self.drafts[key] = draft
        return True


class SchedulingAgent:
    MAX_STEPS = 8

    def __init__(self, tools: Tools):
        self.tools = tools

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        trace: list[dict[str, Any]] = []
        items: list[dict[str, Any]] = []
        steps = 0

        def limit_reached() -> bool:
            return steps >= self.MAX_STEPS

        def incomplete() -> dict[str, Any]:
            return {"status": "incomplete", "draft_saved": False,
                    "message": "达到最大执行步数", "items": items, "trace": trace}
        request_key = request.get("request_id") or hashlib.sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if request_key in self.tools.drafts:
            return {"status": "completed", "draft_saved": True, "duplicate": True,
                    "items": self.tools.drafts[request_key]["items"], "trace": []}

        constraints = request.get("constraints") or {}
        requested = constraints.get("publish_at")
        if not requested:
            return {"status": "needs_input", "draft_saved": False,
                    "message": "缺少硬性发布时间要求", "items": [], "trace": []}

        asset_ids = request.get("asset_ids", [])
        if not asset_ids:
            return {"status": "needs_input", "draft_saved": False,
                    "message": "缺少待处理素材 ID", "items": [], "trace": []}

        for asset_id in asset_ids:
            if limit_reached():
                return incomplete()
            steps += 1
            asset = self.tools.read_asset(asset_id)
            trace.append({"step": steps, "tool": "read_asset", "arguments": {"asset_id": asset_id},
                          "result": "found" if asset else "missing",
                          "next_action": "check_rules" if asset else "block",
                          "reason": "先确认素材信息完整可读取"})
            if not asset:
                items.append({"asset_id": asset_id, "review_status": "blocked",
                              "risk_reasons": ["素材信息不存在"], "suggested_slot": None,
                              "pending_confirmation": False, "draft_save_status": "not_saved"})
                continue
            if limit_reached():
                return incomplete()
            steps += 1
            risks = self.tools.check_rules(asset)
            trace.append({"step": steps, "tool": "check_rules", "arguments": {"asset_id": asset_id},
                          "result": risks or "no risk", "next_action": "block" if risks else "check_slot",
                          "reason": "根据审核规则及素材信息判断风险"})
            if risks:
                items.append({"asset_id": asset_id, "review_status": "risk",
                              "risk_reasons": risks, "suggested_slot": None,
                              "pending_confirmation": False, "draft_save_status": "not_saved"})
                continue
            if limit_reached():
                return incomplete()
            steps += 1
            slot, conflict = self.tools.find_slot(requested, request.get("alternative_slots", []))
            trace.append({"step": steps, "tool": "find_slot", "arguments": {"requested": requested},
                          "result": slot or "no available slot",
                          "next_action": "request_confirmation" if conflict and slot else "block" if not slot else "save_draft",
                          "reason": "硬性时段被占用时仅建议替代时段，等待用户确认" if conflict else "检查请求时段"})
            if not slot:
                items.append({"asset_id": asset_id, "review_status": "blocked",
                              "risk_reasons": ["请求时段及备选时段均不可用"], "suggested_slot": None,
                              "pending_confirmation": False, "draft_save_status": "not_saved"})
            else:
                items.append({"asset_id": asset_id, "review_status": "approved",
                              "risk_reasons": [], "suggested_slot": slot,
                              "pending_confirmation": conflict, "draft_save_status": "pending"})

        if not items or any(item["review_status"] in {"blocked", "risk"} for item in items):
            return {"status": "blocked", "draft_saved": False, "items": items, "trace": trace}
        if limit_reached():
            return incomplete()
        draft = {"request_id": request_key, "items": items, "published": False}
        steps += 1
        saved = self.tools.save_draft(request_key, draft)
        trace.append({"step": steps, "tool": "save_draft", "arguments": {"request_id": request_key},
                      "result": "saved" if saved else "save failed",
                      "next_action": "finish" if saved else "stop",
                      "reason": "只保存排期草稿，不触发发布"})
        if not saved:
            for item in items:
                item["draft_save_status"] = "failed"
            return {"status": "failed", "draft_saved": False, "message": "草稿保存失败",
                    "items": items, "trace": trace}
        for item in items:
            item["draft_save_status"] = "saved"
        self.tools.drafts[request_key]["items"] = items
        return {"status": "completed", "draft_saved": True, "published": False,
                "duplicate": False, "items": items, "trace": trace}


def load_tools(data_dir: str | Path = "data") -> Tools:
    root = Path(data_dir)
    assets = json.loads((root / "assets.json").read_text(encoding="utf-8"))
    rules = json.loads((root / "rules.json").read_text(encoding="utf-8"))
    occupied = set(json.loads((root / "schedule.json").read_text(encoding="utf-8")))
    path = root / "drafts.json"
    drafts = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return Tools(assets, rules, occupied, drafts)


def persist_drafts(tools: Tools, data_dir: str | Path = "data") -> None:
    path = Path(data_dir) / "drafts.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(tools.drafts, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
