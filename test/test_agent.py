import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import SchedulingAgent, Tools

ASSETS = {
    "ok": {"title": "Ready", "status": "ready", "risks": []},
    "risk": {"title": "Risk", "status": "ready", "risks": ["授权待核验"]},
    "missing_title": {"status": "ready", "risks": []},
}
REQUEST = {"request_id": "r1", "asset_ids": ["ok"],
           "constraints": {"publish_at": "18:00"}, "alternative_slots": ["19:00"]}


def make_agent(fail_save=False):
    tools = Tools(ASSETS, {}, {"18:00"}, {}, fail_save)
    return SchedulingAgent(tools), tools


def test_conflict_suggests_alternative_and_waits_for_confirmation():
    agent, tools = make_agent()
    result = agent.run(REQUEST)
    item = result["items"][0]
    assert item["suggested_slot"] == "19:00"
    assert item["pending_confirmation"]
    assert result["draft_saved"]
    assert not result["published"]
    assert "request_confirmation" in [step["next_action"] for step in result["trace"]]
    assert len(tools.drafts) == 1


def test_normal_requested_slot_is_saved_without_confirmation():
    agent, _ = make_agent()
    request = {**REQUEST, "request_id": "r2", "constraints": {"publish_at": "20:00"}}
    result = agent.run(request)
    assert result["status"] == "completed"
    assert result["items"][0]["suggested_slot"] == "20:00"
    assert not result["items"][0]["pending_confirmation"]


def test_missing_asset_is_blocked():
    agent, _ = make_agent()
    request = {**REQUEST, "request_id": "r3", "asset_ids": ["unknown"]}
    result = agent.run(request)
    assert result["status"] == "blocked"
    assert "素材信息不存在" in result["items"][0]["risk_reasons"]
    assert not result["draft_saved"]


def test_missing_metadata_is_risk_block():
    agent, _ = make_agent()
    request = {**REQUEST, "request_id": "r4", "asset_ids": ["missing_title"]}
    result = agent.run(request)
    assert result["status"] == "blocked"
    assert "缺少素材标题" in result["items"][0]["risk_reasons"]


def test_review_risk_blocks_scheduling():
    agent, _ = make_agent()
    request = {**REQUEST, "request_id": "r5", "asset_ids": ["risk"]}
    result = agent.run(request)
    assert result["status"] == "blocked"
    assert "授权待核验" in result["items"][0]["risk_reasons"]


def test_missing_hard_time_requests_input():
    agent, _ = make_agent()
    request = {**REQUEST, "constraints": {}}
    result = agent.run(request)
    assert result["status"] == "needs_input"


def test_repeated_request_is_idempotent():
    agent, tools = make_agent()
    first = agent.run(REQUEST)
    second = agent.run(REQUEST)
    assert first["draft_saved"]
    assert second["duplicate"]
    assert len(tools.drafts) == 1


def test_step_limit_stops_before_tool_budget_is_exceeded():
    agent, tools = make_agent()
    agent.MAX_STEPS = 2
    result = agent.run(REQUEST)
    assert result["status"] == "incomplete"
    assert not result["draft_saved"]
    assert len(result["trace"]) == 2
    assert "save_draft" not in [entry["tool"] for entry in result["trace"]]
    assert tools.drafts == {}


def test_save_failure_is_never_reported_as_success():
    agent, _ = make_agent(fail_save=True)
    result = agent.run(REQUEST)
    assert result["status"] == "failed"
    assert not result["draft_saved"]
    assert result["items"][0]["draft_save_status"] == "failed"
