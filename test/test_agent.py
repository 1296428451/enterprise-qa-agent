import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import SchedulingAgent, Tools


class SchedulingAgentTests(unittest.TestCase):
    def setUp(self):
        self.assets = {
            "ok": {"title": "Ready", "status": "ready", "risks": []},
            "risk": {"title": "Risk", "status": "ready", "risks": ["授权待核验"]},
            "missing_title": {"status": "ready", "risks": []},
        }
        self.request = {"request_id": "r1", "asset_ids": ["ok"],
                        "constraints": {"publish_at": "18:00"}, "alternative_slots": ["19:00"]}

    def make_agent(self, fail_save=False):
        tools = Tools(self.assets, {}, {"18:00"}, {}, fail_save)
        return SchedulingAgent(tools), tools

    def test_conflict_suggests_alternative_and_waits_for_confirmation(self):
        agent, tools = self.make_agent()
        result = agent.run(self.request)
        item = result["items"][0]
        self.assertEqual(item["suggested_slot"], "19:00")
        self.assertTrue(item["pending_confirmation"])
        self.assertTrue(result["draft_saved"])
        self.assertFalse(result["published"])
        self.assertIn("request_confirmation", [step["next_action"] for step in result["trace"]])
        self.assertEqual(len(tools.drafts), 1)

    def test_normal_requested_slot_is_saved_without_confirmation(self):
        agent, _ = self.make_agent()
        request = {**self.request, "request_id": "r2", "constraints": {"publish_at": "20:00"}}
        result = agent.run(request)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["items"][0]["suggested_slot"], "20:00")
        self.assertFalse(result["items"][0]["pending_confirmation"])

    def test_missing_asset_is_blocked(self):
        agent, _ = self.make_agent()
        request = {**self.request, "request_id": "r3", "asset_ids": ["unknown"]}
        result = agent.run(request)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("素材信息不存在", result["items"][0]["risk_reasons"])
        self.assertFalse(result["draft_saved"])

    def test_missing_metadata_is_risk_block(self):
        agent, _ = self.make_agent()
        request = {**self.request, "request_id": "r4", "asset_ids": ["missing_title"]}
        result = agent.run(request)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("缺少素材标题", result["items"][0]["risk_reasons"])

    def test_review_risk_blocks_scheduling(self):
        agent, _ = self.make_agent()
        request = {**self.request, "request_id": "r5", "asset_ids": ["risk"]}
        result = agent.run(request)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("授权待核验", result["items"][0]["risk_reasons"])

    def test_missing_hard_time_requests_input(self):
        agent, _ = self.make_agent()
        request = {**self.request, "constraints": {}}
        result = agent.run(request)
        self.assertEqual(result["status"], "needs_input")

    def test_repeated_request_is_idempotent(self):
        agent, tools = self.make_agent()
        first = agent.run(self.request)
        second = agent.run(self.request)
        self.assertTrue(first["draft_saved"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(len(tools.drafts), 1)

    def test_step_limit_stops_before_tool_budget_is_exceeded(self):
        agent, tools = self.make_agent()
        agent.MAX_STEPS = 2
        result = agent.run(self.request)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["draft_saved"])
        self.assertEqual(len(result["trace"]), 2)
        self.assertNotIn("save_draft", [entry["tool"] for entry in result["trace"]])
        self.assertEqual(tools.drafts, {})

    def test_save_failure_is_never_reported_as_success(self):
        agent, _ = self.make_agent(fail_save=True)
        result = agent.run(self.request)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["draft_saved"])
        self.assertEqual(result["items"][0]["draft_save_status"], "failed")


if __name__ == "__main__":
    unittest.main()
