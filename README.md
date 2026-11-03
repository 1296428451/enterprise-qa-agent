# 素材审核与发布排期 Agent

本项目实现笔试第 1 题。它读取素材元数据、检查审核风险、查询排期，并保存待确认的草稿。项目不会发布内容，也不接入真实模型；Agent 决策使用离线工具模拟，验证的是流程、安全边界和失败处理。

## 运行

要求 Python 3.10+，无第三方依赖。

```bash
python -m unittest discover -s test -v
python src/cli.py examples/requests/conflict.json --fresh-run
python src/api.py
```

API 默认监听 `127.0.0.1:18080`。健康检查 `GET /api/health`；提交排期 `POST /api/plan`，JSON 请求体格式见 `examples/requests/`。

示例请求：

```json
{
  "request_id": "demo-conflict-001",
  "asset_ids": ["M01"],
  "constraints": {"publish_at": "2026-11-05T18:00:00+08:00"},
  "alternative_slots": ["2026-11-05T19:00:00+08:00"]
}
```

该请求的 18:00 已占用，系统建议 19:00 并设置 `pending_confirmation: true`。系统只保存草稿，`published` 为 false。重复 `request_id` 不会新增草稿。

## 评审入口

- 需求：[docs/01-requirements.md](docs/01-requirements.md)
- 方案：[docs/02-solution.md](docs/02-solution.md)
- 验证与限制：[docs/03-evaluation.md](docs/03-evaluation.md)
- AI 使用：[docs/04-ai-usage.md](docs/04-ai-usage.md)
- 复盘：[docs/05-retrospective.md](docs/05-retrospective.md)
- 规范：[.agent/SPEC.md](.agent/SPEC.md)（按仓库策略忽略，不会随 Git 提交）

## 重要限制

`.agent/` 按要求写入 `.gitignore`，测试目录 `test/` 保持 Git 跟踪，确保克隆仓库可复现测试。当前项目为离线 deterministic mock，不证明真实模型驱动 ReAct 决策质量；不能以此声称真实模型决策效果。
