# AI 使用与人工确认

## 使用边界
AI 辅助用于阅读题目与仓库、整理需求、拟定实现/测试方案、编写和修订代码及文档。交付验收以仓库代码、可复现示例和本地测试为准。Provider 的 SSE chat/completions、原生 reranker 与 Query-Aware/答案/核验各阶段真实调用已完成 smoke；收紧首步 instruction 后真实 CLI Agent 整链不再抛 ReAct 协议错，三段真实 CLI 端到端（含可答题 `completed`、冲突题 `conflict_detected`、题库外题 `insufficient_evidence`）均已产出预期状态并记录到 `examples/`。真实模型输出有随机性，不据此声称在线准确率已系统评测。trace 中的 Mock 行为是确定性规则或 mocked provider protocol，不应表述为真实模型推理或在线准确率结果。

## 个人信息与协作记录
- AI 工具：WorkBuddy。
- 使用模型：GPT 6.1 sol、GLM 5.3 Flash。
- 个人贡献：本人负责技术点调研、算法设计，指导 AI 明确具体程序执行逻辑；调研数据集与开源竞赛中的 RAG 检索方案，提出算法实现方案，并定义具体测试点与验收标准。
- AI 协作：AI 编写项目简介/提交文字初稿，辅助代码初稿、测试样例、文档整理与调试建议；本人审查并定义测试与算法验收，不声称独立完成全部代码或测试。
- 时间记录（阶段估计，非秒表记录）：三天考试窗口内，约一天用于调研；核心开发约 3 小时；之后约 1–2 天用于维护、调试、测试复核和文档整理。

## Provider SSE 联调记录
- 网络层统一使用 `httpx.AsyncClient`，chat/completions 发送 `stream=true` 并聚合 SSE `delta.content`；默认相邻块空闲阈值 30 秒，最多额外重试 3 次（首次 + 3 次，共 4 次），partial 每次重试丢弃。
- 4xx、schema/JSON 错误不重试，`CancelledError` 原样传播；TLS verify 保持开启，可配置 CA，不允许 `verify=False`。注入的 AsyncClient 由调用方拥有，内部创建的 client 在请求结束时关闭。
- 真实 SSE smoke（本地 `.env`，标准 TLS 校验）：`post_chat_stream` 与 `ProviderReranker.score_async` 均成功，`QueryAwareExtractor.aextract`、`aanswer_from_evidence`、`averify` 均返回合规 JSON；证书链校验正常（默认 SSL context 有效，SDK/requests/urllib 均 HTTP 200），TLS verify 常开且禁止关闭。
- 本版把 `decide_action_async` instruction 收紧为"observations 为空必须 `action=search`、不得凭常识作答"，`agent.py` 首步 search 硬检查保留兜底；真实 CLI 首步不再抛 `ValueError: the first ReAct action must be search`。
- 真实 CLI 三段端到端（记录见 `examples/{requests,traces,results}/live_*.json`）：`差旅报销需要什么材料？` → `completed`（带 1 条引用 D03，`verify` 通过）；`报销期限多久？` → `conflict_detected`（2 条引用）；`公司有住房补贴吗？` → `insufficient_evidence`（0 条引用，检索至步数上限）。此前 clear-case 多次未 `completed` 的根因是 observation 未回传已核验答案与证据原文；修复只补齐信息流，未放宽引用校验、未加 fence 兼容层、未回退 Mock 或伪造 completed。
- 完整 pytest：68 passed、1 skipped。
