# AI 使用与人工确认

## 使用边界
AI 辅助用于阅读题目与仓库、整理需求、拟定实现/测试方案、编写和修订代码及文档。交付验收以仓库代码、可复现示例和本地测试为准。Provider 的 SSE chat/completions 与原生 reranker 真实调用已完成 smoke；但真实 Provider 驱动的 Agent 端到端问答尚未跑通（模型首步决策未遵循 ReAct 协议）。trace 中的 Mock 行为是确定性规则或 mocked provider protocol，不应表述为真实模型推理或在线准确率结果。

## 个人信息与协作记录
- AI 工具：WorkBuddy。
- 使用模型：GPT 6.1 sol、GLM 5.3 Flash。
- 个人贡献：本人负责技术点调研、算法设计，指导 AI 明确具体程序执行逻辑；调研数据集与开源竞赛中的 RAG 检索方案，提出算法实现方案，并定义具体测试点与验收标准。
- AI 协作：AI 编写项目简介/提交文字初稿，辅助代码初稿、测试样例、文档整理与调试建议；本人审查并定义测试与算法验收，不声称独立完成全部代码或测试。
- 时间记录（阶段估计，非秒表记录）：三天考试窗口内，约一天用于调研；核心开发约 3 小时；之后约 1–2 天用于维护、调试、测试复核和文档整理。

## Provider SSE 联调记录
- 网络层统一使用 `httpx.AsyncClient`，chat/completions 发送 `stream=true` 并聚合 SSE `delta.content`；默认相邻块空闲阈值 30 秒，最多额外重试 3 次（首次 + 3 次，共 4 次），partial 每次重试丢弃。
- 4xx、schema/JSON 错误不重试，`CancelledError` 原样传播；TLS verify 保持开启，可配置 CA，不允许 `verify=False`。注入的 AsyncClient 由调用方拥有，内部创建的 client 在请求结束时关闭。
- 真实 SSE smoke（本地 `.env`，标准 TLS 校验）：`post_chat_stream` 与 `ProviderReranker.score_async` 均成功；证书路径问题已排除（默认 SSL context 校验证书链有效，SDK/requests/urllib 均 HTTP 200），此前"TLS 证书过期"为瞬时/环境差异。
- CLI 真实 Agent 整链本次未返回 `completed`/`insufficient_evidence`：真实模型首个 ReAct 决策未遵循"首步必须为 search"协议而抛 `ValueError`，属模型输出行为问题，非 TLS/SSE 错误；不据此声称真实 Provider Agent E2E 或在线准确率已验证。
- 完整 pytest：65 passed、1 skipped。
