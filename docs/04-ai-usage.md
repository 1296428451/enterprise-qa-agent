# AI 使用与人工确认

## 使用边界
AI 辅助用于阅读题目与仓库、整理需求、拟定实现/测试方案、编写和修订代码及文档。交付验收以仓库代码、可复现示例和本地测试为准。在线 LLM provider 未成功完成调用；trace 中的 Mock 行为是确定性规则或 mocked provider protocol，不应表述为真实模型推理结果。

## 个人信息与协作记录
- AI 工具：WorkBuddy。
- 使用模型：GPT 6.1 sol、GLM 5.3 Flash。
- 个人贡献：本人负责技术点调研、算法设计，指导 AI 明确具体程序执行逻辑；调研数据集与开源竞赛中的 RAG 检索方案，提出算法实现方案，并定义具体测试点与验收标准。
- AI 协作：AI 编写项目简介/提交文字初稿，辅助代码初稿、测试样例、文档整理与调试建议；本人审查并定义测试与算法验收，不声称独立完成全部代码或测试。
- 时间记录（阶段估计，非秒表记录）：三天考试窗口内，约一天用于调研；核心开发约 3 小时；之后约 1–2 天用于维护、调试、测试复核和文档整理。

## Provider 异步联调记录
- 网络层统一使用 `httpx.AsyncClient`，TLS 验证开启并配置显式超时；同步 API 仅在无运行中 event loop 时包装异步流程。
- 本地 `.env` 的异步 smoke 成功调用 LLM `gpt-6.1-sol`、Query-Aware 和 reranker `Qwen/Qwen3-Reranker-0.6B`；reranker 两个候选分数约为 `0.9995`、`0.0011`。
- `EnterpriseQA.answer_async()` 的 ReAct 控制流以 MockTransport 测试覆盖；尚未用真实 provider 运行 Agent clear/conflict/unanswerable E2E 或在线准确率评估。
