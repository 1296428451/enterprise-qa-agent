# AI 使用与人工确认

## 使用边界
AI 辅助用于阅读题目与仓库、整理需求、拟定实现/测试方案、编写和修订代码及文档。交付验收以仓库代码、可复现示例和本地测试为准。在线 LLM provider 未成功完成调用；trace 中的 Mock 行为是确定性规则或 mocked provider protocol，不应表述为真实模型推理结果。

## 个人信息与协作记录
- AI 工具：WorkBuddy。
- 使用模型：GPT 6.1 sol、GLM 5.3 Flash。
- 个人贡献：本人负责技术点调研、算法设计，指导 AI 明确具体程序执行逻辑；调研数据集与开源竞赛中的 RAG 检索方案，提出算法实现方案，并定义具体测试点与验收标准。
- AI 协作：AI 编写项目简介/提交文字初稿，辅助代码初稿、测试样例、文档整理与调试建议；本人审查并定义测试与算法验收，不声称独立完成全部代码或测试。
- 时间记录（阶段估计，非秒表记录）：三天考试窗口内，约一天用于调研；核心开发约 3 小时；之后约 1–2 天用于维护、调试、测试复核和文档整理。

## Provider SSE 联调记录
- 网络层统一使用 `httpx.AsyncClient`，chat/completions 发送 `stream=true` 并聚合 SSE `delta.content`；默认相邻块空闲阈值 30 秒，最多额外重试 3 次，partial 每次重试丢弃。
- 4xx、schema/JSON 错误不重试；TLS verify 保持开启，可配置 CA，不允许 `verify=False`。注入的 AsyncClient 由调用方拥有，内部创建的 client 在请求结束时关闭。
- 本版本 SSE 改造仅完成 MockTransport 离线验收：65 passed、1 skipped；未执行真实 HTTPX Provider smoke，也未验证真实 Provider Agent E2E。
