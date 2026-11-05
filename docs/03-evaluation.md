# 第三题评测与验收

## 测试命令
从仓库任意目录执行：

```bash
python -m pytest C:/Users/ASUS/Desktop/小影笔试/agent-code/test -q
```

Embedding 冒烟测试在未设置 `FRAMES_EMBEDDING_MODEL` 时会跳过；这不等同于向量模型集成验收。

## 可复现示例
请求与实际 CLI 输出/trace 位于 `examples/requests/`、`examples/traces/`：
- `normal.json`：有证据的差旅报销问答与完整输出/trace。
- `missing.json`：题库外住房补贴问题，首轮无证据、改写后仍无证据并拒答。
- `examples/requests/provider_failure.json`、`examples/provider_invalid_action_response.json` 与 `examples/traces/provider_failure.json`：本地 mock provider schema-invalid response；trace 记录 CLI 回放命令、实际异常输出和失败边界。无真实 key/外网调用。

正常回答与题库外路径在不同观察后采取不同动作。前两份 trace 明确标注 `mocked provider protocol / 非在线真实模型`。provider 失败直接抛出异常终止，不重试、不返回 structured failed `Answer`；此为当前限制。示例不包含真实 key。

离线示例可从仓库根目录运行：

```bash
python src/q3/cli.py --input examples/requests/normal.json
python src/q3/cli.py --input examples/requests/missing.json
```

## 评测报告与指标
- `data/q3_eval_report.json`：10 条离线问答案例，报告状态准确率、文档集合 exact accuracy、宏平均文档 precision/recall/F1、可回答题 exact-match、正确拒答准确率。
- `data/q3_react_eval_report.json`：题库外 ReAct 案例，报告状态准确率、文档集合准确率和步数上限遵守率。
- `data/q3_live_accuracy_report.json`：记录运行模式、真实 SSE/Agent smoke 结果与可测的离线指标；在线 provider 准确率仍不可测。
- `data/external/frames_eval_report.json`：FRAMES title-only gold-link proxy，不是全文 RAG 或答案准确率评测。

指标定义：status accuracy 为预测状态匹配期望状态的比例；query/document retrieval exact accuracy 为引用文档集合完全等于期望集合的比例，额外引用和漏引均计错；document precision/recall/F1 按问题分别计算后宏平均；answer exact match 只对有参考答案的问题逐字比对；correct abstention accuracy 检查冲突/不可回答题是否返回预期非回答状态且不输出答案；ReAct step-bound accuracy 检查搜索步数是否不超过配置上限。

离线 10 例状态准确率为 1.00、文档集合 exact accuracy 为 0.90、宏平均 precision/recall/F1 为 0.9667/1.00/0.98、可回答题 exact match 为 1.00（5 例）、正确拒答准确率为 1.00（5 例）。ReAct 3 例报告中的状态准确率、文档集合准确率和步数约束均为 1.00。以上为本地确定性 Mock 与合成题库结果，不是在线模型表现。

## Provider SSE 实现与验证边界
LLM、Query-Aware 与 reranker chat/completions 使用 `httpx.AsyncClient` SSE（`stream=true`），聚合完整 `delta.content` 后解析 JSON。默认相邻数据块空闲阈值为 30 秒，最多额外重试 3 次（首次 + 3 次，共 4 次调用）；超时重试会丢弃此前 partial response。HTTP 4xx、schema/JSON 错误不重试；`CancelledError` 原样传播。TLS 证书校验保持开启；可配置受信任 CA，不允许关闭验证（代码显式拒绝 `verify=False`）。原生 reranker `/rerank` 使用非流式 JSON。

真实 Provider smoke（py312 + 本地 `.env`，标准 TLS 校验）：`post_chat_stream` 成功接收 SSE 数据块、聚合 `delta.content` 并解析为 JSON 对象；`ProviderReranker.score_async` 成功返回两个候选分数（约 `0.9992`、`0.0010`）；`QueryAwareExtractor.aextract`（quote 为候选原文精确子串）、`aanswer_from_evidence`、`averify` 均返回合规 JSON。证书链校验正常：默认 SSL context 可正常校验证书链，OpenAI SDK、requests、urllib 均 HTTP 200，TLS verify 始终开启且代码显式拒绝 `verify=False`。本次将 `src/q3/llm.py` 中 `decide_action_async` 的 instruction 收紧为"observations 为空必须 `action=search`、不得凭常识作答"后，真实 CLI 首步不再抛 `ValueError: the first ReAct action must be search`；`src/q3/agent.py` 首步 search 硬检查保留兜底。

真实 CLI 三段端到端（`--llm openai-compatible --reranker provider --query-aware provider`）：`报销期限多久？` 返回 `conflict_detected`（D01/D02 共 2 条引用），`公司有住房补贴吗？` 返回 `insufficient_evidence`（0 条引用），两者符合预期；`差旅报销需要什么材料？` 4 次真实运行均未返回 `completed`（3 次 `insufficient_evidence`、1 次 `conflict_detected`）：1 次内部已生成带 1 条有效引用的答案且 `verify` 通过、但模型第 3 步仍判 `insufficient_evidence` 导致最终降级，2 次卡在 `答案引用索引缺失、重复或无效`，1 次因 query-aware 抽出多候选冲突信息进入冲突状态。完整 pytest 为 65 passed、1 skipped；离线 MockTransport 覆盖重试、取消、client 生命周期与异步 ReAct 顺序。结论：真实整链已可执行、冲突与拒答行为符合预期，但 clear-case 的 `completed` 未稳定产出，在线答案准确率仍未验证；未放宽 quote 精确子串与 `citation_indices` 校验、未加 markdown fence 兼容层、未静默回退 Mock 或伪造 completed。

冲突检测目前依赖 token/规则 heuristic，不能视为通用语义冲突识别。Provider 异常直接向 CLI 抛出，不重试，也不转换为 structured failed `Answer`。`max_steps` 正值校验及 evaluator 步数配置一致性仍有改进空间。
