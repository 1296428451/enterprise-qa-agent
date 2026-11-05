# 第三题评测与验收

## 测试命令
从仓库任意目录执行：

```bash
python -m pytest tests/ -q
```

Embedding 冒烟测试在未设置 `FRAMES_EMBEDDING_MODEL` 时会跳过；这不等同于向量模型集成验收。

## 可复现示例
请求与实际 CLI 输出/trace 位于 `examples/requests/`、`examples/traces/`：
- `normal.json`：有证据的差旅报销问答与完整输出/trace。
- `missing.json`：题库外住房补贴问题，首轮无证据、改写后仍无证据并拒答。
- `examples/requests/provider_failure.json`、`examples/provider_invalid_action_response.json` 与 `examples/traces/provider_failure.json`：本地 mock provider schema-invalid response；trace 记录 CLI 回放命令、实际异常输出和失败边界。无真实 key/外网调用。

正常回答与题库外路径在不同观察后采取不同动作。前两份 trace 明确标注 `mocked provider protocol / 非在线真实模型`。`provider_failure.json` 回放的是 schema-invalid action 响应：这类协议/schema 错误按设计不重试，异常直接向 CLI 抛出，也不返回 structured failed `Answer`；传输层空闲超时与瞬时网络错误才走 SSE 重试（见下文）。此为当前限制。示例不包含真实 key。

离线示例可从仓库根目录运行：

```bash
python src/q3/cli.py --input examples/requests/normal.json
python src/q3/cli.py --input examples/requests/missing.json
```

### 真实 Provider 运行记录

`python src/q3/cli.py --question "..." --llm openai-compatible --reranker provider --query-aware provider --save-run NAME` 会把请求、ReAct 过程、最终结果分别写入 `examples/{requests,traces,results}/NAME.json`。已提交三组真实运行（均为真实模型决策，非 mock）：

- `live_react`（`差旅报销需要什么材料？`）：`search → answer`，`status=completed`，答案「差旅报销需提交发票、行程单和审批记录。[1]」，1 条引用（D03），`verified=true`。
- `live_conflict`（`报销期限多久？`）：抽取到 D01（30 天）与 D02（15 天）冲突条款，`status=conflict_detected`，2 条引用，不自动选边。
- `live_missing`（`公司有住房补贴吗？`）：模型连续 5 次改写查询继续 `search`，达到 `max_steps` 上限后返回 `insufficient_evidence`，0 引用。

三条记录的可审计性说明：`live_react` 是模型在观察到已核验证据后由 `search` 调整为 `answer`，属模型依据工具结果调整动作的清晰证据；`live_missing` 中模型连续改写查询继续检索，到 `max_steps` 上限后由 `src/q3/agent.py` 强制返回 `insufficient_evidence`，触顶终止是代码行为而非模型选择；`live_conflict` 由冲突检测分支提前返回，`ask_user` 出现在 `verification.action` 而非 ReAct 决策步骤。此前 clear-case 多次未返回 `completed` 的根因是 ReAct 的 observation 只回传 `status`/`evidence_count`，没有把已核验通过的答案与证据原文交给决策模型，导致模型在看不到证据时连续拒答；修复方式是让 observation 携带 `verified`、`answer_preview`、`evidence_quotes`，并在 instruction 中说明已核验证据应返回 `answer`。该修复未改动 `quote` 精确子串校验、`citation_indices` 校验或 `max_steps` 上限。真实模型输出仍有随机性，上述记录是单次实测结果，不代表在线准确率已系统评测。

### 卷面"必须验收"六项对照

| 验收点 | 实际运行记录 | 结果 |
|---|---|---|
| 明确答案 | `examples/results/live_react.json`、`examples/traces/normal.json` | `completed`，引用 D03 原文，`verify` 通过 |
| 文档冲突 | `examples/results/live_conflict.json`、`data/q3_eval_cases.json` | `conflict_detected`，列出 D01/D02 两条原文并要求确认，不选边 |
| 资料外问题 | `examples/results/live_missing.json`、`examples/traces/missing.json` | `insufficient_evidence`，0 引用，不编造 |
| 改写后的相似问题 | `examples/traces/live_missing.json`、`examples/traces/missing.json`、`data/q3_eval_cases.json` | trace 记录多次查询改写；评测含「出差报销要准备哪些材料？」等改写题 |
| 上下文追问 | `examples/requests/q3_clear.json` | 携带 `conversation_history`，`answer(question, history)` 把历史传入 planner/action/verify |
| 工具失败 | `examples/requests/provider_failure.json`、`examples/provider_invalid_action_response.json`、`examples/traces/provider_failure.json` | 回放非法 action 响应，记录实际异常输出与失败边界。该组在首个 action 校验处即抛异常终止，因此 `trace` 为空——异常终止路径没有 ReAct 步骤，属如实披露的限制 |

注：`live_conflict` 属于冲突早返回分支，本版已把外层 ReAct 决策步骤合并进返回 trace（`react_steps` 含 step 1 `search`），保证该路径同样可审计。

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

修复 observation 信息流后，真实 CLI 三段端到端（`--llm openai-compatible --reranker provider --query-aware provider`）均产出预期状态：`差旅报销需要什么材料？` 返回 `completed`（答案带 1 条有效引用 D03，`verify` 通过）；`报销期限多久？` 返回 `conflict_detected`（D01/D02 共 2 条引用）；`公司有住房补贴吗？` 返回 `insufficient_evidence`（0 条引用）。三组运行记录见 `examples/{requests,traces,results}/live_*.json`。修复前 clear-case 曾多次未返回 `completed`，根因是决策模型收不到已核验的答案与证据；修复只补齐 observation 信息，未放宽 quote 精确子串与 `citation_indices` 校验、未加 markdown fence 兼容层、未静默回退 Mock 或伪造 completed。完整 pytest 为 68 passed、1 skipped；离线 MockTransport 覆盖重试、取消、client 生命周期与异步 ReAct 顺序。真实模型输出仍有随机性，上述为单次实测记录，不代表在线答案准确率已系统评测。

冲突检测目前依赖 token/规则 heuristic，不能视为通用语义冲突识别。仅流式空闲超时与瞬时传输错误会重试（最多额外 3 次）；HTTP 4xx、schema/JSON 错误及模型协议违规不重试，异常直接向 CLI 抛出，也不转换为 structured failed `Answer`。`max_steps` 正值校验及 evaluator 步数配置一致性仍有改进空间。
