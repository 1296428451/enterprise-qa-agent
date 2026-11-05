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
- `data/q3_live_accuracy_report.json`：记录运行模式、provider 阻断和可测的离线指标；真实 provider accuracy 不可测。
- `data/external/frames_eval_report.json`：FRAMES title-only gold-link proxy，不是全文 RAG 或答案准确率评测。

指标定义：status accuracy 为预测状态匹配期望状态的比例；query/document retrieval exact accuracy 为引用文档集合完全等于期望集合的比例，额外引用和漏引均计错；document precision/recall/F1 按问题分别计算后宏平均；answer exact match 只对有参考答案的问题逐字比对；correct abstention accuracy 检查冲突/不可回答题是否返回预期非回答状态且不输出答案；ReAct step-bound accuracy 检查搜索步数是否不超过配置上限。

离线 10 例状态准确率为 1.00、文档集合 exact accuracy 为 0.90、宏平均 precision/recall/F1 为 0.9667/1.00/0.98、可回答题 exact match 为 1.00（5 例）、正确拒答准确率为 1.00（5 例）。ReAct 3 例报告中的状态准确率、文档集合准确率和步数约束均为 1.00。以上为本地确定性 Mock 与合成题库结果，不是在线模型表现。

## Provider 阻断与其他限制
LLM endpoint 因 TLS 证书过期未完成在线调用；reranker endpoint 响应缺少预期 `data` 数组，因此未接受分数。真实 provider 整体答案准确率不可测。不得关闭 TLS 校验绕过问题，修复配置后需重跑在线验收。

冲突检测目前依赖 token/规则 heuristic，不能视为通用语义冲突识别。Provider 异常直接向 CLI 抛出，不重试，也不转换为 structured failed `Answer`。`max_steps` 正值校验及 evaluator 步数配置一致性仍有改进空间。
