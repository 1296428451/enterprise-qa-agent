# 第三题评测与验收

## 测试命令
从仓库任意目录执行：

```bash
python -m pytest C:/Users/ASUS/Desktop/小影笔试/agent-code/test -q
```

Embedding 冒烟测试在未设置 `FRAMES_EMBEDDING_MODEL` 时会跳过；这不等同于向量模型集成验收。

## 可复现示例
请求与结果/trace 位于 `examples/requests/`、`examples/traces/`：
- `normal.json`：有证据的差旅报销问答，返回引用并核验通过。
- `missing.json`：题库外住房补贴问题，首轮无证据、改写后仍无证据并拒答。
- `examples/traces/provider_failure.json`：显式 OpenAI-compatible provider 失败路径，不包含凭证。

Trace 明确标注 `mocked provider protocol / 非在线真实模型`。正常回答和题库外路径的观察不同，下一动作因此分别为回答与查询改写/拒答。工具失败材料基于已观察到的 TLS 证书过期阻断，记录请求异常边界；没有伪造成功调用或真实模型决策。可以用以下命令复现离线场景：

```bash
python src/q3/cli.py --input examples/requests/normal.json
python src/q3/cli.py --input examples/requests/missing.json
```

## 评测报告与指标
- `data/q3_eval_report.json`：10 条离线问答案例，报告状态准确率、文档集合 exact accuracy、宏平均文档 precision/recall/F1、可回答题 exact-match、正确拒答准确率。
- `data/q3_react_eval_report.json`：题库外 ReAct 案例，报告状态准确率、文档集合准确率和步数上限遵守率。
- `data/q3_live_accuracy_report.json`：统一记录运行模式、provider 阻断和可测的离线指标；真实 provider accuracy 标记为不可测。
- `data/external/frames_eval_report.json`：FRAMES title-only gold-link proxy，不是全文 RAG 或答案准确率评测。

指标定义：status accuracy 为预测状态匹配期望状态的比例；query/document retrieval exact accuracy 为引用文档集合完全等于期望集合的比例，额外引用和漏引均计错；document precision/recall/F1 按问题分别计算后宏平均；answer exact match 只对有参考答案的问题逐字比对；correct abstention accuracy 检查冲突/不可回答题是否返回预期非回答状态且不输出答案；ReAct step-bound accuracy 检查搜索步数是否不超过配置上限。

当前报告中，离线 10 例的状态准确率为 1.00、文档集合 exact accuracy 为 0.90、宏平均 precision/recall/F1 为 0.9667/1.00/0.98、可回答题 exact match 为 1.00（5 例）、正确拒答准确率为 1.00（5 例）。ReAct 3 例报告中的状态准确率、文档集合准确率和步数约束均为 1.00。以上是本地确定性 Mock 与合成题库结果，不是在线模型表现。

## Provider 阻断
LLM endpoint 因 TLS 证书过期未完成在线调用；reranker endpoint 响应缺少预期 `data` 数组，因此未接受分数。真实 provider 整体答案准确率不可测。不得关闭 TLS 校验来绕过问题，修复 provider 配置后需单独重跑在线验收并标注样本、模型与时间。
