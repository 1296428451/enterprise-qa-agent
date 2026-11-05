# 第三题评测与当前状态

当前分支：`question-3-enterprise-rag`。

## 企业问答 CLI

```bash
python -m pytest tests/ -v
python src/q3/cli.py --question "差旅报销需要什么材料？"
python src/q3/cli.py --input examples/requests/q3_conflict.json
```

系统使用本地合成制度数据、BM25 与语义召回、可替换的 reranker adapter 和结构化 trace。默认模型适配为离线 Mock；明确答案返回引用，制度版本冲突时保留引用并请求用户确认，资料外问题返回 `insufficient_evidence`。

## 模型状态

本地 `all-MiniLM-L6-v2` embedding 已在 `py312` 下生成 384 维归一化向量并用于 FAISS。默认运行使用确定性离线 Mock，仅验证流程；Mock 改写仅追加固定词语，不代表模型自主决策。真实 LLM planner/verifier 需显式传入 `--llm openai-compatible`，Provider reranker 需显式传入 `--reranker provider` 并配置相应环境变量。接口的模拟 HTTP 单测验证协议行为，不代表在线服务已联通。

## 合成问答专项评测

```bash
python src/q3/evaluate.py --cases data/q3_eval_cases.json --db data/knowledge.db --output data/q3_eval_report.json
```

最近一次 10 条专项评测结果：`status_accuracy=1.00`、`query_retrieval_accuracy=0.90`、`document_precision_macro=0.9667`、`document_recall_macro=1.00`、`document_f1_macro=0.9800`、`answer_exact_match_accuracy=1.00`（5 条可回答问题）、`correct_abstention_accuracy=1.00`（5 条冲突/不可回答问题）。逐题报告在 `data/q3_eval_report.json`。

## ReAct 题库外评测

```bash
python src/q3/evaluate_react.py --cases data/q3_react_eval_cases.json --db data/knowledge.db --output data/q3_react_eval_report.json
```

该专项覆盖题库外奥运会问题、住房补贴问题和明确可答问题，检查是否改写查询、是否遵守最大步数并在证据不足时拒答。最新结果为状态准确率 1.00、问题级文档检索准确率 1.00、步数上限遵守率 1.00。它验证本地 Mock 的有界流程，不代表真实 LLM 决策质量。

## Provider 联调状态

已使用项目 `.env` 尝试真实 provider 联调，但本轮未将失败请求回退为 Mock：

- LLM provider：TLS 握手失败，服务端证书已过期；
- Reranker provider：请求返回内容缺少预期的 `data` 数组，系统明确报 provider schema 错误；
- 未关闭 TLS 证书校验，未把异常结果计入真实模型准确率；
- 详细记录见 `data/q3_live_accuracy_report.json`。

因此当前可报告的准确率是离线 Mock + 本地知识库评测，不应表述为真实在线 LLM 准确率。修复 provider 证书和响应格式后，再单独重跑真实模型评测。

## FRAMES title-only gold-link proxy

官方 test split 包含 824 个问题、2,636 个 gold-link occurrences、2,507 个去重原始 gold URL 和 2,507 个候选列原始 URL。规范化后有 2,475 个候选文章 ID；4 个唯一 URL 因缺 scheme、非 Wikipedia host 或 `Special:Search` 被排除。数据及本地 hash 见 `data/external/frames_manifest.json`。

运行完整 BM25、Dense 与 RRF 代理评测：

```powershell
$env:FRAMES_EMBEDDING_MODEL="path/to/all-MiniLM-L6-v2"
python -m pytest tests/ -v
python src/q3/evaluate_frames.py --method rrf --model $env:FRAMES_EMBEDDING_MODEL --output data/external/frames_eval_report.json
```

下表给出 question-macro Recall@5、link-micro Recall@5、MRR@10 与 nDCG@10：

| Method | Macro R@5 | Link R@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 title-only | 0.3601 | 0.3063 | 0.6175 | 0.4095 |
| all-MiniLM-L6-v2 title-only | 0.5321 | 0.4732 | 0.8328 | 0.5999 |
| RRF BM25+dense title-only | 0.5248 | 0.4614 | 0.8107 | 0.5805 |

Dense 在这四项汇总指标上高于 RRF；不应仅因混合方案更复杂就默认选择 RRF。完整 Recall@1/5/10、all-gold-hit@k、reasoning-type 分层和逐题明细见机器可读报告。

**解释边界：**候选文本只有 Wikipedia 标题，无法取得文章正文是因为代理 TLS 验证失败。该 proxy 只评估标题匹配和 gold-link 排序；它不是全文证据检索、答案准确率、引用支持或端到端多跳 RAG 评测。候选集来自 test split 本身，也不构成独立开放语料检索基准。
