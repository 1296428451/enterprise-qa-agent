# 第三题评测与当前状态

当前分支：`question-3-enterprise-rag`。

## 企业问答 CLI

```bash
python -m unittest discover -s test -v
python src/q3/cli.py --question "差旅报销需要什么材料？"
python src/q3/cli.py --input examples/requests/q3_conflict.json
```

系统使用本地合成制度数据、BM25 与语义召回、可替换的 reranker adapter 和结构化 trace。默认模型适配为离线 Mock；明确答案返回引用，制度版本冲突时保留引用并请求用户确认，资料外问题返回 `insufficient_evidence`。

## 模型状态

本地 `all-MiniLM-L6-v2` embedding 已在 `py312` 下生成 384 维归一化向量并用于 FAISS。Reranker 可通过 provider adapter 配置，但默认不发网络请求；小于 1B 的本地 reranker 权重未成功下载并启用。真实 LLM planner/verifier 也尚未接入，因此目前不声称真实模型的规划或答案核验质量。

## 合成问答专项评测

```bash
python src/q3/evaluate.py --cases data/q3_eval_cases.json --db data/knowledge.db --output data/q3_eval_report.json
```

10 条专项样本覆盖明确答案、同义改写、冲突和资料外问题。最近一次确认结果为 `status_accuracy=1.00`、`document_recall_rate=1.00`。该小规模合成评测只验证项目设定场景，不代表外部知识库泛化能力。

## FRAMES title-only gold-link proxy

官方 test split 包含 824 个问题、2,636 个 gold-link occurrences、2,507 个去重原始 gold URL 和 2,507 个候选列原始 URL。规范化后有 2,475 个候选文章 ID；4 个唯一 URL 因缺 scheme、非 Wikipedia host 或 `Special:Search` 被排除。数据及本地 hash 见 `data/external/frames_manifest.json`。

运行完整 BM25、Dense 与 RRF 代理评测：

```powershell
$env:FRAMES_EMBEDDING_MODEL="path/to/all-MiniLM-L6-v2"
python -m unittest discover -s test -v
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