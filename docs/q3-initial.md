# 第三题初步实现说明

当前分支：`question-3-enterprise-rag`。

## 当前可运行能力

```bash
python src/q3/cli.py --question "差旅报销需要什么材料？"
python src/q3/cli.py --input examples/requests/q3_conflict.json
```

也支持 JSON 标准输入。当前版本使用本地合成制度数据、BM25 与轻量语义 Mock、可替换的 Jev reranker 接口和结构化 trace。明确答案会返回引用；D01/D02 冲突会请求用户确认；资料外问题会返回 `insufficient_evidence`。

## 真实模型状态

本地已接入 `all-MiniLM-L6-v2` embedding，并成功生成 384 维归一化向量及 FAISS 索引。Jev 权重未在给定模型目录中发现，因此不声称使用真实 Jev。

## Reranker model status

The preferred open reranker is `Qwen/Qwen3-Reranker-0.6B`; alternatives are `mixedbread-ai/mxbai-rerank-base-v2` and `BAAI/bge-reranker-v2-m3`. All are below 1B parameters and suitable for local pairwise relevance scoring. The current Hugging Face mirror/proxy attempt returned HTTP 502 and left only incomplete download files, so none is enabled yet. The project keeps the tested Jev-compatible adapter and Mock fallback rather than claiming a partial model is usable.

The provider configuration supplied outside the repository identifies `Qwen/Qwen3-Reranker-0.6B` at an OpenAI-style `/v1/rerank` endpoint. `src/q3/provider_reranker.py` implements the compatible request adapter, but no key is stored in the repository and no external request is made by default. Enable it explicitly with `RERANK_BASE_URL`, `RERANK_MODEL`, and `RERANK_API_KEY` in the shell.

## Local semantic index and database

Build both persisted artifacts from the repository root after the test gate:

```bash
python -m unittest discover -s test -v
python src/q3/build_index.py --model "E:/Users/ASUS/PycharmProjects/pythonProject1/MyServer/models/AI-ModelScope/all-MiniLM-L6-v2"
python src/q3/build_database.py
```

The SQLite database contains four synthetic documents and four chunks. Runtime state remains ignored.

## 评测结果（10 cases）

从仓库根目录运行：

```bash
python src/q3/evaluate.py --cases data/q3_eval_cases.json --db data/knowledge.db --output data/q3_eval_report.json
```

当前检索门槛要求多词查询至少命中两个不同非停用词，并对语义 Mock 结果设置最低重叠度，以避免“员工福利积分如何兑换？”因共享“员工”单字误召回。最新结果：

- `status_accuracy`: **1.00**
- `document_recall_rate`: **1.00**
- 全量 unittest：21 passed、1 skipped（本地 embedding 依赖未启用）


**限制：**Wikipedia 文章正文因代理 TLS 过期无法取得，本评测因此采用 title-only gold-link proxy：从每题 `wiki_links` 汇总原始 URL，得到 2,507 个候选 ID，并以 URL 中的 Wikipedia 标题作为候选文本。指标仅衡量题目与 gold 标题的匹配，**不是 full-text RAG**，不应与论文正文检索结果直接比较。来源、Apache-2.0 数据集许可、文件字节数、824 行及 SHA256 见 `data/external/frames_manifest.json`。

复现命令（从仓库根目录执行；dense 需要本地 `py312` embedding 环境）：

```bash
python -m unittest discover -s test -v
python src/q3/evaluate_frames.py --data data/external/frames_test.tsv --model "<local all-MiniLM-L6-v2 path>" --output data/external/frames_eval_report.json
```

评测输出 BM25、all-MiniLM-L6-v2 dense cosine，以及 RRF（k=60）的 Recall@5/10、MRR@10、nDCG@10 与样本数。当前实测（824 queries）：

| Method | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 title-only | 0.3592 | 0.3900 | 0.6135 | 0.4055 |
| all-MiniLM-L6-v2 title-only | 0.5291 | 0.5915 | 0.8274 | 0.5949 |
| RRF BM25+dense title-only | 0.5197 | 0.5793 | 0.8043 | 0.5750 |

详见机器可读报告 `data/external/frames_eval_report.json`。
