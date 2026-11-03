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

## 评测边界

外部 RAG 数据集只用于检索评测，不混入企业知识库。第三题最终验收仍以题目规定的合成 FAQ/制度场景为准，重点检查引用支持、冲突保留、资料外拒答、上下文追问和工具失败。当前 Hugging Face FRAMES 下载受本机 TLS/代理连接失败阻塞，未伪造外部数据集指标。
