# 第三题初步实现说明

当前分支：`question-3-enterprise-rag`。

## 当前可运行能力

```bash
python src/q3/cli.py --question "差旅报销需要什么材料？"
python src/q3/cli.py --input examples/requests/q3_conflict.json
```

也支持 JSON 标准输入。当前版本使用本地合成制度数据、BM25 与轻量语义 Mock、可替换的 Jev reranker 接口和结构化 trace。明确答案会返回引用；D01/D02 冲突会请求用户确认；资料外问题会返回 `insufficient_evidence`。

## 真实模型状态

本地已发现 `bge-m3` 和 `all-MiniLM-L6-v2` 文件，但当前环境尚未完成真实 embedding 适配；Jev 权重也未在给定模型目录中发现。因此当前结果不声称使用真实向量模型或真实 Jev。下一步先在 tests 通过后接入真实模型，并以同一套指标比较 Mock 与真实实现。

## Reranker model status

The preferred open reranker is `Qwen/Qwen3-Reranker-0.6B`; alternatives are `mixedbread-ai/mxbai-rerank-base-v2` and `BAAI/bge-reranker-v2-m3`. All are below 1B parameters and suitable for local pairwise relevance scoring. The current Hugging Face mirror/proxy attempt returned HTTP 502 and left only incomplete download files, so none is enabled yet. The project keeps the tested Jev-compatible adapter and Mock fallback rather than claiming a partial model is usable.

The local `all-MiniLM-L6-v2` model was loaded successfully in the `py312` environment and produced normalized 384-dimensional vectors. Build the persisted FAISS index from the repository root after the test gate:

```bash
python -m unittest discover -s test -v
python src/q3/build_index.py --model "E:/Users/ASUS/PycharmProjects/pythonProject1/MyServer/models/AI-ModelScope/all-MiniLM-L6-v2"
```

The small generated index and metadata are reproducible from `data/q3_documents.json`. Jev remains an adapter because no local Jev weights were found; public Jev references describe a hosted TypeSafe decision API rather than downloadable weights.

## 评测边界

外部 RAG 数据集只用于检索评测，不混入企业知识库。第三题最终验收仍以题目规定的合成 FAQ/制度场景为准，重点检查引用支持、冲突保留、资料外拒答、上下文追问和工具失败。当前 Hugging Face FRAMES 下载受本机 TLS 连接失败阻塞，未伪造外部数据集指标。
