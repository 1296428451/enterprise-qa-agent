# 企业知识问答 Agent（第三题）

本分支实现笔试第三题：基于本地企业文档的知识问答 CLI，包含有界 ReAct 检索循环、BM25 与语义向量混合召回、可配置 reranker adapter、动态证据选择、引用核验及版本冲突追问。当前默认离线运行，使用确定性 Mock，不会默认请求外部模型服务。

## 环境与测试

Python 3.12 环境建议使用项目已配置的 `py312`：

```bash
conda activate py312
pytest test/ -v
```

Embedding 冒烟测试默认跳过；设置本地模型目录后可启用：

```bash
export FRAMES_EMBEDDING_MODEL=path/to/all-MiniLM-L6-v2
pytest test/test_embedding.py -v
```

PowerShell 可用 `$env:FRAMES_EMBEDDING_MODEL="path/to/all-MiniLM-L6-v2"` 设置变量。

## 问答与知识库管理

以下命令均从仓库根目录运行：

```bash
python src/q3/cli.py --question "报销期限多久？"
python src/q3/cli.py --input examples/requests/q3_conflict.json
python src/q3/manage.py --input path/to/docs.json --db data/knowledge.db --rebuild-index
python src/q3/manage.py --list --db data/knowledge.db
```

默认使用离线 Mock。可显式启用 OpenAI-compatible planner/verifier 或 Provider reranker；先在当前 shell 中配置 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY` 和/或 `RERANK_BASE_URL`、`RERANK_MODEL`、`RERANK_API_KEY`，再按需传入 `--llm openai-compatible`、`--reranker provider`。密钥只从环境变量读取，不写入项目文件。真实 provider 请求失败或响应格式不正确时会直接报错，不会静默降级到 Mock。

```bash
python src/q3/cli.py --question "报销期限多久？" --llm openai-compatible --reranker provider
```

新文档以版本方式追加；相同文档的多个版本同时命中时保留各版本引用并请求确认。`--rebuild-index` 仅重建 SQLite 文本索引，不代表 FAISS 向量索引刷新。SQLite 与向量文件属于本地运行状态，不应提交。

## ReAct 检索循环

每轮由模型根据工具观察选择 `search`、`answer`、`ask_user` 或 `insufficient_evidence`。`search` 会执行混合召回、重排和 query-aware 证据抽取；无证据时可以改写查询并再次搜索。循环默认最多 5 步，可在 `config.json` 的 `runtime.max_steps`、输入 JSON 的 `config.max_steps` 或 CLI 的 `--max-steps` 调整。达到上限会停止并返回证据不足；冲突或多版本证据始终请求用户确认。

## Query-Aware Evidence Extraction

重排后取 Top 20 候选，由 LLM 针对用户问题从每条候选中提取相关证据片段，去除无关内容以提高信噪比。输出为编号证据项（source ID + key info + quote），与用户问题组合为最终回答输入。提示词模板见 `src/q3/query_aware.py` 中的 `QUERY_AWARE_PROMPT`。

```bash
python src/q3/evaluate_query_aware.py --output data/q3_query_aware_report.json
```

评测报告包含证据保留率、字符/token 压缩比和逐条提取结果。

## ReAct 搜索循环

系统采用有界 ReAct 工作流：模型根据上一次检索观察选择 `search`、改写查询、`answer` 或 `insufficient_evidence`。最多执行 5 步，避免无限搜索。首次查询没有证据时，Mock planner 会自动追加查询词；真实 LLM 模式由模型返回结构化 action 和 rewritten query。

```bash
python src/q3/evaluate_react.py --cases data/q3_react_eval_cases.json --db data/knowledge.db --output data/q3_react_eval_report.json
```

题库外案例包括“今年奥运会中国有没有举行？”和住房补贴问题，验证无证据时会改写查询并在上限内拒答，不凭空生成答案。`data/react_seed_documents.json` 可用于先导入一条额外知识，再重新查询；导入过程通过 SQLite 版本管理保留。最新 ReAct 评测为 3/3 状态正确、3/3 文档集合正确、步数上限全部遵守。


`data/external/frames_test.tsv` 是 FRAMES test split。因为当前无法取得 Wikipedia 正文，该评测只比较 query 与 article title 的匹配，属于 **title-only gold-link proxy**，不能解读为全文 RAG、答案准确率或多跳推理效果。报告与数据清单见 `data/external/frames_eval_report.json` 和 `data/external/frames_manifest.json`。

```bash
pytest test/ -v
python src/q3/evaluate_frames.py --method bm25
python src/q3/evaluate_frames.py --method rrf --model "path/to/all-MiniLM-L6-v2"
```

报告分别列出 BM25、Dense 和 RRF 的 link-micro/question-macro Recall@1/5/10、all-gold-hit、MRR@10、nDCG@10 及 reasoning-type 分层。分层是重叠多标签统计，不同类别样本数不可相加视为总样本数。

## 项目说明

第一题素材排期 Agent 保留在仓库历史；当前分支的可运行主项目是第三题。固定数据、源代码、测试、示例及评测材料纳入 Git；`.agents/`、`AGENTS.md`、`plan.md`、密钥、个人模型文件和运行缓存不纳入 Git。
