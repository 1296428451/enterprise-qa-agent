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

默认使用确定性的离线 Mock，只用于验证检索流程、控制流和安全边界，不代表真实模型能力或在线准确率。Mock 查询改写是固定规则（追加“相关制度”），不是模型自主改写。显式传入 `--llm openai-compatible` 才会由配置的 provider 决定 planner/verifier action；可另行传入 `--reranker provider` 启用 reranker provider。先在当前 shell 配置相应的 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY` 和/或 `RERANK_BASE_URL`、`RERANK_MODEL`、`RERANK_API_KEY`。密钥只从环境变量读取，不写入项目文件。真实 provider 请求失败或响应格式不正确时会直接报错，不会静默降级到 Mock。当前联调未能验证在线自主决策或准确率：LLM endpoint TLS 证书已过期，reranker 响应缺少预期的 `data` 数组。

```bash
python src/q3/cli.py --question "报销期限多久？" --llm openai-compatible --reranker provider
```

新文档以版本方式追加；相同文档的多个版本同时命中时保留各版本引用并请求确认。`--rebuild-index` 仅重建 SQLite 文本索引，不代表 FAISS 向量索引刷新。SQLite 与向量文件属于本地运行状态，不应提交。

## ReAct 检索循环

循环默认最多 5 步，可在 `config.json` 的 `runtime.max_steps`、输入 JSON 的 `config.max_steps` 或 CLI 的 `--max-steps` 调整。每轮根据工具观察选择 `search`、`answer`、`ask_user` 或 `insufficient_evidence`。默认 Mock 的 action 与查询改写由确定性规则生成，首次无证据时固定追加“相关制度”；只有 `--llm openai-compatible` 才启用 provider 控制的 action 选择。搜索会执行混合召回、重排和 query-aware 证据抽取；冲突或多版本证据请求用户确认，达到步数上限则停止。

`test_q3.py` 中包含模拟 provider 的集成测试，验证 provider 决策下 `search -> observation -> rewritten search -> answer` 流程；它不是在线真实模型测试。当前真实 provider 联调被过期 TLS 证书和 reranker 响应缺少预期 `data` 数组阻断，因此不能据此声称已验证线上自主决策或准确率。

```bash
python src/q3/evaluate_react.py --cases data/q3_react_eval_cases.json --db data/knowledge.db --output data/q3_react_eval_report.json
```

题库外案例包括“今年奥运会中国有没有举行？”和住房补贴问题，验证无证据时会改写查询并在上限内拒答，不凭空生成答案。`data/react_seed_documents.json` 可用于先导入一条额外知识，再重新查询；导入过程通过 SQLite 版本管理保留。最新 ReAct 评测为 3/3 状态正确、3/3 文档集合正确、步数上限全部遵守。

## Query-Aware Evidence Extraction

重排后取 Top 20 候选，由 LLM 针对用户问题从每条候选中提取相关证据片段，去除无关内容以提高信噪比。输出为编号证据项（source ID + key info + quote），与用户问题组合为最终回答输入。提示词模板见 `src/q3/query_aware.py` 中的 `QUERY_AWARE_PROMPT`。

```bash
python src/q3/evaluate_query_aware.py --output data/q3_query_aware_report.json
```

评测报告包含证据保留率、字符/token 压缩比和逐条提取结果。

`data/external/frames_test.tsv` 是 FRAMES test split。因为当前无法取得 Wikipedia 正文，该评测只比较 query 与 article title 的匹配，属于 **title-only gold-link proxy**，不能解读为全文 RAG、答案准确率或多跳推理效果。报告与数据清单见 `data/external/frames_eval_report.json` 和 `data/external/frames_manifest.json`。

```bash
pytest test/ -v
python src/q3/evaluate_frames.py --method bm25
python src/q3/evaluate_frames.py --method rrf --model "path/to/all-MiniLM-L6-v2"
```

报告分别列出 BM25、Dense 和 RRF 的 link-micro/question-macro Recall@1/5/10、all-gold-hit、MRR@10、nDCG@10 及 reasoning-type 分层。分层是重叠多标签统计，不同类别样本数不可相加视为总样本数。

## 项目说明

第一题素材排期 Agent 保留在仓库历史；当前分支的可运行主项目是第三题。固定数据、源代码、测试、示例及评测材料纳入 Git；`.agents/`、`AGENTS.md`、`plan.md`、密钥、个人模型文件和运行缓存不纳入 Git。
