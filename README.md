# 企业知识问答 Agent（第三题）

本分支实现笔试第三题：基于本地企业文档的知识问答 CLI，包含有界 ReAct 检索循环、BM25 与 mock token-overlap semantic signal 混合候选、可配置 reranker adapter、动态证据选择、引用核验及版本冲突追问。默认 CLI 未创建或注入 FAISS semantic retriever；FAISS 是可选索引工具，不在 CLI 在线主链。当前默认离线运行，使用确定性 Mock，不会默认请求外部模型服务。

## 环境与测试
本项目在 Python 3.12（conda env `py312`）下完成全部验收；代码未使用 3.12 专属语法，理论上 3.10+ 均可运行，但只在 3.12 上逐一验证过。当前全量验收结果：模型已下载时 81 passed；未下载时 80 passed、1 skipped（embedding 冒烟项自动跳过）。先安装依赖：

```bash
python -m pip install -r requirements.txt
```

运行全量验收：

```bash
python -m pytest tests/ -q
```

`-q`：精简输出，只打印通过/失败汇总。

Embedding / FAISS 只用于可选索引与评测，主链不需要。要用时先装可选依赖，模型默认放在仓库内的相对目录 `models/all-MiniLM-L6-v2`，缺失时自动下载（权重不入库）：

```bash
python -m pip install faiss-cpu torch transformers sentencepiece safetensors
python -m src.q3.download_model     # 下载 sentence-transformers/all-MiniLM-L6-v2 到 models/
python src/q3/build_index.py        # 不传 --model 时使用上面的默认目录
pytest tests/test_embedding.py -v   # 模型缺失时该冒烟项自动 skip
```

`download_model.py` 无必填参数；可选 `--destination`（仓库相对目录，默认 `models/all-MiniLM-L6-v2`）与 `--model-id`（Hugging Face 模型 id）。`build_index.py` 的 `--model` 省略时用上述默认目录，另有 `--documents`、`--index`、`--metadata` 可改输入与产物路径。`pytest -v` 打印每个用例名。

`--model`（或环境变量 `FRAMES_EMBEDDING_MODEL`）可指向仓库内的其它相对路径或任意 Hugging Face model id；相对路径按仓库根解析，`--model models/my-model` 即 `<repo>/models/my-model`。已存在的目录一律以 `local_files_only=True` 加载。若直连 Hugging Face 不可用，设置镜像即可，例如 `HF_ENDPOINT=https://hf-mirror.com python -m src.q3.download_model`（`huggingface_hub` 的元数据接口被拦时，代码会退回按文件清单直连下载）；完全离线时把权重手工放到 `models/all-MiniLM-L6-v2` 即可。

## 问答与知识库管理

以下命令均从仓库根目录运行：

```bash
python src/q3/cli.py --question "报销期限多久？"
python src/q3/cli.py --input examples/requests/q3_conflict.json
python src/q3/manage.py --list --db data/knowledge.db

# 导入新文档并重建文本索引；把 path/to/docs.json 换成实际文件路径
python src/q3/manage.py --input path/to/docs.json --db data/knowledge.db --rebuild-index
```

`--question`：问题文本，与 `--input` 二选一。`--input`：请求 JSON，可含 `question`、`conversation_history`、`config`。`manage.py --list`：按版本列出知识库，可加 `--document-id ID` 只看单个文档；`--db`：SQLite 路径。`--rebuild-index`：在 `document_chunks(text)` 上建普通 B-tree，不改变检索方式。

默认语义候选由 `MockSemanticRetriever` 以 token overlap 计算；这不是实际 dense embedding 或 FAISS 检索。当前 CLI 没有创建/注入 FAISS semantic retriever。FAISS 仅作为可选索引/评测工具，不在 CLI 在线问答主链。BM25 与 mock token-overlap 共同提供离线候选信号。默认使用确定性离线 Mock 验证检索控制流和安全边界，不代表真实模型能力或在线准确率。Mock 查询改写是固定规则（追加“相关制度”），不是模型自主改写。显式传入 `--llm openai-compatible` 才会由配置的 provider 决定 planner/verifier action；可另行传入 `--reranker provider` 启用 reranker provider。当前 Provider 网络适配使用 `httpx.AsyncClient`，TLS 证书验证开启并配置显式超时。`EnterpriseQA.answer_async()` 可在已有 event loop 中调用；同步 CLI 的 `answer()` 仅作为兼容包装，在无运行中 event loop 时运行异步流程。OpenAI-compatible LLM 和 Query-Aware 均调用 `/chat/completions`；reranker 原生接口支持 `results`/`data`，并保留 chat `scores` 响应模式。

```bash
python src/q3/cli.py --question "报销期限多久？" --llm openai-compatible --reranker provider
```

`--llm openai-compatible`：改由真实 provider 决定 action 与查询改写（默认 `mock`）。`--reranker provider`：调用真实 reranker 打分（默认 `mock`）。

新文档以版本方式追加；相同文档的多个版本同时命中时保留各版本引用并请求确认。`--rebuild-index` 只在 `document_chunks(text)` 上建一个普通 B-tree 索引（返回 `index_type=sqlite_text`），不改变检索方式，也不是 FAISS 向量索引刷新——CLI 仍把全部 chunk 读入内存做 BM25 与 token-overlap 召回。

`data/knowledge.db` 与 `data/q3_vectors.faiss` 是随仓库提交的只读数据快照，保证开箱即可复跑上面所有命令；重新导入或重建索引产生的变更属于本地运行态，提交前请先备份或改用副本，例如 `python src/q3/manage.py --input docs.json --db /tmp/kb.db --rebuild-index`，以免污染快照。

## ReAct 检索循环

循环默认最多 5 步，可在 `config.json` 的 `runtime.max_steps`、输入 JSON 的 `config.max_steps` 或 CLI 的 `--max-steps` 调整。每轮根据工具观察选择 `search`、`answer`、`ask_user` 或 `insufficient_evidence`。默认 Mock 的 action 与查询改写由确定性规则生成，首次无证据时固定追加“相关制度”；只有 `--llm openai-compatible` 才启用 provider 控制的 action 选择。搜索会执行混合召回、重排和 query-aware 证据抽取，证据里带上条文声明的适用范围（`Scope`）；该用哪条、以及条款是否真的矛盾，由作答那一次请求一并判断——判为矛盾时模型返回 `conflict=true`，系统据此返回 `conflict_detected` 并请用户确认。同一文档多版本始终由代码判定并请求确认，达到步数上限则停止。

`tests/test_q3.py` 中包含模拟 provider 的集成测试，验证 provider 决策下 `search -> observation -> rewritten search -> answer` 流程；它不是在线真实模型测试。当前 Provider chat/completions 使用 `httpx.AsyncClient` 的 `stream=true` SSE：聚合 `delta.content` 后解析 JSON；相邻数据块空闲超过 30 秒时最多额外重试 3 次（首次 + 3 次重试，共 4 次）并丢弃 partial；HTTP 4xx、JSON、schema 错误不重试；`CancelledError` 原样传播；TLS 校验始终开启，禁止关闭验证。

真实 Provider smoke（py312 + 本地 `.env`，标准 TLS 校验）：`post_chat_stream` 成功收到 SSE 数据块、聚合 `delta.content` 并解析出 JSON 对象；`ProviderReranker.score_async` 成功对两个候选返回分数（约 `0.9992`、`0.0010`）；`QueryAwareExtractor.aextract`（quote 为候选原文精确子串）、`aanswer_from_evidence`、`averify` 均返回合规 JSON。证书链校验正常：默认 SSL context、OpenAI SDK、requests、urllib 均可成功调用，TLS verify 始终开启且禁止关闭。本次将 `src/q3/llm.py` 中 `decide_action_async` 的 instruction 收紧为"observations 为空必须 `action=search`、不得凭常识作答"后，真实 CLI 首步不再抛 `ValueError: the first ReAct action must be search`；`src/q3/agent.py` 的首步 search 硬检查保留作为兜底。

真实 CLI 三段端到端（`--llm openai-compatible --reranker provider --query-aware provider`，运行记录见 `examples/{requests,traces,results}/live_*.json`）：`差旅报销需要什么材料？` 返回 `completed`（答案「差旅报销需提交发票、行程单和审批记录。[1]」，1 条引用 D03，`verify` 通过）；`报销期限多久？` 返回 `conflict_detected`（2 条引用 D01/D02）；`公司有住房补贴吗？` 连续改写检索至步数上限后返回 `insufficient_evidence`（0 条引用）。三条路径的可审计性：`live_react` 是模型在观察到已核验证据后由 `search` 调整为 `answer`（模型动作调整的清晰证据）；`live_missing` 中模型连续改写查询继续检索，到 `max_steps` 上限后由 `src/q3/agent.py` 强制返回 `insufficient_evidence`（代码在触顶时终止，不是模型选择该动作）；`live_conflict` 由冲突检测分支提前返回 `conflict_detected`（同样是代码终止，`ask_user` 出现在 `verification.action` 而非 ReAct 决策步骤）。此前 clear-case 多次未返回 `completed` 的根因是 ReAct 的 observation 未把已核验答案与证据原文回传决策模型，修复后只补齐该信息流，未放宽 quote 精确子串与 `citation_indices` 校验、未加 markdown fence 兼容层、未静默回退 Mock 或伪造 completed。真实模型输出有随机性，以上为单次实测记录，不声称在线答案准确率已系统评测。完整 pytest 为 81 项。

```bash
python src/q3/evaluate_react.py --cases data/q3_react_eval_cases.json --db data/knowledge.db --output data/q3_react_eval_report.json
```

`--cases`：评测用例 JSON；`--db`：知识库路径；`--output`：报告输出路径；可选 `--seed`：先导入额外文档再评测。

题库外案例包括“今年奥运会中国有没有举行？”和住房补贴问题，验证无证据时会改写查询并在上限内拒答，不凭空生成答案。`data/react_seed_documents.json` 可用于先导入一条额外知识，再重新查询；导入过程通过 SQLite 版本管理保留。最新 ReAct 评测为 3/3 状态正确、3/3 文档集合正确、步数上限全部遵守。

## Query-Aware Evidence Extraction

重排后取 Top 20 候选（由 `config.json` 的 `retrieval.rerank_top_k` 决定，默认 20；输入 JSON 的 `config.rerank_top` 可覆盖），由 LLM 针对用户问题从每条候选中提取相关证据片段，去除无关内容以提高信噪比。输出为编号证据项（source ID + key info + quote），与用户问题组合为最终回答输入。提示词模板见 `src/q3/query_aware.py` 中的 `QUERY_AWARE_PROMPT`。

```bash
python src/q3/evaluate_query_aware.py --output data/q3_query_aware_report.json
```

`--output`：报告输出路径（默认 `data/q3_query_aware_report.json`）。

评测报告包含证据保留率、字符/token 压缩比和逐条提取结果。

`data/external/frames_test.tsv` 是 FRAMES test split。因为当前无法取得 Wikipedia 正文，该评测只比较 query 与 article title 的匹配，属于 **title-only gold-link proxy**，不能解读为全文 RAG、答案准确率或多跳推理效果。报告与数据清单见 `data/external/frames_eval_report.json` 和 `data/external/frames_manifest.json`。

```bash
pytest tests/ -v
python src/q3/evaluate_frames.py --method bm25
python src/q3/evaluate_frames.py --method rrf --model models/all-MiniLM-L6-v2
```

`--method`：`bm25` / `dense` / `rrf`；`--model`：dense 与 rrf 需要，省略时用 `models/` 默认目录并自动下载；另可加 `--batch-size`、`--data`、`--output` 改批大小、输入与报告路径。

报告分别列出 BM25、Dense 和 RRF 的 link-micro/question-macro Recall@1/5/10、all-gold-hit、MRR@10、nDCG@10 及 reasoning-type 分层。分层是重叠多标签统计，不同类别样本数不可相加视为总样本数。

## 第三题示例与项目说明

- 离线 Mock 请求：`examples/requests/normal.json`、`examples/requests/missing.json`、`examples/requests/provider_failure.json`
- 离线 Mock 输出与 trace：`examples/traces/normal.json`、`examples/traces/missing.json`、`examples/traces/provider_failure.json`
- provider schema 失败回放：`python src/q3/cli.py --input examples/requests/provider_failure.json --llm openai-compatible --provider-response-fixture examples/provider_invalid_action_response.json`

### 一条命令跑通真实 Provider 并落盘

`--save-run NAME` 会把请求、ReAct 过程、最终结果分别写入 `examples/requests/NAME.json`、`examples/traces/NAME.json`、`examples/results/NAME.json`，同时在标准错误输出打印从 ReAct 决策到模型回答的完整过程，标准输出仍是原有 JSON。同名文件会被覆盖（覆盖时标准错误会明确提示），仓库内的 `live_*` 是已提交的实测记录，复跑时建议换一个名字，例如 `my_run`：

```bash
python src/q3/cli.py --question "差旅报销需要什么材料？" \
  --llm openai-compatible --reranker provider --query-aware provider \
  --max-steps 5 --save-run my_run
```

`--query-aware provider`：用真实 provider 抽取证据（默认 `mock`）。`--max-steps`：ReAct 步数上限（默认取 `config.json` 的 `runtime.max_steps`，即 5）。`--save-run NAME`：把请求、过程与结果写入 `examples/{requests,traces,results}/NAME.json`；同名覆盖会在 stderr 提示。其余可选参数：`--candidate-limit`、`--max-evidence`、`--db`、`--provider-response-fixture`（回放本地 HTTP fixture）、`--admin-import`（批量导入文档）。

落盘文件只记录 CLI 参数，不写任何环境变量值、API Key、endpoint 或主机名。已提交三条真实 Provider 运行记录：

| NAME | 场景 | 决策路径 | 最终状态 |
|---|---|---|---|
| `live_react` | 有明确证据的差旅报销 | `search → answer` | `completed`，返回带引用答案 |
| `live_conflict` | 报销期限存在两条冲突制度 | 冲突检测后 `ask_user` | `conflict_detected`，2 条引用 |
| `live_missing` | 题库外住房补贴 | 连续 5 次 `search` 改写查询后触顶 | `insufficient_evidence`，0 引用 |

三条路径体现模型依据不同工具观察调整动作；`traces/*.json` 的 `react_steps` 记录 step、action、query、tool_parameters、result_summary、next_action_reason，`pipeline_trace` 按实际经过的阶段记录 plan/检索/rerank/冲突提示/抽取/选证/作答/核验（冲突题在作答阶段即判为 `conflict=true`，不再进入核验）。

示例 trace 标注为 `mocked provider protocol / 非在线真实模型`；失败 trace 记录 provider 阻断，不包含凭证。第一题素材排期 Agent 的历史代码仍以 `src/agent.py`、`src/api.py`、`src/cli.py`、`data/assets.json`、`data/rules.json`、`data/schedule.json`、`tests/test_agent.py`、`tests/test_persistence.py` 形式留在当前仓库树中，**不属于第三题验收主体**；`pytest` 收集 81 项，其中 10 项为上述第一题历史用例、71 项属第三题；第三题内 1 项 embedding 冒烟在模型未下载时自动跳过，故显示 80 passed / 1 skipped。固定数据、源代码、测试、示例及评测材料纳入 Git；`.agents/`、`AGENTS.md`、`plan.md`、密钥、个人模型文件和运行缓存不纳入 Git。

## 凭证安全与提交前自检

`.env`（仓库根）与 `provider.txt`（仓库上一级目录，仅本地存在）保存真实 API Key，二者均不纳入 Git（见 `.gitignore`）；仓库只提供 `.env.example` 占位模板。代码只通过环境变量读取凭证，不会写入日志、trace、示例或报告。TLS 证书校验始终开启，禁止关闭。

提交或推送前可运行自检脚本，它会扫描**全部 Git 历史对象**（含 blob、commit/tag 消息、悬空对象）、工作区与暂存区，检查本地配置中的凭证值以及常见凭证格式（`sk-`、`gh*_`、JWT-like、Bearer 字面量、赋值式 api key）：

```bash
python scripts/check_secrets.py                 # 全历史 + 工作区，退出码 0 表示干净
python scripts/check_secrets.py --staged-only   # 仅暂存区，适合作为 pre-commit 钩子
```

`--staged-only`：只检查暂存区，适合作为 pre-commit 钩子；省略时扫描全部 Git 历史、对象与工作区。

脚本只在内存中比对，输出仅含 SHA-256 短标签、对象短 id 与文件路径，**不打印任何凭证字符、片段、完整 URL 或主机名**；命中凭证时退出码为 1。非凭证的公开模型名（如 reranker 模型 ID）会作为 INFO 单独列出，不算泄露。可选启用本地钩子：

```bash
git config core.hooksPath scripts
```

`core.hooksPath`：把 Git 钩子目录指向仓库内 `scripts/`，使 `scripts/hooks/pre-commit` 生效；不设置则钩子不会被调用。

（钩子位于 `scripts/hooks/pre-commit`，不会随仓库自动生效，需执行上面的配置命令。）

注意：`git` 历史一旦推送到远端便无法靠后续提交彻底抹除，若 `.env` 曾泄露到对话或日志，应先在 Provider 控制台轮换该 Key，再确认历史干净后推送。

## 提交信息
题号：第三题。GitHub 仓库地址与最终提交 SHA 属提交前最后更新字段，应在实际推送/提交前按最终仓库信息填写，文档不硬编码未来 SHA。当前本地 HEAD 仅代表编辑开始前基线，不作为最终提交 SHA。
