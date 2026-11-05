# 第三题需求：企业知识问答 Agent

## 用户与问题
企业员工向内部知识库提问，希望得到可引用、可核验的制度答案；当资料不足或版本/内容冲突时，系统必须拒答或请求用户确认，而不是编造答案。

## 目标
实现一个可离线复现的企业文档问答 CLI/API 核心：混合召回企业文档，经过重排与 query-aware 证据抽取，使用有界 ReAct 循环决定继续检索、回答或拒答，并返回引用、核验结果和完整 trace。

## 范围
- 文档版本化导入与 SQLite 知识库管理。
- BM25 与 mock token-overlap semantic signal 混合候选、Mock JEV 重排、可选 provider reranker。默认 CLI 未注入实际 dense/FAISS semantic retriever；FAISS 不在在线问答主链。
- query-aware 证据片段抽取、引用与冲突/多版本核验。
- ReAct 动作：`search`、`answer`、`ask_user`、`insufficient_evidence`；步数有上限。
- 离线确定性 Mock 运行、OpenAI-compatible provider 协议适配、CLI 与测试。
- 示例请求、原始输出/trace、离线评测报告。

## 非范围与边界
不承诺在线真实模型准确率，不包含真实生产凭证、真实企业数据、自动发布、生产级鉴权/并发治理，也不把 Mock 决策写成真实模型决策。Provider 网络层使用 `httpx.AsyncClient`，chat/completions 采用 `stream=true` SSE 并具备 30 秒 idle 超时重试（最多额外 3 次、丢弃 partial，HTTP/schema/JSON 不重试、`CancelledError` 原样传播），TLS 校验始终开启且禁止关闭。证书链校验正常：默认 SSL context、OpenAI SDK、requests、urllib 均成功调用。真实 smoke 中 SSE chat、原生 reranker 与 Query-Aware/答案生成/核验各阶段调用成功；`decide_action_async` instruction 增加"observations 为空必须 search"约束并保留首步 search 硬检查后，真实 CLI 整链不再因首步协议抛错。冲突题返回 `conflict_detected`（2 条引用）、题库外题返回 `insufficient_evidence`（0 条引用），均符合预期；修复 observation 信息流后，题库内 clear-case 已返回带引用的 `completed` 答案，三段真实运行记录见 `examples/{requests,traces,results}/live_*.json`。真实模型输出有随机性，仍不把单次实测等同于在线准确率系统评测。

## 验收标准
1. 有证据问题返回 `completed`、答案和有效引用。
2. 题库外/资料不足返回 `insufficient_evidence`，不生成答案；首次无证据可改写查询，达到上限后停止。
3. 冲突或多版本返回冲突状态并请求确认。
4. 每步 trace 包含 `step`、`action`、`query`、`tool_parameters`、`result_summary`、`next_action_reason`。
5. ReAct 步数受 `max_steps` 限制；不泄露 key。
6. `python -m pytest tests/ -q` 通过（Embedding 冒烟可因未配置本地模型而跳过）。

第一题素材排期代码仍以 `src/agent.py`、`src/api.py`、`src/cli.py`、`data/assets.json`、`data/rules.json`、`data/schedule.json`、`tests/test_agent.py`、`tests/test_persistence.py` 留在当前仓库树，不属于第三题验收主体；pytest 收集 69 项，其中 10 项为上述历史用例、59 项属第三题（1 项 embedding 冒烟跳过），故显示 68 passed / 1 skipped。
