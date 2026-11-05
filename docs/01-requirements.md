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
不承诺在线真实模型准确率，不包含真实生产凭证、真实企业数据、自动发布、生产级鉴权/并发治理，也不把 Mock 决策写成真实模型决策。Provider 网络层使用 `httpx.AsyncClient`，chat/completions 采用 `stream=true` SSE 并具备 30 秒 idle 超时重试（最多额外 3 次、丢弃 partial），TLS 校验始终开启。真实 smoke 中 SSE chat 与原生 reranker 调用成功，此前"TLS 证书过期"的阻断已排除；但真实 Provider 驱动的 Agent 端到端问答因模型首步决策未遵循 ReAct 协议尚未跑通，不把组件 smoke 等同于在线准确率或 Agent E2E 验证。

## 验收标准
1. 有证据问题返回 `completed`、答案和有效引用。
2. 题库外/资料不足返回 `insufficient_evidence`，不生成答案；首次无证据可改写查询，达到上限后停止。
3. 冲突或多版本返回冲突状态并请求确认。
4. 每步 trace 包含 `step`、`action`、`query`、`tool_parameters`、`result_summary`、`next_action_reason`。
5. ReAct 步数受 `max_steps` 限制；不泄露 key。
6. `python -m pytest C:/Users/ASUS/Desktop/小影笔试/agent-code/test -q` 通过（Embedding 冒烟可因未配置本地模型而跳过）。

第一题素材排期代码仅保留在历史提交，不属于本次第三题交付主体。
