# 第三题方案：企业知识问答 Agent

## 调用流程
请求解析 -> ReAct planner 决定检索动作 -> 混合候选（BM25 + mock token-overlap semantic signal） -> 候选合并 -> 重排 -> query-aware 证据抽取 -> 证据选择 -> 版本/内容冲突检查 -> 基于引用生成答案 -> 引用核验 -> 返回结果与 trace。

```text
Question
  -> bounded ReAct decision
      -> search(query) -> retrieval/rerank/extract -> observation
          -> answer (有可用证据)
          -> search(rewritten query) (首次无证据)
          -> insufficient_evidence (改写后仍无证据)
          -> ask_user (发现冲突/多版本)
```

## ReAct 与运行模式
循环由 `src/q3/agent.py` 实现，默认最多 5 步，可通过配置、输入 `config.max_steps` 或 CLI 参数覆盖。每步 trace 记录 `step`、`action`、`query`、`tool_parameters`、`result_summary`、`next_action_reason`。工具观察中的证据数、冲突和改写状态会影响下一动作：有证据可进入回答，首次无证据触发改写，改写后仍无证据则拒答。默认候选信号为 BM25 + mock token-overlap semantic signal；CLI 未创建或注入实际 dense/FAISS retriever。FAISS 是可选索引工具，不在在线问答主链。

默认离线 Mock 是确定性规则（首次无证据追加“相关制度”），仅验证协议、流程和边界。`--llm openai-compatible` 才请求外部 provider。测试中的模拟 provider 也只验证协议行为，不是在线真实模型决策。Provider 网络层使用 `httpx.AsyncClient`：chat/completions 走 `stream=true` SSE（聚合 `delta.content` 后解析 JSON，30 秒 idle 超时最多额外重试 3 次并丢弃 partial，4xx/schema/JSON 不重试，TLS 校验常开且禁止关闭），原生 `/rerank` 走非流式 JSON（支持 `results`/`data` 及 chat `scores`）。真实 SSE smoke 中各组件调用成功，证书链校验正常（默认 SSL context、SDK、requests、urllib 均通过）；`decide_action_async` instruction 收紧后真实 CLI 首步不再违反 ReAct 协议，整链可执行。冲突题 `conflict_detected`、题库外题 `insufficient_evidence`、可答题 `completed` 均符合预期（见 `examples/{requests,traces,results}/live_*.json`）。此前 clear-case 多次未返回 `completed` 是因为 observation 未回传已核验答案与证据，修复只补齐该信息流，未放宽引用校验或回退 Mock；真实模型输出仍有随机性，不声称在线准确率已系统评测。

## 组件与数据
- `src/q3/agent.py`：ReAct 循环、检索编排、冲突处理和 trace。
- `src/q3/retrieval.py`：混合候选、重排适配、证据选择和冲突检测。
- `src/q3/llm.py`、`src/q3/provider_reranker.py`、`src/q3/query_aware.py`：Mock/provider 接口。
- `src/q3/cli.py`、`src/q3/manage.py`：问答与版本化知识库导入管理。
- `data/`：合成企业制度、评测案例与可再生成报告；`data/knowledge.db` 与 `data/q3_vectors.faiss` 是随仓库提交的只读快照，重新导入或重建产生的变更属于本地运行态。

## 关键边界
缺少引用证据时不输出答案；冲突或同文档多版本不自动选边；provider 配置错误/请求失败直接暴露，不静默降级为 Mock；密钥只从环境变量读取。当前示例 trace 明确标注 `mocked provider protocol / 非在线真实模型`。

## 已知限制
语义检索需要可用的 embedding 模型：默认自动下载到 `models/all-MiniLM-L6-v2`，也可用 `--model` 指向本地目录或其它 hub id，完全离线时需提前放置权重；离线默认结果不代表生产效果。当前评测使用本地合成文档/固定案例，样本规模有限。向量索引刷新、生产并发与鉴权不属于本次交付范围。FRAMES title-only proxy 仅评估 query/article title 链接匹配，不能代表全文 RAG 或答案质量。
