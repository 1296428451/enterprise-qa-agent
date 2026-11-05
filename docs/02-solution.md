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

默认离线 Mock 是确定性规则（首次无证据追加“相关制度”），仅验证协议、流程和边界。`--llm openai-compatible` 才请求外部 provider。测试中的模拟 provider 也只验证协议行为，不是在线真实模型决策。当前真实联调受 LLM endpoint TLS 证书过期阻断，reranker 返回 schema 缺少 `data` 阻断；没有真实 provider 成功结果可报告。

## 组件与数据
- `src/q3/agent.py`：ReAct 循环、检索编排、冲突处理和 trace。
- `src/q3/retrieval.py`：混合候选、重排适配、证据选择和冲突检测。
- `src/q3/llm.py`、`src/q3/provider_reranker.py`、`src/q3/query_aware.py`：Mock/provider 接口。
- `src/q3/cli.py`、`src/q3/manage.py`：问答与版本化知识库导入管理。
- `data/`：合成企业制度、评测案例和可再生成报告；SQLite/index 属本地运行状态。

## 关键边界
缺少引用证据时不输出答案；冲突或同文档多版本不自动选边；provider 配置错误/请求失败直接暴露，不静默降级为 Mock；密钥只从环境变量读取。当前示例 trace 明确标注 `mocked provider protocol / 非在线真实模型`。

## 已知限制
语义检索依赖显式配置的本地模型或可用 provider；离线默认结果不代表生产效果。当前评测使用本地合成文档/固定案例，样本规模有限。向量索引刷新、生产并发与鉴权不属于本次交付范围。FRAMES title-only proxy 仅评估 query/article title 链接匹配，不能代表全文 RAG 或答案质量。
