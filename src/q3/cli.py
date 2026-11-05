import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.q3.settings import load_config, load_env_file

load_env_file(ROOT / ".env")
CONFIG = load_config(ROOT / "config.json")

from src.q3.agent import EnterpriseQA, load_chunks
from src.q3.llm import MockLLM, OpenAICompatibleLLM
from src.q3.manage import import_documents
from src.q3.provider_reranker import ProviderReranker
from src.q3.query_aware import MockQueryAwareExtractor, OpenAICompatibleQueryAwareExtractor
from src.q3.retrieval import MockJevReranker


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_run(name: str, args, question: str, payload: dict, result: dict) -> None:
    """Persist request, ReAct process and final result under examples/.

    Only CLI arguments are recorded: no environment variable values, API keys,
    endpoint URLs or hostnames are written to disk.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise SystemExit("--save-run NAME must match [A-Za-z0-9_-]+")
    base = ROOT / "examples"
    command = "python src/q3/cli.py " + " ".join([
        "--question", json.dumps(question, ensure_ascii=False),
        "--llm", args.llm, "--reranker", args.reranker,
        "--query-aware", args.query_aware, "--max-steps", str(args.max_steps),
        "--save-run", name,
    ])
    mode = ("live provider (openai-compatible)" if args.llm == "openai-compatible"
            else "offline deterministic mock")
    trace = result.get("trace", [])
    react_steps = [item for item in trace if "step" in item]
    pipeline_steps = [item for item in trace if "step" not in item]

    request_doc = {
        "question": question,
        "conversation_history": payload.get("conversation_history", []),
        "config": payload.get("config", {}),
        "mode": {"llm": args.llm, "reranker": args.reranker,
                 "query_aware": args.query_aware, "max_steps": args.max_steps},
    }
    trace_doc = {
        "scenario": "ReAct 决策循环与检索/抽取/核验流水线",
        "provider_mode": mode,
        "command": command,
        "request_file": f"examples/requests/{name}.json",
        "result_file": f"examples/results/{name}.json",
        "react_steps": react_steps,
        "pipeline_trace": pipeline_steps,
        "full_trace": trace,
        "note": "真实 provider 运行时由模型决定 action 与查询改写；离线 mock 为确定性规则，不代表模型自主决策。",
    }
    result_doc = {
        "scenario": "最终回答与引用核验结果",
        "provider_mode": mode,
        "command": command,
        "question": question,
        "status": result.get("status"),
        "completeness": result.get("completeness"),
        "answer": result.get("answer"),
        "formatted": result.get("formatted"),
        "citations": result.get("citations", []),
        "gaps": result.get("gaps", []),
        "retrieval": result.get("retrieval", {}),
        "verification": result.get("verification", {}),
    }
    targets = (
        (base / "requests" / f"{name}.json", request_doc),
        (base / "traces" / f"{name}.json", trace_doc),
        (base / "results" / f"{name}.json", result_doc),
    )
    for path, document in targets:
        if path.exists():
            # Reusing a name silently replaced the committed run record once;
            # say so out loud instead of overwriting evidence without a trace.
            print(f"[save-run] overwriting existing {path.relative_to(ROOT)}",
                  file=sys.stderr)
        _write_json(path, document)
    for path, _ in targets:
        print(f"[save-run] {path.relative_to(ROOT)}", file=sys.stderr)


def print_react_summary(question: str, result: dict) -> None:
    """Print a human-readable ReAct process followed by the model's answer."""
    print("=" * 68, file=sys.stderr)
    print("ReAct 执行过程", file=sys.stderr)
    print("=" * 68, file=sys.stderr)
    print(f"问题：{question}", file=sys.stderr)
    for item in result.get("trace", []):
        if "step" in item:
            print(f"  [step {item['step']}] action={item.get('action')} "
                  f"query={item.get('query')}", file=sys.stderr)
            print(f"      tool_parameters={item.get('tool_parameters')}", file=sys.stderr)
            print(f"      result_summary={item.get('result_summary')} "
                  f"next_action_reason={item.get('next_action_reason')}", file=sys.stderr)
        else:
            extra = " ".join(f"{k}={v}" for k, v in item.items()
                             if k not in {"action", "result", "queries", "intent", "note"})
            print(f"  - {item.get('action')}: {item.get('result')} {extra}", file=sys.stderr)
    print("-" * 68, file=sys.stderr)
    print(f"最终状态：{result.get('status')}", file=sys.stderr)
    print(f"完整度：{result.get('completeness')}"
          f"（complete=完全回答 / partial=部分回答，缺口见 gaps / none=未回答）", file=sys.stderr)
    print(f"模型回答：{result.get('answer')}", file=sys.stderr)
    print("格式化输出：", file=sys.stderr)
    print(result.get("formatted") or "", file=sys.stderr)
    print(f"核验：{result.get('verification')}", file=sys.stderr)
    print(f"缺口：{result.get('gaps')}", file=sys.stderr)
    print("=" * 68, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Question three enterprise QA Agent")
    parser.add_argument("--question")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--candidate-limit", type=int, default=CONFIG["retrieval"]["candidate_limit"])
    parser.add_argument("--max-evidence", type=int, default=CONFIG["retrieval"]["max_evidence"])
    parser.add_argument("--max-steps", type=int, default=CONFIG["runtime"].get("max_steps", 5))
    parser.add_argument("--llm", choices=("mock", "openai-compatible"), default=CONFIG["runtime"]["llm"])
    parser.add_argument("--reranker", choices=("mock", "provider"), default=CONFIG["runtime"]["reranker"])
    parser.add_argument("--query-aware", choices=("mock", "provider"), default=CONFIG["runtime"]["query_aware_mode"])
    parser.add_argument("--provider-response-fixture", type=Path, help="用本地 HTTP response JSON fixture 验证 provider 协议错误")
    parser.add_argument("--admin-import", action="store_true", help="批量导入知识库文档")
    parser.add_argument("--db", type=Path, default=ROOT / CONFIG["database"])
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--save-run", metavar="NAME",
                        help="把本次运行的请求、ReAct 过程与结果保存到 examples/{requests,traces,results}/NAME.json")
    args = parser.parse_args()
    if args.admin_import:
        if not args.input:
            parser.error("--admin-import requires --input")
        print(json.dumps(import_documents(args.input, args.db, args.rebuild_index), ensure_ascii=False, indent=2))
        return
    payload = json.loads(args.input.read_text(encoding="utf-8")) if args.input else None
    if payload is None and not args.question:
        payload = json.load(sys.stdin)
    payload = payload or {"question": args.question}
    question = payload["question"]
    config = payload.get("config", {})
    chunks = load_chunks(args.db if args.db.exists() else ROOT / "data" / "q3_documents.json")
    fixture = json.loads(args.provider_response_fixture.read_text(encoding="utf-8")) if args.provider_response_fixture else None
    if fixture is not None and args.llm != "openai-compatible":
        parser.error("--provider-response-fixture requires --llm openai-compatible")
    llm = OpenAICompatibleLLM(
        base_url="https://provider.example.invalid/v1",
        model="fixture-model",
        api_key="fixture-key",
        response_fixture=fixture,
    ) if fixture is not None else OpenAICompatibleLLM() if args.llm == "openai-compatible" else MockLLM()
    reranker = ProviderReranker.from_environment() if args.reranker == "provider" else MockJevReranker()
    extractor = OpenAICompatibleQueryAwareExtractor.from_environment(
        max_tokens=int(CONFIG["query_aware"]["max_tokens"])
    ) if args.query_aware == "provider" else MockQueryAwareExtractor()
    result = EnterpriseQA(
        chunks,
        candidate_limit=int(config.get("candidate_limit", CONFIG["retrieval"]["candidate_limit"])),
        max_evidence=int(config.get("max_evidence", CONFIG["retrieval"]["max_evidence"])),
        max_steps=int(config.get("max_steps", args.max_steps)),
        rerank_top=int(config.get("rerank_top", CONFIG["retrieval"]["rerank_top_k"])),
        reranker=reranker,
        llm=llm,
        extractor=extractor,
    ).answer(question, payload.get("conversation_history"))
    result_doc = asdict(result)
    if args.save_run:
        save_run(args.save_run, args, question, payload, result_doc)
        print_react_summary(question, result_doc)
    print(json.dumps(result_doc, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
