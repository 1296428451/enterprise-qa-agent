import argparse
import json
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
        reranker=reranker,
        llm=llm,
        extractor=extractor,
    ).answer(question, payload.get("conversation_history"))
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
