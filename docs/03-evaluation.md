# Evaluation

## Test suite
Run `python -m unittest discover -s test -v` from the repository root. The local tests cover normal scheduling, missing asset metadata, policy risk, requested-slot conflict with confirmation, idempotent duplicate submission, missing requested time, and save failure.

## Example evidence
- `examples/traces/conflict.json`: captured actual output and summarized trace for occupied time and confirmed alternative.
- `examples/traces/normal.json`: normal free-slot scenario and expected result.
- `examples/traces/risk.json`: risk scenario and expected block result.
- `examples/traces/missing.json`: missing metadata scenario and expected block result.

Inputs are under `examples/requests/`. Run an example with `python "C:/Users/ASUS/Desktop/小影笔试/src/cli.py" "C:/Users/ASUS/Desktop/小影笔试/examples/requests/conflict.json"`. The CLI prints the actual result and tool trace. HTTP health check: `GET /api/health`; planning: `POST /api/plan`.

The automated tests live in the locally retained `test/` directory, but this directory is intentionally ignored by Git per user instruction. Therefore the tests pass locally but are absent from a clone of the submitted repository. This does not satisfy the exam's requirement to submit executable tests; resolve the ignore-policy conflict before final submission.

## Interpretation
The deterministic offline mock verifies workflow and safety boundaries only. It does not demonstrate real LLM planning quality. Save-failure injection is covered at the tool boundary by unit test.
