# Solution

## Flow
Request -> read asset metadata -> check review rules -> query requested/alternative slot -> block, request human confirmation, or save a draft -> return result and trace.

## Decision policy
The current offline planner uses explicit deterministic decisions so the business boundaries are testable. A requested slot conflict never mutates the request: a free alternative is returned with `pending_confirmation=true`. Risky or missing assets are blocked. The save tool is idempotent by request key. Save failure is surfaced as `failed`.

## Components
- `src/agent.py`: tool contracts, bounded agent loop, trace, and local JSON data access.
- `src/api.py`: standard-library HTTP API on localhost.
- `src/cli.py`: command-line runner.
- `data/`: synthetic assets, policy placeholder, schedule, and local draft state.

## Trade-offs and limitations
No external LLM or framework is required, so the project runs offline and tests reliably. This is a workflow/mock demonstration, not evidence of model-driven action selection. Data persistence is a JSON file and is not safe for concurrent writers or production. Policy data is illustrative and needs owner review.

## Repository constraints
- `data/drafts.json` is runtime state and is intentionally ignored; the service creates it when a draft is first persisted.
- `test/` is tracked and is the acceptance gate for changes.
