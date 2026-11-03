# Retrospective

## Hardest issue
The key boundary is distinguishing a suggested alternative from an authorized schedule change. An occupied hard-requested slot must remain unchanged in the request, while the alternative is explicitly marked as awaiting confirmation.

## Learning
Offline tool simulation makes failure and idempotency behavior reproducible. A trace should report observable tool calls and concise action rationale, not hidden chain-of-thought.

## Next steps
Replace JSON persistence with a transactional store, add policy ownership/versioning, and connect a real planner behind an explicit approval gate. Then evaluate different tool results and verify the planner changes its next action.
