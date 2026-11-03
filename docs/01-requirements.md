# Requirements

## User
Content operations staff coordinating review and publication slots.

## Goal
Find assets requiring action and produce a review-aware, conflict-free schedule draft that a human can confirm.

## Scope
- Read asset metadata, check configured review rules, inspect requested and alternative slots, and save drafts.
- Report review state, risk reasons, suggested time, confirmation requirement, save state, and auditable action trace.
- Cover normal scheduling, missing asset information, policy risk, occupied slot, duplicate submission, and save failure.
- HTTP API and CLI; offline deterministic mock data.

## Out of scope
- Real publication, social platform integrations, real model inference, authentication, concurrent multi-user scheduling, and production policy enforcement.

## Success criteria
- No publication side effects.
- Hard requested times are never silently changed; alternatives require confirmation.
- Missing or risky assets are blocked from scheduling.
- Repeated identical requests produce one draft.
- Save errors are returned as errors.
- Each run is bounded and emits a trace.
- All local tests pass.
