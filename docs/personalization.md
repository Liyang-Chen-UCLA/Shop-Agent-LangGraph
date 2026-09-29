# Personalization MVP

Personalization reads a completed, persisted Market Profile and matching RelationGraph.
It asks up to three product-specific questions, then converts the user's feedback into
a preference overlay. Shared definitions and graph files are never modified.

## Direct use

```python
from shop_agent_langgraph import personalization_agent

history = ["想买一个游戏手柄"]
questions = personalization_agent.prepare_questions(
    "301", history, thread_id="user-1", task_id="controller-1",
)
# Display questions.questions and wait for actual user feedback, unless empty.
history.append("每天玩四小时，续航优先")
result = personalization_agent.personalize(
    "301", history, thread_id="user-1", task_id="controller-1",
)
# Later corrections regenerate the entire overlay from chronological user history.
history.append("续航不用优先考虑了")
result = personalization_agent.personalize(
    "301", history, thread_id="user-1", task_id="controller-1",
)
```

Use the actual resolved taxonomy node for the request; `301` is illustrative.
`aprepare_questions` and `apersonalize` provide async equivalents. Model and store
dependencies can be injected into `PersonalizationAgent` for tests or custom callers.
Direct callers supply authentic user-only messages for one task, including skipped
answers. Empty questions allow immediate finalization; nonempty questions require
another user message. An initial preparation must precede finalization.

## Schemas and validation

`QuestionSet.questions` contains zero to three records with `id`, `text`,
`target_refs`, `claim_ids`, and `purpose`. References must exist; cited claims must
be supported. Missing scenario inputs can motivate conditional questions.

`PersonalizedProfile` contains runtime-assigned `node_id`, `profile_hash`, and
`graph_revision`, plus:

- `context`: graph context ref, typed value, unit, verbatim user quote.
- `criteria`: source dimension ref, natural-language preference, priority
  (high/medium/low/unspecified), strength (hard/soft), user quote, claim IDs.
- `attributes`: source dimension ref, natural-language requirement, strength, user quote.
- `unresolved`: missing knowledge, unmapped requests, skipped or ambiguous inputs.

Source refs retain market IDs even when a descriptive attribute becomes a personal
criterion. Duplicate refs across criteria and attributes are rejected. Quotes must
occur in actual user messages, context types and units must match graph definitions,
and only supported claim IDs may be cited. These checks verify structure and provenance,
not semantic entailment: prompts additionally require scope/condition review, explicit
user support for hard requirements, and no invented thresholds or weights.

The model receives an explicit `allowed_claim_ids` list. Invalid claim citations
trigger up to two correction attempts containing the rejected output and specific
ID/status errors. Each corrected output is fully revalidated. Invalid references
are never silently removed or promoted to supported claims; exhausting corrections
raises a detailed error and leaves the stored session unchanged.

Preferences and requirements are descriptive text, not executable filtering/scoring
rules. Context unavailable in the graph stays unresolved. Text derivations are not
executed. No search tools are exposed; knowledge gaps remain explicit.

## Supervisor and persistence

After Relation Agent completes, Supervisor calls `prepare_personalization`. The
runtime obtains the thread ID and latest user message directly; its stable message
ID becomes the task ID. Supervisor displays the questions and calls
`complete_personalization` on feedback (or immediately for an empty set).
The completion tool obtains the chronological user history beginning at that ID,
without accepting model-generated quotes or user identity arguments.

This MVP supports one active shopping task per conversation. A new purchase starts
a new task. Updating older tasks after switching is outside the MVP; use separate
threads or the direct API with correctly scoped history. It has no multi-scenario
scoring, incremental field editing, or automatic research loop.

Sessions are saved under `.cache/personalization/<thread-hash>/<task-hash>.json`;
`PERSONALIZATION_CACHE_DIR` overrides the root. Each session includes questions,
user history, the latest result, input versions, and a revision. Paths hash identifiers;
writes use atomic replacement and compare the session revision under an exclusive
lock. Validation/model failures do not replace the previous session. There is no
revision archive or automatic expiry. Session files contain user messages.

Changed upstream versions block completion: rerun preparation with the full task
history through the direct API. Upstream inputs are checked again before saving,
but the stores are not a cross-file transaction. As with RelationStore, a leftover
lock after a process crash requires inspection and manual recovery.
