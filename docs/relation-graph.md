# Relation Graph

Relation Agent reads a completed Market Profile from disk for a canonical Route
node ID. It uses Tavily search (as Research does) to collect source-backed
relationships, edits an isolated draft with tools, validates it, and publishes a
versioned graph. It does not modify Market Profile or evaluate product scores.

## Entry points

```python
from shop_agent_langgraph import relation_agent

graph = relation_agent.invoke("301")
updated = relation_agent.invoke("301", refresh=True)
# Async is also supported:
# graph = await relation_agent.ainvoke("301")
```

A Market Profile must already exist, normally written by Supervisor's
`call_market_agent`. Supervisor calls `call_relation_agent` after completed market
results, using the same resolved node. Ambiguous routes and pending market results
must not proceed to relation research. This sequencing is instructed by the
Supervisor prompt; Relation Agent independently enforces the persisted-profile
precondition. Custom callers can invoke Relation Agent without Supervisor.

## Schema

`RelationGraph` contains:

- `schema_version`, `node_id`, `profile_hash`, `revision`, `created_at`, `updated_at`.
- `context_variables`: reusable scenario inputs, without a user's actual values.
- `derived_variables`: named inputs, textual derivation, and evidence references.
- `relations`: conditional influences, associations, and requirements.
- `utility_rules`: qualitative, scenario-dependent utility descriptions.
- `evidence`: URL, title, exact excerpt, retrieval time, source kind, and scope.
- `profile_suggestions`: proposed missing dimensions, without changing the profile.
- `unresolved`: open questions, or resolved questions with an explanation.

References use `criterion:<id>`, `attribute:<id>`, `context:<id>` and
`derived:<id>`. Market nodes reuse profile IDs. Context and derived variables have
`value_type` (numeric, boolean, categorical); only numeric variables have a unit.

A relation contains `id`, `source`, `target`, `kind`, `effect`, `conditions`,
`statement`, `evidence_ids`, `counter_evidence_ids`, `status`, and optional
`supersedes`. `influence` is directional, `association` does not establish
causation, and `requires` means source requires target. Numeric effect is
positive, negative, non_monotonic, or unknown, and is null for requires.
Positive means an increase in the target, not better utility. Numeric signs need
numeric endpoints. Non-numeric relationships may use unknown with an explanatory
statement and conditions.

A utility rule targets a criterion, with `context_refs`, `shape`, optional
`direction` and `threshold_ref`, and the same claim/evidence/status fields.
Shapes are monotonic, diminishing_returns, saturation, target_range, and
context_dependent. Monotonic needs a direction; numerical shapes require numeric
criteria. A threshold must reference a numeric node with the same declared units.
Missing thresholds remain null, accompanied by an open question when relevant.
Ranges and derivations are qualitative descriptions, never executable formulas.

Conditions are ANDed, with `ref`, `operator` (eq/ne/gt/gte/lt/lte), exactly one of
`value` or `value_ref`, and optional `unit`. Ordered comparisons require numeric
nodes. Numeric literals must use a declared unit, if the node has units.
Reference comparisons require the same type and units; implicit conversions are
not performed. A nonnumeric condition does not have a unit.

## Evidence and review

Tavily requests raw content when available. `add_evidence` only accepts a URL
retrieved in the current run and an exact excerpt occurring in its returned
content. Titles, retrieval timestamps and source kind are assigned by the runtime,
not the model. Evidence records are immutable by ID; use a new ID for a new source
or excerpt. A search excerpt is explicitly distinguished from raw source content.
This verifies retrieval provenance, not the truth of the source or entailment of
the claim. The agent must assess applicability and source quality.

Claims are candidate, supported, disputed, or retired. Supported claims require
supporting evidence and no counter evidence; disputed claims require both.
Unknown references, self edges, duplicate IDs, invalid units and types, cyclic
derivations, and supported duplicate/conflicting edges under identical structured
conditions are rejected. Semantic contradictions in differently worded conditions
still require model review. Edges are not required for every profile dimension.

## Editing and finalization

Tools: search_relation_evidence, add_evidence, upsert_context_variable,
upsert_derived_variable, upsert_relation, upsert_utility_rule, retire_relation,
suggest_profile_change, upsert_open_question, inspect_relation_graph,
finalize_relation_graph.

Dependencies must be created first. Each edit validates a copied draft before
committing it, so an invalid edit leaves the previous draft unchanged. Retirement
retains the record. Substantially different replacement claims use a new ID and
supersedes referencing a retired claim. Questions can be marked resolved with a
resolution. Per-run locks protect concurrent tool calls, and drafts are isolated
between invocations. Dependent edits must be sequential.

Finalization requires at least one search and either findings or an unresolved
question, validates references and units, and freezes the draft. On model-loop
completion the runtime checks the current Market Profile again, then saves it.
Stopping without successful finalization does not publish. Research has a budget
of 12 searches and a graph recursion limit of 100. Search/network failures are
surfaced; previously published graphs are retained.

## Persistence and reuse

Default layout (override the root with `RELATION_CACHE_DIR`):

```text
.cache/relation_nodes/<node_id>/
  current.json
  revisions/1.json
  revisions/2.json
```

The hash covers canonical JSON for criteria and attributes sorted by ID; it
excludes query, source product IDs, and save time. Any definition changes,
including unit or description changes, invalidate reuse. A matching graph is
returned without initializing a model or search client. `refresh=True` edits a
copy of the existing matching graph. A changed profile starts a fresh draft and
provides the old graph only for explicit review; no old claims are automatically
carried over. Revision numbering and original creation time are preserved.

Each publish compares the base revision under an exclusive filesystem lock,
creates an immutable revision, and atomically replaces current.json. A competing
writer raises RevisionConflict instead of overwriting changes. A changed Profile
during research also prevents publication. The Profile and graph are separate
files, not a cross-file transaction: every subsequent graph read checks the hash.

There is no automatic expiry. Historical JSON files support inspection and manual
rollback; automatic rollback is not exposed yet. A corrupt current graph is an
explicit error rather than a silent overwrite of history. After a hard process
crash, inspect a leftover .write.lock or orphan revision before manually removing
it and retrying; locks are never broken automatically.

## Current boundary

This version delivers structured knowledge for explanation, not a scoring or
causal simulation engine. It does not execute textual formulas, calculate
thresholds or utility scores, or infer that A causes C from an A-B-C path.
`query.match_conditions` returns matched/unmatched/unknown without guessing missing
values or converting units. `query.supported_paths` returns bounded lists of relation
IDs using only supported edges with matched structured conditions; it never reverses
edges or combines effect signs. Paths are references for inspection, not causal
conclusions. Consumers must additionally check each edge's textual scope and kind. Candidate,
disputed and retired claims remain inspectable but are not established facts.
