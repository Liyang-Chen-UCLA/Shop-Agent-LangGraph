# Eval and Market aggregation protocol

The reduction pipeline separates semantic assessment from canonicalization:

1. Eval compares two `CriteriaAttributeSet` values and returns an `EvalReport`.
2. Market receives the original collections and that report, then runs an isolated
   aggregation graph for the pair.
3. Completed pair results continue through the existing parallel, pairwise reduction.
   A pending pair stops subsequent reduction rounds.

## Eval report

Eval exposes only `submit_eval_report`. It uses the shared parameter-based submission
graph, so the model submits an `EvalReport` directly and runtime validation checks it
against the original inputs.

Each group contains only:

| Field | Meaning |
| --- | --- |
| `status` | `match`, `uncertain`, or `independent` |
| `left_ids` | Source-qualified left field IDs |
| `right_ids` | Source-qualified right field IDs |
| `reason` | Semantic justification for the classification |

`match` means the fields are clearly synonymous. `uncertain` covers containment,
intersection, composite definitions, granularity differences, incompatible qualifiers,
or insufficient evidence. `independent` means a field has no corresponding or
plausibly related field in the other input.

Every input field must occur exactly once in the report. Match and uncertain groups
contain both sides; independent groups contain exactly one side. Connected uncertain
relationships belong in one group. These rules make the report a complete partition,
while leaving all canonical decisions to Market.

Eval never emits an item definition, mapping, merge, or partial decomposition.
`uncertain` is a valid submitted assessment and does not block Eval completion.

## Market aggregation

For each pair, Market creates fresh mutable state and first applies a deterministic
fast path. Independent fields are retained directly. A one-to-one match is merged
directly when both items use the same concrete schema and all structured fields other
than ID, name, description, and aliases agree. The left item is reused and right-side
names and aliases are appended as aliases. If every group takes this path, the Market
aggregation model is not called.

Only uncertain groups, many-to-many matches, and matches that fail structured
compatibility are sent to the aggregation model. It exposes one
`apply_aggregation_plan` tool containing all match patches, independent references,
and partial resolutions. Match decisions select a `base_ref` and supply only changed
fields in `patch`; complete definitions are required only for newly decomposed
outputs. Multiple plan calls in one response are combined and validated as one batch.

An uncertain Eval group can become a match, independent fields, or a partial
decomposition after Market inspects the original definitions. The report does not
force one of these outcomes.

The complete plan is validated before changing state. Validation
covers reference existence and side, duplicate consumption, output kind, source
mapping, full structural coverage, canonical ID uniqueness, and model compatibility.
After a successful plan, the runtime automatically assembles the final collection;
there is no model-authored collection and no separate submit round.

Full definitions, mappings, reasons, and patches are stored in the runtime audit.
Model-visible tool results contain only processed group IDs, remaining group IDs,
pending groups, or concrete validation errors, so completed operations are not
replayed into later model context.

## Pending results and retry limit

When definitions do not contain enough evidence for a safe aggregation decision,
Market uses a `needs_review` partial-resolution group with no outputs and concrete
`missing_evidence`. The aggregation graph returns `status="pending"` immediately.
Each pending group also carries the pair's left and right product-source IDs so its
locally qualified field references remain unambiguous after parallel pairs are joined.
The current reduction round may finish already-running sibling pairs, but no later
round starts and no partial collection is exposed as final.

Invalid calls may be corrected within a bounded number of model attempts. Reaching
the configured limit also returns a pending result containing the unresolved source
IDs. This prevents infinite loops and prevents uncertainty from being silently forced
into independent fields.

## Parallel reduction and state isolation

Market preserves pairwise parallel reduction. Each pair runs Eval first and then its
own Market aggregation graph. Tool closures, unresolved references, accumulated
criteria, and accumulated attributes are created per pair, so concurrent pairs cannot
consume or append to one another's state.

The offline tests cover Eval immutability and exact report coverage, deterministic
model skipping, mixed-plan atomicity, automatic runtime finalization, compact tool
feedback, per-pair state isolation, pairwise orchestration, pending propagation, and
the aggregation retry limit.
