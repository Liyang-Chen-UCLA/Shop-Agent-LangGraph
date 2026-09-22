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

For each pair, Market creates fresh aggregation tools and mutable state. The graph
receives the two original collections plus their validated `EvalReport`. It exposes:

- `match` for canonical merging of synonymous left/right fields;
- `independent` for retaining source definitions unchanged;
- `resolve_partial` for actual containment, overlap, composite, or granularity repair;
- `submit_aggregation({})` for runtime assembly of the final collection.

An uncertain Eval group can become a match, independent fields, or a partial
decomposition after Market inspects the original definitions. The report does not
force one of these outcomes.

All mutation tools validate the complete batch before changing state. Validation
covers reference existence and side, duplicate consumption, output kind, source
mapping, full structural coverage, canonical ID uniqueness, and model compatibility.
Final submission rejects unresolved sources and accepts no model-authored collection.

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

The offline tests cover Eval immutability and exact report coverage, Market tool
atomicity, runtime-only finalization, per-pair state isolation, pairwise orchestration,
pending propagation, and the aggregation retry limit.
