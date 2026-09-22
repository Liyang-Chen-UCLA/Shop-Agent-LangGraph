# Eval partial-resolution protocol

Eval exposes `match`, `resolve_partial`, `independent`, and `submit`.
`get_diff` is no longer a tool. Differences in wording, aliases, convertible units,
or observed value coverage alone belong in `match`, not `resolve_partial`.

## Responsibility

`resolve_partial` repairs containment, overlapping granularity, and composite
properties. It preserves shared dimensions and source-specific dimensions instead
of merging everything into a broader definition. It supports one-to-one,
one-to-many, many-to-one, and many-to-many groups in a single batch.

The request is `{"resolutions": [group, ...]}`. Each group has:

| Field | Meaning |
| --- | --- |
| `group_id` | Unique ID within the batch |
| `left_ids`, `right_ids` | Nonempty lists of unresolved, side-qualified field references |
| `relation` | `left_more_specific`, `right_more_specific`, or `overlap` |
| `reason` | Why the definitions overlap and require repair |
| `status` | `resolved` or `needs_review` |
| `outputs` | Canonical outputs, each with `alignment`, `kind`, `item`, `source_refs`, `preserved_aspects` |
| `missing_evidence` | Empty for resolved groups; nonempty for review groups |

Each output's `item` uses the existing Criterion or Attribute schema. `kind` must
agree with the presence of `direction`. Output `alignment` is final:

- `match`: the same dimension is supported by references from both input sides.
- `independent`: a separate dimension supported by references from one side.
  This means separately retained, not statistically independent. Implication or
  containment may still exist; downstream scoring must avoid double counting.

A source may map to several outputs within a group. It must not occur in two groups
in the same batch. Connected overlapping candidates must therefore form one group.
Every resolved group's source must appear in at least one output. Coverage checks
are structural only: the model is responsible for preserving all source aspects,
not merely citing all source IDs. Neither schema validation nor this tool proves
semantic equivalence or the truth of a merchant claim.

## Many-to-many example

Two left fields (`adjustable_vibration`, `lighting`) and two right fields
(`vibration`, `dynamic_light_bar`) can be repaired into four outputs:

| Output | Alignment | Source refs |
| --- | --- | --- |
| `has_vibration` | match | left adjustable vibration + right vibration |
| `vibration_adjustable` | independent | left adjustable vibration |
| `has_lighting` | match | left lighting + right dynamic light bar |
| `has_dynamic_light_bar` | independent | right dynamic light bar |

This repairs definitions; it does not populate product observations. For example,
false adjustability does not imply absent vibration, and unknown stays unknown.
Preserve protocol, component, state, subtype, and other defining qualifiers.
Do not create a generic parent solely to force a match.

## Batch validation and state changes

A valid call returns `committed: true`, a `results` list containing each group's
status, consumed references, complete outputs/mappings, and missing evidence,
and the remaining `unresolved` references. Tool responses preserve the repair
mapping in the run trace; CriteriaAttributeSet's public result schema is unchanged.

The entire batch is preflighted before state mutation: side and existence checks,
unique source consumption, group IDs, output coverage, alignment, kind, and canonical
ID uniqueness. A semantic-protocol validation error returns `committed: false`
with errors and changes nothing. Malformed schema inputs are rejected by tool
argument validation before the handler runs. Failed `match` and `independent`
batches likewise make no partial writes.

A `needs_review` group must have `outputs: []` and nonempty `missing_evidence`.
Its inputs are retained; other resolved groups in a valid batch can commit.
It is not a completed resolution and blocks final submission. Evidence acquisition
or human-review orchestration is not added by this protocol: the caller must supply
the missing evidence before those items can be correctly resolved. Do not repeatedly
retry an identical review or relabel uncertainty as independence to force completion.

## Finalization

Call `submit` alone with `{}` after all inputs have been consumed. The runtime
executes the submit tool and constructs the result from accepted operations and
the original product-source union. The model cannot provide a replacement collection
or source IDs. Submission combined with other tool calls is rejected before any
of those calls execute.

Eval opts into `execute_submit_tool=True` in the shared graph builder. Existing
agents retain argument-based submissions by default. Input validation is explicit
because LangChain skips validation for zero-field StructuredTool schemas.

Run the offline protocol and graph tests with `uv run pytest -q`. The scripted
model tests exercise the real LangGraph/tool submission path without API calls.
