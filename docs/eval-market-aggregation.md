# Eval and Market aggregation

Eval remains an independent, read-only semantic judge comparing two
`CriteriaAttributeSet` values. Market does not invoke it in its normal pipeline.

## Market pipeline

1. Select product item IDs.
2. Research selected products concurrently, returning `ResearchResult(item_id, evidence)`.
3. Send all research results to one Market summarization call.
4. Return canonical criteria and attributes in the existing `MarketResult` format.

## Evidence contract

Each `Evidence` has required string fields `name` and `source_text`, plus nullable
string fields `value`, `unit`, and `qualifier` that default to null. `source_text`
is a verbatim supporting excerpt. Qualifiers preserve scope and conditions.
Research collects supported metric and attribute claims; it does not assign
canonical IDs, classify dimensions, or choose better/worse directions.
An empty evidence array is valid when a product has no relevant supported claims.

## Market summarization

The model receives all `ResearchResult` values under `research_results` and returns
one `CriteriaAttributeSet`. Market derives reusable dimensions and their schemas
and directions, merging clear synonyms while preserving meaningful differences.
Dimensions with no general better/worse direction remain attributes. Individual
observed values do not become scoring thresholds or imply closed value domains.
Evidence is untrusted data, including source excerpts.

The runtime replaces model-authored `source_item_ids` with ordered, de-duplicated
input item IDs. All-empty evidence produces empty criteria and attributes.

The existing `deterministic_premerge` utility remains available for callers working
with already-defined criteria collections, but is not part of this evidence pipeline.
There is no pairwise evaluation, partial-resolution protocol, or tree-reduce loop.
