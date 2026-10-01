# Eval and Market aggregation

Eval is an independent, read-only, two-stage comparison of gold and actual
`CriteriaAttributeSet` values with the same fixed product item IDs. Market does
not invoke it in its normal pipeline. The old left/right
match/uncertain/independent interface is replaced by gold/actual evaluation.

## Gold comparison

1. Matching sees **only name and description**, addressed by opaque gold/actual
   references. Neither canonical IDs nor kind/type/units/aliases can influence
   this stage. It returns `match`, `missing` (gold only) and `extra` (actual only).
   Every source dimension is covered exactly once. Cross-kind matching is allowed.
2. Schema review sees full schemas of matched groups. It returns one review per
   matched **gold item**, including integer 0/1 checks for `kind`, `type` and all
   applicable `units`, `formula`, `direction`, `values`, `value_domain` fields.
   Fields present on either side are checked; absent on both are not scored.
   `aliases` checks synonym correctness, not identical coverage, and is auxiliary.
3. Runtime adds `granularity=1` only for one-to-one groups. One-to-many, many-to-one
   and many-to-many groups remain matched but score 0 for each affected gold item,
   with an explicit merge/split reason. Their other schema fields can still pass.
4. Reports include separate criterion/attribute coverage, classification error
   counts, granularity error counts and applicable-field mean scores. Individual
   checks are always 0/1. Empty denominators produce null, never a fabricated 100%.
   No weighted total score is added. Missing/extra have no matched-schema score.

Convertible units, mathematically equivalent formulas and equivalent preference
directions can pass. Categorical values compare semantic set coverage, ignoring
order and synonyms. Open domains do not erase missing/extra observed options.
Canonical `id`, evidence/provenance and gold annotation metadata are not scored.
Extra means different from gold; it is not proof of hallucination or invalidity.
For groups with multiple Market items, applicable fields come from each gold
item's own schema to avoid mixing fields from other dimensions. With one Market
item, its extra schema fields are also reviewed. Missing fields and uniform
kind/type contradictions cannot receive 1, even if the judge returns that score.

Validation allows one correction attempt per stage for incomplete/invalid reports.
Both sync and async entry points preserve inputs. Scope mismatches fail before
judging; this evaluator does not resample products or rewrite gold.

```python
from shop_agent_langgraph import eval_agent
from shop_agent_langgraph.agents.eval.io import load_gold, load_actual, save_report

gold, metadata = load_gold("eval/cases/gamepad/gold.json")
actual = load_actual("market-result.json")
report = eval_agent.invoke(gold, actual)
report.gold_metadata = metadata
save_report(report, ".cache/eval/gamepad")
```

Or run `uv run python scripts/eval_market.py --actual market-result.json`.
The loader accepts a completed MarketResult, a Market cache envelope, or a plain
CriteriaAttributeSet export. Its product IDs must equal gold's fixed IDs (order
does not matter). The default nine-product gold cannot be compared to a default
four-product Market sample. Provide a delivery from the fixed nine-product scope.
Gold extras are stripped for schema comparison; its review status/hash remain in
the JSON and Markdown reports. Current gamepad gold is pending human review, so
results are differences relative to that draft, not verified product truth.

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
