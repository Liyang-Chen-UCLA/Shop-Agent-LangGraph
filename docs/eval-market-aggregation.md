# Eval and Market aggregation

Eval and Market aggregation are independent entry points.

`EvalAgent` remains a read-only semantic judge for comparing two
`CriteriaAttributeSet` values. It returns an `EvalReport` and is available for
offline comparisons such as agent output versus gold annotations. Market Agent
does not invoke it during its normal pipeline.

## Market MVP pipeline

Market runs five stages:

1. Select product item IDs.
2. Research all selected products concurrently.
3. Deterministically remove obvious duplicates.
4. Send every remaining collection to one aggregation model call.
5. Return the canonical criteria and attributes.

There is no pairwise evaluation, pair aggregation, partial-resolution protocol,
or tree-reduce loop.

## Deterministic pre-merge

The pre-merge is deliberately conservative. Two fields merge only when all of
the following agree after case and whitespace normalization where applicable:

- criterion versus attribute kind;
- ID, name, and description;
- concrete schema type;
- every non-text semantic field, including units, formula, values, value domain,
  and direction.

Aliases are combined for such duplicates. Similar names, different descriptions,
conflicting schemas, and any other uncertain cases remain separate for the model.
Inputs are deep-copied, so research results are not mutated.

## One-shot aggregation

The aggregation model receives all pre-merged `CriteriaAttributeSet` values in a
single context and returns one final `CriteriaAttributeSet`. It merges clear
synonyms, preserves clearly different items, and keeps conflicting or uncertain
items independent. It does not recurse or request a follow-up resolution round.

The runtime replaces model-authored `source_item_ids` with the ordered, de-duplicated
IDs from the inputs. This keeps provenance deterministic while reusing the existing
criteria and attribute schemas.
