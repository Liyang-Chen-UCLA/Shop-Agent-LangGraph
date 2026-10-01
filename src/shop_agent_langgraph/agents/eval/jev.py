"""Pinned Jev field judgments run alongside, never overwrite, the LLM baseline."""
from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

from ...core.tracing import load_environment

RULES = {
    "kind": "The corresponding dimension must have the same criterion/attribute classification.",
    "type": "The corresponding dimension must have the same numeric/boolean/categorical type.",
    "units": "Units must describe the same physical quantity and be equivalent or convertible without changing schema semantics.",
    "formula": "Formulas must have equivalent mathematical meaning and conditions, including unit conversions; null on both sides is correct.",
    "direction": "Preference direction and its conditions must have equivalent meaning; reversed preferences are incorrect.",
    "values": "Categorical values must preserve the same semantic set, ignoring synonyms and order. Missing or additional non-equivalent options are incorrect, even for open domains.",
    "value_domain": "Open/closed domain semantics must agree.",
    "aliases": "Aliases must refer to the same dimension; identical synonym coverage is not required.",
}


def deterministic_score(field, gold, actual):
    if field == "kind":
        kinds = {entry["kind"] for entry in actual}
        return int(gold["kind"] in kinds) if len(kinds) == 1 else None
    if field == "type":
        types = {entry["item"]["type"] for entry in actual}
        return int(gold["item"]["type"] in types) if len(types) == 1 else None
    if field not in {"kind", "aliases"}:
        if (field in gold["item"]) != any(field in entry["item"] for entry in actual):
            return 0
    if field == "formula" and gold["item"].get(field) is None:
        if all(field in entry["item"] and entry["item"][field] is None for entry in actual):
            return 1
    return None


class JevJudge:
    def __init__(self, *, model="jev-1.13.0", min_confidence=None, client_factory=None, trace_client=None):
        if model == "jev-latest":
            raise ValueError("evaluation must pin a Jev model version")
        if min_confidence is not None and not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        self.model = model
        self.min_confidence = min_confidence
        self.client_factory = client_factory
        self.trace_client = trace_client

    def preflight(self):
        load_environment()
        if self.client_factory is None and not os.getenv("TYPESAFE_API_KEY", "").strip():
            raise RuntimeError("Set TYPESAFE_API_KEY in .env before running Jev")

    async def review(self, groups, baseline):
        self.preflight()
        baseline_scores = {r.gold_ref: {s.field: s.score for s in r.fields} for r in baseline.items}
        decisions = []
        factory = self.client_factory or (lambda: AsyncTypeSafeClient(
            model=self.model, timeout=45, retry=RetryPolicy(max_retries=1)))
        rubric = Path(__file__).with_name("jev_rules.md").read_text(encoding="utf-8")
        async with factory() as client:
            for group in groups:
                for ref, gold in group["gold"].items():
                    actual = list(group["actual"].values())
                    questions = {}
                    for field in group["required_fields"][ref]:
                        known = deterministic_score(field, gold, actual)
                        if known is not None:
                            decisions.append(dict(gold_ref=ref, field=field, source="code",
                                candidate_score=known, baseline_score=baseline_scores[ref][field],
                                status="determined"))
                        else:
                            questions[field] = Choice(instructions=f"{rubric}\nJudge ONLY `{field}`. {RULES[field]}",
                                criteria={"correct": "The actual schema preserves this gold dimension's field semantics.",
                                          "incorrect": "The field is absent, incompatible, or changes this dimension's meaning.",
                                          "unclear": "The supplied state is insufficient or ambiguous for this judgment."})
                    if not questions:
                        continue
                    state = {"gold": gold, "actual": group["actual"]}
                    observation = self.trace_client.start_as_current_observation(
                        name="jev-schema-review", as_type="generation", model=self.model,
                        input={"state": state, "questions": {k: q.model_dump(mode="json") for k, q in questions.items()}}) if self.trace_client else nullcontext(None)
                    with observation as span:
                        response = await client.system_one(state=state, questions=questions, model=self.model)
                        if span is not None:
                            span.update(output=response.model_dump(mode="json"),
                                usage_details={"input": response.usage.input_tokens or 0,
                                               "output": response.usage.output_tokens or 0})
                    if response.model != self.model or set(response.choices) != set(questions):
                        raise ValueError("Jev response changed model version or omitted/added questions")
                    for field, answer in response.choices.items():
                        if answer.choice not in {"correct", "incorrect", "unclear"}:
                            raise ValueError("Jev returned an unknown choice")
                        if set(answer.probabilities) != {"correct", "incorrect", "unclear"} or not 0 <= answer.confidence <= 1:
                            raise ValueError("Jev returned invalid confidence or probabilities")
                        if any(not 0 <= p <= 1 for p in answer.probabilities.values()) or abs(sum(answer.probabilities.values()) - 1) > .01:
                            raise ValueError("Jev probability distribution is invalid")
                        candidate = {"correct": 1, "incorrect": 0}.get(answer.choice)
                        requires_review = (candidate is None or self.min_confidence is None
                                           or answer.confidence < self.min_confidence
                                           or candidate != baseline_scores[ref][field])
                        decisions.append(dict(gold_ref=ref, field=field, source="jev", model=response.model,
                            choice=answer.choice, probabilities=answer.probabilities, confidence=answer.confidence,
                            candidate_score=candidate, baseline_score=baseline_scores[ref][field],
                            status="needs_review" if requires_review else "shadow_only",
                            state=state, instructions=questions[field].instructions,
                            usage=response.usage.model_dump(mode="json") if response.usage else None))
        return {"mode": "shadow", "model": self.model, "min_confidence": self.min_confidence,
                "calibrated": False, "decisions": decisions}
