from __future__ import annotations

from typing import Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr

from ...domain.criteria import SchemaModel, Criterion, Attribute
from ..market.schemas import MarketResult
from ..relation.schemas import Identifier, NodeRef, RelationGraph, Text


class Question(SchemaModel):
    id: Identifier
    text: Text
    target_refs: list[Text] = Field(default_factory=list)
    claim_ids: list[Identifier]
    purpose: Text


class QuestionSet(SchemaModel):
    questions: list[Question] = Field(max_length=3)


class ContextFact(SchemaModel):
    ref: NodeRef
    value: StrictStr | StrictInt | StrictFloat | StrictBool
    unit: Text | None
    user_quote: Text


class PersonalizedCriterion(SchemaModel):
    source_ref: Text
    preference: Text
    priority: Literal["high", "medium", "low", "unspecified"]
    strength: Literal["hard", "soft"]
    user_quote: Text
    claim_ids: list[Identifier]


class PersonalizedAttribute(SchemaModel):
    source_ref: Text
    requirement: Text
    strength: Literal["hard", "soft"]
    user_quote: Text


class Constraint(SchemaModel):
    source_ref: Text
    operator: Literal["eq", "lte", "gte", "in"]
    values: list[StrictStr | StrictInt | StrictFloat | StrictBool] = Field(min_length=1)
    unit: Text | None = None
    strength: Literal["hard", "soft"]
    user_quote: Text


class CandidateScope(SchemaModel):
    node_id: str
    user_quote: Text


class PersonalizationContent(SchemaModel):
    local_criteria: list[Criterion] = Field(default_factory=list)
    local_attributes: list[Attribute] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    candidate_scope: CandidateScope | None = None
    blocking_questions: list[Text] = Field(default_factory=list, max_length=3)
    ready_to_search: bool = False
    context: list[ContextFact]
    criteria: list[PersonalizedCriterion]
    attributes: list[PersonalizedAttribute]
    unresolved: list[Text]


class PersonalizedProfile(PersonalizationContent):
    node_id: str
    profile_hash: str
    graph_revision: int = Field(ge=1)


class PersonalizationSession(SchemaModel):
    schema_version: Literal[1] = 1
    thread_id: Text
    task_id: Text
    node_id: str
    profile_hash: str
    graph_revision: int = Field(ge=1)
    revision: int = Field(ge=1)
    user_messages: list[Text] = Field(min_length=1)
    question_set: QuestionSet
    result: PersonalizedProfile | None = None


class OutputValidationError(ValueError):
    """A model-authored output can be regenerated to satisfy the contract."""


class ClaimReferenceError(OutputValidationError):
    """A model output cites missing or non-supported graph claims."""


def validate_output(output: QuestionSet | PersonalizationContent, profile: MarketResult,
                    graph: RelationGraph, user_messages: list[str]) -> None:
    dimensions = {f"{kind}:{item.id}": item for kind, items in
                  (("criterion", profile.criteria), ("attribute", profile.attributes)) for item in items}
    contexts = {f"context:{item.id}": item for item in graph.context_variables}
    nodes = set(dimensions) | set(contexts) | {f"derived:{v.id}" for v in graph.derived_variables}
    claim_statuses = {c.id: c.status for c in graph.relations + graph.utility_rules}

    def claims(ids, location):
        invalid = {id: claim_statuses.get(id, "unknown ID") for id in ids
                   if claim_statuses.get(id) != "supported"}
        if invalid:
            raise ClaimReferenceError(
                f"{location}: claim IDs must reference supported graph claims; "
                f"invalid IDs and statuses: {invalid}")

    def unique(ids, location):
        seen = set()
        duplicates = []
        for value in ids:
            if value in seen and value not in duplicates:
                duplicates.append(value)
            seen.add(value)
        if duplicates:
            raise OutputValidationError(
                f"duplicate {location}: {', '.join(duplicates)}"
            )

    if isinstance(output, QuestionSet):
        unique([q.id for q in output.questions], "question IDs")
        for question in output.questions:
            if set(question.target_refs) - nodes:
                raise OutputValidationError("unknown question target reference")
            claims(question.claim_ids, f"question {question.id}.claim_ids")
        return
    unique([f.ref for f in output.context], "context references")
    local_items = output.local_criteria + output.local_attributes
    unique([item.id for item in local_items], "local dimension IDs")
    dimensions.update({f"local:{item.id}": item for item in local_items})
    if output.ready_to_search and output.blocking_questions:
        raise OutputValidationError("ready_to_search cannot have blocking questions")
    if output.candidate_scope and not any(output.candidate_scope.user_quote in m for m in user_messages):
        raise OutputValidationError("candidate scope requires a user quote")
    for constraint in output.constraints:
        dimension = dimensions.get(constraint.source_ref)
        if dimension is None:
            raise OutputValidationError("unknown constraint dimension")
        if constraint.operator != "in" and len(constraint.values) != 1:
            raise OutputValidationError("scalar constraint requires exactly one value")
        if dimension.type == "numeric":
            if constraint.unit not in dimension.units or any(type(v) not in (int, float) for v in constraint.values):
                raise OutputValidationError("numeric constraint requires matching units and numeric values")
        else:
            if constraint.operator in {"lte", "gte"} or constraint.unit is not None:
                raise OutputValidationError("non-numeric constraint requires equality/set and no unit")
            expected = bool if dimension.type == "boolean" else str
            if any(type(v) is not expected for v in constraint.values):
                raise OutputValidationError("constraint value type mismatch")
    unique(
        [p.source_ref for p in output.criteria + output.attributes],
        "criteria/attributes source_ref values",
    )
    for item in output.context + output.criteria + output.attributes + output.constraints:
        if not any(item.user_quote in message for message in user_messages):
            raise OutputValidationError("user_quote must occur verbatim in a user message")
    for fact in output.context:
        if fact.ref not in contexts:
            raise OutputValidationError(
                "context facts must reference graph context variables"
            )
        variable = contexts[fact.ref]
        valid_type = {"numeric": type(fact.value) in (int, float),
                      "boolean": type(fact.value) is bool,
                      "categorical": type(fact.value) is str}[variable.value_type]
        if not valid_type or fact.unit != variable.unit:
            raise OutputValidationError("context type or unit does not match graph")
    for item in output.criteria + output.attributes:
        if item.source_ref not in dimensions:
            raise OutputValidationError("unknown profile dimension")
        if isinstance(item, PersonalizedCriterion):
            claims(item.claim_ids, f"criterion {item.source_ref}.claim_ids")
