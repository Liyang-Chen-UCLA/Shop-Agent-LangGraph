from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, HttpUrl, StringConstraints, model_validator

from ...domain.criteria import SchemaModel
from ..market.schemas import MarketResult

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
NodeRef = Annotated[str, StringConstraints(pattern=r"^(criterion|attribute|context|derived):[a-z][a-z0-9_]*$")]
Status = Literal["candidate", "supported", "disputed", "retired"]
Scalar = str | FiniteFloat | bool


class ContextVariable(SchemaModel):
    id: Identifier
    name: Text
    value_type: Literal["numeric", "boolean", "categorical"]
    unit: Text | None = None
    description: Text

    @model_validator(mode="after")
    def unit_type(self):
        if self.unit is not None and self.value_type != "numeric":
            raise ValueError("only numeric variables have units")
        return self


class DerivedVariable(ContextVariable):
    input_refs: list[NodeRef] = Field(min_length=1)
    derivation: Text
    evidence_ids: list[Identifier] = Field(min_length=1)


class Condition(SchemaModel):
    ref: NodeRef
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte"]
    value: Scalar | None = None
    value_ref: NodeRef | None = None
    unit: Text | None = None

    @model_validator(mode="after")
    def one_operand(self):
        if (self.value is None) == (self.value_ref is None):
            raise ValueError("provide exactly one of value or value_ref")
        return self


class RelationEvidence(SchemaModel):
    id: Identifier
    url: HttpUrl
    title: Text
    source_text: Text
    retrieved_at: datetime
    source_kind: Literal["raw_content", "search_excerpt"]
    scope: Text


class Claim(SchemaModel):
    id: Identifier
    conditions: list[Condition] = Field(default_factory=list)
    statement: Text
    evidence_ids: list[Identifier] = Field(default_factory=list)
    counter_evidence_ids: list[Identifier] = Field(default_factory=list)
    status: Status = "candidate"
    supersedes: Identifier | None = None

    @model_validator(mode="after")
    def evidence_status(self):
        if self.status in ("supported", "disputed") and not self.evidence_ids:
            raise ValueError("supported/disputed claims require evidence")
        if self.status == "disputed" and not self.counter_evidence_ids:
            raise ValueError("disputed claims require counter evidence")
        if self.status == "supported" and self.counter_evidence_ids:
            raise ValueError("claims with counter evidence must be disputed")
        if set(self.evidence_ids) & set(self.counter_evidence_ids):
            raise ValueError("support and counter evidence must be distinct")
        return self


class Relation(Claim):
    source: NodeRef
    target: NodeRef
    kind: Literal["influence", "association", "requires"]
    effect: Literal["positive", "negative", "non_monotonic", "unknown"] | None = None

    @model_validator(mode="after")
    def endpoints(self):
        if self.source == self.target:
            raise ValueError("self relations are not allowed")
        if (self.kind == "requires") != (self.effect is None):
            raise ValueError("requires has no numeric effect; other kinds require an effect")
        return self


class UtilityRule(Claim):
    criterion: NodeRef
    context_refs: list[NodeRef] = Field(default_factory=list)
    shape: Literal["monotonic", "diminishing_returns", "saturation", "target_range", "context_dependent"]
    direction: Literal["increasing", "decreasing"] | None = None
    threshold_ref: NodeRef | None = None

    @model_validator(mode="after")
    def criterion_and_direction(self):
        if not self.criterion.startswith("criterion:"):
            raise ValueError("utility rules must target a criterion")
        if any(not ref.startswith("context:") for ref in self.context_refs):
            raise ValueError("context_refs must reference context variables")
        if self.shape == "monotonic" and self.direction is None:
            raise ValueError("monotonic utility requires a direction")
        return self


class ProfileSuggestion(SchemaModel):
    id: Identifier
    kind: Literal["criterion", "attribute"]
    name: Text
    reason: Text
    evidence_ids: list[Identifier] = Field(min_length=1)


class OpenQuestion(SchemaModel):
    id: Identifier
    question: Text
    related_refs: list[NodeRef] = Field(default_factory=list)
    status: Literal["open", "resolved"] = "open"
    resolution: Text | None = None

    @model_validator(mode="after")
    def resolved_explanation(self):
        if self.status == "resolved" and self.resolution is None:
            raise ValueError("resolved questions require a resolution")
        return self


class RelationGraph(SchemaModel):
    schema_version: Literal[1] = 1
    node_id: str
    profile_hash: str
    revision: int = Field(ge=1)
    context_variables: list[ContextVariable] = Field(default_factory=list)
    derived_variables: list[DerivedVariable] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    utility_rules: list[UtilityRule] = Field(default_factory=list)
    evidence: list[RelationEvidence] = Field(default_factory=list)
    profile_suggestions: list[ProfileSuggestion] = Field(default_factory=list)
    unresolved: list[OpenQuestion] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


def profile_hash(profile: MarketResult) -> str:
    payload = {
        kind: sorted([item.model_dump(mode="json") for item in getattr(profile, kind)], key=lambda item: item["id"])
        for kind in ("criteria", "attributes")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def validate_graph(graph: RelationGraph, profile: MarketResult) -> None:
    """Validate references, types, units, evidence, and directly conflicting claims.

    This checks structural consistency, not whether the cited source proves a claim.
    """
    graph = RelationGraph.model_validate(graph.model_dump())
    if profile.status != "completed" or graph.profile_hash != profile_hash(profile):
        raise ValueError("graph requires the matching completed Market Profile")
    nodes = {}
    for kind, items in (("criterion", profile.criteria), ("attribute", profile.attributes),
                        ("context", graph.context_variables), ("derived", graph.derived_variables)):
        for item in items:
            ref = f"{kind}:{item.id}"
            if ref in nodes:
                raise ValueError(f"duplicate node: {ref}")
            value_type = item.type if kind in ("criterion", "attribute") else item.value_type
            units = getattr(item, "units", []) if kind in ("criterion", "attribute") else ([item.unit] if item.unit else [])
            nodes[ref] = (value_type, units)
    for items in (graph.relations + graph.utility_rules, graph.evidence, graph.profile_suggestions, graph.unresolved):
        ids = [item.id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate IDs")
    evidence_ids = {item.id for item in graph.evidence}

    def node(ref):
        if ref not in nodes:
            raise ValueError(f"unknown node: {ref}")
        return nodes[ref]

    def evidence(refs):
        if set(refs) - evidence_ids:
            raise ValueError("unknown evidence reference")

    claims = {item.id: item for item in graph.relations + graph.utility_rules}
    for claim in claims.values():
        evidence(claim.evidence_ids + claim.counter_evidence_ids)
        if claim.supersedes:
            if claim.supersedes not in claims or claims[claim.supersedes].status != "retired":
                raise ValueError("supersedes must reference a retired claim")
        for condition in claim.conditions:
            value_type, units = node(condition.ref)
            if condition.operator not in ("eq", "ne") and value_type != "numeric":
                raise ValueError("ordered comparisons require numeric nodes")
            if condition.value_ref:
                other_type, other_units = node(condition.value_ref)
                if other_type != value_type or set(other_units) != set(units):
                    raise ValueError("condition references require matching types and units")
                if condition.unit is not None:
                    raise ValueError("reference comparisons use the nodes' units")
            else:
                value = condition.value
                if value_type == "numeric" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    raise ValueError("numeric conditions require numeric values")
                if value_type == "boolean" and not isinstance(value, bool):
                    raise ValueError("boolean conditions require boolean values")
                if value_type == "categorical" and not isinstance(value, str):
                    raise ValueError("categorical conditions require strings")
                if units and condition.unit not in units:
                    raise ValueError("numeric condition must specify a declared unit")
                if not units and condition.unit is not None:
                    raise ValueError("unexpected condition unit")
        if isinstance(claim, Relation):
            source_type, _ = node(claim.source)
            target_type, _ = node(claim.target)
            if claim.effect in ("positive", "negative", "non_monotonic") and (source_type != "numeric" or target_type != "numeric"):
                raise ValueError("signed effects require numeric endpoints")
        else:
            value_type, units = node(claim.criterion)
            for ref in claim.context_refs:
                node(ref)
            if claim.shape in ("monotonic", "diminishing_returns", "saturation", "target_range") and value_type != "numeric":
                raise ValueError("numeric utility shapes require numeric criteria")
            if claim.threshold_ref:
                threshold_type, threshold_units = node(claim.threshold_ref)
                if threshold_type != "numeric" or value_type != "numeric" or set(units) != set(threshold_units):
                    raise ValueError("utility threshold requires matching numeric units")
    dependencies = {f"derived:{item.id}": item.input_refs for item in graph.derived_variables}
    def visit(ref, path):
        node(ref)
        if ref in path:
            raise ValueError("cyclic derived variable dependencies")
        for parent in dependencies.get(ref, []):
            visit(parent, path | {ref})
    for item in graph.derived_variables:
        evidence(item.evidence_ids)
        visit(f"derived:{item.id}", set())
    for item in graph.profile_suggestions:
        evidence(item.evidence_ids)
    for item in graph.unresolved:
        for ref in item.related_refs:
            node(ref)
    seen = {}
    for claim in claims.values():
        if claim.status == "retired":
            continue
        conditions = tuple(sorted(c.model_dump_json() for c in claim.conditions))
        key = ((claim.source, claim.target, claim.kind) if isinstance(claim, Relation)
               else (claim.criterion, tuple(sorted(claim.context_refs)))) + (conditions,)
        previous = seen.get(key)
        if previous is not None:
            if claim.status == "supported" or previous.status == "supported":
                raise ValueError("duplicate or conflicting active claims: merge or mark disputed")
        seen[key] = claim
