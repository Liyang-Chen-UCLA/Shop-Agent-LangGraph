from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.language_models import BaseChatModel

from ...core.llm import build_deepseek_model
from ...core.submit_agent import build_submit_agent_graph
from ...domain.criteria import Attribute, CriteriaAttributeSet, Criterion
from .tools import build_eval_tools


PROMPT_PATH = Path(__file__).with_name("prompt.md")


def _unordered_items(items: list[Criterion] | list[Attribute]) -> Counter[str]:
    return Counter(
        json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for item in items
    )


def _validate_submission(
    result: CriteriaAttributeSet,
    *,
    left: CriteriaAttributeSet,
    right: CriteriaAttributeSet,
    unresolved: dict[str, Criterion | Attribute],
    criteria: list[Criterion],
    attributes: list[Attribute],
) -> None:
    if unresolved:
        raise ValueError(f"unresolved item IDs remain: {list(unresolved)}")

    expected_source_item_ids = list(
        dict.fromkeys([*left.source_item_ids, *right.source_item_ids])
    )
    if Counter(result.source_item_ids) != Counter(expected_source_item_ids):
        raise ValueError(
            "source_item_ids must equal the deduplicated union of the left and right "
            f"sources: {expected_source_item_ids}"
        )

    if _unordered_items(result.criteria) != _unordered_items(criteria):
        raise ValueError("submitted criteria do not match the accepted operation results")
    if _unordered_items(result.attributes) != _unordered_items(attributes):
        raise ValueError("submitted attributes do not match the accepted operation results")

    item_ids = [item.id for item in [*result.criteria, *result.attributes]]
    duplicate_ids = sorted(
        item_id for item_id, count in Counter(item_ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"canonical item IDs must be unique: {duplicate_ids}")


class EvalAgent:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def _build_run(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> tuple[Any, dict[str, Any]]:
        tools, unresolved, criteria, attributes = build_eval_tools(left, right)

        def validate_submission(result: CriteriaAttributeSet) -> None:
            _validate_submission(
                result,
                left=left,
                right=right,
                unresolved=unresolved,
                criteria=criteria,
                attributes=attributes,
            )

        graph = build_submit_agent_graph(
            model=self.model,
            tools=tools,
            system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
            result_schema=CriteriaAttributeSet,
            submit_tool_name="submit",
            name="eval_agent",
            validate_submission=validate_submission,
        )
        context = {
            "left_source_item_ids": left.source_item_ids,
            "right_source_item_ids": right.source_item_ids,
            "unresolved": {
                item_id: item.model_dump(mode="json") for item_id, item in unresolved.items()
            },
        }
        return graph, {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(context, ensure_ascii=False),
                }
            ]
        }

    def invoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> CriteriaAttributeSet:
        graph, graph_input = self._build_run(left, right)
        state = graph.invoke(graph_input)
        result = state["submitted_result"]
        return (
            result
            if isinstance(result, CriteriaAttributeSet)
            else CriteriaAttributeSet.model_validate(result)
        )

    async def ainvoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> CriteriaAttributeSet:
        graph, graph_input = self._build_run(left, right)
        state = await graph.ainvoke(graph_input)
        result = state["submitted_result"]
        return (
            result
            if isinstance(result, CriteriaAttributeSet)
            else CriteriaAttributeSet.model_validate(result)
        )


def build_eval_agent(model: BaseChatModel | None = None) -> EvalAgent:
    return EvalAgent(model or build_deepseek_model())


class LazyEvalAgent:
    def __init__(self) -> None:
        self._instance: EvalAgent | None = None
        self._lock = Lock()

    def _get(self) -> EvalAgent:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = build_eval_agent()
        return self._instance

    def invoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> CriteriaAttributeSet:
        return self._get().invoke(left, right)

    async def ainvoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> CriteriaAttributeSet:
        return await self._get().ainvoke(left, right)


eval_agent = LazyEvalAgent()
