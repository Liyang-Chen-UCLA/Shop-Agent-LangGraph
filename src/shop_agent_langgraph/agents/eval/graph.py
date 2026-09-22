from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_core.language_models import BaseChatModel

from ...core.llm import build_deepseek_model
from ...core.submit_agent import build_submit_agent_graph
from ...domain.criteria import CriteriaAttributeSet
from .schemas import EvalReport
from .tools import source_references, submit_eval_report, validate_eval_report


PROMPT_PATH = Path(__file__).with_name("prompt.md")


class EvalAgent:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def _build_run(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> tuple[Any, dict[str, Any]]:
        items, _left_refs, _right_refs = source_references(left, right)

        def validate_submission(report: EvalReport) -> None:
            validate_eval_report(report, left, right)

        graph = build_submit_agent_graph(
            model=self.model,
            tools=[submit_eval_report],
            system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
            result_schema=EvalReport,
            submit_tool_name="submit_eval_report",
            name="eval_agent",
            validate_submission=validate_submission,
        )
        context = {
            "left_source_item_ids": left.source_item_ids,
            "right_source_item_ids": right.source_item_ids,
            "items": {
                reference: item.model_dump(mode="json")
                for reference, item in items.items()
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
    ) -> EvalReport:
        graph, graph_input = self._build_run(left, right)
        state = graph.invoke(graph_input)
        result = state["submitted_result"]
        return result if isinstance(result, EvalReport) else EvalReport.model_validate(result)

    async def ainvoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> EvalReport:
        graph, graph_input = self._build_run(left, right)
        state = await graph.ainvoke(graph_input)
        result = state["submitted_result"]
        return result if isinstance(result, EvalReport) else EvalReport.model_validate(result)


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
    ) -> EvalReport:
        return self._get().invoke(left, right)

    async def ainvoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> EvalReport:
        return await self._get().ainvoke(left, right)


eval_agent = LazyEvalAgent()
