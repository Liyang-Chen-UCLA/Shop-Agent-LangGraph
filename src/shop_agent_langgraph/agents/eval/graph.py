from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel

from ...core.llm import build_deepseek_model
from ...domain.criteria import CriteriaAttributeSet
from .schemas import EvalReport
from .tools import source_references, validate_eval_report


PROMPT_PATH = Path(__file__).with_name("prompt.md")


class EvalAgent:
    def __init__(self, model: BaseChatModel) -> None:
        self.graph = create_agent(
            model=model,
            tools=[],
            system_prompt=PROMPT_PATH.read_text(encoding="utf-8"),
            response_format=ToolStrategy(EvalReport),
            name="eval_agent",
        )

    def _build_run(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> dict[str, Any]:
        items, _left_refs, _right_refs = source_references(left, right)
        context = {
            "left_source_item_ids": left.source_item_ids,
            "right_source_item_ids": right.source_item_ids,
            "items": {
                reference: item.model_dump(mode="json")
                for reference, item in items.items()
            },
        }
        return {
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
        state = self.graph.invoke(self._build_run(left, right))
        raw = state["structured_response"]
        result = raw if isinstance(raw, EvalReport) else EvalReport.model_validate(raw)
        validate_eval_report(result, left, right)
        return result

    async def ainvoke(
        self,
        left: CriteriaAttributeSet,
        right: CriteriaAttributeSet,
    ) -> EvalReport:
        state = await self.graph.ainvoke(self._build_run(left, right))
        raw = state["structured_response"]
        result = raw if isinstance(raw, EvalReport) else EvalReport.model_validate(raw)
        validate_eval_report(result, left, right)
        return result


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
