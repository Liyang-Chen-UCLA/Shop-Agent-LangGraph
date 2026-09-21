from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from ..agents.intent.schemas import IntentResult
from ..agents.market.schemas import MarketResult
from ..agents.route.schemas import RouteResult


class SupervisorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    # 本轮解析结果
    intent: IntentResult | None

    # taxonomy 路由状态
    route: RouteResult | None

    # 当前商品市场分析结果
    market: MarketResult | None
