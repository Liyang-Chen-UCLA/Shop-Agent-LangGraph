import importlib
import json

import pytest

from shop_agent_langgraph.agents.market.schemas import MarketResult
from shop_agent_langgraph.domain.market_cache import MarketCache


tools = importlib.import_module("shop_agent_langgraph.supervisor.tools")


def completed() -> MarketResult:
    return MarketResult(
        query="游戏手柄", item_ids=["product_1"], status="completed",
        criteria=[{
            "id": "battery_life", "name": "续航", "description": "无线模式续航",
            "aliases": [], "type": "numeric", "units": ["h"],
            "direction": {"type": "larger_better"},
        }],
        attributes=[{
            "id": "connection", "name": "连接方式", "description": "支持的连接方式",
            "aliases": [], "type": "categorical", "values": ["USB", "2.4G"],
            "value_domain": "open",
        }],
    )


def test_cache_miss_saves_then_new_instance_reuses_without_market(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_CACHE_DIR", str(tmp_path))
    calls = []

    def research(query, **kwargs):
        calls.append(query)
        return completed()

    monkeypatch.setattr(tools.market_agent, "invoke", research)
    first = tools.call_market_agent.invoke({"node_id": "301"})
    payload = json.loads((tmp_path / "301.json").read_text(encoding="utf-8"))
    assert payload["node"] == MarketCache.node("301").model_dump()
    assert payload["result"] == completed().model_dump(mode="json")
    assert payload["saved_at"]
    # Each tool invocation constructs a new store; disk survives agent/server recreation.
    second = tools.call_market_agent.invoke({"node_id": "301"})
    assert json.loads(first) == json.loads(second) == completed().model_dump(mode="json")
    assert calls == ["游戏手柄"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_cache_hit_does_not_require_dataset_mapping(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_CACHE_DIR", str(tmp_path))
    cache = MarketCache()
    cache.save(cache.node("301"), completed())

    def unexpected(*args):
        pytest.fail("cache hit must skip dataset lookup and market execution")

    monkeypatch.setattr(tools, "dataset_category_for_node", unexpected)
    monkeypatch.setattr(tools.market_agent, "invoke", unexpected)
    assert json.loads(tools.call_market_agent.invoke({"node_id": "301"}))["criteria"]


def test_nodes_sharing_dataset_still_have_separate_cache(tmp_path):
    cache = MarketCache(tmp_path)
    cache.save(cache.node("5598"), completed())
    assert cache.load(cache.node("5322")) is None


@pytest.mark.parametrize("contents", ["{broken", '{"schema_version": 999}'])
def test_invalid_cache_is_rebuilt(tmp_path, monkeypatch, contents):
    monkeypatch.setenv("MARKET_CACHE_DIR", str(tmp_path))
    (tmp_path / "301.json").write_text(contents, encoding="utf-8")
    monkeypatch.setattr(tools.market_agent, "invoke", lambda query, **kwargs: completed())
    tools.call_market_agent.invoke({"node_id": "301"})
    cache = MarketCache()
    assert cache.load(cache.node("301")) == completed()


def test_mismatched_node_is_not_reused(tmp_path):
    cache = MarketCache(tmp_path)
    cache.save(cache.node("301"), completed())
    (tmp_path / "5322.json").write_bytes((tmp_path / "301.json").read_bytes())
    assert cache.load(cache.node("5322")) is None


def test_pending_result_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_CACHE_DIR", str(tmp_path))
    result = MarketResult(
        query="游戏手柄", item_ids=["p1"], status="pending", criteria=[], attributes=[],
        pending_groups=[{
            "group_id": "g1", "left_source_item_ids": ["p1"],
            "right_source_item_ids": ["p2"], "left_ids": ["a"], "right_ids": ["b"],
            "reason": "missing units", "missing_evidence": ["units"],
        }],
    )
    monkeypatch.setattr(tools.market_agent, "invoke", lambda query, **kwargs: result)
    assert json.loads(tools.call_market_agent.invoke({"node_id": "301"}))["status"] == "pending"
    assert not (tmp_path / "301.json").exists()
    cache = MarketCache()
    with pytest.raises(ValueError, match="only completed"):
        cache.save(cache.node("301"), result)


def test_unknown_node_and_market_failure_leave_no_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_CACHE_DIR", str(tmp_path))

    def fail(query, **kwargs):
        raise RuntimeError("market unavailable")

    monkeypatch.setattr(tools.market_agent, "invoke", fail)
    with pytest.raises(ValueError, match="unknown taxonomy"):
        tools.call_market_agent.invoke({"node_id": "../../outside"})
    with pytest.raises(RuntimeError, match="market unavailable"):
        tools.call_market_agent.invoke({"node_id": "301"})
    assert list(tmp_path.iterdir()) == []
