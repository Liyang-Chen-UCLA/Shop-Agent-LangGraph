import asyncio
import importlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from shop_agent_langgraph.agents.market.graph import MarketAgent, build_market_agent
from shop_agent_langgraph.agents.market.schemas import MarketSelection
from shop_agent_langgraph.agents.market.screening import ACTIVE_SCREENING, screening_session
from shop_agent_langgraph.agents.market.tools import (
    get_market_product_info, read_market_product_full_context,
    read_market_product_pages, search_market_products,
)
from shop_agent_langgraph.agents.research.graph import ResearchAgent
from shop_agent_langgraph.agents.research.schemas import ResearchResult
from shop_agent_langgraph.domain import market_env
from test_market_aggregation import RecordingAggregator
from test_relation_agent import ScriptedChatModel, call


def product(item_id, rank, text, category="游戏手柄"):
    pages = [dict(page_id=f"page-{i:04d}", page_index=i, text=value,
                  source_image_paths=[f"{item_id}/{i:03d}.jpg"], text_sha256=None)
             for i, value in enumerate(text, 1)]
    return dict(item_id=item_id, rank=rank, title=f"{item_id} 游戏手柄",
                category=category, ocr_pages=pages,
                context_text="\n".join(text))


@pytest.fixture
def products(monkeypatch):
    rows = [product(f"clip{i}", i, ["手机触屏辅助按键，不是完整控制器"])
            for i in range(1, 6)]
    rows += [product("controller", 6, ["店铺新品广告", "目标型号为完整游戏控制器"]),
             product("unknown", 7, ["混合广告，身份不确定", "没有目标型号说明"])]
    monkeypatch.setattr(market_env, "load_products", lambda: tuple(rows))
    return rows


def decision(item_id, relevance, page_id, quote):
    return dict(item_id=item_id, relevance=relevance, reason="Product identity from source OCR",
                evidence=[dict(page_id=page_id, source_text=quote)])


def test_search_exposes_later_candidates_and_summary_does_not_return_full_ocr(products):
    results = search_market_products.invoke({"query": "游戏手柄"})
    assert [r['item_id'] for r in results] == [r['item_id'] for r in products]
    summary = get_market_product_info.invoke({"item_id": "controller"})
    assert summary['page_count'] == 2
    assert 'context_text' not in summary
    assert all('text' not in p for p in summary['pages'])
    with pytest.raises(ValueError, match="positive"):
        search_market_products.invoke({"query": "游戏手柄", "limit": -1})


def test_page_reads_preserve_source_and_enforce_budgets(products):
    products[-1] = product("unknown", 7, [f"original page {i}" for i in range(1, 7)])
    with screening_session() as session:
        search_market_products.invoke({"query": "游戏手柄"})
        with pytest.raises(ValueError, match="focused"):
            read_market_product_full_context.invoke({"item_id": "unknown"})
        with pytest.raises(ValueError, match="at most"):
            read_market_product_pages.invoke({"item_id": "unknown", "page_ids": [f"page-{i:04d}" for i in range(1, 6)]})
        with pytest.raises(ValueError, match="unknown OCR"):
            read_market_product_pages.invoke({"item_id": "unknown", "page_ids": ["page-9999"]})
        assert session.page_reads == {}
        for i in range(1, 4):
            data = read_market_product_pages.invoke({"item_id": "unknown", "page_ids": [f"page-{i:04d}"]})
            assert data['pages'][0]['text'] == f"original page {i}"
            assert data['pages'][0]['source_image_paths'] == [f"unknown/{i:03d}.jpg"]
        with pytest.raises(ValueError, match="budget"):
            read_market_product_pages.invoke({"item_id": "unknown", "page_ids": ["page-0004"]})
        data = read_market_product_full_context.invoke({"item_id": "unknown"})
        assert len(data['pages']) == 6
        assert len(session.pages['unknown']) == 6
        with pytest.raises(ValueError, match="once"):
            read_market_product_full_context.invoke({"item_id": "unknown"})
    assert ACTIVE_SCREENING.get() is None


def test_reading_requires_current_search_and_cannot_reread(products):
    with screening_session():
        search_market_products.invoke({"query": "游戏手柄", "limit": 1})
        with pytest.raises(ValueError, match="search results"):
            get_market_product_info.invoke({"item_id": "controller"})
        read_market_product_pages.invoke({"item_id": "clip1", "page_ids": ["page-0001"]})
        with pytest.raises(ValueError, match="already read"):
            read_market_product_pages.invoke({"item_id": "clip1", "page_ids": ["page-0001"]})


def test_real_selection_tool_loop_handoff_rejections_and_uncertain_audit(products, monkeypatch):
    model = ScriptedChatModel(responses=[
        call("search_market_products", {"query": "游戏手柄"}),
        call("read_market_product_pages", {"item_id": "clip1", "page_ids": ["page-0001"]}),
        call("read_market_product_pages", {"item_id": "controller", "page_ids": ["page-0002"]}),
        call("read_market_product_pages", {"item_id": "unknown", "page_ids": ["page-0001"]}),
        call("read_market_product_full_context", {"item_id": "unknown"}),
        call("MarketSelection", dict(item_ids=["controller"], screenings=[
            decision("clip1", "irrelevant", "page-0001", "手机触屏辅助按键"),
            decision("controller", "relevant", "page-0002", "目标型号为完整游戏控制器"),
            decision("unknown", "uncertain", "page-0002", "没有目标型号说明"),
        ])),
    ])
    captured = []

    async def research(item_id, target, *, pre_read):
        captured.append((item_id, target, pre_read))
        return ResearchResult(item_id=item_id, relevance="relevant", evidence=[])

    module = importlib.import_module("shop_agent_langgraph.agents.market.graph")
    monkeypatch.setattr(module.research_agent, "ainvoke", research)
    agent = build_market_agent(model)
    agent.aggregator = RecordingAggregator()
    result = asyncio.run(agent.ainvoke("游戏手柄", "电子产品 > 输入设备 > 游戏手柄"))
    assert result.item_ids == ['controller']
    assert captured[0][1] == "电子产品 > 输入设备 > 游戏手柄"
    assert captured[0][2]['pages'][0]['page_id'] == 'page-0002'
    assert captured[0][2]['screening']['relevance'] == 'relevant'
    assert len(agent.aggregator.calls) == 1
    screened = {a['item_id']: a for a in result.audit if a['stage'] == 'screening'}
    assert screened['clip1']['relevance'] == 'irrelevant'
    assert screened['unknown']['relevance'] == 'uncertain'
    assert screened['unknown']['full_context_read']
    assert screened['clip2']['relevance'] == 'not_assessed'
    assert ACTIVE_SCREENING.get() is None


@pytest.mark.parametrize('failure', ['unread', 'forged_quote', 'duplicate', 'uncertain_without_full', 'unconfirmed', 'omitted_rejection'])
def test_invalid_screening_cannot_enter_research(products, failure):
    class SelectionGraph:
        async def ainvoke(self, value, config):
            search_market_products.invoke({'query': '游戏手柄'})
            item = 'unknown' if failure == 'uncertain_without_full' else 'controller'
            page = 'page-0001' if item == 'unknown' else 'page-0002'
            quote = '混合广告，身份不确定' if item == 'unknown' else '目标型号为完整游戏控制器'
            if failure != 'unread':
                read_market_product_pages.invoke({'item_id': item, 'page_ids': [page]})
            if failure == 'omitted_rejection':
                read_market_product_pages.invoke({'item_id': 'clip1', 'page_ids': ['page-0001']})
            return {'structured_response': MarketSelection(item_ids=[item, item] if failure == 'duplicate' else [item],
                screenings=[] if failure == 'unread' else [decision(item,
                    'uncertain' if failure == 'uncertain_without_full' else 'irrelevant' if failure == 'unconfirmed' else 'relevant',
                    page, 'invented text' if failure == 'forged_quote' else quote)])}

    aggregator = RecordingAggregator()
    with pytest.raises(ValueError):
        asyncio.run(MarketAgent(SelectionGraph(), aggregator).ainvoke('游戏手柄'))
    assert not aggregator.calls
    assert ACTIVE_SCREENING.get() is None


def test_concurrent_selections_keep_read_context_separate(products):
    class SelectionGraph:
        async def ainvoke(self, value, config):
            search_market_products.invoke({'query': '游戏手柄'})
            await asyncio.sleep(0)
            await read_market_product_pages.ainvoke({'item_id': 'controller', 'page_ids': ['page-0002']})
            return {'structured_response': MarketSelection(item_ids=['controller'], screenings=[
                decision('controller', 'relevant', 'page-0002', '完整游戏控制器')])}

    async def run():
        agent = MarketAgent(SelectionGraph(), RecordingAggregator())
        return await asyncio.gather(agent._select('游戏手柄'), agent._select('游戏手柄'))

    left, right = asyncio.run(run())
    assert left._read_context['controller']['page_read_calls'] == 1
    assert right._read_context['controller']['page_read_calls'] == 1
    assert left._read_context is not right._read_context


def test_legacy_parquet_remains_readable_with_page_heading_fallback(tmp_path, monkeypatch):
    text = '# Context\n\n## OCR Page 0001\n\noriginal identity\n\n## OCR Page 0002\n\noriginal specification'
    path = tmp_path / 'legacy.parquet'
    pq.write_table(pa.Table.from_pylist([dict(item_id='old', category='游戏手柄', rank=1,
                                             title='legacy', context_text=text)]), path)
    monkeypatch.setenv('MARKET_DATASET_PATH', str(path))
    # fixture's monkeypatch of load_products is deliberately not used here.
    market_env.load_products.cache_clear()
    try:
        assert market_env.get_product_raw_text('old') == text
        assert market_env.read_product_pages('old', ['page-0002'])['pages'][0]['text'] == 'original specification'
    finally:
        market_env.load_products.cache_clear()


def test_research_input_contains_node_and_pre_read_provenance(products):
    prior = {'screening': {'relevance': 'relevant'}, 'pages': [{'page_id': 'page-0002', 'text': '完整游戏控制器'}]}
    message = ResearchAgent._input('controller', 'canonical node', prior)['messages'][0].content
    assert 'Target category: canonical node' in message
    assert 'market_screening_context' in message and 'page-0002' in message
    assert '店铺新品广告' in message  # full context is still available for independent verification


def test_uncertain_after_complete_read_returns_pending_without_research(products, monkeypatch):
    model = ScriptedChatModel(responses=[
        call('search_market_products', {'query': '游戏手柄'}),
        call('read_market_product_pages', {'item_id': 'unknown', 'page_ids': ['page-0001', 'page-0002']}),
        call('MarketSelection', {'item_ids': [], 'screenings': [
            decision('unknown', 'uncertain', 'page-0002', '没有目标型号说明')]}),
    ])
    module = importlib.import_module('shop_agent_langgraph.agents.market.graph')

    async def unexpected(*args, **kwargs):
        pytest.fail('uncertain product must not enter research')

    monkeypatch.setattr(module.research_agent, 'ainvoke', unexpected)
    agent = build_market_agent(model)
    agent.aggregator = RecordingAggregator()
    result = asyncio.run(agent.ainvoke('游戏手柄'))
    assert result.status == 'pending'
    assert not agent.aggregator.calls
    assert any(a.get('relevance') == 'uncertain' for a in result.audit)


def test_prior_screening_policy_cache_is_invalidated(tmp_path):
    import json
    from shop_agent_langgraph.domain.market_cache import MarketCache
    from test_market_cache import completed
    cache = MarketCache(tmp_path)
    node = cache.node('301')
    cache.save(node, completed())
    path = tmp_path / '301.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload['schema_version'] == 3
    payload['schema_version'] = 2
    path.write_text(json.dumps(payload), encoding='utf-8')
    assert cache.load(node) is None
