# Shop Agent LangGraph

The project incrementally ports Shop Agent features to LangGraph. The first
available component resolves one product name against the bundled Google
product taxonomy.

Set the DeepSeek API key in the environment, then invoke the exported agent:

```powershell
$env:DEEPSEEK_API_KEY = "..."
uv run python -c 'from shop_agent_langgraph import route_agent; print(route_agent.invoke("机械键盘"))'
```

`route_agent.invoke(...)` accepts a product string or `{"product": "..."}` and
returns a `RouteResult` Pydantic model. `DEEPSEEK_MODEL` and
`DEEPSEEK_BASE_URL` may optionally override `deepseek-flash` and
`https://api.deepseek.com`.

The shared model explicitly disables DeepSeek thinking mode. Agents use
LangChain `ToolStrategy` for structured output, which sends
`tool_choice="required"`; DeepSeek rejects that choice in thinking mode.
Restart the running server after changing model configuration so cached agent
instances are rebuilt.

The intent agent is an independent entry point for parsing user requests:

```python
from shop_agent_langgraph import intent_agent

result = intent_agent.invoke("预算改成 800，最好轻一点")
```

It returns an `IntentResult` with `action`, `category`,
`criteria_preferences`, and `attribute_preferences`.

The user-facing `create_agent` supervisor orchestrates Intent, Route, and Market
agents through high-level tools. The outer LangGraph only retains conversation
messages and keeps them isolated by `thread_id`:

```python
from shop_agent_langgraph import supervisor

reply = supervisor.invoke("我想买机械键盘，预算 500", thread_id="user-1")
print(reply)

reply = supervisor.invoke("最好是无线的", thread_id="user-1")
print(reply)
```

Start a continuous terminal conversation with the Supervisor:

```powershell
uv run shop-agent-langgraph
```

Alternatively:

```powershell
uv run python -m shop_agent_langgraph --thread-id local-chat
```

Enter `exit`, `quit`, or `退出` to end the conversation.

For LangGraph Studio, copy the environment template and start the local server:

```powershell
Copy-Item .env.example .env
langgraph dev
```

Fill in `DEEPSEEK_API_KEY` and `LANGSMITH_API_KEY` in `.env` before starting.

## Market analysis MVP

Configure Tavily for Research Agent when external evidence is needed:

```dotenv
TAVILY_API_KEY=tvly-...
```

Run the complete asynchronous pipeline:

```python
import asyncio

from shop_agent_langgraph import market_agent

result = asyncio.run(market_agent.ainvoke("游戏手柄"))
print(result.model_dump())
```

Market Agent searches local item IDs and chooses a sample. Research Agent
collects metric and attribute evidence for each selected product concurrently.
Each `ResearchResult` contains `item_id` and `evidence: list[Evidence]`.
Evidence carries `name`, optional `value`, `unit`, and `qualifier`, and required
verbatim `source_text`. Research does not define criteria or preference directions.
Market summarizes all evidence in one model call into canonical criteria and
attributes. Global runtime limits are defined in `core/config.py`; the current
market search and selection maximum is 4 products.

### Reuse market definitions by taxonomy node

In the Supervisor flow, `call_market_agent(node_id)` first checks the local cache
using the exact node ID returned by Route. A hit returns the saved result without
running Market, Research, or aggregation. A miss runs Market and saves only a
completed result. Pending results and failed runs are not cached.

Files default to `.cache/market_nodes/<node_id>.json` in the project directory
(for example, `.cache/market_nodes/301.json`). Set `MARKET_CACHE_DIR` to override
the directory. Each JSON contains a schema version, canonical node ID/name/path,
save time, and the Market result with `criteria`, `attributes`, and source item IDs.
Writes are atomic; invalid or mismatched cache entries are treated as misses.
Different node IDs remain separate even when they map to the same dataset category.

The cache persists across conversations and server restarts and has no automatic
expiry. Delete a node's JSON file to regenerate its definitions on the next request.
Existing definitions are not regenerated after prompt changes unless their file is
removed. Direct `market_agent.invoke(query)` calls remain uncached because they do
not receive a taxonomy node ID. Local cache files are excluded from Git.

## Eval and Market aggregation

Eval remains an independent, read-only semantic judge for comparing two
`CriteriaAttributeSet` values, but Market Agent no longer invokes it.

Market receives every product's evidence without pre-merging it. It derives
reusable dimensions, merges clear synonyms, and preserves meaningful conditions
and units. Descriptive dimensions without a general better/worse direction remain
attributes. There is no pairwise evaluation or tree reduce.

See [the Eval and Market aggregation protocol](docs/eval-market-aggregation.md).

## Relation Agent

After Market Profile is saved for a resolved taxonomy node, Supervisor invokes
Relation Agent. It searches Tavily for evidence, constructs conditional relations
and scenario-dependent utility rules through validated draft tools, and publishes
a versioned `RelationGraph`. Matching node/Profile hashes are reused without model
or search calls. Profile changes require rebuilding and rechecking the graph.

```python
from shop_agent_langgraph import relation_agent

graph = relation_agent.invoke("301")
# Explicitly research updates while preserving revision history:
# graph = relation_agent.invoke("301", refresh=True)
```

Graphs are saved under `.cache/relation_nodes/<node_id>/current.json` and
`revisions/<revision>.json`; override the root with `RELATION_CACHE_DIR`.
A completed local Market Profile and Tavily/DeepSeek credentials are required on a
cache miss. Supported findings, uncertainty, contrary evidence, and profile change
suggestions remain distinct. Numeric scoring and automatic causal propagation
are outside this version.

See [Relation Graph schemas and delivery protocol](docs/relation-graph.md).

## Personalization Agent

After Relation Agent, Supervisor asks up to three product-specific questions and
uses the user's feedback to save personalized criteria and attributes. Existing
market definitions and relation graphs stay read-only. Later preference changes
regenerate the result from the task's user messages.

The independent `personalization_agent` exposes `prepare_questions` and
`personalize` (and async equivalents), with explicit thread/task IDs. Questions,
feedback and the latest result are saved under `.cache/personalization`, configurable
with `PERSONALIZATION_CACHE_DIR`. Input hashes, references, user quotes and context
types/units are validated before publication. The MVP uses natural-language
preferences, not executable scoring rules, and supports one active task per conversation.

See [Personalization MVP schemas and usage](docs/personalization.md).
