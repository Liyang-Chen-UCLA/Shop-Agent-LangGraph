# Shop Agent LangGraph

## Langfuse tracing

The application uses the official LangChain callback integration with Langfuse
Python SDK 4.15.6. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and
`LANGFUSE_BASE_URL` in `.env` (or the process environment). `LANGFUSE_HOST` is
accepted as a fallback. Shell variables take precedence over `.env`.

CLI and LangGraph Studio conversations automatically emit one `shopping-turn`
trace per user turn. The conversation's `thread_id` is the Langfuse session ID.
Nested agents, tools, model names and token usage are captured by framework
callbacks, including parallel research calls and validation retries. Existing
callbacks and LangSmith tracing are preserved. Direct standalone agent calls
can be traced by passing `langfuse.langchain.CallbackHandler` in their config.

Set `LANGFUSE_TRACING_ENABLED=false` to disable tracing. Missing credentials also
leave tracing disabled. Use `LANGFUSE_ENVIRONMENT` and `LANGFUSE_RELEASE` to
distinguish environments and deployments. Export-stage masking removes credential
fields, configured secrets, email addresses and mainland China mobile numbers;
extend `core/tracing.py` for additional application-specific sensitive data.

Long-running servers export in the background. The CLI flushes on exit and the
client shuts down at process exit. Short-lived scripts should call
`shop_agent_langgraph.core.tracing.flush_traces()` before exiting. No credentials
are stored in application code or logs.

Verify a real upload with `uv run python scripts/verify_langfuse.py`. It runs a
small Supervisor request and prints a trace URL. Inspect the exported observations
for that trace (not just authentication success) to confirm model, tokens,
parent/child structure, session ID and masked inputs/outputs.


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
research sample maximum is 4 products. Search can inspect up to 20 candidates,
so unrelated early results do not consume all sample slots.

Market screening uses the canonical target node, independently of personal
preferences. `search_market_products` returns titles and OCR page directories
with short source previews; `get_market_product_info` returns the same summary
for one candidate. `read_market_product_pages` exposes original OCR and image
paths for up to 4 pages per call, with at most 3 calls per product. Unresolved
identity can escalate once to `read_market_product_full_context`. Products still
uncertain after full reading stay out of the sample.

Selection must cite text from pages actually read and provide a relevance reason
for every individually inspected product. Runtime rejects unsearched, unread,
duplicate or unconfirmed selected IDs. Research receives the prior reading
context and independently verifies relevance before evidence aggregation.
The result audit includes screening decisions, read-page provenance and research
decisions; candidates left uninspected are explicitly marked `not_assessed`.
Legacy custom Parquet files with only `context_text` remain supported through
OCR heading parsing, or a single page when no headings exist.

Market cache schema version 3 invalidates older profiles so the next node lookup
uses this screening policy. Caching remains by node, not by personal preferences.

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

Eval now treats the first input as gold and the second as actual, requiring the
same fixed product IDs. Name/description matching yields `match`, `missing` and
`extra`; matched schemas receive per-field 0/1 scores, including classification
and a deterministic merge/split penalty. Criteria and attributes are both covered.
Run `uv run python scripts/eval_market.py --actual market-result.json` to save
JSON and Markdown reports against the independent-controller gold. The actual
delivery must cover the same nine product IDs; the default four-item sample is
not interchangeable. See the protocol below for field semantics and counting.

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

## Market Evaluation

Fixed-product experiments evaluate Research and Market delivery against a versioned
gold draft, with TypeSafe Jev shadow judgments and native Langfuse annotation tasks.
See [the evaluation lifecycle and commands](docs/market-evaluation-lifecycle.md).
