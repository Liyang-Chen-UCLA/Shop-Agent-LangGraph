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

The intent agent is an independent entry point for parsing user requests:

```python
from shop_agent_langgraph import intent_agent

result = intent_agent.invoke("预算改成 800，最好轻一点")
```

It returns an `IntentResult` with `action`, `category`,
`criteria_preferences`, and `attribute_preferences`.

The user-facing supervisor runs intent analysis first on every turn and keeps
conversation state isolated by `thread_id`. For supported resolved taxonomy
routes it also runs Market Agent before replying:

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

Market Agent searches local item IDs and chooses a sample, Research Agent
extracts each selected product concurrently, and Eval Agent merges adjacent
results 2-to-2 in parallel at every reduction layer. Global runtime limits are
defined in `core/config.py`; the current market search and selection maximum is
4 products.
