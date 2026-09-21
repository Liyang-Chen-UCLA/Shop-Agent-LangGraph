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
conversation state isolated by `thread_id`:

```python
from shop_agent_langgraph import supervisor

reply = supervisor.invoke("我想买机械键盘，预算 500", thread_id="user-1")
print(reply)

reply = supervisor.invoke("最好是无线的", thread_id="user-1")
print(reply)
```
