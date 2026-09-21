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
`DEEPSEEK_BASE_URL` may optionally override `deepseek-chat` and
`https://api.deepseek.com`.
