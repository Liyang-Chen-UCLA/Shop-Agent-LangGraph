"""Send a real, small application request; print only trace IDs and the UI URL.

Run with: uv run python scripts/verify_langfuse.py
Then inspect that trace via the Langfuse observations CLI/API.
"""
from uuid import uuid4

from shop_agent_langgraph.core.tracing import get_tracing_client
from shop_agent_langgraph.supervisor.graph import supervisor


def main():
    client = get_tracing_client()
    if client is None:
        raise SystemExit("Langfuse credentials/base URL missing or tracing disabled; check .env.")
    if not client.auth_check():
        raise SystemExit("Langfuse authentication failed; check keys and service URL.")
    session = f"instrumentation-check-{uuid4().hex}"
    with client.start_as_current_observation(name="instrumentation-check",
        input="请简要介绍你能帮助用户做什么。") as span:
        answer = supervisor.invoke("请简要介绍你能帮助用户做什么。", thread_id=session)
        span.update(output=answer)
        trace_id = span.trace_id
    client.flush()
    print(f"Trace ID: {trace_id}")
    print(f"Session ID: {session}")
    print(f"Trace URL: {client.get_trace_url(trace_id=trace_id)}")


if __name__ == "__main__":
    main()
