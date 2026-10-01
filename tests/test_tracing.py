import asyncio
import json

import pytest
from langchain_core.messages import AIMessage
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from shop_agent_langgraph.core import tracing
from shop_agent_langgraph.supervisor.graph import Supervisor, build_supervisor_graph
from test_relation_agent import ScriptedChatModel


def test_redaction_preserves_product_data_and_masks_nested_credentials(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-test-key-123")
    original = json.dumps({"api_key": "secret-test-key-123",
        "content": "Contact alice@example.com or 13812345678", "budget": 500})
    masked = json.loads(tracing.redact(original))
    assert masked["api_key"] == "[REDACTED]"
    assert "alice@example.com" not in masked["content"]
    assert "13812345678" not in masked["content"]
    assert masked["budget"] == 500


def test_unconfigured_tracing_does_not_initialize_client(monkeypatch):
    monkeypatch.setattr(tracing, "load_environment", lambda: None)
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tracing, "_client", lambda *args: pytest.fail("must stay disabled"))
    assert tracing.get_tracing_client() is None


@pytest.mark.parametrize("async_run", [False, True])
def test_supervisor_exports_nested_session_trace_without_network(monkeypatch, async_run):
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key=f"pk-test-{async_run}", secret_key="sk-test",
        base_url="https://example.invalid", tracer_provider=TracerProvider(),
        span_exporter=exporter, mask_otel_spans=tracing.mask_spans)
    monkeypatch.setattr(tracing, "get_tracing_client", lambda: client)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", f"pk-test-{async_run}")
    model = ScriptedChatModel(responses=[AIMessage(content="预算已记录",
        usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16})])
    supervisor = Supervisor(build_supervisor_graph(model))
    try:
        if async_run:
            result = asyncio.run(supervisor.ainvoke("预算500内", thread_id="trace-test",
                config={"metadata": {"ls_model_name": "test-model"}}))
        else:
            result = supervisor.invoke("预算500内", thread_id="trace-test",
                config={"metadata": {"ls_model_name": "test-model"}})
        client.flush()
        spans = exporter.get_finished_spans()
        assert result == "预算已记录"
        root = next(s for s in spans if s.name == "shopping-turn")
        assert root.attributes["langfuse.observation.input"] == "预算500内"
        assert root.attributes["langfuse.observation.output"] == "预算已记录"
        assert any(s.attributes.get("session.id") == "trace-test" for s in spans)
        assert all(s.context.trace_id == root.context.trace_id for s in spans)
        assert any(s.attributes.get("langfuse.observation.type") == "generation" for s in spans)
        assert any(s.attributes.get("langfuse.observation.type") == "agent" for s in spans)
        generation = next(s for s in spans if s.attributes.get("langfuse.observation.type") == "generation")
        assert generation.attributes["langfuse.observation.model.name"] == "test-model"
        usage = json.loads(generation.attributes["langfuse.observation.usage_details"])
        assert usage["input"] == 12 and usage["output"] == 4
    finally:
        client.shutdown()
