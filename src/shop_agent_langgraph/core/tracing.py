"""Request-scoped LangChain callbacks backed by a process-wide Langfuse client."""
from __future__ import annotations

import atexit
from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv


def load_environment():
    # Shell/Studio configuration takes precedence over local development defaults.
    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


def redact(value):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if re.search(
            r"api[_-]?key|secret|password|authorization|access[_-]?token", str(k), re.I)
            else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        # OTEL input/output attributes are JSON strings, so redact nested fields too.
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, (dict, list)):
            return json.dumps(redact(parsed), ensure_ascii=False)
        for name, secret in os.environ.items():
            if re.search(r"KEY|SECRET|PASSWORD|TOKEN", name) and len(secret) >= 8:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[REDACTED EMAIL]", value)
        return re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[REDACTED PHONE]", value)
    return value


def mask_spans(*, params):
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
    patches = {}
    for identifier, span in params.spans.items():
        replacements = {k: redact(v) for k, v in span.attributes.items()
                        if isinstance(v, str) and redact(v) != v}
        if replacements:
            patches[identifier] = OtelSpanPatch(set_attributes=replacements)
    return MaskOtelSpansResult(span_patches=patches)


@lru_cache(maxsize=1)
def _client(public_key, secret_key, base_url):
    from langfuse import Langfuse
    client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url,
                      mask_otel_spans=mask_spans)
    atexit.register(client.shutdown)
    return client


def get_tracing_client():
    load_environment()
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in {"false", "0"}:
        return None
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    if not (public and secret and base_url):
        return None
    return _client(public, secret, base_url)


def flush_traces():
    client = get_tracing_client()
    if client is not None:
        client.flush()


@contextmanager
def trace_turn(messages, config):
    client = get_tracing_client()
    if client is None:
        yield config, None
        return
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler
    from langchain_core.callbacks import BaseCallbackManager

    latest = next((m.content for m in reversed(messages) if m.type == "human"), "")
    session_id = str(config.get("configurable", {}).get("thread_id", "default"))
    with client.start_as_current_observation(name="shopping-turn", input=latest) as span:
        with propagate_attributes(trace_name="shopping-turn", session_id=session_id,
                                  tags=["shop-agent", "shopping"]):
            traced = dict(config)
            callbacks = config.get("callbacks")
            handler = CallbackHandler(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"))
            if isinstance(callbacks, BaseCallbackManager):
                callbacks = callbacks.copy()
                callbacks.add_handler(handler, inherit=True)
            else:
                callbacks = list(callbacks or []) + [handler]
            traced["callbacks"] = callbacks
            traced["metadata"] = {**config.get("metadata", {}),
                                  "langfuse_session_id": session_id,
                                  "langfuse_trace_name": "shopping-turn"}
            yield traced, span
