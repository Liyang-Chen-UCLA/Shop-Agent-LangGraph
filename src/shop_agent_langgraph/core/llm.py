from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def build_deepseek_model() -> ChatOpenAI:
    """Build the shared DeepSeek chat model from environment variables."""
    from .tracing import load_environment
    load_environment()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        # ToolStrategy forces tool_choice="required", which DeepSeek rejects
        # in thinking mode. Disable it explicitly instead of relying on defaults.
        extra_body={"thinking": {"type": "disabled"}},
    )
