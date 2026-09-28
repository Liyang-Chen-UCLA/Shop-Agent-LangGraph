import json

import httpx

from shop_agent_langgraph.agents.intent.graph import build_intent_agent
from shop_agent_langgraph.agents.intent.schemas import IntentResult
from shop_agent_langgraph.core.llm import build_deepseek_model


def test_structured_agent_request_disables_thinking(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.test")
    expected = IntentResult(
        action="create", category="controller",
        criteria_preferences=[], attribute_preferences=[],
    )
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["tool_choice"] == "required"
        assert payload["model"] == "deepseek-flash"
        return httpx.Response(200, json={
            "id": "test-completion", "object": "chat.completion",
            "created": 0, "model": "deepseek-flash",
            "choices": [{
                "index": 0, "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "test-result", "type": "function",
                        "function": {
                            "name": "IntentResult",
                            "arguments": expected.model_dump_json(),
                        },
                    }],
                },
            }],
        })

    # Exercise the real LangChain graph and OpenAI request serialization offline.
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        from openai import OpenAI

        model = build_deepseek_model()
        model.client = OpenAI(
            api_key="test-key", base_url="https://deepseek.test", http_client=client,
        ).chat.completions
        result = build_intent_agent(model).invoke("controller")

    assert result == expected
    assert len(requests) == 1
