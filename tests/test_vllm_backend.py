"""openai_backend against a mock OpenAI-compatible server (the API vLLM
exposes): full answers, streaming, token usage, thinking switched off."""

import json

import httpx

from egyptian_civil_code_rag.backends import openai_backend

SEEN = []


def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    SEEN.append(body)
    assert request.url.path == "/v1/chat/completions"
    if not body.get("stream"):
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "<think>x</think>Article 147."},
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127},
            },
        )

    def event(delta=None, usage=None):
        chunk = {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": body["model"],
            "choices": [] if delta is None else [{"index": 0, "delta": {"content": delta}}],
        }
        if usage:
            chunk["usage"] = usage
        return f"data: {json.dumps(chunk)}\n\n"

    stream = "".join(event(p) for p in ["Art", "icle ", "147."])
    stream += event(usage={"prompt_tokens": 120, "completion_tokens": 3, "total_tokens": 123})
    stream += "data: [DONE]\n\n"
    return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})


def backend():
    SEEN.clear()
    return openai_backend(
        "http://vllm.test/v1",
        "Qwen/Qwen3-8B",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_full_answer_usage_and_no_thinking():
    gen = backend()
    assert gen("prompt") == "Article 147."  # <think> block stripped
    assert gen.last_usage == {"input": 120, "output": 7}
    assert gen.model_name == "Qwen/Qwen3-8B"
    sent = SEEN[-1]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["temperature"] == 0.0


def test_streaming_pieces_and_usage():
    gen = backend()
    assert list(gen.stream("prompt")) == ["Art", "icle ", "147."]
    assert gen.last_usage == {"input": 120, "output": 3}
    assert SEEN[-1]["stream"] is True
    assert SEEN[-1]["stream_options"] == {"include_usage": True}
