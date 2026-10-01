from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from egyptian_civil_code_rag.query import strip_thinking


def transformers_backend(model_name: str, max_new_tokens: int = 400):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    def generate(prompt: str) -> str:
        """Returns the answer text. Token counts for the last call are left
        on generate.last_usage ({"input": n, "output": m}) -- an attribute
        rather than a changed return type, so every existing caller
        (RAGQueryEngine.ask, the RAGAS harness) keeps working unchanged."""
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        n_in = inputs["input_ids"].shape[1]
        raw = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
        generate.last_usage = {"input": int(n_in), "output": int(out.shape[1] - n_in)}
        return strip_thinking(raw)

    def stream(prompt: str):
        """Yields answer text piece by piece as the model produces it.
        Sets generate.last_usage when the stream ends, like generate()."""
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        n_in = int(inputs["input_ids"].shape[1])
        result = {}

        def run():
            # model.generate blocks until done, so it runs in a thread while
            # this generator drains the streamer. Keeping its return value
            # gives exact output token counts (same as generate()).
            result["ids"] = model.generate(
                **inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False
            )

        worker = Thread(target=run)
        worker.start()
        yield from streamer
        worker.join()
        generate.last_usage = {"input": n_in, "output": int(result["ids"].shape[1] - n_in)}

    generate.model_name = model_name
    generate.last_usage = None
    generate.stream = stream
    return generate


def openai_backend(
    base_url: str,
    model: str,
    max_tokens: int = 400,
    api_key: str = "EMPTY",
    temperature: float = 0.0,
    http_client=None,
):
    """Generation through an OpenAI-compatible server -- vLLM's
    `vllm serve` / `python -m vllm.entrypoints.openai.api_server`.

    Same interface as transformers_backend: call it for a full answer, use
    .stream(prompt) for pieces, read .last_usage for token counts (exact,
    from the server's own usage report). Qwen3's thinking mode is switched
    off server-side via chat_template_kwargs, matching the local backend."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    extra = {"chat_template_kwargs": {"enable_thinking": False}}

    def _messages(prompt: str):
        return [{"role": "user", "content": prompt}]

    def generate(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=_messages(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra,
        )
        if resp.usage is not None:
            generate.last_usage = {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            }
        return strip_thinking(resp.choices[0].message.content or "")

    def stream(prompt: str):
        chunks = client.chat.completions.create(
            model=model,
            messages=_messages(prompt),
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in chunks:
            if chunk.usage is not None:
                generate.last_usage = {
                    "input": chunk.usage.prompt_tokens,
                    "output": chunk.usage.completion_tokens,
                }
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    generate.model_name = model
    generate.last_usage = None
    generate.stream = stream
    return generate


def backend_from_env(default_model: str = "Qwen/Qwen3-1.7B"):
    """Picks the generation backend from the environment, so the same API
    code runs on a laptop (CPU, transformers) or against a GPU vLLM server:

        VLLM_BASE_URL=http://host:8000/v1 GEN_MODEL=Qwen/Qwen3-8B  -> vLLM
        (unset)                                                   -> local transformers
    """
    import os

    model = os.environ.get("GEN_MODEL", default_model)
    base_url = os.environ.get("VLLM_BASE_URL")
    if base_url:
        return openai_backend(base_url, model)
    return transformers_backend(model)
