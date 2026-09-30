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
