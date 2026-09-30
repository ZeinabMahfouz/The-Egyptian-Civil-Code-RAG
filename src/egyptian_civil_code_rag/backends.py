import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    generate.model_name = model_name
    generate.last_usage = None
    return generate
