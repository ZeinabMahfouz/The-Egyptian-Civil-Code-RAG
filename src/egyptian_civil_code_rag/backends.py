import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from egyptian_civil_code_rag.query import strip_thinking


def transformers_backend(model_name: str, max_new_tokens: int = 400):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return strip_thinking(raw)

    return generate
