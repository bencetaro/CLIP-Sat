import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()


class SimpleLLM:
    def __init__(self, model_id:str, temperature:float, max_tokens:int, device="cpu"):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        try:
            print(f"[info] - LLM client ({self.model_id}) will be initialized...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            load_kwargs = {}
            if str(self.device).lower() in {"cuda", "gpu"}:
                load_kwargs["torch_dtype"] = torch.float16
                try:
                    import accelerate  # noqa: F401
                    load_kwargs["device_map"] = "auto"
                except Exception:
                    pass
            else:
                load_kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
            if str(self.device).lower() not in {"cuda", "gpu"}:
                self.model.to("cpu")
            print("[info] - LLM client has been loaded in successfully.")
        except Exception as e:
            raise ValueError(f"Failed to initialize HF LLM. model_id={self.model_id} device={self.device}. Error: {e}") from e

    def llm_response(self, prompt:str):
        try:
            target_device = "cuda" if str(self.device).lower() in {"cuda", "gpu"} else "cpu"
            use_chat = bool(getattr(self.tokenizer, "chat_template", None)) and hasattr(self.tokenizer, "apply_chat_template")
            if use_chat:
                messages = [
                    {"role": "system", "content": "Output valid JSON only, no markdown and no extra text."},
                    {"role": "user", "content": prompt},
                ]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(target_device)
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=(self.temperature is not None and float(self.temperature) > 0),
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
                generated_ids = outputs[0][input_ids.shape[-1]:]
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(target_device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=(self.temperature is not None and float(self.temperature) > 0),
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
                prompt_len = inputs["input_ids"].shape[-1]
                generated_ids = outputs[0][prompt_len:]

            result = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return result
        except Exception as e:
            raise AttributeError(
                f"LLM generation failed. model_id={self.model_id} temperature={self.temperature} "
                f"max_tokens={self.max_tokens} device={self.device}. Error: {e}"
            ) from e

if __name__ == "__main__":
    model_id = os.getenv("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    max_tokens = int(os.getenv("MAX_TOKENS", 700))
    temperature = float(os.getenv("TEMPERATURE", 0.2))
    device = os.getenv("DEVICE", "cpu")

    example = {c: np.random.rand() for c in ["Agriculture", "Airport", "Beach", "Desert",
                                             "Forest", "Grassland", "Highway", "Lake",
                                             "Mountain", "Parking", "Port", "Railway",
                                             "Residential", "River"]}

    top_preds = sorted(example.items(), key=lambda x: x[1], reverse=True)[:5]

    prompt = f"""
    You are analyzing the prediction probabilities of the following CLIP model, which have been finetuned on satellite images:

    Predictions:
    {top_preds}

    Instructions:
    1. Provide your guesses about the image content (land-use patterns).
    2. Explain confidence levels:
    - Warn if probabilities are very high (>0.8)
    - Warn if probabilities are very low (<0.3)
    3. Give a brief summary of similar land-use patterns in real world environments.
    4. Respond ONLY in JSON, exactly in this format:

    {{
    "guesses": [...],
    "confidence_analysis": [...],
    "summary": "..."
    }}
    """

    simpllm = SimpleLLM(model_id, temperature, max_tokens, device)
    resp = simpllm.llm_response(prompt)
    print(resp)
