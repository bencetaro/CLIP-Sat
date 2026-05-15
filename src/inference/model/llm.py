import os
import json
import numpy as np
from llama_cpp import Llama
from dotenv import load_dotenv
load_dotenv()


class SimpleLLM:
    def __init__(
        self,
        model_path: str,
        temperature: float,
        max_tokens: int,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        try:
            self.llm = self._load_llama()
            print("[info] - llama.cpp model loaded successfully.")

        except Exception as e:
            raise ValueError(
                f"Failed to initialize llama.cpp model. "
                f"model_path={self.model_path}. Error: {e}"
            ) from e

    def _load_llama(self) -> Llama:
        common_kwargs = {
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "verbose": self.verbose,
            "use_mmap": True,
            "use_mlock": False,
            "n_batch": 512,
            "n_gpu_layers": self.n_gpu_layers,
        }

        if os.path.exists(self.model_path):
            print(f"[info] - Loading GGUF model from local path: {self.model_path}")
            return Llama(model_path=self.model_path, **common_kwargs)

        raise FileNotFoundError(
            f"Model path does not exist: {self.model_path}. "
            "Set LLM_CPP_MODEL_PATH to a mounted GGUF file path."
        )

    def llm_response(self, prompt: str):

        system_prompt = (
            "You are a helpful assistant.\n"
            "You MUST output valid JSON only.\n"
            "Do not output markdown.\n"
            "Do not explain.\n"
            "Return JSON only."
        )

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],

                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.9,              # deterministic generation
                repeat_penalty=1.1,     # faster + safer JSON
                stream=False,           # no streaming needed
                response_format={
                    "type": "json_object"
                }
            )

            result = response["choices"][0]["message"]["content"].strip()
            parsed = json.loads(result)
            return parsed

        except Exception as e:
            raise AttributeError(
                f"LLM generation failed. "
                f"model_path={self.model_path} "
                f"temperature={self.temperature} "
                f"max_tokens={self.max_tokens}. "
                f"Error: {e}"
            ) from e


if __name__ == "__main__":

    model_path = os.getenv("LLM_MODEL_ID", "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", 128))
    temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))
    n_threads = int(os.getenv("LLM_THREADS", 8))

    example = {
        c: np.random.rand()
        for c in [
            "Agriculture",
            "Airport",
            "Beach",
            "Desert",
            "Forest",
            "Grassland",
            "Highway",
            "Lake",
            "Mountain",
            "Parking",
            "Port",
            "Railway",
            "Residential",
            "River",
        ]
    }

    top_preds = sorted(
        example.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    prompt = f"""
Analyze these satellite image predictions.

Predictions:
{top_preds}

Return ONLY valid JSON in this format:

{{
  "guesses": [
    {{
      "label": "string",
      "reason": "string"
    }}
  ],
  "confidence_analysis": [
    "string"
  ],
  "summary": "string"
}}
"""

    simpllm = SimpleLLM(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        n_threads=n_threads,
    )

    response = simpllm.llm_response(prompt)

    print(json.dumps(response, indent=2))
