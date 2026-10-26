"""LLM judge + local embeddings. No Streamlit dependency, so it works in scripts, CI and notebooks.

To use another provider, subclass `Judge` and implement `ask(prompt) -> dict`.
"""
import json
import time
from functools import lru_cache

import numpy as np

DEFAULT_MODEL = "openai/gpt-oss-120b"  # OpenAI open-weight model, free on Groq
MODELS = {
    "OpenAI gpt-oss-120b (best quality)": "openai/gpt-oss-120b",
    "OpenAI gpt-oss-20b (faster, lighter)": "openai/gpt-oss-20b",
}


class Judge:
    """Interface every judge must follow."""

    def ask(self, prompt: str) -> dict:  # pragma: no cover
        raise NotImplementedError


class GroqJudge(Judge):
    """Groq free tier, JSON output, throttling and rate-limit retry."""

    def __init__(self, api_key, model=DEFAULT_MODEL, delay=2.0):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model, self.delay = model, delay

    def ask(self, prompt):
        from groq import RateLimitError
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a strict evaluator. Reply with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    reasoning_effort="low",
                    response_format={"type": "json_object"},
                )
                time.sleep(self.delay)
                return json.loads(resp.choices[0].message.content)
            except RateLimitError:
                time.sleep(15 * (attempt + 1))
            except json.JSONDecodeError:
                time.sleep(1)
        raise RuntimeError("LLM call failed after retries (rate limit or bad JSON).")


@lru_cache(maxsize=1)
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding("BAAI/bge-small-en-v1.5")  # free, local, small


def embed(texts):
    vecs = np.array(list(_embedder().embed(texts)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
