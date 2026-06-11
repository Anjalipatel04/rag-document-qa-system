"""
llm_client.py
Thin wrapper around OpenAI SDK that supports both:
  - OpenAI (GPT-3.5 / GPT-4o-mini)
  - Groq (free Llama3 / Mixtral inference — recommended for students)

Groq gives 6000 free tokens/min — plenty for this project without paying.
"""

import os
from typing import List, Dict
from openai import OpenAI


def build_llm_client() -> "LLMClient":
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL", "llama3-8b-8192")
    max_tokens = int(os.getenv("MAX_TOKENS", "1024"))

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    else:  # openai
        api_key = os.getenv("OPENAI_API_KEY", "")
        client = OpenAI(api_key=api_key)

    return LLMClient(client, model=model, max_tokens=max_tokens)


class LLMClient:
    def __init__(self, client: OpenAI, model: str, max_tokens: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
