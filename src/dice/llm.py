"""LLM providers used by DICE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LanguageModel(Protocol):
    def generate(self, prompt: str, stop: list[str] | None = None) -> str:
        """Generate text from a prompt."""


@dataclass
class OpenAIChatModel:
    """Thin wrapper around the OpenAI chat completions API."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 128
    system_message: str = "You are a helpful assistant."

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai package to use OpenAIChatModel.") from exc
        self.client = OpenAI()

    def generate(self, prompt: str, stop: list[str] | None = None) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "developer", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
        )
        return completion.choices[0].message.content or ""


@dataclass
class GemmaModel:
    """Local Gemma wrapper matching the original notebook experiment."""

    model_name: str = "google/gemma-2-2b-it"
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install torch and transformers to use GemmaModel.") from exc

        self.torch = torch
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        self.model.to(self.device)

    def generate(self, prompt: str, stop: list[str] | None = None) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=self.max_new_tokens)
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = generated.replace(prompt, "", 1).strip()
        if stop:
            for marker in stop:
                if marker in text:
                    text = text.split(marker, 1)[0]
        return text.strip()

