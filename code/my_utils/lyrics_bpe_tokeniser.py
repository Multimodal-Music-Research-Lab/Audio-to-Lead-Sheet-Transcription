"""Hugging Face ID adapter for lexical ``**text`` cells."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from pathlib import Path


class LyricsBPETokeniser:
    """Keep Hugging Face IDs disjoint from the project's music vocabulary."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def tokenizer(self):
        try:
            from transformers import AutoTokenizer

            local_copy = (
                Path(__file__).resolve().parent.parent / "tokenizers" / self.model_name
            )
            if (local_copy / "tokenizer_config.json").is_file():
                return AutoTokenizer.from_pretrained(str(local_copy))
            return AutoTokenizer.from_pretrained(self.model_name)
        except Exception as error:
            raise RuntimeError(
                f"Could not load lyrics BPE tokenizer {self.model_name!r}. "
                "Confirm the model name and Hugging Face access."
            ) from error

    @staticmethod
    def unwrap(token: str) -> int | None:
        if not token.startswith("<LYR_BPE_") or not token.endswith(">"):
            return None
        value = token[len("<LYR_BPE_") : -1]
        return int(value) if value.isdigit() else None

    def decode_wrappers(self, tokens: Iterable[str]) -> str:
        ids = [self.unwrap(token) for token in tokens]
        if any(token_id is None for token_id in ids):
            raise ValueError("Expected only lyric BPE wrapper tokens")
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)
