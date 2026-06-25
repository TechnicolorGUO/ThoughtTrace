"""Token counting. The paper uses tiktoken's gpt-4o encoding for all length
stats; we keep that unchanged (it's just counting). tiktoken is imported lazily
so machines without it can still run turn-count stats — they fall back to a
whitespace word count, clearly flagged as approximate via ``is_exact``.
"""

from __future__ import annotations


class TokenCounter:
    def __init__(self, name: str = "gpt-4o"):
        self.name = name
        self._enc = None
        self.is_exact = False
        try:
            import tiktoken

            self._enc = tiktoken.encoding_for_model(name)
            self.is_exact = True
        except Exception:
            self._enc = None  # fall back to word count

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        return len(text.split())  # approximate fallback

    def __call__(self, text: str | None) -> int:
        return self.count(text)
