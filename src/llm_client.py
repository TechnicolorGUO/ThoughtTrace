"""OpenAI-compatible LLM client with retry + on-disk response caching.

Every LLM-as-X call in Phases 1-4 (labeling, scoring, judging, prediction,
rewriting) goes through this one client so a single Qwen3-8B vLLM endpoint is
reused. Responses are cached on disk keyed by ``(model, messages, params)`` so
reruns are free and deterministic — essential because Phases 2-4 make tens of
thousands of calls.

Serve the backbone once on the server:

    vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 32768

Then, anywhere in the pipeline:

    from src.llm_client import LLMClient
    from src.config import load_config
    client = LLMClient.from_config(load_config()["llm"])
    out = client.complete("Classify this reason: ...")

`openai` is imported lazily inside ``_call`` so this module imports cleanly on a
machine without the package (cache-only inspection, tests of the keying logic).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def _stable_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        cache_dir: str | Path = "outputs/llm_cache",
        max_retries: int = 5,
        retry_backoff: float = 2.0,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None  # lazily constructed openai client

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "LLMClient":
        """Build from a config dict (the ``llm`` or ``judge`` block)."""
        params = dict(
            model=cfg["model"],
            base_url=cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=cfg.get("api_key", "EMPTY"),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1024),
            enable_thinking=cfg.get("enable_thinking", False),
            cache_dir=cfg.get("cache_dir", "outputs/llm_cache"),
        )
        params.update(overrides)
        return cls(**params)

    # --- public API ---------------------------------------------------------

    def complete(
        self,
        prompt: str | None = None,
        *,
        system: str | None = None,
        messages: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> str:
        """Return the assistant text for a single prompt (or full message list)."""
        if messages is None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt or ""})

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            # Qwen3 thinking toggle; harmless extra field for other servers.
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            },
        }

        key = _stable_key(params)
        cache_file = self.cache_dir / f"{key}.json"
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["completion"]

        text = self._call(params)

        if use_cache:
            cache_file.write_text(
                json.dumps({"params": params, "completion": text}, ensure_ascii=False),
                encoding="utf-8",
            )
        return text

    def complete_json(self, prompt: str, **kw) -> dict | list | None:
        """Like :meth:`complete` but parse the response as JSON.

        Tolerates models that wrap JSON in prose / ``` fences by extracting the
        first balanced ``{...}`` or ``[...]`` span. Returns ``None`` on failure
        so callers can drop unparseable rows.
        """
        text = self.complete(prompt, **kw)
        return _extract_json(text)

    # --- internals ----------------------------------------------------------

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _call(self, params: dict) -> str:
        client = self._ensure_client()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(**params)
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retry on any transport error
                last_err = e
                time.sleep(self.retry_backoff ** attempt)
        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_err}"
        )


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    # try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # fall back to first balanced object/array
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    return None
