"""Адаптеры Ollama: эмбеддинги и чат."""

from __future__ import annotations

import re

import httpx
import numpy as np

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
THINK_CLOSE = "</think>"


def strip_thinking(content: str) -> str:
    """Вырезает рассуждение модели.

    Thinking-модели (например, qwen3:4b = Qwen3-4B-Thinking-2507) игнорируют `think: false`;
    открывающий `<think>` у них стоит в шаблоне, поэтому в ответ приходит только `</think>`.
    """
    content = THINK_RE.sub("", content)
    if THINK_CLOSE in content:
        content = content.rsplit(THINK_CLOSE, 1)[1]
    return content.strip()


class OllamaError(RuntimeError):
    pass


def _client(base_url: str, timeout_s: float) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=timeout_s)


def list_models(base_url: str, timeout_s: float = 10) -> list[str]:
    with _client(base_url, timeout_s) as c:
        r = c.get("/api/tags")
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]


def server_version(base_url: str, timeout_s: float = 10) -> str | None:
    try:
        with _client(base_url, timeout_s) as c:
            return c.get("/api/version").json().get("version")
    except (httpx.HTTPError, ValueError):
        return None


def model_available(name: str, available: list[str]) -> bool:
    """`bge-m3` совпадает с `bge-m3:latest`."""
    full = name if ":" in name else f"{name}:latest"
    return name in available or full in available


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, batch_size: int, timeout_s: float):
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
        self.timeout_s = timeout_s

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        with _client(self.base_url, self.timeout_s) as c:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                r = c.post("/api/embed", json={"model": self.model, "input": batch})
                if r.status_code != 200:
                    raise OllamaError(f"/api/embed {r.status_code}: {r.text[:300]}")
                vectors.extend(r.json()["embeddings"])
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        num_ctx: int,
        temperature: float,
        think: bool,
        timeout_s: float,
    ):
        self.base_url = base_url
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        self.timeout_s = timeout_s
        self._think_supported = True

    def chat(self, system: str, user: str) -> str:
        body: dict = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # num_ctx передаём явно: иначе Ollama молча обрежет контекст.
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        with _client(self.base_url, self.timeout_s) as c:
            if self._think_supported:
                r = c.post("/api/chat", json={**body, "think": self.think})
                if r.status_code == 400 and "think" in r.text.lower():
                    self._think_supported = False
            if not self._think_supported:
                r = c.post("/api/chat", json=body)
            if r.status_code != 200:
                raise OllamaError(f"/api/chat {r.status_code}: {r.text[:300]}")
            content = r.json()["message"]["content"]
        return strip_thinking(content)
