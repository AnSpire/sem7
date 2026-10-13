"""Интерфейсы адаптеров. Сервисы зависят только от них."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from mentor.config import DocumentSpec
from mentor.domain import Block, Chunk, ScoredChunk


class DocumentLoader(Protocol):
    def load(self, doc: DocumentSpec) -> list[Block]: ...


class Chunker(Protocol):
    def split(self, doc: DocumentSpec, blocks: list[Block]) -> list[Chunk]: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...  # L2-нормированные векторы


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, top_k: int) -> list[ScoredChunk]: ...
    def save(self) -> None: ...
    def load(self) -> None: ...


class LLM(Protocol):
    def chat(self, system: str, user: str) -> str: ...
