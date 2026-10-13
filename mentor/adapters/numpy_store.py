"""NumpyVectorStore: файловый векторный индекс (vectors.npy + chunks.jsonl + manifest.json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mentor.domain import Chunk, ScoredChunk

# Поля манифеста, которые должны совпадать с текущим конфигом (дата и размерность — нет).
MANIFEST_KEYS_TO_MATCH = ("kb_id", "embed_model", "documents", "chunking", "parsing")


class IndexMismatchError(RuntimeError):
    pass


class NumpyVectorStore:
    def __init__(self, index_dir: Path, manifest: dict):
        """`manifest` — ожидаемые параметры индекса для текущего конфига."""
        self.index_dir = index_dir
        self.manifest = manifest
        self.chunks: list[Chunk] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    @property
    def chunks_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"

    @property
    def vectors_path(self) -> Path:
        return self.index_dir / "vectors.npy"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("число чанков и векторов не совпадает")
        vectors = np.asarray(vectors, dtype=np.float32)
        self.vectors = vectors if not self.chunks else np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        scores = self.vectors @ np.asarray(vector, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores)[:top_k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(scores[i])) for i in order]

    def save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.vectors_path, self.vectors)
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")
        manifest = {
            **self.manifest,
            "dim": int(self.vectors.shape[1]),
            "n_chunks": len(self.chunks),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self) -> None:
        hint = "пересоберите индекс: `uv run mentor ingest`"
        if not self.manifest_path.exists():
            raise IndexMismatchError(f"Индекс не найден в {self.index_dir}; {hint}")
        stored = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        diff = [k for k in MANIFEST_KEYS_TO_MATCH if stored.get(k) != self.manifest.get(k)]
        if diff:
            raise IndexMismatchError(
                "Индекс не соответствует текущему конфигу или PDF "
                f"(расходятся: {', '.join(diff)}); {hint}"
            )
        self.vectors = np.load(self.vectors_path)
        with self.chunks_path.open(encoding="utf-8") as f:
            self.chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        if len(self.chunks) != len(self.vectors):
            raise IndexMismatchError(f"chunks.jsonl и vectors.npy рассинхронизированы; {hint}")
