"""IngestService: парсинг → нарезка → эмбеддинги → сохранение индекса."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from mentor.config import DocumentSpec, Settings
from mentor.domain import Block, Chunk
from mentor.ports import Chunker, DocumentLoader, Embedder, VectorStore


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def build_manifest(settings: Settings) -> dict:
    """Параметры индекса: по ним проверяется, что индекс собран из текущих PDF и конфига."""
    return {
        "kb_id": settings.knowledge_base.kb_id,
        "embed_model": settings.ollama.embed_model,
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "path": d.path,
                "sha256": sha256_file(settings.resolve(d.path)),
            }
            for d in settings.knowledge_base.documents
        ],
        "chunking": settings.chunking.model_dump(),
        "parsing": settings.parsing.model_dump(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def render_parsed_markdown(doc: DocumentSpec, blocks: list[Block]) -> str:
    out = [f"# {doc.doc_id}: {doc.title}\n"]
    for b in blocks:
        out.append(f"<!-- page {b.page} | section {b.section} | {b.kind} -->")
        out.append(f"```\n{b.text}\n```\n" if b.kind == "code" else f"{b.text}\n")
    return "\n".join(out)


@dataclass
class IngestResult:
    blocks: dict[str, list[Block]] = field(default_factory=dict)
    chunks: dict[str, list[Chunk]] = field(default_factory=dict)


class IngestService:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder | None,
        store: VectorStore | None,
        parsed_dir: Path,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.parsed_dir = parsed_dir

    def parse(self, docs: list[DocumentSpec]) -> IngestResult:
        result = IngestResult()
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            blocks = self.loader.load(doc)
            (self.parsed_dir / f"{doc.doc_id}.md").write_text(
                render_parsed_markdown(doc, blocks), encoding="utf-8"
            )
            result.blocks[doc.doc_id] = blocks
            result.chunks[doc.doc_id] = self.chunker.split(doc, blocks)
        return result

    def run(self, docs: list[DocumentSpec]) -> IngestResult:
        if self.embedder is None or self.store is None:
            raise RuntimeError("для индексации нужны embedder и store")
        result = self.parse(docs)
        all_chunks = [c for chunks in result.chunks.values() for c in chunks]
        vectors = self.embedder.embed([c.text_for_embedding for c in all_chunks])
        self.store.add(all_chunks, vectors)
        self.store.save()
        return result
