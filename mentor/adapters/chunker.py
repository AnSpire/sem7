"""SectionChunker: блоки → чанки по разделам с перекрытием."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from itertools import groupby

from mentor.config import ChunkingConfig, DocumentSpec
from mentor.domain import Block, Chunk

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…:;])\s+")
SEP = "\n\n"


@dataclass(frozen=True)
class _Unit:
    text: str
    kind: str
    page: int
    section: str | None
    section_title: str | None
    overlap: bool = False  # копия хвоста предыдущего чанка


def _parent(section: str | None) -> str | None:
    if section is None or "." not in section:
        return section
    return section.rsplit(".", 1)[0]


def _related(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a == b
    return _parent(a) == _parent(b) or a.startswith(b + ".") or b.startswith(a + ".")


def _units_len(units: list[_Unit]) -> int:
    return sum(len(u.text) for u in units) + len(SEP) * max(0, len(units) - 1)


class SectionChunker:
    def __init__(self, cfg: ChunkingConfig):
        self.cfg = cfg

    def split(self, doc: DocumentSpec, blocks: list[Block]) -> list[Chunk]:
        pieces: list[list[_Unit]] = []
        for _, group in groupby(blocks, key=lambda b: b.section):
            units = [u for b in group for u in self._explode(b)]
            pieces.extend(self._pack(units))
        pieces = self._merge_small(pieces)
        return [self._to_chunk(doc, i, units) for i, units in enumerate(pieces, start=1)]

    # ---------- длинные блоки ----------

    def _explode(self, block: Block) -> list[_Unit]:
        """Блок длиннее max_chars режется: текст по предложениям, код по строкам."""
        base = _Unit(block.text, block.kind, block.page, block.section, block.section_title)
        if len(block.text) <= self.cfg.max_chars:
            return [base]
        if block.kind == "code":
            parts = self._greedy(block.text.split("\n"), "\n", self.cfg.max_chars)
        else:
            # Оставляем место под перекрытие, иначе оно не влезет в следующий чанк.
            limit = max(self.cfg.max_chars - self.cfg.overlap_chars, self.cfg.max_chars // 2)
            sentences: list[str] = []
            for s in SENTENCE_SPLIT_RE.split(block.text):
                if len(s) <= limit:
                    sentences.append(s)
                else:
                    sentences.extend(self._greedy(s.split(" "), " ", limit))
            parts = self._greedy(sentences, " ", limit)
        return [replace(base, text=p) for p in parts]

    @staticmethod
    def _greedy(items: list[str], sep: str, limit: int) -> list[str]:
        out: list[str] = []
        cur = ""
        for item in items:
            candidate = f"{cur}{sep}{item}" if cur else item
            if cur and len(candidate) > limit:
                out.append(cur)
                cur = item
            else:
                cur = candidate
        if cur:
            out.append(cur)
        return out

    # ---------- упаковка раздела ----------

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        pieces: list[list[_Unit]] = []
        cur: list[_Unit] = []
        for u in units:
            if any(not x.overlap for x in cur) and _units_len([*cur, u]) > self.cfg.max_chars:
                pieces.append(cur)
                cur = self._overlap(cur)
                if _units_len([*cur, u]) > self.cfg.max_chars:
                    cur = []
            cur.append(u)
        if cur:
            pieces.append(cur)
        return pieces

    def _overlap(self, prev: list[_Unit]) -> list[_Unit]:
        """Хвост предыдущего чанка: целые блоки, а если не влезают — конец последнего абзаца."""
        limit = self.cfg.overlap_chars
        if limit <= 0 or not prev:
            return []
        taken: list[_Unit] = []
        for u in reversed(prev):
            if _units_len([u, *taken]) > limit:
                break
            taken.insert(0, u)
        if taken:
            return [replace(u, overlap=True) for u in taken]
        last = prev[-1]
        if last.kind != "text":
            return []  # код не режем ради перекрытия
        tail = last.text[-limit:]
        sentences = SENTENCE_SPLIT_RE.split(tail)
        tail = " ".join(sentences[1:]) if len(sentences) > 1 else tail.partition(" ")[2]
        return [replace(last, text=tail, overlap=True)] if tail.strip() else []

    # ---------- склейка коротких ----------

    def _merge_small(self, pieces: list[list[_Unit]]) -> list[list[_Unit]]:
        pieces = [p for p in pieces if p]
        i = 0
        while i < len(pieces) and len(pieces) > 1:
            if _units_len(pieces[i]) >= self.cfg.min_chars:
                i += 1
                continue
            options = []
            for j, is_next in ((i - 1, False), (i + 1, True)):
                if 0 <= j < len(pieces):
                    a, b = (pieces[i], pieces[j]) if is_next else (pieces[j], pieces[i])
                    # Сравниваем разделы на стыке: конец левого куска и начало правого.
                    left, right = a[-1].section, b[0].section
                    merged = self._concat(a, b)
                    size = _units_len(merged)
                    options.append(
                        (
                            size > self.cfg.max_chars + self.cfg.min_chars,
                            left != right,
                            not _related(left, right),
                            size > self.cfg.max_chars,
                            0 if is_next else 1,
                            j,
                            merged,
                        )
                    )
            *_, j, merged = min(options, key=lambda o: o[:5])
            lo = min(i, j)
            pieces[lo : lo + 2] = [merged]
            i = max(0, lo - 1)
        return pieces

    @staticmethod
    def _concat(a: list[_Unit], b: list[_Unit]) -> list[_Unit]:
        # Перекрытие в начале b — копия хвоста a; при склейке оно не нужно.
        return a + [u for u in b if not u.overlap]

    # ---------- сборка ----------

    def _to_chunk(self, doc: DocumentSpec, n: int, units: list[_Unit]) -> Chunk:
        parts: list[str] = []
        sections: list[str] = []
        for k, u in enumerate(units):
            if u.section is not None and u.section not in sections:
                sections.append(u.section)
                if k > 0:
                    parts.append(f"{u.section} {u.section_title or ''}".strip())
            parts.append(u.text)
        text = SEP.join(parts)
        first = units[0]
        heading = f"{first.section or ''} {first.section_title or ''}".strip()
        text_for_embedding = (
            f"{doc.title} › {heading}{SEP}{text}" if self.cfg.heading_prefix else text
        )
        return Chunk(
            chunk_id=f"{doc.doc_id}-{n:04d}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            section=first.section,
            section_title=first.section_title,
            page_from=min(u.page for u in units),
            page_to=max(u.page for u in units),
            text=text,
            text_for_embedding=text_for_embedding,
            sections=sections,
        )
