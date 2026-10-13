"""AnswerService: поиск → промпт → генерация → разбор ссылок."""

from __future__ import annotations

import re
import time

from mentor.domain import Answer, Citation, ScoredChunk
from mentor.ports import LLM, Embedder, VectorStore
from mentor.prompts import REFUSAL_TEXT, SYSTEM_PROMPT_NO_RAG, SYSTEM_PROMPT_RAG

# [1], [1][3], [1, 2]
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
REFUSAL_MARKER = "в материалах курса этого нет"


def parse_citations(text: str, n_fragments: int) -> tuple[list[int], list[int]]:
    """Возвращает (валидные номера, невалидные номера) в порядке первого появления."""
    valid: list[int] = []
    invalid: list[int] = []
    for m in CITATION_RE.finditer(text):
        for num in (int(x) for x in m.group(1).split(",")):
            target = valid if 1 <= num <= n_fragments else invalid
            if num not in target:
                target.append(num)
    return valid, invalid


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_refusal(text: str) -> bool:
    return REFUSAL_MARKER in normalize(text)


def format_pages(page_from: int, page_to: int) -> str:
    return str(page_from) if page_from == page_to else f"{page_from}–{page_to}"


def build_context(question: str, retrieved: list[ScoredChunk]) -> str:
    parts = ["Фрагменты материалов курса:", ""]
    for i, sc in enumerate(retrieved, start=1):
        c = sc.chunk
        title = f" «{c.section_title}»" if c.section_title else ""
        parts.append(
            f"[{i}] {c.doc_title} — раздел {c.section}{title}, "
            f"стр. {format_pages(c.page_from, c.page_to)}"
        )
        parts.append(c.text)
        parts.append("")
    parts.append(f"Вопрос студента: {question}")
    return "\n".join(parts)


class AnswerService:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        model_name: str,
        top_k: int,
        min_score: float | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.model_name = model_name
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        vector = self.embedder.embed([question])[0]
        return self.store.search(vector, top_k or self.top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        started = time.perf_counter()
        retrieved = self.retrieve(question, top_k)
        top_score = retrieved[0].score if retrieved else None

        if self.min_score is not None and (top_score is None or top_score < self.min_score):
            text = REFUSAL_TEXT
        else:
            text = self.llm.chat(SYSTEM_PROMPT_RAG, build_context(question, retrieved))

        valid, invalid = parse_citations(text, len(retrieved))
        citations = [
            Citation(
                n=n,
                chunk_id=retrieved[n - 1].chunk.chunk_id,
                doc_id=retrieved[n - 1].chunk.doc_id,
                doc_title=retrieved[n - 1].chunk.doc_title,
                section=retrieved[n - 1].chunk.section,
                page_from=retrieved[n - 1].chunk.page_from,
                page_to=retrieved[n - 1].chunk.page_to,
            )
            for n in valid
        ]
        return Answer(
            question=question,
            answer_text=text,
            refused=is_refusal(text),
            citations=citations,
            invalid_citation_numbers=invalid,
            retrieved=retrieved,
            top_score=top_score,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=self.model_name,
            mode="rag",
        )

    def ask_no_rag(self, question: str) -> Answer:
        started = time.perf_counter()
        text = self.llm.chat(SYSTEM_PROMPT_NO_RAG, question)
        return Answer(
            question=question,
            answer_text=text,
            refused=is_refusal(text),
            citations=[],
            invalid_citation_numbers=[],
            retrieved=[],
            top_score=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=self.model_name,
            mode="no_rag",
        )
