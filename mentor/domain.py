"""Доменные объекты конвейера."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Block(BaseModel):
    doc_id: str
    text: str
    page: int  # с 1
    section: str | None
    section_title: str | None
    kind: Literal["text", "code"] = "text"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str | None
    section_title: str | None
    page_from: int
    page_to: int
    text: str
    text_for_embedding: str
    # Все разделы, чьи блоки попали в чанк (после склейки коротких их может быть несколько).
    sections: list[str] = []


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    n: int
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str | None
    page_from: int
    page_to: int


class Answer(BaseModel):
    question: str
    answer_text: str
    refused: bool
    citations: list[Citation]
    invalid_citation_numbers: list[int]
    retrieved: list[ScoredChunk]
    top_score: float | None
    latency_ms: int
    model: str
    mode: Literal["rag", "no_rag"]


class EvalRecord(BaseModel):
    id: str
    type: str
    question: str
    answer_text: str | None = None
    refused: bool | None = None
    citations: list[Citation] = []
    invalid_citation_numbers: list[int] = []
    retrieved: list[dict] = []
    top_score: float | None = None
    latency_ms: int | None = None
    model: str
    mode: str
    checks: dict = {}
    suggested_error_codes: list[str] = []
    error: str | None = None
    manual_score: int | None = None
    manual_error_codes: list[str] = []
    manual_comment: str = ""
