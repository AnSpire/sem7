"""Автоматические проверки — подсказки для ручной оценки, не итоговая оценка."""

from __future__ import annotations


def section_match(chunk_section: str | None, gold_section: str) -> bool:
    """Раздел совпадает, если равен эталонному или вложен в него (в любую сторону)."""
    if chunk_section is None:
        return False
    a, b = chunk_section, gold_section
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def _chunk_sections(chunk: dict) -> list[str]:
    return chunk.get("sections") or ([chunk["section"]] if chunk.get("section") else [])


def chunk_matches(chunk: dict, gold: dict, check_pages: bool = False) -> bool:
    if chunk["doc_id"] != gold["doc_id"]:
        return False
    if not any(section_match(s, gold["section"]) for s in _chunk_sections(chunk)):
        return False
    pages = gold.get("pages") or []
    if check_pages and pages:
        return any(chunk["page_from"] <= p <= chunk["page_to"] for p in pages)
    return True


def _matches_any(chunk: dict, golds: list[dict], check_pages: bool = False) -> bool:
    return any(chunk_matches(chunk, g, check_pages) for g in golds)


def _cited_chunks(citations: list[dict], retrieved: list[dict]) -> list[dict]:
    by_id = {c["chunk_id"]: c for c in retrieved}
    return [by_id[c["chunk_id"]] for c in citations if c["chunk_id"] in by_id]


def run_checks(
    question: dict,
    *,
    mode: str,
    retrieved: list[dict],
    citations: list[dict],
    refused: bool,
) -> tuple[dict, list[str]]:
    """Возвращает (checks, suggested_error_codes). retrieved — словари полей чанка."""
    qtype = question["type"]
    golds = question.get("gold") or []
    rag = mode == "rag"
    cited = _cited_chunks(citations, retrieved)
    checks: dict = {}
    codes: list[str] = []

    if qtype == "provocation":
        checks = {"refused": refused, "pass": refused}
        if not refused:
            codes.append("G2?")
        return checks, codes

    if qtype == "main":
        if rag:
            ranks = [i for i, c in enumerate(retrieved, start=1) if _matches_any(c, golds)]
            hit = bool(ranks)
            hit_page = any(_matches_any(c, golds, check_pages=True) for c in retrieved)
            has_citation = bool(citations)
            cites_gold = any(_matches_any(c, golds) for c in cited)
            checks = {
                "retrieval_hit": hit,
                "retrieval_hit_page": hit_page,
                "gold_rank": ranks[0] if ranks else None,
                "has_citation": has_citation,
                "cites_gold": cites_gold,
                "refused": refused,
                "false_refusal": refused,
            }
            if not hit:
                codes.append("R1")
            if retrieved and not hit:
                codes.append("R2")
            if not refused and not has_citation:
                codes.append("S1")
            if has_citation and not cites_gold and hit:
                codes.append("S2")
        else:
            checks = {"refused": refused, "false_refusal": refused}
        return checks, codes

    if qtype == "multi":
        gold_docs = list(dict.fromkeys(g["doc_id"] for g in golds))
        gold_secs = [f"{g['doc_id']}:{g['section']}" for g in golds]
        if rag:
            docs_retrieved = [d for d in gold_docs if any(c["doc_id"] == d for c in retrieved)]
            secs_retrieved = [
                key
                for key, g in zip(gold_secs, golds, strict=True)
                if any(chunk_matches(c, g) for c in retrieved)
            ]
            docs_cited = [d for d in gold_docs if any(c["doc_id"] == d for c in cited)]
            any_gold_retrieved = any(_matches_any(c, golds) for c in retrieved)
            any_gold_cited = any(_matches_any(c, golds) for c in cited)
            checks = {
                "docs_retrieved": docs_retrieved,
                "all_docs_retrieved": len(docs_retrieved) == len(gold_docs),
                "sections_retrieved": secs_retrieved,
                "sections_coverage": round(len(secs_retrieved) / len(gold_secs), 3),
                "docs_cited": docs_cited,
                "all_docs_cited": len(docs_cited) == len(gold_docs),
                "has_citation": bool(citations),
                "refused": refused,
                "false_refusal": refused,
            }
            if not checks["all_docs_retrieved"]:
                codes.append("R1")
            if retrieved and not any_gold_retrieved:
                codes.append("R2")
            if not refused and not checks["all_docs_cited"]:
                codes.append("S1")
            if citations and not any_gold_cited and any_gold_retrieved:
                codes.append("S2")
        else:
            checks = {"refused": refused, "false_refusal": refused}
        return checks, codes

    raise ValueError(f"неизвестный тип вопроса: {qtype}")
