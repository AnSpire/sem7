"""PyMuPdfLoader: PDF → список блоков с номером страницы и разделом."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from mentor.config import DocumentSpec, ParsingConfig
from mentor.domain import Block

# Номер раздела в начале строки. В этих PDF пробел после номера иногда «съеден»
# inline-кодом («4.1.if Заявления»), поэтому кроме `\s+` допускаем сразу букву.
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s*(?=[^\d\s.])(.*)$")

URL_RE = re.compile(r"^(https?://\S+|\S*docs\.python\.org\S*)$")
PAGE_COUNTER_RE = re.compile(r"^\d+\s*/\s*\d+$")
DATETIME_RE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4},?\s+\d{1,2}:\d{2}(:\d{2})?$")
CODE_PREFIX_RE = re.compile(r"^(>>>|\.\.\.)(\s|$)")

SOFT_HYPHENS = ("‐", "­")
MONO_FONT_HINTS = ("consolas", "mono", "courier", "menlo")
TOGGLE_BUTTON_TEXT = ">>>"  # кнопка «скрыть приглашение» справа от примеров кода


@dataclass
class _Line:
    page: int
    x0: float
    y0: float
    size: float
    text: str
    mono: bool


def _is_mono(span: dict) -> bool:
    return bool(span["flags"] & 8) or any(h in span["font"].lower() for h in MONO_FONT_HINTS)


INVISIBLE_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", INVISIBLE_RE.sub("", text)).strip()


def join_lines(lines: list[str]) -> str:
    """Склеивает строки абзаца: перенос строки → пробел, дефисный перенос → слитно."""
    out = ""
    for raw in lines:
        line = _norm(raw)
        if not line:
            continue
        if not out:
            out = line
        elif out.endswith(SOFT_HYPHENS):
            out = out[:-1] + line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out + line
        else:
            out = out + " " + line
    return out


class PyMuPdfLoader:
    def __init__(self, root: Path, cfg: ParsingConfig):
        self.root = root
        self.cfg = cfg
        # doc_id → [(section, title, page)] — для диагностики и чекпоинта.
        self.headings: dict[str, list[tuple[str, str, int]]] = {}

    # ---------- извлечение строк ----------

    def _read_lines(self, path: Path) -> tuple[list[_Line], int]:
        lines: list[_Line] = []
        with pymupdf.open(path) as pdf:
            n_pages = pdf.page_count
            for page_no, page in enumerate(pdf, start=1):
                width = page.rect.width
                for block in page.get_text("dict")["blocks"]:
                    if block["type"] != 0:
                        continue
                    for line in block["lines"]:
                        spans = [s for s in line["spans"] if s["text"].strip()]
                        if not spans:
                            continue
                        text = INVISIBLE_RE.sub("", "".join(s["text"] for s in line["spans"]))
                        text = text.rstrip()
                        x0 = line["bbox"][0]
                        if text.strip() == TOGGLE_BUTTON_TEXT and x0 > width * 0.7:
                            continue
                        lines.append(
                            _Line(
                                page=page_no,
                                x0=x0,
                                y0=line["bbox"][1],
                                size=max(s["size"] for s in spans),
                                text=text,
                                mono=all(_is_mono(s) for s in spans),
                            )
                        )
        lines.sort(key=lambda ln: (ln.page, round(ln.y0, 0), ln.x0))
        return lines, n_pages

    def _drop_page_furniture(self, lines: list[_Line], n_pages: int) -> list[_Line]:
        """Удаляет колонтитулы: повторяющиеся строки, URL, счётчики страниц, дату/время."""
        pages_per_text: dict[str, set[int]] = {}
        for ln in lines:
            if not ln.mono:  # строки кода вроде «...» повторяются, но это не колонтитул
                pages_per_text.setdefault(_norm(ln.text), set()).add(ln.page)
        threshold = max(2, self.cfg.repeated_line_ratio * n_pages)
        repeated = {t for t, pages in pages_per_text.items() if len(pages) >= threshold}

        kept = []
        for ln in lines:
            t = _norm(ln.text)
            if n_pages > 1 and not ln.mono and t in repeated:
                continue
            if URL_RE.match(t) or PAGE_COUNTER_RE.match(t) or DATETIME_RE.match(t):
                continue
            kept.append(ln)
        return kept

    # ---------- классификация ----------

    def _match_heading(self, ln: _Line, chapter: int, body_size: float) -> tuple[str, str] | None:
        if ln.mono or ln.size < body_size * self.cfg.heading_size_ratio:
            return None
        text = _norm(ln.text)
        if len(text) > self.cfg.max_heading_chars or CODE_PREFIX_RE.match(text):
            return None
        m = HEADING_RE.match(text)
        if not m:
            return None
        number = m.group(1)
        parts = number.split(".")
        if int(parts[0]) != chapter or len(parts) > self.cfg.max_section_depth:
            return None
        return number, m.group(2).replace("¶", "").strip()

    def _is_code(self, ln: _Line) -> bool:
        if CODE_PREFIX_RE.match(ln.text.lstrip()) and ln.mono:
            return True
        return ln.mono and ln.x0 >= self.cfg.code_indent_min_x

    # ---------- сборка блоков ----------

    def load(self, doc: DocumentSpec) -> list[Block]:
        path = self.root / doc.path
        lines, n_pages = self._read_lines(path)
        lines = self._drop_page_furniture(lines, n_pages)

        body_size = Counter(round(ln.size, 1) for ln in lines if not ln.mono).most_common(1)[0][0]
        kinds = []
        for ln in lines:
            heading = self._match_heading(ln, doc.chapter, body_size)
            if heading:
                kinds.append(("heading", heading))
            elif self._is_code(ln):
                kinds.append(("code", None))
            else:
                kinds.append(("text", None))

        gap_limit = self._gap_limits(lines, kinds)

        blocks: list[Block] = []
        headings: list[list] = []
        self.headings[doc.doc_id] = headings  # type: ignore[assignment]
        section: str | None = None
        section_title: str | None = None
        cur_kind: str | None = None
        cur_lines: list[str] = []
        cur_page = 0
        prev: _Line | None = None
        prev_kind: str | None = None

        def flush() -> None:
            nonlocal cur_lines, cur_kind
            if cur_lines:
                if cur_kind == "code":
                    text = "\n".join(cur_lines).strip("\n")
                else:
                    text = join_lines(cur_lines)
                if text.strip():
                    blocks.append(
                        Block(
                            doc_id=doc.doc_id,
                            text=text,
                            page=cur_page,
                            section=section,
                            section_title=section_title,
                            kind=cur_kind,
                        )
                    )
            cur_lines, cur_kind = [], None

        for ln, (kind, heading) in zip(lines, kinds, strict=True):
            if kind == "heading":
                flush()
                section, section_title = heading
                headings.append([section, section_title, ln.page])
            elif (
                kind == "text"
                and prev_kind == "heading"
                and prev is not None
                and ln.page == prev.page
                and abs(ln.size - prev.size) < 0.5
            ):
                # Перенос длинного заголовка на вторую строку.
                section_title = f"{section_title} {_norm(ln.text)}".strip()
                headings[-1][1] = section_title
                prev, prev_kind = ln, "heading"
                continue
            else:
                if section is None:
                    section, section_title = str(doc.chapter), None
                same_run = (
                    cur_kind == kind
                    and prev is not None
                    and ln.page == prev.page
                    and ln.y0 - prev.y0 <= gap_limit[kind]
                )
                if not same_run:
                    flush()
                    cur_kind, cur_page = kind, ln.page
                cur_lines.append(ln.text if kind == "code" else ln.text.strip())
            prev, prev_kind = ln, kind
        flush()

        return _fix_cross_page_hyphens(blocks)

    def _gap_limits(self, lines: list[_Line], kinds: list[tuple]) -> dict[str, float]:
        """Порог разрыва между строками для нового абзаца — от медианного интервала."""
        steps: dict[str, list[float]] = {"text": [], "code": []}
        for a, b, (ka, _), (kb, _) in zip(lines, lines[1:], kinds, kinds[1:], strict=False):
            if ka == kb and ka in steps and a.page == b.page and b.y0 > a.y0:
                steps[ka].append(b.y0 - a.y0)
        ratio = {"text": self.cfg.paragraph_gap_ratio, "code": self.cfg.code_gap_ratio}
        return {k: statistics.median(v) * ratio[k] if v else float("inf") for k, v in steps.items()}


def _fix_cross_page_hyphens(blocks: list[Block]) -> list[Block]:
    """Слово, перенесённое через границу страниц, целиком переносится в следующий блок."""
    for a, b in zip(blocks, blocks[1:], strict=False):
        if a.kind == b.kind == "text" and b.page == a.page + 1 and a.text.endswith(SOFT_HYPHENS):
            head, _, fragment = a.text[:-1].rpartition(" ")
            a.text = head
            b.text = fragment + b.text
    return [b for b in blocks if b.text.strip()]
