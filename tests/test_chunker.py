import re

import pytest

from mentor.adapters.chunker import SectionChunker
from mentor.config import ChunkingConfig, DocumentSpec
from mentor.domain import Block

CFG = ChunkingConfig(max_chars=600, overlap_chars=100, min_chars=150, heading_prefix=True)
TOLERANCE = CFG.min_chars  # склейка короткого хвоста может немного превысить max_chars
DOC = DocumentSpec(doc_id="T", title="Тест", path="t.pdf", chapter=9)


def _sentences(n: int, prefix: str) -> str:
    return " ".join(f"{prefix} предложение номер {i} про классы и объекты." for i in range(n))


def _b(text: str, page: int, section: str, kind: str = "text") -> Block:
    return Block(
        doc_id="T",
        text=text,
        page=page,
        section=section,
        section_title=f"Раздел {section}",
        kind=kind,
    )


def _blocks() -> list[Block]:
    code = "\n".join(f">>> x{i} = {i}" for i in range(20))  # ~250 символов
    return [
        _b(_sentences(3, "Введение"), 1, "9"),
        _b(_sentences(8, "Альфа"), 1, "9.1"),
        _b(_sentences(8, "Бета"), 2, "9.1"),
        _b(code, 2, "9.1", kind="code"),
        _b(_sentences(30, "Гамма"), 3, "9.2"),
        _b("Короткий хвост.", 4, "9.2"),
        _b(_sentences(5, "Дельта"), 4, "9.3"),
    ]


@pytest.fixture
def chunks():
    return SectionChunker(CFG).split(DOC, _blocks())


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_chunk_size_limit(chunks):
    assert chunks
    for c in chunks:
        assert len(c.text) <= CFG.max_chars + TOLERANCE, c.chunk_id


def test_no_text_lost(chunks):
    """Каждое предложение и каждая строка кода исходных блоков есть хотя бы в одном чанке."""
    joined = " ".join(_flat(c.text) for c in chunks)
    for b in _blocks():
        pieces = b.text.split("\n") if b.kind == "code" else re.split(r"(?<=\.)\s+", b.text)
        for piece in pieces:
            assert _flat(piece) in joined, piece


def test_page_metadata(chunks):
    for b in _blocks():
        first_sentence = _flat(re.split(r"(?<=\.)\s+", b.text)[0])
        holders = [c for c in chunks if first_sentence in _flat(c.text)]
        assert holders, first_sentence
        assert any(c.page_from <= b.page <= c.page_to for c in holders)
    for c in chunks:
        assert c.page_from <= c.page_to


def test_code_block_not_split(chunks):
    code = next(b for b in _blocks() if b.kind == "code")
    assert any(code.text in c.text for c in chunks)


def test_ids_prefix_and_sections(chunks):
    assert [c.chunk_id for c in chunks] == [f"T-{i:04d}" for i in range(1, len(chunks) + 1)]
    for c in chunks:
        assert c.section == c.sections[0]
        assert c.text_for_embedding.startswith(f"Тест › {c.section}")
        assert not c.text.startswith("Тест ›")


def test_short_tail_merges_within_same_section(chunks):
    tail = [c for c in chunks if "Короткий хвост." in c.text]
    assert tail and all("9.2" in c.sections for c in tail)
    assert all(c.section == "9.2" for c in tail)


def test_overlap_between_consecutive_chunks_of_long_section(chunks):
    sec = [c for c in chunks if c.sections == ["9.2"]]
    assert len(sec) >= 2
    a, b = sec[0], sec[1]
    head = _flat(b.text)[:50]
    assert head in _flat(a.text), "следующий чанк начинается с перекрытия"
