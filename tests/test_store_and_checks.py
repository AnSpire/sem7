import numpy as np
import pytest

from mentor.adapters.numpy_store import IndexMismatchError, NumpyVectorStore
from mentor.domain import Chunk
from mentor.eval.checks import chunk_matches, run_checks, section_match


def _chunk(cid: str, doc: str, section: str, pages=(1, 1), sections=None) -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        doc_title=doc,
        section=section,
        section_title=None,
        page_from=pages[0],
        page_to=pages[1],
        text=f"текст {cid}",
        text_for_embedding=f"текст {cid}",
        sections=sections or [section],
    )


MANIFEST = {"kb_id": "t", "embed_model": "m", "documents": [], "chunking": {}, "parsing": {}}


def test_store_roundtrip_and_search(tmp_path):
    store = NumpyVectorStore(tmp_path, MANIFEST)
    vecs = np.eye(3, dtype=np.float32)
    store.add([_chunk(f"T-{i}", "T", "9") for i in range(3)], vecs)
    store.save()

    loaded = NumpyVectorStore(tmp_path, MANIFEST)
    loaded.load()
    hits = loaded.search(np.array([0.1, 0.9, 0.0]), top_k=2)
    assert [h.chunk.chunk_id for h in hits] == ["T-1", "T-0"]
    assert hits[0].score == pytest.approx(0.9)


def test_store_detects_manifest_mismatch(tmp_path):
    store = NumpyVectorStore(tmp_path, MANIFEST)
    store.add([_chunk("T-0", "T", "9")], np.eye(1, dtype=np.float32))
    store.save()
    changed = {**MANIFEST, "embed_model": "другая"}
    with pytest.raises(IndexMismatchError, match="uv run mentor ingest"):
        NumpyVectorStore(tmp_path, changed).load()


def test_store_missing_index(tmp_path):
    with pytest.raises(IndexMismatchError, match="uv run mentor ingest"):
        NumpyVectorStore(tmp_path / "nope", MANIFEST).load()


@pytest.mark.parametrize(
    ("chunk_sec", "gold", "expected"),
    [
        ("9.3.1", "9.3.1", True),
        ("9.3.1", "9.3", True),
        ("9.3", "9.3.1", True),
        ("9.2", "9", True),
        ("9.3", "9.30", False),
        ("4.3", "4.4", False),
        (None, "9", False),
    ],
)
def test_section_match(chunk_sec, gold, expected):
    assert section_match(chunk_sec, gold) is expected


def test_chunk_matches_uses_all_sections_and_pages():
    c = _chunk("P4-1", "P4", "4.3", pages=(2, 3), sections=["4.3", "4.4"]).model_dump()
    assert chunk_matches(c, {"doc_id": "P4", "section": "4.4", "pages": [3]}, check_pages=True)
    assert not chunk_matches(c, {"doc_id": "P4", "section": "4.4", "pages": [5]}, check_pages=True)
    assert chunk_matches(c, {"doc_id": "P4", "section": "4.4", "pages": []}, check_pages=True)
    assert not chunk_matches(c, {"doc_id": "P5", "section": "4.4", "pages": []})


def test_run_checks_main_codes():
    q = {"type": "main", "gold": [{"doc_id": "P9", "section": "9.3.2", "pages": [6]}]}
    gold = {**_chunk("P9-1", "P9", "9.3.2", (5, 6)).model_dump(), "score": 0.6}
    other = {**_chunk("P4-1", "P4", "4.3").model_dump(), "score": 0.5}
    cite_other = [{"chunk_id": "P4-1"}]

    checks, codes = run_checks(
        q, mode="rag", retrieved=[other, gold], citations=cite_other, refused=False
    )
    assert checks["retrieval_hit"] and checks["retrieval_hit_page"]
    assert checks["gold_rank"] == 2
    assert not checks["cites_gold"]
    assert codes == ["S2"]

    checks, codes = run_checks(q, mode="rag", retrieved=[other], citations=[], refused=False)
    assert codes == ["R1", "R2", "S1"]


def test_run_checks_provocation_and_multi():
    checks, codes = run_checks(
        {"type": "provocation"}, mode="rag", retrieved=[], citations=[], refused=False
    )
    assert checks == {"refused": False, "pass": False} and codes == ["G2?"]

    q = {
        "type": "multi",
        "gold": [
            {"doc_id": "P4", "section": "4.2", "pages": []},
            {"doc_id": "P5", "section": "5.6", "pages": []},
        ],
    }
    r = [{**_chunk("P4-1", "P4", "4.2").model_dump(), "score": 0.5}]
    checks, codes = run_checks(
        q, mode="rag", retrieved=r, citations=[{"chunk_id": "P4-1"}], refused=False
    )
    assert checks["docs_retrieved"] == ["P4"] and not checks["all_docs_retrieved"]
    assert checks["sections_coverage"] == 0.5
    assert codes == ["R1", "S1"]
