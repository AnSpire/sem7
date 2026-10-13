"""Индекс не должен содержать методику тестирования и эталонные ответы."""

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mentor.__main__ import app
from mentor.config import (
    FORBIDDEN_KB_FILES,
    ForbiddenDocumentError,
    ensure_allowed_documents,
    load_settings,
)

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = load_settings(ROOT / "config.yaml")
ALLOWED_DOC_IDS = {"P4", "P5", "P9"}
METHODOLOGY_MARKERS = (
    "Эталонный ответ",
    "Критерий ошибки",
    "Провокационный вопрос",
    "Ожидаемое поведение",
)

index_required = pytest.mark.skipif(
    not (SETTINGS.index_dir / "manifest.json").exists(),
    reason="индекс не собран: uv run mentor ingest",
)


def _manifest() -> dict:
    return json.loads((SETTINGS.index_dir / "manifest.json").read_text(encoding="utf-8"))


def _chunks() -> list[dict]:
    with (SETTINGS.index_dir / "chunks.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_config_lists_only_pilot_documents():
    assert {d.doc_id for d in SETTINGS.knowledge_base.documents} == ALLOWED_DOC_IDS
    ensure_allowed_documents(SETTINGS)


@index_required
def test_manifest_contains_only_allowed_documents():
    manifest = _manifest()
    assert {d["doc_id"] for d in manifest["documents"]} <= ALLOWED_DOC_IDS
    raw = json.dumps(manifest, ensure_ascii=False)
    for forbidden in ("Otsenka_raboty_sistemy.pdf", "questions.yaml"):
        assert forbidden not in raw


@index_required
def test_chunks_contain_only_allowed_documents():
    chunks = _chunks()
    assert chunks
    assert {c["doc_id"] for c in chunks} <= ALLOWED_DOC_IDS


@index_required
def test_chunks_have_no_methodology_markers():
    for c in _chunks():
        for marker in METHODOLOGY_MARKERS:
            assert marker.lower() not in c["text"].lower(), (c["chunk_id"], marker)
            assert marker.lower() not in c["text_for_embedding"].lower(), (c["chunk_id"], marker)


@index_required
def test_reference_answers_not_in_index():
    """Эталонные формулировки, которых нет в документации, не должны встречаться в чанках."""
    questions = yaml.safe_load(SETTINGS.questions_path.read_text(encoding="utf-8"))
    texts = " ".join(c["text"] for c in _chunks())
    for q in questions:
        if q["type"] == "provocation":
            assert q["expected"] not in texts


@pytest.mark.parametrize("forbidden", FORBIDDEN_KB_FILES)
def test_ingest_rejects_forbidden_files(tmp_path, forbidden):
    cfg = {
        "knowledge_base": {
            "kb_id": "leak",
            "documents": [{"doc_id": "X", "title": "X", "path": forbidden, "chapter": 1}],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ForbiddenDocumentError):
        ensure_allowed_documents(load_settings(config_path))

    result = CliRunner().invoke(app, ["ingest", "--config", str(config_path), "--parse-only"])
    assert result.exit_code != 0
    assert "нельзя загружать в базу знаний" in " ".join(result.output.split())
    assert not (tmp_path / "data").exists()


def test_answer_service_does_not_know_reference():
    source = (ROOT / "mentor" / "services" / "answer.py").read_text(encoding="utf-8")
    assert "reference" not in source
    assert "gold" not in source
