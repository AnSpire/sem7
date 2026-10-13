"""Загрузка config.yaml в pydantic-модели."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG = Path("config.yaml")

# Файлы, которые нельзя загружать в базу знаний: в них эталонные ответы.
FORBIDDEN_KB_FILES = ("Otsenka_raboty_sistemy.pdf", "mentor/eval/questions.yaml")


class DocumentSpec(BaseModel):
    doc_id: str
    title: str
    path: str
    chapter: int


class KnowledgeBaseConfig(BaseModel):
    kb_id: str
    documents: list[DocumentSpec]


class ParsingConfig(BaseModel):
    repeated_line_ratio: float = 0.5
    max_heading_chars: int = 120
    max_section_depth: int = 4
    heading_size_ratio: float = 1.1
    code_indent_min_x: float = 44.0
    paragraph_gap_ratio: float = 1.35
    code_gap_ratio: float = 2.5


class ChunkingConfig(BaseModel):
    max_chars: int = 1800
    overlap_chars: int = 250
    min_chars: int = 300
    heading_prefix: bool = True


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embed_batch_size: int = 16
    llm_model: str = "qwen3:8b"
    num_ctx: int = 8192
    temperature: float = 0.1
    think: bool = False
    timeout_s: float = 300


class RetrievalConfig(BaseModel):
    top_k: int = 6
    min_score: float | None = None


class EvalConfig(BaseModel):
    questions: str = "mentor/eval/questions.yaml"
    snippet_chars_cli: int = 200
    snippet_chars_report: int = 300


class PathsConfig(BaseModel):
    parsed_dir: str = "data/parsed"
    index_dir: str = "data/index"
    runs_dir: str = "runs"


class Settings(BaseModel):
    knowledge_base: KnowledgeBaseConfig
    parsing: ParsingConfig = ParsingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    ollama: OllamaConfig = OllamaConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    eval: EvalConfig = EvalConfig()
    paths: PathsConfig = PathsConfig()
    # Корень проекта: относительно него разрешаются все пути. Не читается из YAML.
    root: Path = Path(".")

    def resolve(self, relative: str) -> Path:
        return (self.root / relative).resolve()

    @property
    def parsed_dir(self) -> Path:
        return self.resolve(self.paths.parsed_dir)

    @property
    def index_dir(self) -> Path:
        return self.resolve(self.paths.index_dir)

    @property
    def runs_dir(self) -> Path:
        return self.resolve(self.paths.runs_dir)

    @property
    def questions_path(self) -> Path:
        return self.resolve(self.eval.questions)


def load_settings(path: Path = DEFAULT_CONFIG) -> Settings:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    settings = Settings.model_validate(raw)
    settings.root = path.resolve().parent
    return settings


class ForbiddenDocumentError(ValueError):
    pass


def ensure_allowed_documents(settings: Settings) -> None:
    """Падает, если в базу знаний попал файл с эталонами."""
    forbidden = {settings.resolve(f) for f in FORBIDDEN_KB_FILES}
    forbidden_names = {Path(f).name.lower() for f in FORBIDDEN_KB_FILES}
    for doc in settings.knowledge_base.documents:
        p = settings.resolve(doc.path)
        if p in forbidden or p.name.lower() in forbidden_names:
            raise ForbiddenDocumentError(
                f"Файл {doc.path!r} (doc_id={doc.doc_id}) нельзя загружать в базу знаний: "
                "в нём эталонные ответы методики тестирования. Уберите его из "
                "knowledge_base.documents в config.yaml."
            )
