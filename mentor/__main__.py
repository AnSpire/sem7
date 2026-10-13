"""CLI наставника. Здесь же — единственное место сборки адаптеров."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from mentor.adapters.chunker import SectionChunker
from mentor.adapters.numpy_store import IndexMismatchError, NumpyVectorStore
from mentor.adapters.ollama import (
    OllamaEmbedder,
    OllamaLLM,
    list_models,
    model_available,
    server_version,
)
from mentor.adapters.pdf_loader import PyMuPdfLoader
from mentor.config import (
    DEFAULT_CONFIG,
    ForbiddenDocumentError,
    Settings,
    ensure_allowed_documents,
    load_settings,
)
from mentor.domain import Answer, ScoredChunk
from mentor.services.answer import AnswerService, format_pages
from mentor.services.ingest import IngestService, build_manifest, sha256_file

app = typer.Typer(help="ИИ-наставник: пилотный RAG по материалам курса.", no_args_is_help=True)
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Путь к config.yaml")]
ModelOpt = Annotated[str | None, typer.Option("--model", help="LLM-модель Ollama")]
TopKOpt = Annotated[int | None, typer.Option("--top-k", help="Сколько фрагментов искать")]
MinScoreOpt = Annotated[float | None, typer.Option("--min-score", help="Порог отказа по score")]


# ---------- фабрика ----------


def _settings(
    config: Path, model: str | None = None, top_k: int | None = None, min_score: float | None = None
) -> Settings:
    s = load_settings(config)
    if model:
        s.ollama.llm_model = model
    if top_k:
        s.retrieval.top_k = top_k
    if min_score is not None:
        s.retrieval.min_score = min_score
    return s


def make_embedder(s: Settings) -> OllamaEmbedder:
    o = s.ollama
    return OllamaEmbedder(o.base_url, o.embed_model, o.embed_batch_size, o.timeout_s)


def make_llm(s: Settings) -> OllamaLLM:
    o = s.ollama
    return OllamaLLM(o.base_url, o.llm_model, o.num_ctx, o.temperature, o.think, o.timeout_s)


def make_store(s: Settings) -> NumpyVectorStore:
    return NumpyVectorStore(s.index_dir, build_manifest(s))


def make_answer_service(s: Settings) -> AnswerService:
    store = make_store(s)
    try:
        store.load()
    except IndexMismatchError as e:
        console.print(f"[red]{e}")
        raise typer.Exit(1) from e
    return AnswerService(
        embedder=make_embedder(s),
        store=store,
        llm=make_llm(s),
        model_name=s.ollama.llm_model,
        top_k=s.retrieval.top_k,
        min_score=s.retrieval.min_score,
    )


def _require_ollama(s: Settings, models: list[str]) -> None:
    try:
        available = list_models(s.ollama.base_url)
    except httpx.HTTPError as e:
        console.print(
            f"[red]Ollama недоступна по {s.ollama.base_url} ({e}).[/] "
            "Установите и запустите Ollama: "
            "https://ollama.com/download (на macOS: `brew install ollama && ollama serve`)."
        )
        raise typer.Exit(1) from e
    missing = [m for m in models if not model_available(m, available)]
    if missing:
        console.print("[red]Не хватает моделей Ollama. Выполните:[/]")
        for m in missing:
            console.print(f"  ollama pull {m}")
        raise typer.Exit(1)


# ---------- вывод ----------


def _print_chunks(retrieved: list[ScoredChunk], snippet: int) -> None:
    table = Table(show_lines=True)
    for col in ("#", "score", "док", "раздел", "стр.", "текст"):
        table.add_column(col)
    for i, sc in enumerate(retrieved, start=1):
        c = sc.chunk
        table.add_row(
            str(i),
            f"{sc.score:.3f}",
            c.doc_id,
            f"{c.section} {c.section_title or ''}",
            format_pages(c.page_from, c.page_to),
            " ".join(c.text.split())[:snippet],
        )
    console.print(table)


def _print_answer(a: Answer, snippet: int) -> None:
    console.rule(f"[bold]{a.model} · {a.mode} · {a.latency_ms} мс")
    console.print(a.answer_text)
    console.print(f"\n[bold]Отказ:[/] {a.refused}   [bold]top_score:[/] {a.top_score}")
    if a.citations:
        console.print("[bold]Ссылки:[/]")
        for c in a.citations:
            console.print(
                f"  [{c.n}] {c.doc_title} — раздел {c.section}, стр. "
                f"{format_pages(c.page_from, c.page_to)} ({c.chunk_id})"
            )
    if a.invalid_citation_numbers:
        console.print(f"[yellow]Невалидные номера ссылок: {a.invalid_citation_numbers}")
    if a.retrieved:
        console.print("[bold]Найденные фрагменты:[/]")
        _print_chunks(a.retrieved, snippet)


# ---------- команды ----------


@app.command()
def doctor(config: ConfigOpt = DEFAULT_CONFIG, model: ModelOpt = None) -> None:
    """Проверить, что Ollama запущена и нужные модели скачаны."""
    s = _settings(config, model)
    _require_ollama(s, [s.ollama.embed_model, s.ollama.llm_model])
    console.print(
        f"[green]Ollama {server_version(s.ollama.base_url)} доступна, модели "
        f"{s.ollama.embed_model}, {s.ollama.llm_model} на месте."
    )


@app.command()
def ingest(
    config: ConfigOpt = DEFAULT_CONFIG,
    parse_only: Annotated[
        bool, typer.Option("--parse-only", help="Только парсинг и нарезка, без эмбеддингов")
    ] = False,
) -> None:
    """Парсинг PDF → нарезка → эмбеддинги → сохранение индекса."""
    s = _settings(config)
    try:
        ensure_allowed_documents(s)
    except ForbiddenDocumentError as e:
        console.print(f"[red]Ошибка: {e}")
        raise typer.Exit(2) from e
    loader = PyMuPdfLoader(s.root, s.parsing)
    chunker = SectionChunker(s.chunking)
    docs = s.knowledge_base.documents
    if parse_only:
        service = IngestService(loader, chunker, None, None, s.parsed_dir)
        result = service.parse(docs)
    else:
        _require_ollama(s, [s.ollama.embed_model])
        service = IngestService(loader, chunker, make_embedder(s), make_store(s), s.parsed_dir)
        with console.status("Парсинг, нарезка и эмбеддинги…"):
            result = service.run(docs)

    table = Table(title=f"База знаний «{s.knowledge_base.kb_id}»")
    for col in ("doc_id", "блоков", "разделов", "чанков", "ср. длина", "мин", "макс"):
        table.add_column(col, justify="right")
    for doc in docs:
        blocks, chunks = result.blocks[doc.doc_id], result.chunks[doc.doc_id]
        lengths = [len(c.text) for c in chunks]
        table.add_row(
            doc.doc_id,
            str(len(blocks)),
            str(len(loader.headings[doc.doc_id])),
            str(len(chunks)),
            f"{mean(lengths):.0f}",
            str(min(lengths)),
            str(max(lengths)),
        )
    console.print(table)
    console.print(f"Разобранный текст: {s.parsed_dir}")
    if not parse_only:
        console.print(f"[green]Индекс сохранён: {s.index_dir}")


@app.command()
def search(
    query: str,
    config: ConfigOpt = DEFAULT_CONFIG,
    top_k: Annotated[int, typer.Option("--top-k")] = 3,
) -> None:
    """Поиск без LLM: показать top-k фрагментов (smoke-проверка индекса)."""
    s = _settings(config)
    _require_ollama(s, [s.ollama.embed_model])
    service = make_answer_service(s)
    _print_chunks(service.retrieve(query, top_k), s.eval.snippet_chars_cli)


@app.command()
def ask(
    question: str,
    config: ConfigOpt = DEFAULT_CONFIG,
    model: ModelOpt = None,
    top_k: TopKOpt = None,
    min_score: MinScoreOpt = None,
    no_rag: Annotated[bool, typer.Option("--no-rag", help="Голая модель без фрагментов")] = False,
) -> None:
    """Задать вопрос наставнику."""
    s = _settings(config, model, top_k, min_score)
    if no_rag:
        _require_ollama(s, [s.ollama.llm_model])
        answer = AnswerService(
            make_embedder(s), make_store(s), make_llm(s), s.ollama.llm_model, s.retrieval.top_k
        ).ask_no_rag(question)
    else:
        _require_ollama(s, [s.ollama.embed_model, s.ollama.llm_model])
        answer = make_answer_service(s).ask(question)
    _print_answer(answer, s.eval.snippet_chars_cli)


@app.command(name="eval")
def eval_(
    config: ConfigOpt = DEFAULT_CONFIG,
    model: ModelOpt = None,
    top_k: TopKOpt = None,
    min_score: MinScoreOpt = None,
    no_rag: Annotated[bool, typer.Option("--no-rag")] = False,
    only: Annotated[str | None, typer.Option("--only", help="Например: 1,5,M1")] = None,
    tag: Annotated[str | None, typer.Option("--tag", help="Метка машины, например mac-m4")] = None,
) -> None:
    """Прогнать контрольный набор вопросов."""
    from mentor.eval.report import summarize
    from mentor.eval.runner import git_sha, host_info, load_questions, make_run_dir, run_eval

    s = _settings(config, model, top_k, min_score)
    mode = "no_rag" if no_rag else "rag"
    questions = load_questions(s.questions_path)
    if only:
        wanted = {x.strip() for x in only.split(",")}
        questions = [q for q in questions if str(q["id"]) in wanted]

    if no_rag:
        _require_ollama(s, [s.ollama.llm_model])
        service = AnswerService(
            make_embedder(s), make_store(s), make_llm(s), s.ollama.llm_model, s.retrieval.top_k
        )
        ask_fn = service.ask_no_rag
        manifest = None
    else:
        _require_ollama(s, [s.ollama.embed_model, s.ollama.llm_model])
        service = make_answer_service(s)
        ask_fn = service.ask
        manifest = json.loads((s.index_dir / "manifest.json").read_text(encoding="utf-8"))

    run_dir = make_run_dir(s.runs_dir, s.ollama.llm_model, mode, tag)
    run_config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": s.ollama.llm_model,
        "mode": mode,
        "top_k": s.retrieval.top_k,
        "tag": tag,
        "only": only,
        "settings": s.model_dump(mode="json", exclude={"root"}),
        "questions_path": str(s.questions_path),
        "questions_sha256": sha256_file(s.questions_path),
        "index_manifest": manifest,
        "git_sha": git_sha(s.root),
        "host": host_info(),
        "ollama_version": server_version(s.ollama.base_url),
    }
    records = run_eval(
        ask_fn,
        questions,
        run_dir=run_dir,
        run_config=run_config,
        model=s.ollama.llm_model,
        mode=mode,
        console=console,
    )

    summary = summarize(run_config, [r.model_dump(mode="json") for r in records])
    table = Table(title=f"Сводка: {run_dir.name}")
    table.add_column("метрика")
    table.add_column("значение")
    for k, v in summary.items():
        if k != "multi":
            table.add_row(k, "—" if v is None else str(v))
    for k, v in summary["multi"].items():
        table.add_row(k, v)
    console.print(table)
    console.print(f"Результаты: {run_dir}")


@app.command()
def report(
    run_dirs: Annotated[list[Path], typer.Argument(help="Папки прогонов (runs/*)")],
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Сформировать results.csv / results.md (и comparison.md для нескольких прогонов)."""
    from mentor.eval.report import build_reports

    s = _settings(config)
    dirs = sorted(d for d in run_dirs if (d / "records.jsonl").exists())
    if not dirs:
        console.print("[red]Не найдено ни одной папки прогона с records.jsonl")
        raise typer.Exit(1)
    for p in build_reports(dirs, s.questions_path, s.eval.snippet_chars_report):
        console.print(f"записан {p}")


if __name__ == "__main__":
    app()
