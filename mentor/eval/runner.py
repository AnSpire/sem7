"""Прогон контрольного набора: каждый вопрос отдельно, без истории."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from mentor.domain import Answer, EvalRecord
from mentor.eval.checks import run_checks


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _slug(value: str) -> str:
    # «:» недопустим в именах файлов Windows.
    return re.sub(r"[^\w.\-]+", "-", value).strip("-")


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def make_run_dir(runs_dir: Path, model: str, mode: str, tag: str | None) -> Path:
    name = "_".join(
        p
        for p in (datetime.now().strftime("%Y%m%d-%H%M"), _slug(model), mode, tag and _slug(tag))
        if p
    )
    run_dir = runs_dir / name
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = runs_dir / f"{name}-{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def retrieved_to_dicts(answer: Answer) -> list[dict]:
    out = []
    for sc in answer.retrieved:
        c = sc.chunk
        out.append(
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "section": c.section,
                "section_title": c.section_title,
                "sections": c.sections,
                "page_from": c.page_from,
                "page_to": c.page_to,
                "score": round(sc.score, 4),
                "text": c.text,
            }
        )
    return out


def run_eval(
    ask: Callable[[str], Answer],
    questions: list[dict],
    *,
    run_dir: Path,
    run_config: dict,
    model: str,
    mode: str,
    console: Console,
) -> list[EvalRecord]:
    """`ask` получает только текст вопроса — эталон в сервис не передаётся."""
    (run_dir / "config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    records: list[EvalRecord] = []
    records_path = run_dir / "records.jsonl"
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    with progress, records_path.open("w", encoding="utf-8") as f:
        task = progress.add_task("eval", total=len(questions))
        for q in questions:
            progress.update(task, description=f"вопрос {q['id']}")
            rec = EvalRecord(
                id=str(q["id"]), type=q["type"], question=q["question"], model=model, mode=mode
            )
            try:
                answer = ask(q["question"])
                retrieved = retrieved_to_dicts(answer)
                citations = [c.model_dump() for c in answer.citations]
                checks, codes = run_checks(
                    q, mode=mode, retrieved=retrieved, citations=citations, refused=answer.refused
                )
                rec = rec.model_copy(
                    update={
                        "answer_text": answer.answer_text,
                        "refused": answer.refused,
                        "citations": answer.citations,
                        "invalid_citation_numbers": answer.invalid_citation_numbers,
                        "retrieved": retrieved,
                        "top_score": answer.top_score,
                        "latency_ms": answer.latency_ms,
                        "checks": checks,
                        "suggested_error_codes": codes,
                    }
                )
            except Exception as e:  # noqa: BLE001 — ошибка одного вопроса не валит прогон
                rec.error = f"{type(e).__name__}: {e}"
                console.print(f"[red]вопрос {q['id']}: {rec.error}")
            records.append(rec)
            f.write(rec.model_dump_json() + "\n")
            f.flush()
            progress.advance(task)
    return records


def host_info() -> dict:
    return {
        "node": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
