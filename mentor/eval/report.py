"""Отчёты по прогонам: results.csv (шаблон методики), results.md, comparison.md."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mentor.eval.runner import load_questions


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    with (run_dir / "records.jsonl").open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return config, records


def _pages(c: dict) -> str:
    return (
        str(c["page_from"])
        if c["page_from"] == c["page_to"]
        else f"{c['page_from']}–{c['page_to']}"
    )


def _fragment_label(c: dict) -> str:
    return f"{c['doc_id']} §{c['section']} стр.{_pages(c)} ({c['score']:.3f})"


def _citation_label(c: dict) -> str:
    return f"[{c['n']}] {c['doc_id']} §{c['section']} стр.{_pages(c)}"


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(config: dict, records: list[dict]) -> dict:
    ok = [r for r in records if not r.get("error")]
    by_type = {t: [r for r in ok if r["type"] == t] for t in ("main", "provocation", "multi")}
    rag = config["mode"] == "rag"
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms") is not None]
    answered = by_type["main"] + by_type["multi"]
    summary = {
        "n_questions": len(records),
        "n_errors": len(records) - len(ok),
        "hit_at_k": _rate([r["checks"]["retrieval_hit"] for r in by_type["main"]]) if rag else None,
        "hit_at_k_page": (
            _rate([r["checks"]["retrieval_hit_page"] for r in by_type["main"]]) if rag else None
        ),
        "citation_rate": _rate([bool(r["citations"]) for r in answered]) if rag else None,
        "provocation_refusal_rate": _rate([r["refused"] for r in by_type["provocation"]]),
        "false_refusal_rate": _rate([r["refused"] for r in answered]),
        "latency_mean_ms": int(np.mean(latencies)) if latencies else None,
        "latency_p95_ms": int(np.percentile(latencies, 95)) if latencies else None,
        "multi": {},
    }
    for r in by_type["multi"]:
        ch = r["checks"]
        if rag:
            summary["multi"][r["id"]] = (
                f"документы найдены: {','.join(ch['docs_retrieved']) or '—'}; "
                f"разделы: {ch['sections_coverage']:.0%}; "
                f"процитированы: {','.join(ch['docs_cited']) or '—'}"
            )
    return summary


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.0%}" if v <= 1 else f"{v:.1f}"
    return str(v)


def _checks_hint(checks: dict) -> str:
    return "; ".join(f"{k}={v}" for k, v in checks.items())


def write_csv(run_dir: Path, records: list[dict]) -> Path:
    path = run_dir / "results.csv"
    header = [
        "№",
        "Вопрос",
        "Найденные фрагменты",
        "Источники",
        "Ответ LLM",
        "Оценка",
        "Комментарий / тип ошибки",
        "Подсказки проверок",
        "Предварительные коды",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        for r in records:
            w.writerow(
                [
                    r["id"],
                    r["question"],
                    "\n".join(_fragment_label(c) for c in r["retrieved"]),
                    "\n".join(_citation_label(c) for c in r["citations"]),
                    r["answer_text"] if not r.get("error") else f"ОШИБКА: {r['error']}",
                    "",
                    "",
                    _checks_hint(r["checks"]),
                    ", ".join(r["suggested_error_codes"]),
                ]
            )
    return path


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in (text or "").splitlines()) or "> —"


def write_md(
    run_dir: Path, config: dict, records: list[dict], questions: dict, snippet: int
) -> Path:
    s = summarize(config, records)
    ch = config["settings"]["chunking"]
    host = config["host"]
    embed_model = config["settings"]["ollama"]["embed_model"]
    lines = [
        f"# Прогон {run_dir.name}",
        "",
        f"- Модель: `{config['model']}`, режим: `{config['mode']}`",
        f"- Хост: {host['node']} ({host['system']} {host['machine']}),"
        f" Ollama {config.get('ollama_version') or '?'}",
        f"- top_k: {config['top_k']}; нарезка: max_chars={ch['max_chars']}, "
        f"overlap={ch['overlap_chars']}, min_chars={ch['min_chars']}",
        f"- Эмбеддинги: `{embed_model}`; дата: {config['started_at']}",
        f"- git: `{config.get('git_sha') or '—'}`",
        "",
        "## Сводка",
        "",
        "| Метрика | Значение |",
        "|---|---|",
        f"| Hit@{config['top_k']} (основные, раздел) | {_fmt(s['hit_at_k'])} |",
        f"| Hit@{config['top_k']} (основные, раздел + страница) | {_fmt(s['hit_at_k_page'])} |",
        f"| Доля ответов со ссылкой (основные + мульти) | {_fmt(s['citation_rate'])} |",
        f"| Доля провокаций с отказом | {_fmt(s['provocation_refusal_rate'])} |",
        f"| Доля ложных отказов (основные + мульти) | {_fmt(s['false_refusal_rate'])} |",
        f"| Время ответа, среднее / p95, мс | {_fmt(s['latency_mean_ms'])} / "
        f"{_fmt(s['latency_p95_ms'])} |",
        f"| Ошибки выполнения | {s['n_errors']} из {s['n_questions']} |",
    ]
    for qid, text in s["multi"].items():
        lines.append(f"| {qid} | {text} |")
    lines += ["", "## Вопросы", ""]
    for r in records:
        q = questions.get(r["id"], {})
        reference = q.get("reference") or q.get("expected") or "—"
        lines += [
            f"### {r['id']}. {r['question']}",
            "",
            f"*Тип:* {r['type']}"
            + (f" · *категория:* {q['category']}" if q.get("category") else ""),
            "",
            f"**Эталон:** {reference}",
            "",
        ]
        if q.get("gold"):
            gold = ", ".join(
                f"{g['doc_id']} §{g['section']}" + (f" стр.{g['pages']}" if g.get("pages") else "")
                for g in q["gold"]
            )
            lines += [f"**Эталонные источники:** {gold}", ""]
        if r.get("error"):
            lines += [f"**ОШИБКА:** `{r['error']}`", ""]
            continue
        lines += [
            f"**Ответ модели** ({r['latency_ms']} мс, отказ: {r['refused']}):",
            "",
            _quote(r["answer_text"]),
            "",
            "**Ссылки:** "
            + (", ".join(_citation_label(c) for c in r["citations"]) or "—")
            + (
                f"; невалидные номера: {r['invalid_citation_numbers']}"
                if r["invalid_citation_numbers"]
                else ""
            ),
            "",
        ]
        if r["retrieved"]:
            lines += ["**Найденные фрагменты:**", ""]
            for i, c in enumerate(r["retrieved"], start=1):
                preview = " ".join(c["text"].split())[:snippet]
                lines.append(
                    f"{i}. `{c['chunk_id']}` {c['doc_id']} §{c['section']} "
                    f"«{c['section_title'] or ''}», стр. {_pages(c)}, "
                    f"score {c['score']:.3f} — {preview}…"
                )
            lines.append("")
        lines += [
            f"**Проверки:** {_checks_hint(r['checks'])}",
            "",
            f"**Предварительные коды:** {', '.join(r['suggested_error_codes']) or '—'}",
            "",
        ]
    path = run_dir / "results.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _mark(v) -> str:
    return "—" if v is None else ("✓" if v else "✗")


def write_comparison(runs: list[tuple[Path, dict, list[dict]]], out: Path) -> Path:
    lines = [
        "# Сравнение прогонов",
        "",
        "| Прогон | Модель | Режим | top_k | Hit@k | Hit@k+стр | Со ссылкой | Отказ на провокации |"
        " Ложные отказы | Время ср. / p95, мс | Ошибки |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run_dir, config, records in runs:
        s = summarize(config, records)
        lines.append(
            f"| {run_dir.name} | {config['model']} | {config['mode']} | {config['top_k']} | "
            f"{_fmt(s['hit_at_k'])} | {_fmt(s['hit_at_k_page'])} | {_fmt(s['citation_rate'])} | "
            f"{_fmt(s['provocation_refusal_rate'])} | {_fmt(s['false_refusal_rate'])} | "
            f"{_fmt(s['latency_mean_ms'])} / {_fmt(s['latency_p95_ms'])} | {s['n_errors']} |"
        )
    multi_rows = [
        (run_dir.name, qid, text)
        for run_dir, config, records in runs
        for qid, text in summarize(config, records)["multi"].items()
    ]
    if multi_rows:
        lines += [
            "",
            "## Мультидокументные вопросы",
            "",
            "| Прогон | Вопрос | Покрытие |",
            "|---|---|---|",
        ]
        lines += [f"| {a} | {b} | {c} |" for a, b, c in multi_rows]

    lines += [
        "",
        "## Вопрос × прогон",
        "",
        "Отметки: **Н** — эталон найден поиском (для M — найдены все документы), "
        "**С** — есть ссылка, **О** — отказ. ✓ да, ✗ нет, — неприменимо; "
        "`ERR` — ошибка выполнения.",
        "",
        "| № | " + " | ".join(f"{i + 1}" for i in range(len(runs))) + " |",
        "|---|" + "---|" * len(runs),
    ]
    by_run = [{r["id"]: r for r in records} for _, _, records in runs]
    ids = list(dict.fromkeys(r["id"] for _, _, records in runs for r in records))
    for qid in ids:
        cells = []
        for recs in by_run:
            r = recs.get(qid)
            if r is None:
                cells.append(" ")
            elif r.get("error"):
                cells.append("ERR")
            else:
                ch = r["checks"]
                found = ch.get("retrieval_hit", ch.get("all_docs_retrieved"))
                cite = (
                    None
                    if r["type"] == "provocation" or not r["retrieved"]
                    else bool(r["citations"])
                )
                cells.append(f"Н{_mark(found)} С{_mark(cite)} О{_mark(r['refused'])}")
        lines.append(f"| {qid} | " + " | ".join(cells) + " |")
    lines += ["", "Прогоны:", ""]
    lines += [f"{i + 1}. `{run_dir.name}`" for i, (run_dir, _, _) in enumerate(runs)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def build_reports(run_dirs: list[Path], questions_path: Path, snippet: int) -> list[Path]:
    questions = {str(q["id"]): q for q in load_questions(questions_path)}
    written: list[Path] = []
    runs = []
    for run_dir in run_dirs:
        config, records = load_run(run_dir)
        runs.append((run_dir, config, records))
        written.append(write_csv(run_dir, records))
        written.append(write_md(run_dir, config, records, questions, snippet))
    if len(runs) > 1:
        written.append(write_comparison(runs, run_dirs[0].parent / "comparison.md"))
    return written
