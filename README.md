# mentor — ИИ-наставник (этап 1: пилотный RAG)

Наставник отвечает студенту строго по загруженным материалам курса со ссылками `[n]` на фрагменты.
Если ответа в материалах нет, он отвечает «В материалах курса этого нет.»

Конвейер: PDF → блоки (страница, раздел) → чанки → эмбеддинги `bge-m3` → файловый индекс (numpy) →
поиск top-k → ответ локальной LLM через Ollama → разбор ссылок. Логика сервисов зависит только от
интерфейсов в `mentor/ports.py`; адаптеры собираются в `mentor/__main__.py`.

## Установка

1. Установите [uv](https://docs.astral.sh/uv/getting-started/installation/):
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Разверните проект: `uv sync` (Python 3.12 скачается сам, зависимости — из `uv.lock`).
3. Установите [Ollama](https://ollama.com/download) нативно (не в Docker: на Mac Docker не даёт доступа
   к GPU Apple Silicon). macOS: `brew install ollama && brew services start ollama`.
4. Скачайте модели:
   ```bash
   ollama pull bge-m3      # эмбеддинги
   ollama pull qwen3:8b    # LLM (Mac M4 / RTX 2060)
   ollama pull qwen3:4b    # вторая LLM; на Mac с 24 ГБ+ можно qwen3:30b-a3b
   ```
5. Проверьте: `uv run mentor doctor` (или `uv run mentor doctor --model qwen3:4b`).

Зависимости добавляются только через `uv add` (`uv add --dev` для инструментов). `pip`, `venv`,
`requirements.txt` не используются.

### macOS + iCloud «Документы»

Если проект лежит в синхронизируемой с iCloud папке, iCloud ставит флаг `hidden` на файлы `.venv`,
а Python 3.12 игнорирует скрытые `.pth` — `uv run mentor` падает с `No module named 'mentor'`.
Решение — держать окружение в папке `.nosync`, которую iCloud не трогает:

```bash
rm -rf .venv .venv.nosync && mkdir .venv.nosync && ln -s .venv.nosync .venv && uv sync
```

## Команды

```bash
uv run mentor ingest                 # парсинг → нарезка → эмбеддинги → data/index/
uv run mentor ingest --parse-only    # только парсинг и нарезка (data/parsed/*.md для проверки)
uv run mentor search "метод __init__"           # поиск без LLM (top-3)
uv run mentor ask "Что такое множество?"        # ответ со ссылками и найденными фрагментами
uv run mentor ask "Что такое множество?" --model qwen3:4b --top-k 8
uv run mentor ask "Кто создал Python?" --no-rag # голая модель без материалов
uv run mentor eval --model qwen3:8b --tag mac-m4             # 22 контрольных вопроса
uv run mentor eval --model qwen3:8b --no-rag --tag mac-m4
uv run mentor eval --only 1,5,M1 --top-k 8
uv run mentor report runs/*          # results.csv, results.md в каждом прогоне + runs/comparison.md
```

Любой параметр из `config.yaml` можно переопределить: `--model`, `--top-k`, `--min-score`,
`--config путь/к/config.yaml`.

Проверки качества кода: `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`.

## Результаты прогонов

`runs/<YYYYMMDD-HHMM>_<model>_<mode>_<tag>/`:

- `config.json` — эффективный конфиг, манифест индекса, git sha, хост, версия Ollama;
- `records.jsonl` — по строке на вопрос: ответ, ссылки, найденные фрагменты с полным текстом и
  метаданными, `top_score`, время, автоматические проверки, предварительные коды ошибок
  (R1, R2, S1, S2, G2?) и пустые поля для ручной оценки;
- `results.csv` — таблица по шаблону методики (`;`, UTF-8 с BOM, открывается в Excel); колонки
  «Оценка» и «Комментарий / тип ошибки» заполняются вручную по таблицам 5–7 методики;
- `results.md` — сводка и разбор каждого вопроса.

Автоматические проверки — подсказки, а не оценка. Совпадение раздела: раздел чанка равен эталонному
или вложен в него в любую сторону (`9.3.1` ~ `9.3`, эталон `9` покрывает любой `9.x`).

## Как добавить документы в базу знаний

1. Положите PDF в проект.
2. В `config.yaml` задайте новый `knowledge_base.kb_id` и перечислите файлы явно (glob не
   используется):
   ```yaml
   knowledge_base:
     kb_id: course-v1
     documents:
       - {doc_id: L1, title: "Лекция 1. Введение", path: "materials/lecture1.pdf", chapter: 1}
   ```
   `chapter` — номер главы: заголовок раздела распознаётся как строка вида `1.2.3 Заголовок`, первое
   число которой равно `chapter`, набранная крупнее основного текста.
3. Выполните `uv run mentor ingest --parse-only`, проверьте разделы и страницы в `data/parsed/*.md`,
   затем `uv run mentor ingest`.

Если параметры нарезки, модель эмбеддингов или сами PDF изменились, индекс нужно пересобрать — при
расхождении с `data/index/manifest.json` команды `ask`/`eval` об этом сообщат.

⚠️ `Otsenka_raboty_sistemy.pdf` (методика с эталонными ответами) и `mentor/eval/questions.yaml`
нельзя добавлять в базу знаний: `ingest` откажется работать, а `tests/test_index_isolation.py`
проверяет, что их нет в индексе.
