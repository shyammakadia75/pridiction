# Football Intelligence

A modular, staged-build Python project for football (soccer) analytics: match
prediction, betting-value identification, and adaptive learning.

> ## Part 1 — Engineering Foundation
> This phase establishes the project **scaffolding only**: configuration, logging,
> an SQLite data layer with enforced constraints, a provider-agnostic ingestion
> interface, a reusable validation framework, a CLI, tests, and reproducible
> setup. **No real football data is ingested and no fake data is inserted in this
> phase.** Parts 2+ (StatsBomb ingestion, modeling, betting logic) are **not**
> implemented yet.

## Requirements

- Python 3.10+ (developed and verified on **Python 3.13.14**)
- Runtime: `pandas`, `numpy`, `python-dotenv`
- Dev/test: `pytest`

## Setup (Windows PowerShell)

A virtual environment is recommended. Because `pandas`/`numpy` are often already
installed system-wide, the environment below inherits them with
`--system-site-packages` to avoid re-downloading large wheels; if they are not
present on your machine, **omit** that flag and install `pandas`/`numpy`
explicitly.

```powershell
# 1. Create the virtual environment (reuse system pandas/numpy when available)
python -m venv .venv --system-site-packages

# 2. Install test + dotenv dependencies, then the project in editable mode
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install pytest python-dotenv
.venv\Scripts\python -m pip install -e .

# 3. Activate (process-scoped policy change — does not modify system settings)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.venv\Scripts\Activate.ps1
```

> **Activation troubleshooting:** if `.\Activate.ps1` is blocked by the
> execution policy, the `-Scope Process` command above permits it for the current
> shell only, without weakening system-wide security. If you prefer **not to
> activate**, skip it and call the venv binaries directly instead, e.g.
> `.venv\Scripts\python -m football_intelligence init-db`.

## Usage

```powershell
# Initialize the database schema (idempotent — safe to re-run)
.venv\Scripts\python -m football_intelligence init-db

# Run all project health checks
.venv\Scripts\python -m football_intelligence health

# Show the CLI version
.venv\Scripts\football-intelligence --version
```

After activation you can also use the `football-intelligence` console script
directly. The configured database defaults to `data/football_intelligence.db`.

`init-db` creates the SQLite schema **without inserting any data**. `health`
verifies configuration, directories, database accessibility, schema (all 12
tables), foreign-key enforcement, and logging.

## Testing

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

All tests run against disposable databases in pytest's `tmp_path`; no real or
production database is ever touched, and no network access is required.

## Configuration

Defaults come from environment variables with the `FI_` prefix. Copy the example
and adjust as needed:

```powershell
Copy-Item .env.example .env
```

| Variable           | Default                                  | Description                          |
| ------------------ | ---------------------------------------- | ------------------------------------ |
| `FI_ENVIRONMENT`   | `development`                            | `development` / `testing` / `production` |
| `FI_DATABASE_PATH` | `data/football_intelligence.db`          | SQLite path (relative to project root) |
| `FI_LOG_LEVEL`     | `INFO`                                   | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `FI_LOG_DIR`       | `logs`                                   | Log directory                        |
| `FI_LOG_FILE`      | `football_intelligence.log`              | Log file name                        |
| `FI_LOG_FORMAT`    | `%(asctime)s [%(levelname)s] %(name)s: %(message)s` | Logging format              |

Invalid values (e.g. a bad log level or environment) raise `ConfigError` at load
time.

## Project layout

```
pridiction/
├── data/{raw,processed,external}/.gitkeep   # data directories (gitignored contents)
├── models/.gitkeep
├── notebooks/.gitkeep
├── logs/.gitkeep
├── config/.gitkeep
├── scripts/.gitkeep
├── src/football_intelligence/
│   ├── __init__.py        # package exports & version
│   ├── __main__.py        # enables `python -m football_intelligence`
│   ├── cli.py             # init-db / health CLI
│   ├── config.py          # env-driven Settings (frozen)
│   ├── exceptions.py      # shared exception hierarchy
│   ├── logging_config.py  # console + file logger
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py   # FK-enabled sqlite3 connection factory + context mgr
│   │   ├── initialize.py   # idempotent schema init + migration version tracking
│   │   └── schema.py       # table DDL, constraints, indexes, migrations
│   └── data/
│       ├── __init__.py
│       ├── cleaning/__init__.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   └── base.py      # BaseIngestor + IngestionResult (provider-agnostic)
│       └── validation/
│           ├── __init__.py
│           └── validators.py # ValidationResult + field/record validators
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_database.py
│   └── test_validation.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Verified state (Part 1)

- Editable install works; `football-intelligence --version` reports `0.1.0`.
- `init-db` initializes the schema idempotently (schema version 1, FKs on).
- `health` reports **Overall: OK** (configuration, directories, schema, foreign
  keys, database, and logging all green).
- `pytest tests/` — **57 passed** (run against disposable `tmp_path` databases;
  no real database and no network access required).

## License

MIT
