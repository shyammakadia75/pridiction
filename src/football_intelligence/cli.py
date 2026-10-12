"""Command-line interface for football-intelligence.

Commands: ``init-db`` (create the SQLite schema, idempotent) and ``health``
(verify configuration, directories, database accessibility, schema, and
logging). Health checks are reported as ok / warning / error. The CLI never
claims an external data source is reachable unless a real check is performed.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import load_settings
from .database import get_database_version, initialize_database, resolve_db_path
from .database.connection import get_connection
from .database.schema import TABLE_NAMES
from .exceptions import ConfigError, DatabaseError
from .logging_config import get_logger, setup_logging

logger = get_logger("cli")


@dataclass
class CheckResult:
    """Outcome of a single health check."""

    name: str
    status: str  # "ok" | "warning" | "error"
    message: str = ""


@dataclass
class HealthReport:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(c.status == "error" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "warning" for c in self.checks)

    def summary(self) -> str:
        lines = ["Health report:"]
        for c in self.checks:
            lines.append(f"  [{c.status.upper():7s}] {c.name}: {c.message}")
        overall = "ERROR" if self.has_errors else ("WARNING" if self.has_warnings else "OK")
        lines.append(f"Overall: {overall}")
        return "\n".join(lines)


def _check_configuration(report: HealthReport) -> None:
    try:
        s = load_settings()
    except ConfigError as exc:
        report.checks.append(CheckResult("configuration", "error", str(exc)))
        return
    report.checks.append(
        CheckResult("configuration", "ok", f"FI_ENVIRONMENT={s.environment}; database={s.database_path}")
    )


def _check_directories(report: HealthReport) -> None:
    from .config import (
        CONFIG_DIR, DATA_DIR, EXTERNAL_DATA_DIR, LOGS_DIR,
        PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR,
    )

    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR, LOGS_DIR, CONFIG_DIR, PROJECT_ROOT,
    ]
    for d in directories:
        rel = d.relative_to(PROJECT_ROOT) if d.is_absolute() else d
        if d.exists():
            report.checks.append(CheckResult(f"directory:{rel}", "ok", str(d)))
        else:
            try:
                d.mkdir(parents=True, exist_ok=True)
                report.checks.append(CheckResult(f"directory:{rel}", "warning", f"created {d}"))
            except OSError as exc:
                report.checks.append(CheckResult(f"directory:{rel}", "error", f"cannot create {d}: {exc}"))


def _check_database(report: HealthReport, db_path: Path) -> None:
    if not db_path.exists():
        report.checks.append(
            CheckResult("database", "warning", f"file '{db_path}' does not exist (run 'football-intelligence init-db')")
        )
        return
    try:
        conn = get_connection(db_path)
        try:
            version = get_database_version(db_path)
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
            missing = [t for t in TABLE_NAMES if t not in tables]
            if missing:
                report.checks.append(CheckResult("schema", "error", f"missing table(s): {missing}"))
            else:
                report.checks.append(
                    CheckResult("schema", "ok", f"all {len(TABLE_NAMES)} tables present; schema_version={version}")
                )
            fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
            report.checks.append(
                CheckResult("foreign_keys", "ok", "PRAGMA foreign_keys = ON")
                if fk else CheckResult("foreign_keys", "error", "foreign keys disabled")
            )
            report.checks.append(CheckResult("database", "ok", f"accessible '{db_path}'"))
        finally:
            conn.close()
    except DatabaseError as exc:
        report.checks.append(CheckResult("database", "error", str(exc)))


def _log_file_path() -> Path:
    from .config import settings as s

    return s.log_dir / s.log_file


def _check_logging(report: HealthReport) -> None:
    try:
        lg = setup_logging()
        lg.info("Health-check logging test.")
        log_file = _log_file_path()
        if log_file.exists():
            report.checks.append(CheckResult("logging", "ok", f"console + file '{log_file.name}'"))
        else:
            report.checks.append(CheckResult("logging", "warning", "file handler not active"))
    except Exception as exc:  # pragma: no cover - defensive
        report.checks.append(CheckResult("logging", "error", str(exc)))


def run_health_checks(db_path=None) -> HealthReport:
    """Run all health checks and return the assembled report."""
    s = load_settings()
    target = resolve_db_path(db_path)
    report = HealthReport()
    _check_configuration(report)
    _check_directories(report)
    _check_database(report, target)
    _check_logging(report)
    return report
def cmd_init_db(args: argparse.Namespace) -> int:
    setup_logging()
    try:
        version = initialize_database(args.db_path)
    except DatabaseError as exc:
        logger.error("Database initialization failed: %s", exc)
        print(f"ERROR: database initialization failed: {exc}", file=sys.stderr)
        return 1
    logger.info("Database initialized to schema version %d.", version)
    target = args.db_path if args.db_path else "configured database"
    print(f"OK: database initialized to schema version {version} ({target}).")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    setup_logging()
    report = run_health_checks(args.db_path)
    logger.info(
        "Health check complete (overall: %s).",
        "ERROR" if report.has_errors else ("WARNING" if report.has_warnings else "OK"),
    )
    print(report.summary())
    return 1 if report.has_errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="football-intelligence",
        description="Football Intelligence project CLI.",
    )
    parser.add_argument("--version", action="version", version=f"football-intelligence {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-db", help="Initialize the SQLite database schema.")
    init.add_argument(
        "--db-path", default=None,
        help="Path to the SQLite database file (defaults to configured path).",
    )
    init.set_defaults(func=cmd_init_db)

    health = subparsers.add_parser("health", help="Run project health checks.")
    health.add_argument(
        "--db-path", default=None,
        help="Database path to check (defaults to configured path).",
    )
    health.set_defaults(func=cmd_health)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point used by the console script and ``python -m``."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
