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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from . import __version__
from .config import PROJECT_ROOT, load_settings
from .data.ingestion.base import IngestionResult
from .data.ingestion.odds import OddsIngestor, DEFAULT_TIMEOUT, list_leagues as list_odds_leagues
from .data.ingestion.football_data_results import (
    DATASET_NAME,
    PROVIDER,
    FootballDataResultsIngestor,
    list_result_leagues,
)
from .data.ingestion.statsbomb import (
    StatsBombIngestor,
    list_competitions,
    record_ingestion_run,
)
from .database import get_database_version, initialize_database, resolve_db_path
from .database.connection import get_connection
from .database.schema import TABLE_NAMES
from .exceptions import ConfigError, DatabaseError, IngestionError
from .logging_config import get_logger, setup_logging
from .prediction import (
    EvaluationReport,
    save_report,
    run_prediction_evaluation,
)
from .prediction.elo import (
    EloModel,
    EloReport,
    run_elo_evaluation,
    save_elo_report,
)
from .prediction.poisson import (
    PoissonModel,
    PoissonFixturePrediction,
    run_poisson_evaluation,
    save_poisson_report,
    DEFAULT_MAX_GOALS,
    DEFAULT_MIN_HISTORY,
    DEFAULT_PRIOR_STRENGTH,
)

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


def _safe_record_run(db_path, result: IngestionResult) -> None:
    """Record an ingestion-run audit row, surfacing provenance failures.

    A failure to write the ``ingestion_runs`` audit trail is logged at ERROR
    and re-raised as :class:`DatabaseError` (chained from the cause) so a
    missing audit record is never silently swallowed.  The original ingestion
    outcome is preserved on ``result`` and by the caller; provenance failures
    are surfaced as a distinct exception.  No credentials are logged.
    """
    try:
        record_ingestion_run(db_path, result)
    except Exception as exc:  # noqa: BLE001 -- surface, do not swallow
        logger.error("Could not record ingestion run (audit row lost): %s", exc)
        raise DatabaseError(
            f"Failed to record ingestion run for "
            f"{result.provider}/{result.dataset_name}: {exc}"
        ) from exc


def cmd_statsbomb_list(args: argparse.Namespace) -> int:
    """List available StatsBomb competitions and seasons."""
    setup_logging()
    try:
        df = list_competitions(
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            refresh=args.refresh,
            timeout=args.timeout,
        )
    except IngestionError as exc:
        logger.error("Failed to list competitions: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if df.empty:
        print("No competitions found in the StatsBomb Open Data repository.")
        return 0
    display_cols = [
        "competition_id", "season_id",
        "competition_name", "season_name",
        "country_name", "competition_gender",
    ]
    cols = [c for c in display_cols if c in df.columns]
    source = "cached" if not args.refresh and not args.cache_dir else "network"
    print(
        f"StatsBomb Open Data - {len(df)} competition/season entries ({source}):"
    )
    print(df[cols].to_string(index=False))
    return 0


def cmd_statsbomb_import(args: argparse.Namespace) -> int:
    """Import matches for a StatsBomb competition and season."""
    setup_logging()
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    ingestor = StatsBombIngestor(
        competition_id=args.competition_id,
        season_id=args.season_id,
        db_path=args.db_path,
        cache_dir=cache_dir,
        refresh=args.refresh,
        timeout=args.timeout,
    )
    logger.info(
        "Importing StatsBomb competition %s season %s",
        ingestor.competition_id, ingestor.season_id,
    )
    try:
        result = ingestor.ingest()
    except IngestionError as exc:
        failed = IngestionResult(
            provider=ingestor.provider,
            dataset_name=ingestor.dataset_name,
            status="failed",
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            _safe_record_run(args.db_path, failed)
        except DatabaseError as pexc:
            logger.error("Failed-import audit recording also failed: %s", pexc)
            print(f"ERROR: failed to record audit trail: {pexc}", file=sys.stderr)
        logger.error("Import failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    source = "cache" if ingestor.was_cached else "network"
    try:
        _safe_record_run(args.db_path, result)
    except DatabaseError as pexc:
        logger.error("Import succeeded but the audit trail was not recorded: %s", pexc)
        print(f"ERROR: {pexc}", file=sys.stderr)
        return 1
    print(
        f"Selected: competition={ingestor.competition_id}, "
        f"season={ingestor.season_id}"
    )
    print(f"Source: {ingestor.source_url} ({source})")
    print(f"Records read: {result.records_read}")
    print(f"Records inserted: {result.records_inserted}")
    print(f"Records updated: {result.records_updated}")
    print(f"Status: {result.status}")
    return 0 if result.is_success else 1


# -- Odds subcommands ---------------------------------------------------


def cmd_odds_list(args: argparse.Namespace) -> int:
    """Print the list of supported Football-Data league codes."""
    leagues = list_odds_leagues()
    print("Available Football-Data league codes:")
    for code, name in leagues:
        print(f"  {code:4s}  {name}")
    return 0


def cmd_odds_import(args: argparse.Namespace) -> int:
    """Import historical odds for a league and season."""
    setup_logging()
    league_code = args.league_code
    season_code = str(args.season_code)
    ingestor = OddsIngestor(
        league_code=league_code,
        season_code=season_code,
        db_path=args.db_path,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        timeout=args.timeout,
    )
    logger.info(
        "Importing odds for league %s season %s",
        ingestor.league_code, ingestor.season_code,
    )
    try:
        result = ingestor.ingest()
    except IngestionError as exc:
        failed = IngestionResult(
            provider=ingestor.provider,
            dataset_name=ingestor.dataset_name,
            status="failed",
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            _safe_record_run(args.db_path, failed)
        except DatabaseError as pexc:
            logger.error("Failed-import audit recording also failed: %s", pexc)
            print(f"ERROR: failed to record audit trail: {pexc}", file=sys.stderr)
        logger.error("Import failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    source = "cache" if ingestor.was_cached else "network"
    try:
        _safe_record_run(args.db_path, result)
    except DatabaseError as pexc:
        logger.error("Import succeeded but the audit trail was not recorded: %s", pexc)
        print(f"ERROR: {pexc}", file=sys.stderr)
        return 1
    print(
        f"Selected: league={ingestor.league_code}, season={ingestor.season_code}"
    )
    print(f"Source: {ingestor.source_url} ({source})")
    print(f"Records read:    {result.records_read}")
    print(f"Records inserted: {result.records_inserted}")
    print(f"Records updated:  {result.records_updated}")
    print(f"Matches matched:  {result.records_matched}")
    print(f"Matches unmatched: {result.records_unmatched}")
    print(f"Matches ambiguous: {result.records_ambiguous}")
    print(f"Status: {result.status}")
    return 0 if result.is_success else 1


# -- Football-Data results subcommands --------------------------------


def cmd_results_list(args: argparse.Namespace) -> int:
    """Print the list of supported Football-Data league codes for results."""
    leagues = list_result_leagues()
    print("Available Football-Data league codes (results):")
    for code, name in leagues:
        print(f"  {code:4s}  {name}")
    return 0


def cmd_results_import(args: argparse.Namespace) -> int:
    """Import historical match results for a league and season."""
    setup_logging()
    league_code = str(args.league_code).strip().upper()
    season_code = str(args.season_code)
    try:
        ingestor = FootballDataResultsIngestor(
            league_code=league_code,
            season_code=season_code,
            db_path=args.db_path,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            refresh=args.refresh,
            timeout=args.timeout,
            source_url=args.source_url,
        )
        logger.info(
            "Importing results for league %s season %s",
            ingestor.league_code, ingestor.season_code,
        )
        result = ingestor.ingest()
    except IngestionError as exc:
        failed = IngestionResult(
            provider=PROVIDER,
            dataset_name=DATASET_NAME,
            status="failed",
            error_message=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            _safe_record_run(args.db_path, failed)
        except DatabaseError as pexc:
            logger.error("Failed-import audit recording also failed: %s", pexc)
            print(f"ERROR: failed to record audit trail: {pexc}", file=sys.stderr)
        logger.error("Import failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    source = "cache" if ingestor.was_cached else "network"
    try:
        _safe_record_run(args.db_path, result)
    except DatabaseError as pexc:
        logger.error("Import succeeded but the audit trail was not recorded: %s", pexc)
        print(f"ERROR: {pexc}", file=sys.stderr)
        return 1
    print(
        f"Selected: league={ingestor.league_code}, season={ingestor.season_code}"
    )
    print(f"Source: {ingestor.source_url} ({source})")
    print(f"Records read:    {result.records_read}")
    print(f"Records inserted: {result.records_inserted}")
    print(f"Records updated:  {result.records_updated}")
    print(f"Records skipped:  {result.records_skipped}")
    print(f"Records ambiguous: {result.records_ambiguous}")
    print(f"Status: {result.status}")
    return 0 if result.is_success else 1


def _print_eval_summary(report: EvaluationReport) -> None:
    """Print a compact per-model train/val/test metric summary to stdout."""
    d = report.dataset
    print(f"Evaluation: provider={d.get('provider')} "
          f"competition={d.get('provider_competition_id')} "
          f"season={d.get('season_name')} n_matches={d.get('n_matches')} "
          f"n_features={d.get('n_features')}")
    counts = report.splits.get("raw_counts", {})
    print(f"  split counts: {counts}")
    for m in report.models:
        cfg = m.get("config", {})
        splits = m.get("splits", {})
        print(f"Model: {m.get('name')} (type={cfg.get('type')}) "
              f"train={m.get('n_train')} val={m.get('n_val')} test={m.get('n_test')}")
        for sp in ("train", "val", "test"):
            r = splits.get(sp, {})
            if r.get("n_samples", 0) == 0:
                print(f"  [{sp}] n=0 (skipped)")
                continue
            print(f"  [{sp}] n={r.get('n_samples')} acc={r.get('accuracy'):.4f} "
                  f"brier={r.get('brier_score'):.4f} logloss={r.get('log_loss'):.4f} "
                  f"macroF1={r.get('macro_f1'):.4f}")
    print(f"production DB sha256: {report.production_db_sha256}")
    if report.production_db_unchanged is None:
        unchanged = "UNKNOWN (no production db hashed)"
    elif report.production_db_unchanged:
        unchanged = "YES (sha256 stable: unchanged)"
    else:
        unchanged = "NO (sha256 changed! -- violates leakage contract)"
    print(f"production DB unchanged: {unchanged}")


def cmd_model_evaluate(args: argparse.Namespace) -> int:
    """football-intelligence model evaluate -- leak-safe baseline vs. LR."""
    setup_logging()
    try:
        db_path = resolve_db_path(args.db_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: cannot resolve database path: {exc}", file=sys.stderr)
        return 1
    if not Path(db_path).exists():
        print(f"ERROR: database not found at '{db_path}' "
              f"(run 'football-intelligence init-db').", file=sys.stderr)
        return 1

    seasons = args.season if args.season else [None]
    rc = 0
    for season_name in seasons:
        label = season_name or "all-seasons"
        try:
            report = run_prediction_evaluation(
                db_path,
                provider=args.provider,
                provider_competition_id=args.competition_id,
                season_name=season_name,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                lr_learning_rate=args.lr_learning_rate,
                lr_n_iterations=args.lr_n_iterations,
                lr_l2_lambda=args.lr_l2_lambda,
                lr_seed=args.lr_seed,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"ERROR: evaluation failed for {label} "
                  f"(provider={args.provider}, competition={args.competition_id}): {exc}",
                  file=sys.stderr)
            logger.exception("Phase-18 evaluation failed for %s", label)
            rc = 1
            continue
        _print_eval_summary(report)

        suffix = season_name.replace("/", "-") if season_name else "all"
        name = f"phase18_evaluation_{args.competition_id}_{suffix}"
        try:
            artifacts = save_report(report, args.output_dir, name=name)
        except OSError as exc:
            print(f"ERROR: could not write report to '{args.output_dir}': {exc}",
                  file=sys.stderr)
            rc = 1
            continue
        print(f"Saved: {artifacts['json']}")
        print(f"Saved: {artifacts['txt']}")
    return rc


def cmd_ratings_elo(args: argparse.Namespace) -> int:
    """football-intelligence ratings elo -- chronological Elo ratings (read-only)."""
    setup_logging()
    try:
        db_path = resolve_db_path(args.db_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: cannot resolve database path: {exc}", file=sys.stderr)
        return 1
    if not Path(db_path).exists():
        print(f"ERROR: database not found at '{db_path}' "
              f"(run 'football-intelligence init-db').", file=sys.stderr)
        return 1
    seasons = args.season if args.season else [None]
    rc = 0
    for season_name in seasons:
        label = season_name or "all-seasons"
        try:
            rep = run_elo_evaluation(
                db_path,
                provider=args.provider,
                provider_competition_id=args.competition_id,
                season_name=season_name,
                K=args.K,
                home_advantage=args.home_advantage,
                initial_rating=args.initial_rating,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"ERROR: Elo evaluation failed for {label}: {exc}", file=sys.stderr)
            logger.exception("Phase-19 Elo evaluation failed for %s", label)
            rc = 1
            continue
        d = rep.dataset
        print(f"Elo: provider={d.get('provider')} competition={d.get('provider_competition_id')} "
              f"season={d.get('season_name')} n_matches={d.get('n_matches')} n_teams={d.get('n_teams')}")
        em = rep.eval_metrics
        print(f"  online eval (predicted from prior ratings only): n={em.get('n')} "
              f"acc={em.get('accuracy'):.4f} macroF1={em.get('macro_f1'):.4f} "
              f"brier_expected_score={em.get('brier_expected_score'):.4f}")
        top = sorted(rep.ratings.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("  top 10 by Elo:")
        for i, (team, rating) in enumerate(top, 1):
            print(f"    {i:>2d}. {str(team):<24s} {rating:.1f}")
        print(f"  production DB unchanged: {rep.production_db_unchanged}")
        suffix = season_name.replace("/", "-") if season_name else "all"
        name = f"phase19_elo_{args.competition_id}_{suffix}"
        try:
            art = save_elo_report(rep, args.output_dir, name=name)
        except OSError as exc:
            print(f"ERROR: could not write report to '{args.output_dir}': {exc}",
                  file=sys.stderr)
            rc = 1
            continue
        print(f"  Saved: {art['json']}")
        print(f"  Saved: {art['txt']}")
    return rc

def cmd_ratings_poisson(args: argparse.Namespace) -> int:
    """football-intelligence ratings poisson -- Poisson goal model (read-only).

    Generates leak-free, chronological Poisson predictions for a competition
    and writes a JSON + TXT report.  The production database is treated as
    strictly read-only (SHA-256 re-checked after load).
    """
    setup_logging()
    try:
        db_path = resolve_db_path(args.db_path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"ERROR: cannot resolve database path: {exc}", file=sys.stderr)
        return 1
    if not Path(db_path).exists():
        print(f"ERROR: database not found at '{db_path}' "
              f"(run 'football-intelligence init-db').", file=sys.stderr)
        return 1
    seasons = args.season if args.season else [None]
    rc = 0
    for season_name in seasons:
        label = season_name or "all-seasons"
        try:
            rep = run_poisson_evaluation(
                db_path,
                provider=args.provider,
                provider_competition_id=args.competition_id,
                season_name=season_name,
                max_goals=args.max_goals,
                min_history=args.min_history,
                prior_strength=args.prior_strength,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            print(f"ERROR: Poisson evaluation failed for {label}: {exc}",
                  file=sys.stderr)
            logger.exception("Phase-20 Poisson evaluation failed for %s", label)
            rc = 1
            continue
        d = rep.dataset
        print(f"Poisson: provider={d.get('provider')} "
              f"competition={d.get('provider_competition_id')} "
              f"season={d.get('season_name')} n_matches={d.get('n_matches')} "
              f"n_teams={d.get('n_teams')}")
        m = rep.eval_metrics
        print(f"  online eval (predicted from prior matches only): "
              f"n={m.get('n')} n_scored={m.get('n_scored', m.get('n'))} "
              f"acc={m.get('accuracy'):.4f} macroF1={m.get('macro_f1'):.4f} "
              f"brier={m.get('brier_score'):.4f} logloss={m.get('log_loss'):.4f}")
        gm = m.get("goal_metrics", {})
        if gm:
            print(f"  goal MAE(total)={gm.get('expected_total_goals_mae', 0.0):.4f} "
                  f"over2.5_brier={gm.get('over_2_5_brier', 0.0):.4f} "
                  f"btts_brier={gm.get('btts_brier', 0.0):.4f}")
        print(f"  production DB unchanged: {rep.production_db_unchanged}")
        suffix = season_name.replace("/", "-") if season_name else "all"
        name = f"phase20_poisson_{args.competition_id}_{suffix}"
        try:
            art = save_poisson_report(rep, args.output_dir, name=name)
        except OSError as exc:
            print(f"ERROR: could not write report to '{args.output_dir}': {exc}",
                  file=sys.stderr)
            rc = 1
            continue
        print(f"  Saved: {art['json']}")
        print(f"  Saved: {art['txt']}")
    return rc



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

    # -- StatsBomb subcommand group ----------------------------------------
    sb = subparsers.add_parser(
        "statsbomb", help="StatsBomb Open Data ingestion.",
    )
    sb_sub = sb.add_subparsers(dest="action", required=True)

    sb_list = sb_sub.add_parser(
        "list", help="List available StatsBomb competitions and seasons.",
    )
    sb_list.add_argument(
        "--refresh", action="store_true",
        help="Re-download competitions.json even if cached.",
    )
    sb_list.add_argument(
        "--cache-dir", default=None,
        help="Override the raw-file cache directory.",
    )
    sb_list.add_argument(
        "--timeout", type=int, default=30,
        help="HTTP timeout in seconds (default 30).",
    )
    sb_list.set_defaults(func=cmd_statsbomb_list)

    sb_import = sb_sub.add_parser(
        "import", help="Import matches for a StatsBomb competition and season.",
    )
    sb_import.add_argument(
        "--competition-id", type=int, required=True,
        help="StatsBomb competition ID (positive integer).",
    )
    sb_import.add_argument(
        "--season-id", type=int, required=True,
        help="StatsBomb season ID (positive integer).",
    )
    sb_import.add_argument(
        "--db-path", default=None,
        help="Database path (defaults to configured path).",
    )
    sb_import.add_argument(
        "--cache-dir", default=None,
        help="Override the raw-file cache directory.",
    )
    sb_import.add_argument(
        "--refresh", action="store_true",
        help="Re-download even if cached.",
    )
    sb_import.add_argument(
        "--timeout", type=int, default=30,
        help="HTTP timeout in seconds (default 30).",
    )
    sb_import.set_defaults(func=cmd_statsbomb_import)

    # -- Odds subcommand group --------------------------------------------
    odds = subparsers.add_parser(
        "odds", help="Football-Data.co.uk historical odds ingestion.",
    )
    odds_sub = odds.add_subparsers(dest="action", required=True)

    odds_list = odds_sub.add_parser(
        "list", help="List supported Football-Data league codes.",
    )
    odds_list.set_defaults(func=cmd_odds_list)

    odds_import = odds_sub.add_parser(
        "import", help="Import historical odds for a league and season.",
    )
    odds_import.add_argument(
        "--league-code", type=str, required=True,
        help="Football-Data league code (e.g. E0 for Premier League).",
    )
    odds_import.add_argument(
        "season_code", type=str,
        help="4-digit season code, e.g. 2324 for the 2023/24 season.",
    )
    odds_import.add_argument(
        "--db-path", default=None,
        help="Database path (defaults to configured path).",
    )
    odds_import.add_argument(
        "--cache-dir", default=None,
        help="Override the raw-file cache directory.",
    )
    odds_import.add_argument(
        "--refresh", action="store_true",
        help="Re-download even if cached.",
    )
    odds_import.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds (default 30).",
    )
    odds_import.set_defaults(func=cmd_odds_import)

    # -- Football-Data results subcommand group ---------------------------
    fdata = subparsers.add_parser(
        "football-data", help="Football-Data.co.uk ingestion (odds/results).",
    )
    fdata_sub = fdata.add_subparsers(dest="action", required=True)

    # results: import
    results = fdata_sub.add_parser(
        "results", help="Historical match results ingestion.",
    )
    results_sub = results.add_subparsers(dest="action", required=True)

    results_list = results_sub.add_parser(
        "list-leagues", help="List supported Football-Data league codes.",
    )
    results_list.set_defaults(func=cmd_results_list)

    results_import = results_sub.add_parser(
        "import", help="Import historical results for a league and season.",
    )
    results_import.add_argument(
        "--league-code", "--league", dest="league_code", type=str, required=True,
        help="Football-Data league code (e.g. E0 for Premier League).",
    )
    results_import.add_argument(
        "--season-code", "--season", dest="season_code", type=str, required=True,
        help="4-digit season code, e.g. 1617 for the 2016/17 season.",
    )
    results_import.add_argument(
        "--db-path", default=None,
        help="Database path (defaults to configured path).",
    )
    results_import.add_argument(
        "--cache-dir", default=None,
        help="Override the raw-file cache directory.",
    )
    results_import.add_argument(
        "--refresh", action="store_true",
        help="Re-download even if cached.",
    )
    results_import.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    results_import.add_argument(
        "--source-url", default=None,
        help="Override the results CSV URL (testing).",
    )
    results_import.set_defaults(func=cmd_results_import)
    # -- Model subcommand group (Phase 18 evaluation) ------------------------
    model = subparsers.add_parser(
        'model', help='Model evaluation: leakage-safe baseline vs. logistic regression.',
    )
    model_sub = model.add_subparsers(dest='action', required=True)
    model_eval = model_sub.add_parser(
        'evaluate', help='Run Phase-18 leakage-safe model evaluation (read-only).',
    )
    model_eval.add_argument('--db-path', default=None,
        help='Database path (defaults to the configured production database, read-only).')
    model_eval.add_argument('--provider', default='statsbomb',
        help='Data provider name (default: statsbomb).')
    model_eval.add_argument('--competition-id', dest='competition_id', type=str, default='2',
        help="Provider competition id (default: '2' = StatsBomb Premier League).")
    model_eval.add_argument('--season', dest='season', action='append', default=None,
        help="Season name, e.g. '2015/2016'. Repeat for multiple seasons; "
             'omit to evaluate all seasons for the competition present in the database.')
    model_eval.add_argument('--output-dir', dest='output_dir',
        default=str(PROJECT_ROOT / 'reports' / 'phase18'),
        help='Directory for JSON/TXT report artifacts (created if missing).')
    model_eval.add_argument('--train-frac', dest='train_frac', type=float, default=0.60,
        help='Fraction of rows for the chronological train split (default: 0.60).')
    model_eval.add_argument('--val-frac', dest='val_frac', type=float, default=0.20,
        help='Fraction of rows for the chronological validation split (default: 0.20).')
    model_eval.add_argument('--lr-learning-rate', dest='lr_learning_rate', type=float, default=0.05)
    model_eval.add_argument('--lr-n-iterations', dest='lr_n_iterations', type=int, default=1000)
    model_eval.add_argument('--lr-l2-lambda', dest='lr_l2_lambda', type=float, default=0.01)
    model_eval.add_argument('--lr-seed', dest='lr_seed', type=int, default=42)
    model_eval.set_defaults(func=cmd_model_evaluate)

    # -- Ratings subcommand group (Phase 19) -----------------------------------
    ratings = subparsers.add_parser(
        'ratings', help='Rating models (e.g. Elo ratings).',
    )
    ratings_sub = ratings.add_subparsers(dest='action', required=True)
    elo = ratings_sub.add_parser(
        'elo', help='Compute chronological Elo ratings over a competition (read-only).',
    )
    elo.add_argument('--db-path', default=None,
        help='Database path (defaults to the configured production database, read-only).')
    elo.add_argument('--provider', default='statsbomb',
        help='Data provider name (default: statsbomb).')
    elo.add_argument('--competition-id', dest='competition_id', type=str, default='2',
        help="Provider competition id (default: '2' = StatsBomb Premier League).")
    elo.add_argument('--season', dest='season', action='append', default=None,
        help="Season name, e.g. '2015/2016'. Repeat for multiple seasons; "
             'omit to evaluate all seasons for the competition present in the database.')
    elo.add_argument('--output-dir', dest='output_dir',
        default=str(PROJECT_ROOT / 'reports' / 'phase19'),
        help='Directory for JSON/TXT report artifacts (created if missing).')
    elo.add_argument('--k-factor', dest='K', type=float, default=20.0,
        help='Elo K-factor in rating points (default: 20).')
    elo.add_argument('--home-advantage', dest='home_advantage', type=float, default=50.0,
        help='Home advantage in rating points (default: 50).')
    elo.add_argument('--initial-rating', dest='initial_rating', type=float, default=1500.0,
        help='Initial rating for unseen teams (default: 1500).')
    elo.set_defaults(func=cmd_ratings_elo)
    poisson = ratings_sub.add_parser(
        'poisson', help='Poisson goal model: chronological match probabilities (read-only).',
    )
    poisson.add_argument('--db-path', default=None,
        help='Database path (defaults to the configured production database, read-only).')
    poisson.add_argument('--provider', default='statsbomb',
        help='Data provider name (default: statsbomb).')
    poisson.add_argument('--competition-id', dest='competition_id', type=str, default='2',
        help="Provider competition id (default: '2' = StatsBomb Premier League).")
    poisson.add_argument('--season', dest='season', action='append', default=None,
        help="Season name, e.g. '2015/2016'. Repeat for multiple seasons; "
             'omit to evaluate all seasons for the competition present in the database.')
    poisson.add_argument('--output-dir', dest='output_dir',
        default=str(PROJECT_ROOT / 'reports' / 'phase20'),
        help='Directory for JSON/TXT report artifacts (created if missing).')
    poisson.add_argument('--max-goals', dest='max_goals', type=int, default=8,
        help='Maximum goals in the scoreline grid (default: 8).')
    poisson.add_argument('--min-history', dest='min_history', type=int, default=5,
        help='Min pre-match matches before a team strength is trusted (default: 5).')
    poisson.add_argument('--prior-strength', dest='prior_strength', type=float, default=1.0,
        help='Fallback strength for teams with insufficient history (default: 1.0).')
    poisson.set_defaults(func=cmd_ratings_poisson)


    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point used by the console script and ``python -m``."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
