"""SQLite schema definitions.

The schema is applied idempotently (``CREATE TABLE IF NOT EXISTS``) so that
re-running initialization never destroys existing data. A ``schema_version``
table records the highest applied migration version, giving future parts a
safe extension point for evolving the schema.

StatsBomb Open Data does not necessarily supply every column above. Columns
that may be absent from a source are declared ``NULL``-able; unavailable
information is stored as missing, never fabricated.
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------
SCHEMA_VERSION: Final[int] = 1

# ---------------------------------------------------------------------------
# SQL statements
# ---------------------------------------------------------------------------
SCHEMA_VERSION_TABLE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT    NOT NULL
);
"""

TEAMS_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS teams (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    provider          TEXT    NOT NULL,
    provider_team_id  TEXT    NOT NULL,
    team_name         TEXT,
    country           TEXT,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    CONSTRAINT uq_teams_provider_id UNIQUE (provider, provider_team_id)
);
"""

PLAYERS_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS players (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    provider          TEXT    NOT NULL,
    provider_player_id TEXT   NOT NULL,
    player_name       TEXT,
    date_of_birth     TEXT,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    CONSTRAINT uq_players_provider_id UNIQUE (provider, provider_player_id)
);
"""

MATCHES_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS matches (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    provider          TEXT    NOT NULL,
    provider_match_id TEXT    NOT NULL,
    competition_id    INTEGER NOT NULL,
    season_name       TEXT,
    match_date        TEXT,
    home_team_id      INTEGER NOT NULL,
    away_team_id      INTEGER NOT NULL,
    home_goals        INTEGER,
    away_goals        INTEGER,
    match_status      TEXT,
    created_at        TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL,
    CONSTRAINT uq_matches_provider_id UNIQUE (provider, provider_match_id),
    CONSTRAINT fk_matches_competition FOREIGN KEY (competition_id) REFERENCES competitions (id),
    CONSTRAINT fk_matches_home_team  FOREIGN KEY (home_team_id)  REFERENCES teams (id),
    CONSTRAINT fk_matches_away_team  FOREIGN KEY (away_team_id)  REFERENCES teams (id),
    CONSTRAINT chk_home_goals_nonneg CHECK (home_goals IS NULL OR home_goals >= 0),
    CONSTRAINT chk_away_goals_nonneg CHECK (away_goals IS NULL OR away_goals >= 0),
    CONSTRAINT chk_teams_distinct    CHECK (home_team_id <> away_team_id)
);
"""

PREDICTIONS_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id              INTEGER NOT NULL,
    model_version_id      INTEGER NOT NULL,
    prediction_timestamp  TEXT    NOT NULL,
    home_win_probability  REAL,
    draw_probability      REAL,
    away_win_probability  REAL,
    feature_snapshot_json TEXT,
    data_cutoff_timestamp TEXT,
    prediction_status     TEXT,
    created_at            TEXT    NOT NULL,
    CONSTRAINT fk_predictions_match FOREIGN KEY (match_id) REFERENCES matches (id),
    CONSTRAINT fk_predictions_model FOREIGN KEY (model_version_id) REFERENCES model_versions (id),
    CONSTRAINT chk_home_prob_range CHECK (home_win_probability IS NULL OR (home_win_probability >= 0 AND home_win_probability <= 1)),
    CONSTRAINT chk_draw_prob_range CHECK (draw_probability IS NULL OR (draw_probability >= 0 AND draw_probability <= 1)),
    CONSTRAINT chk_away_prob_range CHECK (away_win_probability IS NULL OR (away_win_probability >= 0 AND away_win_probability <= 1))
);
"""

ODDS_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS odds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER NOT NULL,
    bookmaker       TEXT,
    market          TEXT,
    selection       TEXT,
    decimal_odds    REAL    NOT NULL,
    observed_at     TEXT    NOT NULL,
    source_reference TEXT,
    created_at      TEXT    NOT NULL,
    CONSTRAINT fk_odds_match FOREIGN KEY (match_id) REFERENCES matches (id),
    CONSTRAINT chk_decimal_odds_positive CHECK (decimal_odds > 0)
);
"""

PREDICTION_ERRORS_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS prediction_errors (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id         INTEGER NOT NULL,
    actual_result         TEXT,
    evaluation_timestamp  TEXT    NOT NULL,
    brier_score           REAL,
    log_loss              REAL,
    error_metadata_json   TEXT,
    created_at            TEXT    NOT NULL,
    CONSTRAINT fk_pred_errors_prediction FOREIGN KEY (prediction_id) REFERENCES predictions (id),
    CONSTRAINT chk_brier_score_range CHECK (brier_score IS NULL OR (brier_score >= 0 AND brier_score <= 1))
);
"""

MODEL_PERFORMANCE_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS model_performance (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version_id        INTEGER NOT NULL,
    evaluation_period_start TEXT,
    evaluation_period_end   TEXT,
    sample_size             INTEGER,
    accuracy                REAL,
    log_loss                REAL,
    brier_score             REAL,
    metrics_json            TEXT,
    created_at              TEXT    NOT NULL,
    CONSTRAINT fk_perf_model FOREIGN KEY (model_version_id) REFERENCES model_versions (id),
    CONSTRAINT chk_accuracy_range CHECK (accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1))
);
"""

# Ordered collection of every table definition. schema_version is first so its
# version queries can run during initialization.
ALL_TABLES: Final[tuple[str, ...]] = (
    SCHEMA_VERSION_TABLE_SQL,
    TEAMS_SQL,
    PLAYERS_SQL,
    COMPETITIONS_SQL,
    MATCHES_SQL,
    DATA_SOURCES_SQL,
    INGESTION_RUNS_SQL,
    MODEL_VERSIONS_SQL,
    PREDICTIONS_SQL,
    ODDS_SQL,
    PREDICTION_ERRORS_SQL,
    MODEL_PERFORMANCE_SQL,
)

# Indexes to support common query patterns; created idempotently.
INDEX_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE INDEX IF NOT EXISTS idx_matches_competition ON matches (competition_id);",
    "CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches (home_team_id);",
    "CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches (away_team_id);",
    "CREATE INDEX IF NOT EXISTS idx_matches_match_date ON matches (match_date);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions (match_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions (model_version_id);",
    "CREATE INDEX IF NOT EXISTS idx_odds_match ON odds (match_id);",
    "CREATE INDEX IF NOT EXISTS idx_pred_errors_prediction ON prediction_errors (prediction_id);",
    "CREATE INDEX IF NOT EXISTS idx_model_perf_model ON model_performance (model_version_id);",
)

# Migrations keyed by the version they introduce. Version 1 is the baseline
# applied by ALL_TABLES, so this dict starts empty and is ready for future
# schema evolution (Part 2+) without dropping data.
MIGRATIONS: Final[dict[int, tuple[str, ...]]] = {}

# Convenience: the set of table names managed by this schema, useful for
# health checks and tests.
TABLE_NAMES: Final[tuple[str, ...]] = (
    "schema_version",
    "teams",
    "players",
    "competitions",
    "matches",
    "data_sources",
    "ingestion_runs",
    "model_versions",
    "predictions",
    "odds",
    "prediction_errors",
    "model_performance",
)


