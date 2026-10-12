"""Update README.md with StatsBomb documentation."""
import pathlib

p = pathlib.Path("README.md")
content = p.read_text(encoding="utf-8")

# 1. Add StatsBomb Ingestion section after the Usage section (before Testing)
old_usage_end = """`init-db` creates the SQLite schema **without inserting any data**. `health`
verifies configuration, directories, database accessibility, schema (all 12
tables), foreign-key enforcement, and logging.

## Testing"""

new_usage_end = """`init-db` creates the SQLite schema **without inserting any data**. `health`
verifies configuration, directories, database accessibility, schema (all 12
tables), foreign-key enforcement, and logging.

## StatsBomb Ingestion

```powershell
# Initialize the database (if not already done)
.venv\\Scripts\\python -m football_intelligence init-db

# List available competition/season pairs (fetches from StatsBomb Open Data)
.venv\\Scripts\\python -m football_intelligence statsbomb list

# Import Champions League 2018/2019 into the default database
.venv\\Scripts\\python -m football_intelligence statsbomb import --competition-id 16 --season-id 4

# Force a fresh download (bypass the local JSON cache)
.venv\\Scripts\\python -m football_intelligence statsbomb import --competition-id 16 --season-id 4 --refresh
```

### How it works

1. **Download** -- fetches JSON from the [StatsBomb Open Data](https://github.com/statsbomb/open-data)
   GitHub repository using Python's stdlib `urllib.request` (no extra HTTP dependency).
2. **Cache** -- the raw JSON is stored deterministically under `data/raw/statsbomb/`
   (e.g. `data/raw/statsbomb/matches/16/4.json`).  Subsequent imports reuse the
   cache unless `--refresh` is passed.  Path-traversal is blocked: competition
   and season IDs are validated as positive integers.
3. **Normalize** -- each match record is mapped to the project's SQLite schema
   (`competitions`, `teams`, `matches`), with `provider = "statsbomb"` and the
   StatsBomb match ID stored in `provider_match_id`.
4. **Validate** -- records pass through the shared validation framework before
   persistence.
5. **Load** -- an idempotent upsert (insert-or-update) pattern writes
   competitions, teams, and matches in a **single transaction** that rolls back
   on any error.  Re-importing the same competition simply updates existing rows.
6. **Provenance** -- every run is recorded in `data_sources` (source URL, retrieval
   timestamp, data version) and `ingestion_runs` (status, record counts, errors).

### CLI reference

| Flag                | Short | Description                                              |
| ------------------- | ----- | -------------------------------------------------------- |
| `--competition-id`  | `-c`  | StatsBomb competition ID (positive integer)            |
| `--season-id`       | `-s`  | StatsBomb season ID (positive integer)                   |
| `--db-path`         | `-d`  | Override the SQLite database path                       |
| `--cache-dir`       |       | Override the JSON cache directory                       |
| `--refresh`         |       | Re-download from the network instead of using the cache |
| `--timeout`         |       | HTTP timeout in seconds (default 15)                     |

### Notable competition / season pairs

| Competition          | Competition ID | Season ID | Season        |
| -------------------- | -------------- | --------- | ------------- |
| Champions League     | 16             | 4         | 2018/2019     |
| Premier League       | 2              | 44        | 2003/2004     |
| La Liga              | 11             | 23        | 2011/2012     |
| FIFA World Cup       | 43             | 3         | 2018          |
| NWSL                 | 49             | 3         | 2018          |

## Testing"""

if old_usage_end in content:
    content = content.replace(old_usage_end, new_usage_end)
    print("OK: Added StatsBomb Ingestion section")
else:
    print("WARN: Usage section end not found -- skipping")

p.write_text(content, encoding="utf-8")
print("README.md updated (phase 1).")
