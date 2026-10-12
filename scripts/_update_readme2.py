"""Update README.md: project tree and verified state."""
import pathlib

p = pathlib.Path("README.md")
content = p.read_text(encoding="utf-8")

# 2. Update project tree to include statsbomb.py
old_tree_line = "\u2502   \u2502   \u2514\u2500\u2500 base.py      # BaseIngestor + IngestionResult (provider-agnostic)"
new_tree_line = "\u2502   \u2502   \u251c\u2500\u2500 base.py      # BaseIngestor + IngestionResult (provider-agnostic)\n\u2502   \u2502   \u2514\u2500\u2500 statsbomb.py # StatsBombIngestor adapter"
if old_tree_line in content:
    content = content.replace(old_tree_line, new_tree_line)
    print("OK: Added statsbomb.py to project tree")
else:
    print("WARN: Tree base.py line not found")

# 3. Update tests tree to include test_statsbomb.py
old_test_line = "\u2502   \u2514\u2500\u2500 test_validation.py"
new_test_line = "\u2502   \u251c\u2500\u2500 test_statsbomb.py\n\u2502   \u2514\u2500\u2500 test_validation.py"
if old_test_line in content:
    content = content.replace(old_test_line, new_test_line)
    print("OK: Added test_statsbomb.py to tests in tree")
else:
    print("WARN: test_validation.py tree line not found")

# 4. Update Verified state section
old_verified = """## Verified state (Part 1)

- Editable install works; `football-intelligence --version` reports `0.1.0`.
- `init-db` initializes the schema idempotently (schema version 1, FKs on).
- `health` reports **Overall: OK** (configuration, directories, schema, foreign
  keys, database, and logging all green).
- `pytest tests/` -- **57 passed** (run against disposable `tmp_path` databases;
  no real database and no network access required)."""

new_verified = """## Verified state

### Part 1 -- Engineering Foundation
- Editable install works; `football-intelligence --version` reports `0.1.0`.
- `init-db` initializes the schema idempotently (schema version 1, FKs on).
- `health` reports **Overall: OK** (configuration, directories, schema, foreign
  keys, database, and logging all green).
- `pytest tests/` -- **57 passed** (run against disposable `tmp_path` databases;
  no real database and no network access required).

### Part 2 -- StatsBomb Ingestion
- `StatsBombIngestor` downloads, normalizes, validates, and persists match data
  from the real StatsBomb Open Data repository (verified with Champions League
  2018/2019 -- Liverpool vs Tottenham, match 22912).
- JSON cache under `data/raw/statsbomb/` with path-traversal protection.
- Idempotent re-import (0 inserted, 1 updated on second run).
- Provenance recorded in `data_sources` and `ingestion_runs` tables.
- `pytest tests/` -- **112 passed** (57 Part 1 + 55 Part 2; all network calls
  mocked, no real data downloaded during tests)."""

if old_verified in content:
    content = content.replace(old_verified, new_verified)
    print("OK: Updated Verified state section")
else:
    print("WARN: Verified state section not found")

p.write_text(content, encoding="utf-8")
print("README.md updated (phase 2).")
