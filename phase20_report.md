# Phase 20 — Independent Poisson Goal Model

## 1. Status
- Phase 19 COMPLETE (Elo ratings + `ratings elo` CLI; see `phase19_report.md`).
- Phase 20 = a **standalone, independent Poisson goal model** (no reuse of Elo/logistic
  parameters), strictly read‑only against the production DB, deterministic, leakage‑free,
  and fully tested. New module: `src/football_intelligence/prediction/poisson.py`.
- CLI: `ratings poisson` (subcommand of `ratings`). Artifacts written only to
  `reports/phase20`; **the production DB is never written**.
- Working tree preserved this phase.

## 2. Dataset (genuine, read‑only)
- Prod DB `data/football_intelligence.db`, SHA‑256 (before == after):
  `725af574b88dc033bd3b00928de503f9bd9a836fbe4252832b7812dce0ab3122`
  (verified externally and via the internal `_sha256` guard).
- Load filter (load time only, via the shared `load_matches_for_features`):
  provider=`statsbomb`, provider_competition_id=`2` (Premier League). No `season_name`
  filter -> all Premier League seasons present in the DB are used.
- **n_matches = 380, n_teams = 20, n_scored = 380** (all labelled; 0 NULL goals).
  Date range: `2015-08-08` -> `2016-05-17`.
- Matches sorted ascending by `match_date`; each fixture is scored from history
  containing **strictly earlier dates only** (see §5).

## 3. Model
An independent‑Poisson (bivariate) model: home and away goal counts are independent
Poisson variables with rates `λh`, `λ_a`.

Scoreline grid (renormalised over `[0..max_goals]²`):

    P(i, j) = Pois(i; λh) · Pois(j; λ_a) / Z,   Z = Σ_{i,j≤max_goals} Pois(i; λh)·Pois(j; λ_a)

Outcomes (read off the renormalised grid):

    P(home win) = Σ_{i>j} P(i,j)   P(draw) = Σ_{i=j} P(i,j)   P(away win) = Σ_{i<j} P(i,j)

Over/Under (exact, grid‑independent, via the Poisson CDF of the sum):

    P(total > line) = 1 − CDF_Poisson(line; λh + λ_a)        # line = 2.5

Both teams to score (independence, survival function):

    P(BTTS) = P(home ≥ 1) · P(away ≥ 1) = sf(0; λh) · sf(0; λ_a)

Rates are team‑strength–estimated from the eligible history (`λ = league_mean ×
attack × defence`, see `estimate_team_strengths`); teams below `min_history` fall back
to `prior_strength` (1.0 = league mean); the rate floor `1e-4` prevents log/0 and
degenerate all‑zero grids.

## 4. Reuse audit (no duplicated metric code)
- `OUTCOME_LABELS` / outcome labelling convention shared with `prediction/baseline.py`.
- `compute_accuracy`, `compute_log_loss`, `compute_brier_score`, `build_confusion_matrix`,
  `class_distribution` — reused from `prediction/baseline.py`.
- `compute_macro_f1`, `compute_precision_recall`, `compute_calibration` — reused from
  `prediction/evaluation.py`.
- `load_matches_for_features` + `resolve_db_path` — reused (SELECT‑only loader; the
  read‑only SHA‑256 guard is mirrored from `evaluation.py`).
- `poisson.py` contains **only** Poisson‑specific logic plus thin orchestration; no
  metric or loader code is duplicated.

## 5. Cutoff / leakage protocol
For each fixture `m` (chronological order):
1. Eligible history = matches with `match_date` **strictly less than** `m.match_date`.
   Same‑day results are deliberately excluded (multiple fixtures share a match date), so
   a fixture’s parameters never absorb concurrent results.
2. Strengths are re‑estimated per fixture from that history.
3. Probabilities are generated **before** reading `m.home_goals`/`m.away_goals` for scoring.
4. Only then is the result read; `_outcome_label` returns `None` for NaN/non‑numeric/
   negative goals, so invalid rows are excluded from the scored set (not dropped from
   prediction).

This is a strict prefix with **no same‑day, no future, no cross‑competition** leakage.

## 6. Genuine‑data evaluation (prod, read‑only)
Reproduction command:

    football-intelligence ratings poisson \
        --db-path data/football_intelligence.db \
        --competition-id 2 \
        --output-dir reports/phase20

Online evaluation (each of the 380 matches predicted from prior matches only):

| metric | value |
|---|---|
| n / n_scored | 380 / 380 |
| accuracy | 0.4395 |
| macro-F1 | 0.3581 |
| Brier (3‑class) | 0.6660 |
| log-loss | 1.1758 |

Class distribution (true): `home_win=157 (41.3%)`, `draw=107 (28.2%)`, `away_win=116 (30.5%)`.

Confusion matrix (rows = true [home/draw/away], cols = predicted [home/draw/away])
→ accuracy (100+4+63)/380 = 0.4395:

    [[100,   5,  52],
     [ 53,   4,  50],
     [ 47,   6,  63]]

Per‑class (precision / recall / F1):
- `home_win`: 0.500 / 0.637 / 0.560
- `draw`:      0.267 / 0.037 / 0.066
- `away_win`:  0.382 / 0.543 / 0.448

Goal metrics: `expected_total_goals_mae = 1.499`, `mean_actual_total_goals = 2.70`,
`mean_predicted_total_goals = 2.595`, `over_2.5_brier = 0.280`, `btts_brier = 0.272`.

Calibration (5‑bin, mean predicted vs observed frequency) is reasonable on `home_win`
and `away_win` bins but very weak on `draw` (only 4/107 draws caught) — see §11.

Artifacts (under `reports/phase20`):
`phase20_production_<ts>.{json,txt}`. SHA‑256 in the report header equals the prod
digest above; `production_db_unchanged == True`.

## 7. Synthetic two‑season DB smoke test (deterministic validation)
Disposable DB built by `tests/_phase18_db.py` (two seasons, 6 teams, 40 matches).
Defaults (max_goals=8, min_history=5, prior_strength=1.0):

| metric | value |
|---|---|
| n / n_scored | 40 / 40 |
| accuracy | 0.5250 |
| macro-F1 | 0.3449 |
| Brier | 0.6620 |
| log-loss | 1.2453 |
| total‑goals MAE | 1.2688 |
| mean actual / predicted total | 2.55 / 2.6004 |
| over 2.5 Brier | 0.2620 |
| BTTS Brier | 0.2458 |

`db_unchanged = True` (read‑only). Used as a fast, hermetic regression harness; the
genuine production numbers in §6 are the primary reported results.

## 8. Safety
- Production DB write‑checked: SHA‑256 before == after every step
  (`725af574…3122`), both externally and via the `_sha256` guard.
- Only read‑only PRAGMA/SELECT/hash issued against the prod DB; a mismatch raises
  `RuntimeError` and aborts.
- All writes go to `reports/phase20` (and test temp dirs). CLI returns exit code 1 on a
  missing DB / invalid competition / evaluation failure.

## 9. Tests
`tests/test_prediction_poisson.py` — **34 tests**, all green (full suite: 417 passed):

- **PMF correctness** vs `scipy.stats.poisson`; negative/NaN means clamped to the rate floor.
- **Grid**: sums to 1, shape `(max_goals+1)²`, incl. small `max_goals=3`.
- **H/D/A aggregation**: sum to 1; home‑win dominant when `λh >> λ_a`, away‑win dominant the
  reverse; symmetric rates give symmetric home/away (home slightly favoured by grid mass);
  zero rates fall back to the floor with a valid distribution.
- **Over/Under**: exact match to `1 − CDF_Poisson(line; λh+λ_a)`; higher rates → higher over.
- **BTTS**: exact match to the independence formula `sf(0;λh)·sf(0;λ_a)`; near‑zero when one
  rate is ~0.
- **Model**: deterministic for identical inputs; home/away ordering matters; unknown teams
  and empty/missing‑column history degrade to the league prior (valid probs, `home_rate>0`);
  insufficient history uses `prior_strength`.
- **Leakage**: first fixture `history_size==0`; same‑day results never leak (two fixtures on
  one date both get `history_size==0`); determinism end‑to‑end.
- **E2E eval**: read‑only (SHA‑256 equal before/after), `n_scored==n_matches`, metric
  range sanity, confusion‑matrix/class‑distribution shape, goal‑metric keys present;
  determinism across runs; invalid competition raises `ValueError`.
- **Report artefacts**: both JSON + TXT written with expected content.
- **CLI**: `ratings poisson` subparser resolves (`elo` subparser still works); default
  flags; all‑seasons + single‑season smoke; missing DB, invalid competition and negative
  `max_goals` (clamped to 0) handled with correct exit codes.

Run: `.venv\Scripts\python.exe -m pytest tests/test_prediction_poisson.py -q` (34 passed).

## 10. Reproduction
```
cd C:\pridection\pridiction
.venv\Scripts\python.exe -m pytest -q                                          # 417 passed
.venv\Scripts\python.exe -m pytest tests/test_prediction_poisson.py -q          # 34 passed
football-intelligence ratings poisson \
    --db-path data/football_intelligence.db \
    --competition-id 2 --output-dir reports/phase20
```

## 11. Limitations / honest caveats
- **Independent‑Poisson assumption** ignores the well‑known negative score correlation
  (the “0‑0” undercount); this depresses draw probability (only 4/107 draws predicted,
  F1 0.066) and is the main calibration weakness.
- **Attack/defence model is simple**: team strength = `league_mean × attack × defence`
  with a single `prior_strength` fallback; no home‑advantage parameter is estimated
  (home field enters only through the home‑rate being modelled directly).
- **No parameter tuning**: all hyper‑parameters are fixed defaults (max_goals=8,
  min_history=5, prior_strength=1.0). Accuracy 0.44 on a hard 3‑class task with a ~41/28/30
  class prior is weak but honest for an untuned single‑season online model.
- **Single competition, single (two‑season) window**: results do not generalise beyond
  the statsbomb Premier League data present in the DB.
- **Brier > 0.66 / log‑loss ~1.18** reflect poor calibration relative to a trivial class
  prior; the model is a transparent baseline, not a production forecaster.
- `max_goals` truncation is re‑normalised (mass above the grid is redistributed), which
  slightly upweights the `[0..8]` cells; with `max_goals=8` and these rates the truncation
  tail is negligible.

## 12. Next phase
Phase 21 — **model comparison**: run Poisson vs. Phase‑18 logistic regression vs. Phase‑19
Elo on an identical set of chronological test fixtures (same splits, same metrics),
reporting relative accuracy / Brier / log‑loss / macro‑F1 and a paired significance note.
No production DB writes; reuse the read‑only harness already in place.
