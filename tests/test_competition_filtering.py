"""Tests for competition and season filtering in the feature pipeline.

These are synthetic, disposable-database tests (created in pytest's ``tmp_path``).
The production database is never read or written here -- see the read-only
real-data smoke test (``p8_smoke.py``) for the genuine-database verification.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from football_intelligence.features.engineering import (
    FeatureEngineer,
    load_matches_for_features,
)
from football_intelligence.prediction.baseline import (
    chronological_split,
    compute_outcome_label,
    prepare_training_data,
)
from _fixtures import (
    _connect,
    _insert_match,
    _seed_competitions,
    _seed_teams,
    make_initialized_db,
)

STATS = "statsbomb"


def _build_db(tmp_path: Path):
    """Disposable DB with 2 leagues x 2 seasons + 1 CL 'final'.

    Returns (db_path, epl_id, cl_id, ids) where ids[name] = (match_id, provider_match_id).
    """
    db = make_initialized_db(tmp_path)
    with _connect(db) as conn:
        comps = _seed_competitions(conn, [
            (STATS, "2", "Premier League"),
            (STATS, "7", "Champions League"),
        ])
        teams = _seed_teams(conn, [
            (STATS, "T1", "Team A"),
            (STATS, "T2", "Team B"),
            (STATS, "T3", "Team C"),
            (STATS, "T4", "Team D"),
        ])
    epl_id = comps[(STATS, "2")]
    cl_id = comps[(STATS, "7")]
    t1, t2, t3, t4 = (teams[(STATS, f"T{n}")] for n in ("1", "2", "3", "4"))
    rows = [
        ("e1", epl_id, "2015/16", "2016-08-13T15:00:00+00:00", t1, t2, 2, 1),
        ("e2", epl_id, "2015/16", "2016-08-13T17:00:00+00:00", t3, t4, 1, 1),
        ("e_same", epl_id, "2015/16", "2016-08-14T17:00:00+00:00", t1, t4, 3, 0),
        ("e3", epl_id, "2015/16", "2016-08-14T17:00:00+00:00", t1, t3, 0, 2),
        ("e4", epl_id, "2015/16", "2016-08-15T15:00:00+00:00", t1, t4, 3, 0),
        ("e5", epl_id, "2015/16", "2016-08-16T15:00:00+00:00", t2, t3, 1, 1),
        ("e6", epl_id, "2016/17", "2017-08-13T15:00:00+00:00", t1, t2, 1, 0),
        ("e7", epl_id, "2016/17", "2017-08-14T15:00:00+00:00", t3, t4, 2, 2),
        ("e8", epl_id, "2016/17", "2017-08-15T15:00:00+00:00", t1, t3, 2, 1),
        ("c1", cl_id, "2018/19", "2019-06-01T15:00:00+00:00", t1, t4, 1, 2),
    ]
    ids: dict = {}
    with _connect(db) as conn:
        for pmid, comp_id, season, date, h, a, hg, ag in rows:
            mid = _insert_match(
                conn, provider=STATS, provider_match_id=pmid,
                competition_id=comp_id, season_name=season, match_date=date,
                home_team_id=h, away_team_id=a, home_goals=hg, away_goals=ag,
                match_status="available",
            )
            ids[pmid] = (int(mid), pmid)
    return db, epl_id, cl_id, ids


def _count_matches(db) -> int:
    with _connect(db) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])


# 1. Premier League only
def test_filter_premier_league_only(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(db_path=db, provider_competition_id="2", provider=STATS)
    assert len(df) == 9
    assert set(df["competition_id"].unique()) == {epl_id}
    assert ids["c1"][1] not in set(df["provider_match_id"])
    df2 = load_matches_for_features(db_path=db, provider_competition_id="2")
    assert len(df2) == 9
    assert set(df2["competition_id"].unique()) == {epl_id}


# 2. Premier League 2015/16 only
def test_filter_premier_league_2015_16_only(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(
        db_path=db, provider_competition_id="2", season_name="2015/16", provider=STATS
    )
    assert len(df) == 6
    assert set(df["season_name"].unique()) == {"2015/16"}
    assert set(df["competition_id"].unique()) == {epl_id}


# 3. Champions League 'final' excluded from EPL selection
def test_champions_league_final_excluded_from_epl(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(db_path=db, provider_competition_id="2", provider=STATS)
    assert ids["c1"][1] not in set(df["provider_match_id"])
    cl_df = load_matches_for_features(db_path=db, provider_competition_id="7", provider=STATS)
    assert len(cl_df) == 1
    assert ids["c1"][1] in set(cl_df["provider_match_id"])


# 4. Combining competition (local id) + season
def test_combine_competition_id_and_season(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(db_path=db, competition_id=epl_id, season_name="2016/17")
    assert len(df) == 3
    assert set(df["season_name"].unique()) == {"2016/17"}
    assert set(df["competition_id"].unique()) == {epl_id}


# 5. Invalid / unknown / contradictory selection
def test_unknown_provider_competition_raises(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    with pytest.raises(ValueError):
        load_matches_for_features(db_path=db, provider_competition_id="NOPE", provider=STATS)


def test_unknown_competition_id_raises(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    with pytest.raises(ValueError):
        load_matches_for_features(db_path=db, competition_id=999999)


def test_contradictory_filters_raise(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    with pytest.raises(ValueError):
        load_matches_for_features(
            db_path=db, competition_id=epl_id, provider_competition_id="2", provider=STATS
        )


# 6. Empty selection (valid competition, absent season)
def test_empty_selection_returns_empty_dataframe(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(
        db_path=db, provider_competition_id="2", season_name="1999/00", provider=STATS
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# 7. Backward compatibility (no filters -> full set)
def test_no_filters_returns_all_matches(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(db_path=db)
    assert len(df) == 10
    assert set(df["provider_match_id"]) == set(pid for (_, pid) in ids.values())


# 8. Chronological order preserved through the filtered path
def test_chronological_split_disjoint_and_ordered(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(db_path=db, provider_competition_id="2", provider=STATS)
    train_df, val_df, test_df = chronological_split(df, 0.6, 0.2)
    assert train_df["match_date"].max() <= val_df["match_date"].min()
    assert val_df["match_date"].max() <= test_df["match_date"].min()
    ids_in = lambda d: set(d["provider_match_id"])
    assert ids_in(train_df).isdisjoint(ids_in(val_df))
    assert ids_in(val_df).isdisjoint(ids_in(test_df))
    assert ids_in(train_df).isdisjoint(ids_in(test_df))


# 9. Leakage prevention through the filtered feature path
def test_filtered_features_one_row_per_match_no_cl(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    fe = FeatureEngineer()
    feats = fe.compute_all_features_from_db(
        db_path=db, provider_competition_id="2", provider=STATS
    )
    assert len(feats) == 9
    assert feats["features_valid"].all()
    assert ids["c1"][0] not in set(feats["target_match_id"].astype(int))
    assert len(set(feats["target_match_id"])) == 9


def test_same_datetime_excluded_from_history(tmp_path):
    """Target e3 (2016-08-14T17:00): earlier e1 (2016-08-13 win = 3 pts) counts;
    the same-datetime e_same fixture and all later fixtures are excluded."""
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    fe = FeatureEngineer()
    df = load_matches_for_features(db_path=db, provider_competition_id="2", provider=STATS)
    feats = fe.compute_features(df, ids["e3"][0])
    assert feats["features_valid"] is True
    assert feats["home_points_last_5"] == 3
    assert feats["home_matches_played_last_5"] == 1


def test_labels_valid_and_dates_present(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(db_path=db, provider_competition_id="2", provider=STATS)
    assert df["match_date"].notna().all()
    for _, row in df.iterrows():
        assert compute_outcome_label(row["home_goals"], row["away_goals"]) is not None


# 10. The loader is read-only on a disposable database
def test_load_matches_for_features_is_read_only(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    before_hash = hashlib.sha256(Path(db).read_bytes()).hexdigest()
    before_count = _count_matches(db)
    load_matches_for_features(db_path=db, provider_competition_id="2", provider=STATS)
    load_matches_for_features(db_path=db)
    after_hash = hashlib.sha256(Path(db).read_bytes()).hexdigest()
    after_count = _count_matches(db)
    assert before_hash == after_hash
    assert before_count == after_count


# Pipeline integration: training-data construction on the filtered scope
def test_prepare_training_data_on_filtered_scope(tmp_path):
    db, epl_id, cl_id, ids = _build_db(tmp_path)
    df = load_matches_for_features(db_path=db, provider_competition_id="2", provider=STATS)
    train_df, _, _ = chronological_split(df.sort_values("match_date"), 0.6, 0.2)
    valid, X, y = prepare_training_data(train_df, engineer=FeatureEngineer())
    assert len(X) == len(y)
    assert len(X) > 0
    assert set(y.tolist()).issubset({0, 1, 2})