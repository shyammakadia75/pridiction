"""Tests for the feature-engineering module.

All tests use synthetic DataFrames with clearly-labelled test data.
No real data is read or written.

Test coverage follows Section 7 of the Part 4 specification:
 1. Valid historical matches
 2. Missing required fields
 3. Duplicate provider identifiers
 4. Invalid scores
 5. Home team equals away team
 6. Teams with no prior matches
 7. Teams with fewer than N previous matches
 8. Correct rolling points and goal statistics
 9. Correct home/away splits
10. Chronological ordering independent of database row order
11. Future-match leakage prevention
12. Target-match score leakage prevention
13. Same-day fixtures with unknown kickoff order
14. Missing dates and malformed records
15. Deterministic output across repeated runs
16. Regression: quality checks still function
"""
from __future__ import annotations

import pandas as pd
import pytest

from football_intelligence.features.engineering import FeatureEngineer
from football_intelligence.data.quality import check_match_data_quality
from _fixtures import _insert_match, make_seeded_db, _connect


def _make_history_df():
    """Return a small synthetic match history as a DataFrame.

    Four teams (IDs 1-4), six completed matches spread across several days.
    All dates are clearly synthetic test data.

    Match timeline:
      m1: 2024-08-01, team1 2-1 team2  (team1 home win, 3 pts)
      m2: 2024-08-01, team3 1-1 team4  (draw, 1 pt each)
      m3: 2024-08-05, team1 0-3 team3  (team3 away win)
      m4: 2024-08-05, team2 1-2 team4  (team4 home win)
      m5: 2024-08-08, team1 3-0 team4  (team1 home win)
      m6: 2024-08-10, team1 1-1 team2  (draw)

    Target fixture (not in history):
      t99: 2024-08-12, team1 (home) vs team2 (away)
    """
    return pd.DataFrame([
        {"match_id": 1, "provider": "test", "provider_match_id": "h1",
         "competition_id": 1, "season_name": "2024/25",
         "match_date": "2024-08-01T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 1, "match_status": "available"},
        {"match_id": 2, "provider": "test", "provider_match_id": "h2",
         "competition_id": 1, "season_name": "2024/25",
         "match_date": "2024-08-01T17:00:00+00:00",
         "home_team_id": 3, "away_team_id": 4,
         "home_goals": 1, "away_goals": 1, "match_status": "available"},
        {"match_id": 3, "provider": "test", "provider_match_id": "h3",
         "competition_id": 1, "season_name": "2024/25",
         "match_date": "2024-08-05T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 3,
         "home_goals": 0, "away_goals": 3, "match_status": "available"},
        {"match_id": 4, "provider": "test", "provider_match_id": "h4",
         "competition_id": 1, "season_name": "2024/25",
         "match_date": "2024-08-05T18:00:00+00:00",
         "home_team_id": 2, "away_team_id": 4,
         "home_goals": 1, "away_goals": 2, "match_status": "available"},
        {"match_id": 5, "provider": "test", "provider_match_id": "h5",
         "competition_id": 1, "season_name": "2024/25",
         "match_date": "2024-08-08T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 4,
         "home_goals": 3, "away_goals": 0, "match_status": "available"},
        {"match_id": 6, "provider": "test", "provider_match_id": "h6",
         "competition_id": 1, "season_name": "2024/25",
         "match_date": "2024-08-10T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 1, "away_goals": 1, "match_status": "available"},
    ])


def _make_history_df_shuffled():
    """Same as _make_history_df but rows in reverse order."""
    df = _make_history_df()
    return df.iloc[::-1].reset_index(drop=True)


def _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                       home=1, away=2, hg=None, ag=None, status="scheduled"):
    """Append a target fixture row to the DataFrame."""
    row = {"match_id": mid, "provider": "test", "provider_match_id": f"t{mid}",
           "competition_id": 1, "season_name": "2024/25",
           "match_date": date, "home_team_id": home, "away_team_id": away,
           "home_goals": hg, "away_goals": ag, "match_status": status}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)



# ---------------------------------------------------------------------------
# 1. Valid historical matches
# ---------------------------------------------------------------------------

def test_valid_history_produces_features():
    """A valid target match with prior history produces valid features."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
    assert feats["features_valid"] is True
    assert feats["target_match_id"] == 99
    assert feats["home_team_id"] == 1
    assert feats["away_team_id"] == 2


def test_computed_features_match_expected_values():
    """Spot-check specific feature values against hand-computed expectations."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 1 (home): m1(W,2-1), m3(L,0-3), m5(W,3-0), m6(D,1-1)
    # Points: 3+0+3+1=7, GS: 2+0+3+1=6, GC: 1+3+0+1=5
    assert feats["home_matches_played_last_5"] == 4
    assert feats["home_points_last_5"] == 7
    assert feats["home_wins_last_5"] == 2
    assert feats["home_draws_last_5"] == 1
    assert feats["home_losses_last_5"] == 1
    assert feats["home_goals_scored_last_5"] == 6
    assert feats["home_goals_conceded_last_5"] == 5
    assert feats["home_goal_diff_last_5"] == 1

    # Team 2 (away): m1(L,1-2), m4(L,2-1), m6(D,1-1)
    assert feats["away_matches_played_last_5"] == 3
    assert feats["away_points_last_5"] == 1
    assert feats["away_wins_last_5"] == 0
    assert feats["away_losses_last_5"] == 2

        # H2H (team1 vs team2): m1(W), m6(D)
    assert feats["h2h_total_matches"] == 2
    assert feats["h2h_home_wins"] == 1


# ---------------------------------------------------------------------------
# 2. Missing required fields
# ---------------------------------------------------------------------------

def test_missing_match_date_features_invalid():
    """A target match with NULL match_date cannot compute features."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date=None, hg=2, ag=1)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
    assert feats["features_valid"] is False


# ---------------------------------------------------------------------------
# 3. Duplicate provider identifiers
# ---------------------------------------------------------------------------

def test_duplicate_provider_id_does_not_affect_features():
    """Duplicate provider_match_id in history does not crash feature extraction."""
    df = _make_history_df()
    dup = df.iloc[[0]].copy()
    dup["match_id"] = 999
    df = pd.concat([df, dup], ignore_index=True)
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
    assert feats["features_valid"] is True


# ---------------------------------------------------------------------------
# 4. Invalid scores (negative goals)
# ---------------------------------------------------------------------------

def test_negative_goals_prevented_by_schema(tmp_path):
    """The DB schema's CHECK constraint rejects negative goals."""
    db = make_seeded_db(tmp_path)
    with _connect(db) as conn:
        with pytest.raises(Exception):
            _insert_match(conn, provider_match_id="bad", home_team_id=1,
                           away_team_id=2, match_date="2024-08-01T15:00:00+00:00",
                           home_goals=-1, away_goals=1)


def test_negative_goals_in_dataframe_handled():
    """Negative goals in a DataFrame are still processed (caller's responsibility)."""
    df = _make_history_df()
    df.loc[df["match_id"] == 1, "home_goals"] = -1
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
    assert feats["features_valid"] is True


# ---------------------------------------------------------------------------
# 5. Home team equals away team
# ---------------------------------------------------------------------------

def test_same_home_away_team_handled():
    """Feature engine processes self-play fixtures without crashing."""
    df = _make_history_df()
    df = _add_target_match(df, home=1, away=1, hg=2, ag=1)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
    assert feats["features_valid"] is True
    assert feats["target_match_id"] == 99


# ---------------------------------------------------------------------------
# 6. Teams with no prior matches (cold start)
# ---------------------------------------------------------------------------

def test_team_with_no_prior_matches():
    """A team with no completed matches before the target gets NaN stats."""
    df = _make_history_df()
    target = _add_target_match(df, mid=99, home=5, away=6, hg=1, ag=0)
    fe = FeatureEngineer()
    feats = fe.compute_features(target, 99)
    assert feats["home_matches_played_last_5"] == 0
    assert feats["away_matches_played_last_5"] == 0
    assert pd.isna(feats["home_points_last_5"])
    assert pd.isna(feats["away_points_last_5"])


# ---------------------------------------------------------------------------
# 7. Teams with fewer than N previous matches
# ---------------------------------------------------------------------------

def test_team_with_fewer_than_n_matches():
    """A team with only 3 prior matches gets correct stats."""
    df = _make_history_df()
    target = _add_target_match(df, mid=99, home=2, away=3, hg=1, ag=0)
    fe = FeatureEngineer()
    feats = fe.compute_features(target, 99)
    assert feats["home_matches_played_last_5"] == 3
    assert feats["home_points_last_5"] == 1
    assert feats["home_wins_last_5"] == 0
    assert feats["home_draws_last_5"] == 1
    assert feats["home_losses_last_5"] == 2
    assert feats["home_goals_scored_last_5"] == 3
    assert feats["home_goals_conceded_last_5"] == 5
    assert feats["home_goal_diff_last_5"] == -2


# ---------------------------------------------------------------------------
# 8. Window 5 vs Window 10
# ---------------------------------------------------------------------------

def test_window_5_vs_10_different_results():
    """With 11+ prior matches, last_5 and last_10 windows differ."""
    df = _make_history_df()
    extra = pd.DataFrame([{
        "match_id": 100 + i, "provider": "test",
        "provider_match_id": f"extra-{i}",
        "competition_id": 1, "season_name": "2024/25",
        "match_date": f"2024-07-{i+1:02d}T15:00:00+00:00",
        "home_team_id": 1, "away_team_id": 3,
        "home_goals": 1, "away_goals": 0, "match_status": "available",
    } for i in range(8)])
    df = pd.concat([df, extra], ignore_index=True)
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
    assert feats["home_matches_played_last_5"] == 5
    assert feats["home_matches_played_last_10"] == 10
    assert feats["home_points_last_5"] != feats["home_points_last_10"]


# ---------------------------------------------------------------------------
# 9. Correct home/away venue splits
# ---------------------------------------------------------------------------

def test_correct_home_away_splits():
    """Venue-specific features match hand-computed expectations."""
    df = _make_history_df()
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
        # Team 1 home matches: m1(W,2-1), m3(L,0-3), m5(W,3-0), m6(D,1-1)
    # All 4 matches at home, 7 pts, GS=6, GC=5
    assert feats["home_matches_at_home_last_5"] == 4
    assert feats["home_points_at_home_last_5"] == 7
    assert feats["home_goals_scored_at_home_last_5"] == 6
    assert feats["home_goals_conceded_at_home_last_5"] == 5
    # Team 2 away matches: m1(L,1-2), m6(D,1-1) -> 1 pt, GS=2, GC=3
    assert feats["away_matches_at_away_last_5"] == 2
    assert feats["away_points_at_away_last_5"] == 1
    assert feats["away_goals_scored_at_away_last_5"] == 2
    assert feats["away_goals_conceded_at_away_last_5"] == 3


# ---------------------------------------------------------------------------
# 10. Row order independence
# ---------------------------------------------------------------------------

def test_row_order_does_not_affect_features():
    """Features are the same regardless of DataFrame row order."""
    df = _make_history_df()
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats_normal = fe.compute_features(df, 99)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    feats_shuffled = fe.compute_features(df_shuffled, 99)
    for key in ("home_points_last_5", "home_goals_scored_last_5",
                "away_points_last_5", "away_goals_conceded_last_5"):
        assert feats_normal[key] == feats_shuffled[key], (
            f"Mismatch in {key}: {feats_normal[key]} vs {feats_shuffled[key]}"
        )


# ---------------------------------------------------------------------------
# 11. Future-match leakage prevention
# ---------------------------------------------------------------------------

def test_future_match_does_not_leak():
    """Adding a future match must not change earlier features."""
    df = _make_history_df()
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats_without = fe.compute_features(df, 99)
    future = pd.DataFrame([{
        "match_id": 100, "provider": "test", "provider_match_id": "future1",
        "competition_id": 1, "season_name": "2024/25",
        "match_date": "2024-08-15T15:00:00+00:00",
        "home_team_id": 1, "away_team_id": 2,
        "home_goals": 5, "away_goals": 0, "match_status": "available",
    }])
    df_with = pd.concat([df, future], ignore_index=True)
    feats_with = fe.compute_features(df_with, 99)
    for key in feats_without:
        assert feats_without[key] == feats_with[key], (
            f"Future match leaked into feature '{key}': "
            f"{feats_without[key]} vs {feats_with[key]}"
        )


# ---------------------------------------------------------------------------
# 12. Target-match score leakage prevention
# ---------------------------------------------------------------------------

def test_target_match_score_not_used():
    """The target match's own score must not appear in its features."""
    df = _make_history_df()
    df_no_score = _add_target_match(df, mid=99, hg=None, ag=None)
    feats_no = FeatureEngineer().compute_features(df_no_score, 99)
    df_with_score = _add_target_match(df, mid=99, hg=9, ag=8)
    feats_yes = FeatureEngineer().compute_features(df_with_score, 99)
    for key in feats_no:
        if key == "target_match_id":
            continue
        assert feats_no[key] == feats_yes[key], (
            f"Target score leaked into feature '{key}': "
            f"{feats_no[key]} vs {feats_yes[key]}"
        )



# ---------------------------------------------------------------------------
# 13. Same-day fixtures with unknown kickoff order
# ---------------------------------------------------------------------------

def test_same_day_match_excluded_from_history():
    """A match with the same datetime as the target is excluded from history."""
    df = _make_history_df()
    # Target has SAME datetime as m6 (2024-08-10T15:00:00)
    target = _add_target_match(
        df, mid=99, home=1, away=2,
        date="2024-08-10T15:00:00+00:00", hg=2, ag=1
    )
    fe = FeatureEngineer()
    feats = fe.compute_features(target, 99)
    # m6 excluded (same datetime) -> team1 has 3: m1(W), m3(L), m5(W) -> 6 pts
    assert feats["home_matches_played_last_5"] == 3
    assert feats["home_points_last_5"] == 6
    # m6 excluded -> team2 has 2: m1(L), m4(L) -> 0 pts
    assert feats["away_matches_played_last_5"] == 2
    assert feats["away_points_last_5"] == 0


# ---------------------------------------------------------------------------
# 14. Missing dates and malformed records
# ---------------------------------------------------------------------------

def test_missing_date_in_history_excluded():
    """A history match with a NULL date is excluded from features."""
    df = _make_history_df()
    df.loc[df["match_id"] == 1, "match_date"] = None
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
    # m1 excluded -> team1 has 3 matches: m3(L,0-3), m5(W,3-0), m6(D,1-1) -> 4 pts
    assert feats["home_matches_played_last_5"] == 3
    assert feats["home_points_last_5"] == 4


def test_malformed_date_in_history_excluded():
    """A history match with an unparseable date is excluded."""
    df = _make_history_df()
    df.loc[df["match_id"] == 3, "match_date"] = "garbage"
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)
    # m3 excluded -> team1 has 3 matches: m1(W), m5(W), m6(D) -> 7 pts
    assert feats["home_points_last_5"] == 7


# ---------------------------------------------------------------------------
# 15. Deterministic output across repeated runs
# ---------------------------------------------------------------------------

def test_deterministic_output():
    """Running feature extraction twice yields identical results."""
    df = _make_history_df()
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    feats1 = fe.compute_features(df, 99)
    feats2 = fe.compute_features(df, 99)
    assert feats1 == feats2


def test_deterministic_all_features():
    """compute_all_features produces identical DataFrames on repeated runs."""
    df = _make_history_df()
    df = _add_target_match(df, hg=2, ag=1)
    fe = FeatureEngineer()
    result1 = fe.compute_all_features(df)
    result2 = fe.compute_all_features(df)
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# 16. DB integration and regression
# ---------------------------------------------------------------------------

def test_compute_all_features_from_db(tmp_path):
    """End-to-end: initialize DB, insert matches, compute features."""
    db = make_seeded_db(tmp_path)
    with _connect(db) as conn:
        _insert_match(conn, provider_match_id="m1", home_team_id=1,
                       away_team_id=2, match_date="2024-08-01T15:00:00+00:00",
                       home_goals=2, away_goals=1, match_status="available")
        _insert_match(conn, provider_match_id="m2", home_team_id=3,
                       away_team_id=4, match_date="2024-08-02T15:00:00+00:00",
                       home_goals=1, away_goals=0, match_status="available")
        _insert_match(conn, provider_match_id="m3", home_team_id=1,
                       away_team_id=3, match_date="2024-08-05T15:00:00+00:00",
                       home_goals=0, away_goals=2, match_status="available")
    fe = FeatureEngineer()
    df = fe.compute_all_features_from_db(db)
    assert len(df) == 3
    assert "features_valid" in df.columns
    assert "home_points_last_5" in df.columns
        # First match has no history -> NaN
    m1 = df[df["target_match_id"] == 1].iloc[0]
    assert pd.isna(m1["home_points_last_5"])


def test_existing_quality_checks_still_work(tmp_path):
    """Regression: data quality checks still function."""
    db = make_seeded_db(tmp_path)
    with _connect(db) as conn:
        _insert_match(conn, provider_match_id="ok", home_team_id=1,
                       away_team_id=2, match_date="2024-08-12T15:00:00+00:00",
                       home_goals=2, away_goals=1)
    report = check_match_data_quality(db)
    assert report.is_valid is True


# ---------------------------------------------------------------------------
# 17. Phase 26.5: last-five-match form features (per-match averages,
#    clean sheets, venue form, recency-weighted points)
# ---------------------------------------------------------------------------


def test_per_match_averages_for_both_teams():
    """points_per_match, avg_goals_scored/conceded computed for last 5."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 1 (home): 4 matches, pts=7, GS=6, GC=5
    assert feats["home_matches_played_last_5"] == 4
    assert feats["home_points_per_match_last_5"] == 7 / 4
    assert feats["home_avg_goals_scored_last_5"] == 6 / 4
    assert feats["home_avg_goals_conceded_last_5"] == 5 / 4

    # Team 2 (away): 3 matches, pts=1, GS=3, GC=5
    assert feats["away_matches_played_last_5"] == 3
    assert feats["away_points_per_match_last_5"] == 1 / 3
    assert feats["away_avg_goals_scored_last_5"] == 3 / 3
    assert feats["away_avg_goals_conceded_last_5"] == 5 / 3


def test_clean_sheets_and_matches_scored():
    """Clean sheets = matches where the team conceded 0; scored = matches where GF > 0."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 1 (home): m1(2-1,CS=no), m3(0-3,CS=no), m5(3-0,CS=yes), m6(1-1,CS=no)
    assert feats["home_clean_sheets_last_5"] == 1
    assert feats["home_matches_scored_last_5"] == 3  # m3 had 0 goals

    # Team 2 (away): m1(1-2,CS=no), m4(1-2,CS=no), m6(1-1,CS=no)
    assert feats["away_clean_sheets_last_5"] == 0
    assert feats["away_matches_scored_last_5"] == 3  # all scored 1




# ---------------------------------------------------------------------------
# 17. Phase 26.5: venue-specific and recency-weighted form features
# ---------------------------------------------------------------------------


def test_venue_specific_form_home_team():
    """Home-team venue features reflect only that team's home matches."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 1 home matches: m1(W,2-1), m3(L,0-3), m5(W,3-0), m6(D,1-1)
    assert feats["home_matches_at_home_last_5"] == 4
    assert feats["home_points_at_home_last_5"] == 7
    assert feats["home_wins_at_home_last_5"] == 2
    assert feats["home_draws_at_home_last_5"] == 1
    assert feats["home_losses_at_home_last_5"] == 1
    assert feats["home_goals_scored_at_home_last_5"] == 6
    assert feats["home_goals_conceded_at_home_last_5"] == 5
    assert feats["home_goal_diff_at_home_last_5"] == 1
    assert feats["home_clean_sheets_at_home_last_5"] == 1
    assert feats["home_points_per_match_at_home_last_5"] == 7 / 4


def test_venue_specific_form_away_team():
    """Away-team venue features reflect only that team's away matches."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 2 away matches: m1(L,1-2), m6(D,1-1)
    assert feats["away_matches_at_away_last_5"] == 2
    assert feats["away_points_at_away_last_5"] == 1
    assert feats["away_wins_at_away_last_5"] == 0
    assert feats["away_draws_at_away_last_5"] == 1
    assert feats["away_losses_at_away_last_5"] == 1
    assert feats["away_goals_scored_at_away_last_5"] == 2
    assert feats["away_goals_conceded_at_away_last_5"] == 3
    assert feats["away_goal_diff_at_away_last_5"] == -1
    assert feats["away_clean_sheets_at_away_last_5"] == 0
    assert feats["away_points_per_match_at_away_last_5"] == 1 / 2


def test_recency_weighted_points():
    """Recency-weighted points use linear weights 1..n chronologically."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 1: pts=[3, 0, 3, 1], weights=[1, 2, 3, 4] -> 16/10 = 1.6
    assert feats["home_recency_weighted_points_last_5"] == 1.6
    # Team 2: pts=[0, 0, 1], weights=[1, 2, 3] -> 3/6 = 0.5
    assert feats["away_recency_weighted_points_last_5"] == 0.5


def test_no_history_returns_none_for_form_features():
    """When a team has no prior matches, form features are NaN."""
    df = _add_target_match(
        _make_history_df().iloc[0:0], mid=99, hg=None, ag=None,
        home=999, away=888, status="available",
    )
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    assert feats["home_matches_played_last_5"] == 0
    assert pd.isna(feats["home_points_per_match_last_5"])
    assert pd.isna(feats["home_avg_goals_scored_last_5"])
    assert pd.isna(feats["home_recency_weighted_points_last_5"])


def test_fewer_than_five_matches_returns_correct_count():
    """Teams with fewer than 5 prior matches still get full stats."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=None, ag=None,
                           home=3, away=4, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 3 appears in m2 (draw) and m3 (loss) -> 2 matches
    assert feats["home_matches_played_last_5"] == 2


def test_future_matches_excluded():
    """Matches on or after the fixture date are excluded from history."""
    df = _make_history_df()
    df = _add_target_match(df, mid=98, hg=5, ag=0,
                           date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, status="available")
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    assert feats["home_matches_played_last_5"] == 4
    assert feats["home_points_last_5"] == 7


def test_form_features_with_zero_goals_conceded():
    """Clean sheet counted when a team concedes exactly 0."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    assert feats["home_clean_sheets_at_home_last_5"] == 1


def test_all_new_form_features_present_in_feature_columns():
    """All Phase 26.5 feature names appear in compute_all_features()."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    cols = list(fe.compute_all_features(df).columns)

    expected_both = [
    "points_per_match_last_5", "avg_goals_scored_last_5",
    "avg_goals_conceded_last_5", "clean_sheets_last_5",
    "matches_scored_last_5", "recency_weighted_points_last_5",
    "recency_matches_last_5",
    ]
    expected_home_only = [
        "points_at_home_last_5", "wins_at_home_last_5",
        "draws_at_home_last_5", "losses_at_home_last_5",
        "goals_scored_at_home_last_5", "goals_conceded_at_home_last_5",
        "goal_diff_at_home_last_5", "clean_sheets_at_home_last_5",
        "points_per_match_at_home_last_5", "matches_at_home_last_5",
    ]
    expected_away_only = [
        "points_at_away_last_5", "wins_at_away_last_5",
        "draws_at_away_last_5", "losses_at_away_last_5",
        "goals_scored_at_away_last_5", "goals_conceded_at_away_last_5",
        "goal_diff_at_away_last_5", "clean_sheets_at_away_last_5",
        "points_per_match_at_away_last_5", "matches_at_away_last_5",
    ]
    for base in expected_both:
        assert f"home_{base}" in cols, f"home_{base} missing"
        assert f"away_{base}" in cols, f"away_{base} missing"
    for base in expected_home_only:
        assert f"home_{base}" in cols, f"home_{base} missing"
    for base in expected_away_only:
        assert f"away_{base}" in cols, f"away_{base} missing"
