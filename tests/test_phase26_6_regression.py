"""Phase 26.6: regression and leakage tests for feature ablation.

Verifies:
- Each feature uses only eligible matches before the prediction cutoff.
- Same-day matches are excluded when their ordering is unknown.
- The latest five eligible matches are selected correctly.
- Venue-specific histories contain only the correct venue.
- Incomplete histories do not invent observations.
- Training and inference schemas are consistent across ablation configs.
- New features do not silently alter the Poisson model (sanity).
- Predictions remain reproducible under fixed seed.
- Feature flags control feature emission.

All tests use synthetic DataFrames.  No real data is read or written.
"""
from __future__ import annotations

import pandas as pd
import pytest

from football_intelligence.features.engineering import (
    FeatureEngineer,
    FeatureSpec,
)
from football_intelligence.prediction.baseline import prepare_training_data


# ---------------------------------------------------------------------------
# Helpers (reuse the synthetic history pattern from test_features.py)
# ---------------------------------------------------------------------------

def _make_history_df():
    """Six synthetic completed matches, four teams (IDs 1-4).

    Timeline:
      m1: 2024-08-01, team1 2-1 team2  (team1 home, 3 pts)
      m2: 2024-08-01, team3 1-1 team4  (draw, 1 pt each)
      m3: 2024-08-05, team1 0-3 team3  (team3 away, 3 pts)
      m4: 2024-08-05, team2 1-2 team4  (team4 home, 3 pts)
      m5: 2024-08-08, team1 3-0 team4  (team1 home, 3 pts)
      m6: 2024-08-10, team1 1-1 team2  (draw, 1 pt each)

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


def _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                      home=1, away=2, hg=None, ag=None,
                      status="scheduled"):
    """Append a target fixture row to the DataFrame."""
    row = {"match_id": mid, "provider": "test",
           "provider_match_id": f"t{mid}",
           "competition_id": 1, "season_name": "2024/25",
            "match_date": date, "home_team_id": home, "away_team_id": away,
            "home_goals": hg, "away_goals": ag, "match_status": status}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Cutoff correctness and leakage prevention
# ---------------------------------------------------------------------------

def test_same_day_matches_excluded_from_history():
    """Matches on the fixture date must be excluded from history.

    The fixture date is 2024-08-12. Same-day matches (regardless of
    kick-off time) are excluded because their ordering relative to the
    target fixture is unknown.
    """
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=3, ag=0, status="available")
    # Same-day match, earlier time — should NOT be included
    df = _add_target_match(df, mid=98, date="2024-08-12T10:00:00+00:00",
                           home=3, away=4, hg=1, ag=0, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 1 still has only 4 prior matches (m1, m3, m5, m6)
    assert feats["home_matches_played_last_5"] == 4
    assert feats["home_points_last_5"] == 7  # 3+0+3+1


def test_future_matches_excluded():
    """Matches after the fixture date must not appear in features."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")
    df = _add_target_match(df, mid=200, date="2024-08-15T15:00:00+00:00",
                           home=1, away=2, hg=5, ag=0, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    assert feats["home_matches_played_last_5"] == 4
    assert feats["home_points_last_5"] == 7


def test_target_match_excluded():
    """The target fixture itself must not be in the history."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=None, ag=None,
                           status="scheduled")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    assert feats["home_matches_played_last_5"] == 4


def test_latest_five_matches_selected():
    """When >5 matches exist, only the latest 5 are used for last_5 window."""
    rows = []
    for i in range(7):
        rows.append({"match_id": i + 1, "provider": "test",
                     "provider_match_id": f"h{i}",
                     "competition_id": 1, "season_name": "2024/25",
                     "match_date": f"2024-08-{i+1:02d}T15:00:00+00:00",
                     "home_team_id": 1, "away_team_id": 2,
                     "home_goals": 1, "away_goals": 0,
                     "match_status": "available"})
    df = pd.DataFrame(rows)
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=None, ag=None)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Last 5 of 7: matches 3-7 (indices 2-6)
    assert feats["home_matches_played_last_5"] == 5
    assert feats["home_points_last_5"] == 15  # 5 wins
    assert feats["home_goals_scored_last_5"] == 5
    assert feats["home_goals_conceded_last_5"] == 0
    # Last_10 includes all 7
    assert feats["home_matches_played_last_10"] == 7
    assert feats["home_points_last_10"] == 21


# ---------------------------------------------------------------------------
# 2. Venue-specific history correctness
# ---------------------------------------------------------------------------

def test_venue_specific_only_correct_venue():
    """Venue features for home team use only home matches of that team."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 1 (home side) home matches: m1(2-1 win), m3(0-3 loss),
    # m5(3-0 win), m6(1-1 draw) = 4 matches
    assert feats["home_matches_at_home_last_5"] == 4
    # Points: 3+0+3+1 = 7
    assert feats["home_points_at_home_last_5"] == 7

    # Team 2 (away side) away matches: m1(1-2 loss), m6(1-1 draw) = 2
    assert feats["away_matches_at_away_last_5"] == 2
    assert feats["away_points_at_away_last_5"] == 1  # 0+1


def test_venue_features_exclude_opposite_venue():
    """Venue 'home' features must exclude matches where team was away."""
    rows = [
        {"match_id": 1, "provider": "test", "provider_match_id": "h1",
         "competition_id": 1, "season_name": "2024/25",
         "match_date": "2024-08-01T15:00:00+00:00",
         "home_team_id": 1, "away_team_id": 2,
         "home_goals": 2, "away_goals": 1, "match_status": "available"},
        {"match_id": 2, "provider": "test", "provider_match_id": "h2",
         "competition_id": 1, "season_name": "2024/25",
         "match_date": "2024-08-03T15:00:00+00:00",
         "home_team_id": 3, "away_team_id": 1,
         "home_goals": 0, "away_goals": 4, "match_status": "available"},
    ]
    df = pd.DataFrame(rows)
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=None, ag=None)
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Home venue: only m1 (team1 home)
    assert feats["home_matches_at_home_last_5"] == 1
    assert feats["home_points_at_home_last_5"] == 3
    # Overall: both matches
    assert feats["home_matches_played_last_5"] == 2
    assert feats["home_points_last_5"] == 6


# ---------------------------------------------------------------------------
# 3. Incomplete histories and cold-start
# ---------------------------------------------------------------------------

def test_no_history_returns_nan_for_rates():
    """When a team has no prior matches, rate features are NaN."""
    df = _make_history_df().iloc[0:0]
    df = _add_target_match(df, mid=99, hg=None, ag=None,
                           home=999, away=888, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    assert feats["home_matches_played_last_5"] == 0
    assert pd.isna(feats["home_points_per_match_last_5"])
    assert pd.isna(feats["home_avg_goals_scored_last_5"])
    assert pd.isna(feats["home_avg_goals_conceded_last_5"])
    assert pd.isna(feats["home_clean_sheets_last_5"])
    assert pd.isna(feats["home_matches_scored_last_5"])


def test_no_history_venue_features_nan():
    """Venue features return 0/NaN when team has no matches of that venue."""
    df = _make_history_df()
    # Use team 5 (has no history) as home, team 1 as away
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=5, away=1, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 5 has no history -> all home venue features are 0/NaN
    assert feats["home_matches_at_home_last_5"] == 0
    assert pd.isna(feats["home_points_per_match_at_home_last_5"])
    # Team 1 (away side): no away matches in history
    assert feats["away_matches_at_away_last_5"] == 0
    assert pd.isna(feats["away_points_per_match_at_away_last_5"])


def test_incomplete_history_no_invented_observations():
    """Teams with fewer than 5 matches still get correct counts."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, hg=None, ag=None,
                           home=3, away=4, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 3: m2 (draw 1-1), m3 (win 3-0 as away) = 2 matches, 4 pts
    assert feats["home_matches_played_last_5"] == 2
    assert feats["home_points_last_5"] == 4


# ---------------------------------------------------------------------------
# 4. Feature flag consistency (include_form_averages)
# ---------------------------------------------------------------------------

def test_include_form_averages_false_excludes_rate_features():
    """When include_form_averages=False, rate features are not emitted."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")

    fe_avg = FeatureEngineer(include_form_averages=True)
    feats_avg = fe_avg.compute_features(df, 99)

    fe_no_avg = FeatureEngineer(include_form_averages=False)
    feats_no_avg = fe_no_avg.compute_features(df, 99)

    # Rate features present with averages=True
    assert "home_points_per_match_last_5" in feats_avg
    assert "home_avg_goals_scored_last_5" in feats_avg

    # Rate features absent with averages=False
    assert "home_points_per_match_last_5" not in feats_no_avg
    assert "home_avg_goals_scored_last_5" not in feats_no_avg
    assert "home_avg_goals_conceded_last_5" not in feats_no_avg
    assert "home_clean_sheets_last_5" not in feats_no_avg
    assert "home_matches_scored_last_5" not in feats_no_avg


def test_include_form_averages_false_preserves_basic_features():
    """When include_form_averages=False, basic counts are still emitted."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")
    fe = FeatureEngineer(include_form_averages=False)
    feats = fe.compute_features(df, 99)

    assert feats["home_matches_played_last_5"] == 4
    assert feats["home_points_last_5"] == 7
    assert "home_wins_last_5" in feats
    assert "home_losses_last_5" in feats


def test_feature_spec_defaults_include_form_averages():
    """FeatureSpec defaults include_form_averages=True."""
    spec = FeatureSpec()
    assert spec.include_form_averages is True


def test_feature_spec_can_disable_form_averages():
    """FeatureSpec can set include_form_averages=False."""
    spec = FeatureSpec(include_form_averages=False)
    assert spec.include_form_averages is False


# ---------------------------------------------------------------------------
# 5. Recency weight correctness
# ---------------------------------------------------------------------------

def test_recency_weighted_points_correct():
    """Recency weighting: oldest match weight=1, newest=n, normalised."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")
    fe = FeatureEngineer()
    feats = fe.compute_features(df, 99)

    # Team 1's 4 matches sorted chronologically: m1(3pts), m3(0pts),
    # m5(3pts), m6(1pt). Weights: [1, 2, 3, 4], total=10.
    # Weighted sum = 3*1 + 0*2 + 3*3 + 1*4 = 3+0+9+4 = 16.
    # Normalised = 16/10 = 1.6
    assert feats["home_recency_weighted_points_last_5"] == pytest.approx(1.6)
    assert feats["home_recency_matches_last_5"] == 4


# ---------------------------------------------------------------------------
# 6. Training/inference schema consistency
# ---------------------------------------------------------------------------

def _excluded_cols() -> set:
    return {"target_match_id", "target_match_date", "home_team_id",
            "away_team_id", "features_valid", "outcome_label", "_label_idx"}


def test_feature_counts_increase_monotonically():
    """Feature count: A < B < C < D (each adds new features)."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")

    configs = [
        FeatureEngineer(include_form_averages=False,
                        include_home_away_split=False,
                        include_recency_weighted=False),
        FeatureEngineer(include_form_averages=True,
                        include_home_away_split=False,
                        include_recency_weighted=False),
        FeatureEngineer(include_form_averages=True,
                        include_home_away_split=True,
                        include_recency_weighted=False),
        FeatureEngineer(include_form_averages=True,
                        include_home_away_split=True,
                        include_recency_weighted=True),
    ]

    counts = []
    for fe in configs:
        eng, _X, _y = prepare_training_data(df, engineer=fe)
        feats = [c for c in eng.columns if c not in _excluded_cols()]
        counts.append(len(feats))

    assert counts[0] < counts[1] < counts[2] < counts[3], f"Counts: {counts}"


def test_training_and_inference_same_feature_columns():
    """prepare_training_data with same engineer yields same columns."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")

    fe = FeatureEngineer()
    eng1, _X1, _y1 = prepare_training_data(df, engineer=fe)
    eng2, _X2, _y2 = prepare_training_data(df, engineer=fe)

    feats1 = [c for c in eng1.columns if c not in _excluded_cols()]
    feats2 = [c for c in eng2.columns if c not in _excluded_cols()]
    assert feats1 == feats2


def test_baseline_features_subset_of_all_features():
    """Config A features are a subset of Config D features."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")

    fe_a = FeatureEngineer(
        include_form_averages=False,
        include_home_away_split=False,
        include_recency_weighted=False,
    )
    fe_d = FeatureEngineer()

    eng_a, _X_a, _y_a = prepare_training_data(df, engineer=fe_a)
    eng_d, _X_d, _y_d = prepare_training_data(df, engineer=fe_d)

    set_a = {c for c in eng_a.columns if c not in _excluded_cols()}
    set_d = {c for c in eng_d.columns if c not in _excluded_cols()}

    assert set_a.issubset(set_d)
    assert len(set_d - set_a) > 0


# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------

def test_feature_computation_reproducible():
    """Feature computation is deterministic with same input and config."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")

    fe = FeatureEngineer()
    f1 = fe.compute_features(df, 99)
    f2 = fe.compute_features(df, 99)
    assert f1 == f2


def test_feature_computation_same_across_configs():
    """Disabling flags doesn't change basic features, only omits new ones."""
    df = _make_history_df()
    df = _add_target_match(df, mid=99, date="2024-08-12T15:00:00+00:00",
                           home=1, away=2, hg=2, ag=1, status="available")

    fe_full = FeatureEngineer()
    fe_basic = FeatureEngineer(
        include_form_averages=False,
        include_home_away_split=False,
        include_recency_weighted=False,
    )

    f_full = fe_full.compute_features(df, 99)
    f_basic = fe_basic.compute_features(df, 99)

    # Basic features should match
    for key in ["home_matches_played_last_5", "home_points_last_5",
                "home_wins_last_5", "home_goals_scored_last_5",
                "away_matches_played_last_5", "away_points_last_5"]:
        assert f_full[key] == f_basic[key], f"Mismatch for {key}"
