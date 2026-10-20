#!/usr/bin/env python3
"""Phase 26 tests for the match prediction service.

All tests use verified project data (production DB or disposable eval DB)
or synthetic data.  No test writes to the production database.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from football_intelligence.prediction.service import (
    DEFAULT_MODEL,
    KNOWN_PROD_HASH,
    PROD_DB_PATH,
    SUPPORTED_MODELS,
    MatchPrediction,
    PredictionRequest,
    PredictionExplanation,
    build_explanation,
    build_pre_match_history,
    double_chance_probabilities,
    over_under_probabilities,
    predict_match,
    resolve_competition,
    resolve_fixture_date,
    resolve_team,
    team_lookup,
    validate_probability_vector,
)
from football_intelligence.prediction.baseline import OUTCOME_LABELS, OUTCOME_INDEX
from football_intelligence.database import resolve_db_path

PROD_DB = REPO / "data" / "football_intelligence.db"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _prod_ok() -> bool:
    return PROD_DB.exists() and _sha256(PROD_DB) == KNOWN_PROD_HASH


prod_available = pytest.mark.skipif(
    not _prod_ok(),
    reason="Production database not available or hash mismatch",
)


# ---------------------------------------------------------------------------
# 1. Probability vector normalization
# ---------------------------------------------------------------------------


class TestProbabilityValidation:
    """Validate that probability vectors are finite, non-negative, normalized."""

    def test_valid_vector_passes(self):
        validate_probability_vector([0.3, 0.4, 0.3])

    def test_valid_vector_with_tolerance(self):
        validate_probability_vector([0.33, 0.34, 0.33], tol=1e-2)

    def test_negative_values_raise(self):
        with pytest.raises(ValueError, match="Negative"):
            validate_probability_vector([0.3, -0.1, 0.8])

    def test_non_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to"):
            validate_probability_vector([0.3, 0.3, 0.3])

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="Non-finite"):
            validate_probability_vector([0.3, float("nan"), 0.7])

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="Non-finite"):
            validate_probability_vector([0.3, float("inf"), 0.7])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_probability_vector([])

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            validate_probability_vector(np.array([[0.5, 0.5]]))


# ---------------------------------------------------------------------------
# 2. Team resolution and lookups
# ---------------------------------------------------------------------------


@prod_available
class TestTeamResolution:
    """Tests for team resolution and name lookup."""

    def test_resolve_team_known_team(self):
        team = resolve_team(PROD_DB, "Arsenal", provider="statsbomb")
        assert team is not None
        assert team["team_name"] == "Arsenal"

    def test_resolve_team_case_insensitive(self):
        team = resolve_team(PROD_DB, "arsenal", provider="statsbomb")
        assert team is not None
        assert team["team_name"] == "Arsenal"

    def test_resolve_team_unknown_raises(self):
        with pytest.raises(ValueError, match="No team found"):
            resolve_team(PROD_DB, "FakeTeam", provider="statsbomb")

    def test_resolve_team_normalizes_spaces(self):
        """Team names with surrounding whitespace should still resolve."""
        team = resolve_team(PROD_DB, " Manchester United ",
                            provider="statsbomb")
        assert team is not None
        assert "United" in team["team_name"]

    def test_resolve_team_returns_provider_id(self):
        team = resolve_team(PROD_DB, "Liverpool", provider="statsbomb")
        assert team["provider_team_id"] is not None

    def test_resolve_team_wrong_provider_raises(self):
        with pytest.raises(ValueError, match="No team found"):
            resolve_team(PROD_DB, "Arsenal", provider="nonexistent")


# ---------------------------------------------------------------------------
# 3. Database path resolution and integrity
# ---------------------------------------------------------------------------


class TestDbResolution:
    """Tests for database path resolution and SHA-256 integrity."""

    def test_resolve_default_db_path(self):
        path = resolve_db_path(None)
        assert Path(path).exists()

    def test_resolve_default_matches_production_hash(self):
        """Default DB must be the known production DB with verified hash."""
        path = Path(resolve_db_path(None))
        actual = _sha256(path)
        assert actual == KNOWN_PROD_HASH

    def test_resolve_explicit_path(self):
        explicit = str(PROD_DB)
        path = resolve_db_path(explicit)
        assert Path(path).resolve() == PROD_DB.resolve()

    def test_resolve_nonexistent_path(self):
        """Non-existent path should resolve to a Path that does not exist."""
        path = resolve_db_path("/nonexistent/db.db")
        assert not Path(path).exists()

    def test_resolve_db_path_returns_path(self):
        path = resolve_db_path(None)
        assert isinstance(path, Path)


# ---------------------------------------------------------------------------
# 4. Competition resolution
# ---------------------------------------------------------------------------


@prod_available
class TestCompetitionResolution:
    """Tests for competition resolution from request."""

    def test_resolve_competition_valid(self):
        comp = resolve_competition(PROD_DB, "2", provider="statsbomb")
        assert comp is not None
        assert comp["competition_name"] == "Premier League"

    def test_resolve_competition_unknown_raises(self):
        with pytest.raises(ValueError, match="No competition"):
            resolve_competition(PROD_DB, "999", provider="statsbomb")

    def test_resolve_competition_returns_dict(self):
        comp = resolve_competition(PROD_DB, "2", provider="statsbomb")
        assert "id" in comp
        assert "provider" in comp
        assert comp["provider_competition_id"] == "2"


# ---------------------------------------------------------------------------
# 5. Fixture date resolution (no future leakage)
# ---------------------------------------------------------------------------


@prod_available
class TestFixtureDateResolution:
    """Tests for fixture date resolution ensuring no future leakage."""

    def test_resolve_date_explicit(self):
        date = resolve_fixture_date(
            PROD_DB, "2",
            home_team_id=4, away_team_id=2,
            provider="statsbomb",
            season_name="2015/2016",
            target_date="2015-08-24",
        )
        assert date == "2015-08-24"

    def test_resolve_date_from_fixture(self):
        """When target_date is None, the actual fixture date is used."""
        date = resolve_fixture_date(
            PROD_DB, "2",
            home_team_id=4, away_team_id=2,
            provider="statsbomb",
            season_name="2015/2016",
            target_date=None,
        )
        assert date is not None
        assert date.startswith("2015")

    def test_resolve_date_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid date"):
            predict_match(
                provider="statsbomb",
                provider_competition_id="2",
                home_team="Arsenal",
                away_team="Liverpool",
                target_date="not-a-date",
                db_path=PROD_DB,
            )

    def test_resolve_date_future_fixture_not_in_db_raises(self):
        """A future fixture not in the DB requires target_date."""
        with pytest.raises(ValueError, match="Fixture not found"):
            resolve_fixture_date(
                PROD_DB, "2",
                home_team_id=4, away_team_id=2,
                provider="statsbomb",
                season_name="2099/2100",
                target_date=None,
            )


# ---------------------------------------------------------------------------
# 6. Pre-match history (no same-day / future leakage)
# ---------------------------------------------------------------------------


class TestPreMatchHistory:
    """Tests that build_pre_match_history excludes same-day and future matches."""

    @pytest.fixture
    def synthetic_matches(self):
        """Create synthetic match data with dates straddling the fixture date."""
        return pd.DataFrame({
            "match_date": [
                "2015-08-10", "2015-08-20", "2015-08-23", "2015-08-24",
                "2015-08-25", "2015-08-30",
            ],
            "home_team_id": [4, 4, 4, 4, 4, 4],
            "away_team_id": [2, 2, 2, 2, 2, 2],
            "home_team_name": ["Arsenal"] * 6,
            "away_team_name": ["Liverpool"] * 6,
            "home_goals": [1, 2, 0, 3, 1, 2],
            "away_goals": [0, 1, 1, 2, 3, 0],
        })

    def test_history_excludes_target_fixture(self, synthetic_matches):
        history = build_pre_match_history(
            synthetic_matches, "2015-08-24", home_team_id=4, away_team_id=2
        )
        if len(history) > 0:
            fixture_mask = history["match_date"] == "2015-08-24"
            assert not fixture_mask.any()

    def test_history_excludes_same_day_matches(self, synthetic_matches):
        history = build_pre_match_history(
            synthetic_matches, "2015-08-24", home_team_id=4, away_team_id=2
        )
        if len(history) > 0:
            same_day = history["match_date"] == "2015-08-24"
            assert not same_day.any()

    def test_history_excludes_future_matches(self, synthetic_matches):
        history = build_pre_match_history(
            synthetic_matches, "2015-08-24", home_team_id=4, away_team_id=2
        )
        future = pd.to_datetime(history["match_date"]) > pd.to_datetime("2015-08-24")
        assert not future.any()

    def test_history_only_contains_past_matches(self, synthetic_matches):
        history = build_pre_match_history(
            synthetic_matches, "2015-08-24", home_team_id=4, away_team_id=2
        )
        dates = pd.to_datetime(history["match_date"])
        cutoff = pd.to_datetime("2015-08-24")
        assert (dates < cutoff).all()


    def test_history_columns_present(self, synthetic_matches):
        history = build_pre_match_history(
            synthetic_matches, "2015-08-24", home_team_id=4, away_team_id=2
        )
        required_cols = {"match_date", "home_team_name", "away_team_name",
                         "home_goals", "away_goals"}
        assert required_cols.issubset(set(history.columns))


# ---------------------------------------------------------------------------
# 7. Poisson computation (score, over/under, BTTS, DC)
# ---------------------------------------------------------------------------


class TestPoissonComputation:
    """Tests for raw Poisson probability computations."""

    def test_over_under_complementary(self):
        home_rate, away_rate = 1.5, 1.2
        result = over_under_probabilities(
            home_rate, away_rate, lines=[0.5, 1.5, 2.5, 3.5], max_goals=15
        )
        for probs in result.values():
            assert abs(probs["over"] + probs["under"] - 1.0) < 1e-6
            assert 0.0 <= probs["over"] <= 1.0

    def test_over_under_higher_line_lower_over_prob(self):
        home_rate, away_rate = 2.0, 1.0
        result = over_under_probabilities(
            home_rate, away_rate, lines=[2.5, 3.5], max_goals=15
        )
        assert result["3.5"]["over"] < result["2.5"]["over"]

    def test_double_chance_complementary(self):
        home_p, draw_p, away_p = 0.4, 0.3, 0.3
        dc = double_chance_probabilities(home_p, draw_p, away_p)
        assert abs(dc["1X"] - (home_p + draw_p)) < 1e-6
        assert abs(dc["X2"] - (draw_p + away_p)) < 1e-6
        assert abs(dc["12"] - (home_p + away_p)) < 1e-6

    def test_double_chance_valid_range(self):
        dc = double_chance_probabilities(0.5, 0.2, 0.3)
        for v in dc.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# 8. End-to-end predict_match on verified production data
# ---------------------------------------------------------------------------


@prod_available
class TestPredictMatchProduction:
    """End-to-end tests using the verified production database."""

    def test_predict_valid_fixture(self):
        result = predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )
        assert result.home_team_name == "Arsenal"
        assert result.away_team_name == "Liverpool"
        assert result.model == DEFAULT_MODEL

    def test_predict_outcome_probs_sum_to_one(self):
        result = predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )
        total = result.home_win_prob + result.draw_prob + result.away_win_prob
        assert abs(total - 1.0) < 1e-4

    def test_predict_all_probs_in_range(self):
        result = predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )
        assert 0 <= result.home_win_prob <= 1
        assert 0 <= result.draw_prob <= 1
        assert 0 <= result.away_win_prob <= 1
        assert 0 <= result.btts["yes"] <= 1
        assert 0 <= result.btts["no"] <= 1
        for probs in result.over_under.values():
            assert 0 <= probs["over"] <= 1
            assert 0 <= probs["under"] <= 1

    def test_predict_over_under_complementary(self):
        result = predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )
        for probs in result.over_under.values():
            assert abs(probs["over"] + probs["under"] - 1.0) < 1e-6

    def test_predict_double_chance_complementary(self):
        result = predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )
        total_dc = sum(result.double_chance.values())
        assert abs(total_dc - 2.0) < 1e-4

    def test_predict_top_scorelines_normalized(self):
        result = predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )
        d = result.to_dict()
        total = sum(s["probability"] for s in d["top_scorelines"])
        assert total < 1.0  # top-N, not full grid
        # Full grid should sum close to 1.0
        grid_total = sum(d["total_goals_distribution"].values())
        assert abs(grid_total - 1.0) < 0.05
    def test_predict_expected_goals_positive(self):
        result = predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )
        assert result.expected_home_goals >= 0
        assert result.expected_away_goals >= 0
        assert (result.expected_home_goals + result.expected_away_goals) > 0

    def test_predict_model_version_set(self):
        result = predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )
        assert result.model_version is not None
        assert isinstance(result.model_version, str)


# ---------------------------------------------------------------------------
# 9. Error handling / input validation
# ---------------------------------------------------------------------------


@prod_available
class TestPredictMatchErrors:
    """predict_match should raise ValueError for invalid inputs."""

    def test_unknown_team_raises(self):
        with pytest.raises(ValueError, match="No team found"):
            predict_match(
                provider="statsbomb",
                provider_competition_id="2",
                home_team="FakeTeam",
                away_team="Liverpool",
                season_name="2015/2016",
                db_path=PROD_DB,
            )

    def test_same_team_for_both_sides_raises(self):
        with pytest.raises(ValueError, match="different teams"):
            predict_match(
                provider="statsbomb",
                provider_competition_id="2",
                home_team="Arsenal",
                away_team="Arsenal",
                season_name="2015/2016",
                db_path=PROD_DB,
            )

    def test_unknown_competition_raises(self):
        with pytest.raises(ValueError):
            predict_match(
                provider="statsbomb",
                provider_competition_id="999",
                home_team="Arsenal",
                away_team="Liverpool",
                season_name="2015/2016",
                db_path=PROD_DB,
            )

    def test_nonexistent_db_raises(self):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            predict_match(
                provider="statsbomb",
                provider_competition_id="2",
                home_team="Arsenal",
                away_team="Liverpool",
                season_name="2015/2016",
                db_path=Path("/nonexistent/db.db"),
            )


# ---------------------------------------------------------------------------
# 10. Model configuration / supported models
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for model enumeration and defaults."""

    def test_default_model_is_poisson(self):
        assert DEFAULT_MODEL == "poisson"

    def test_supported_models_includes_poisson(self):
        assert "poisson" in SUPPORTED_MODELS

    def test_supported_models_is_not_empty(self):
        assert len(SUPPORTED_MODELS) >= 1

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            predict_match(
                provider="statsbomb",
                provider_competition_id="2",
                home_team="Arsenal",
                away_team="Liverpool",
                season_name="2015/2016",
                model="nonexistent_model",
                db_path=PROD_DB,
            )


# ---------------------------------------------------------------------------
# 11. MatchPrediction dataclass serialization
# ---------------------------------------------------------------------------


class TestMatchPredictionSerialization:
    """Tests for MatchPrediction.to_dict() and JSON serialization."""

    @pytest.fixture
    def prediction(self):
        if not _prod_ok():
            pytest.skip("Production DB not available")
        return predict_match(
            provider="statsbomb",
            provider_competition_id="2",
            home_team="Arsenal",
            away_team="Liverpool",
            season_name="2015/2016",
            db_path=PROD_DB,
        )

    def test_to_dict_keys(self, prediction):
        d = prediction.to_dict()
        required = {
            "model", "model_version", "competition",
            "fixture_date", "outcome_probabilities",
            "expected_goals", "over_under", "both_teams_to_score",
            "double_chance", "most_probable_scoreline",
            "top_scorelines", "notes",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_serializable(self, prediction):
        d = prediction.to_dict()
        json.dumps(d)

    def test_outcome_probabilities_class_order(self, prediction):
        d = prediction.to_dict()
        classes = d["outcome_probabilities"]["class_order"]
        assert classes == ["home_win", "draw", "away_win"]

    def test_home_away_teams_in_dict(self, prediction):
        d = prediction.to_dict()
        assert d["home_team"]["name"] == "Arsenal"
        assert d["away_team"]["name"] == "Liverpool"


# ---------------------------------------------------------------------------
# 12. Explanation / transparency
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_prediction():
    if not _prod_ok():
        pytest.skip("Production DB not available")
    return predict_match(
        provider="statsbomb",
        provider_competition_id="2",
        home_team="Arsenal",
        away_team="Liverpool",
        season_name="2015/2016",
        db_path=PROD_DB,
    )


class TestPredictionExplanation:
    """Tests for the explanation / transparency layer."""

    def test_build_explanation_returns_object(self, sample_prediction):
        expl = build_explanation(
            None, sample_prediction,
            training_cutoff=sample_prediction.fixture_date,
        )
        assert isinstance(expl, PredictionExplanation)

    def test_explanation_has_model_name(self, sample_prediction):
        expl = build_explanation(
            None, sample_prediction,
            training_cutoff=sample_prediction.fixture_date,
        )
        assert expl.model == "poisson"

    def test_explanation_has_limitations(self, sample_prediction):
        expl = build_explanation(
            None, sample_prediction,
            training_cutoff=sample_prediction.fixture_date,
        )
        assert len(expl.limitations) > 0

    def test_explanation_has_features(self, sample_prediction):
        expl = build_explanation(
            None, sample_prediction,
            training_cutoff=sample_prediction.fixture_date,
        )
        assert len(expl.features_available) > 0
