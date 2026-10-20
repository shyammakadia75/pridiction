import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import phase23_run as m
from football_intelligence.prediction.elo import EloModel
raw = m.load_raw_matches()
em = _elo_history_df = m._elo_history_df
eh = em(EloModel(K=20.0, home_advantage=50.0).fit(raw))
print("elo_history_df columns:", list(eh.columns), "len:", len(eh))
# build a small test_df
eng, feats, (tr, va, te) = m.build_feature_dataset(raw, use_interactions=True)
te = te.sort_values(["target_match_date", "home_team_id", "away_team_id"]).reset_index(drop=True)
mer = m._elo_aligned(eh, te)
print("merged columns:", list(mer.columns))
print("has outcome_label:", "outcome_label" in mer.columns)
print(mer[["target_match_date","expected_home","outcome_label"]].head(3).to_string())
