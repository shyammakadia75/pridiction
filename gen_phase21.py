import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
import hashlib

from football_intelligence.config import load_settings
from football_intelligence.features.engineering import load_matches_for_features
from football_intelligence.prediction.comparison import (
    run_model_comparison,
    save_comparison_report,
)


def sha(path):
    return hashlib.sha256(Path(path).resolve().read_bytes()).hexdigest()


db = Path(load_settings().database_path)
sha_before = sha(db)

raw = load_matches_for_features(db, provider="statsbomb", provider_competition_id="2")
counts = raw.groupby("match_date").size()
multi = counts[counts > 1]

rep = run_model_comparison(
    db_path=str(db),
    provider="statsbomb",
    provider_competition_id="2",
        n_boot=500,
    boot_seed=42,
)
arts = save_comparison_report(rep, "reports/phase21", name="phase21_comparison_2_all")
sha_after = sha(db)

print("sha_before:", sha_before)
print("sha_after :", sha_after)
print("production_db_unchanged:", sha_before == sha_after)
print("artifacts:", arts["json"], "|", arts["txt"])
print(f"n_matches={rep.dataset['n_matches']} n_train={rep.dataset['n_train']} "
      f"n_val={rep.dataset['n_val']} n_test={rep.dataset['n_test']}")
print(f"test window {rep.dataset['test_start_date']} -> {rep.dataset['test_end_date']}")
print(f"prod PL: distinct_dates={int(counts.shape[0])} "
          f"multi_fixture_dates={int(len(multi))} max_per_date={int(counts.max())}")
