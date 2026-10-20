import sys
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import phase23_run as m
raw = m.load_raw_matches()
print("raw rows:", len(raw), "cols:", list(raw.columns)[:12])
eng, feats, (tr, va, te) = m.build_feature_dataset(raw, use_interactions=True)
print("eng rows:", len(eng), "feats:", len(feats))
print("split counts:", len(tr), len(va), len(te), "=", len(tr)+len(va)+len(te))
print("train range:", tr["target_match_date"].min(), "..", tr["target_match_date"].max())
print("val   range:", va["target_match_date"].min(), "..", va["target_match_date"].max())
print("test  range:", te["target_match_date"].min(), "..", te["target_match_date"].max())
print("prod hash ok:", m.prod_hash_ok())
print("interaction cols present:", [c for c in feats if "minus" in c or "advantage" in c])
print("nan in feats:", int(eng[feats].isna().sum().sum()), "of", len(feats), "cols")
