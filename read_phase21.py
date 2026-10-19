import json
d = json.load(open("reports/phase21/phase21_comparison_2_all.json"))
print("dataset:", json.dumps(d["dataset"]))
print("split:", json.dumps(d["split"]))
print("bootstrap:", d["config"]["bootstrap"])
print("production_db_unchanged:", d["production_db_unchanged"])
print("production_db_sha256:", d["production_db_sha256"])
print("notes:", len(d["notes"]), "| first:", d["notes"][0][:80])
for n in ("logistic_regression", "poisson", "elo"):
    m = d["models"][n]
    print("  %s: acc=%s macro_f1=%s brier=%s logloss=%s brier_exp=%s"
          % (n, m.get("accuracy"), m.get("macro_f1"), m.get("brier_score"),
             m.get("log_loss"), m.get("brier_expected_score")))
    print("       cm=", m.get("confusion_matrix"))
print("paired:")
for p in d["paired"]:
    mc = p["mcnemar_accuracy"]
    ab = p["bootstrap_accuracy"]
    bb = p.get("bootstrap_brier")
    print("  %s vs %s | mcnemar p=%.4f (n01=%d n10=%d) | "
          "boot dAcc mean=%.4f CI=[%.4f,%.4f] p=%.4f | brier %s"
          % (p["pair"][0], p["pair"][1], mc["p_value"], mc["n01"], mc["n10"],
             ab["mean_diff"], ab["ci_low_95"], ab["ci_high_95"], ab["p_value"],
             "n/a" if bb is None else
             "mean=%.4f CI=[%.4f,%.4f] p=%.4f" % (bb["mean_diff"], bb["ci_low_95"], bb["ci_high_95"], bb["p_value"])))
print("n_predictions:", len(d["predictions"]))
print("pred keys:", sorted(d["predictions"][0].keys()))
