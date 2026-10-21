#!/usr/bin/env python3
"""Phase 26.8 runner: extended chronological evaluation with intra-season folds.

Usage:
    .venv\\Scripts\\python.exe scripts/phase26_8_extended_eval.py [--fast] [--n-boot N]
"""
from __future__ import annotations

import argparse
import sys

from football_intelligence.prediction.walkforward_extended import (
    run_walkforward_extended,
    write_extended_report_txt,
)
from football_intelligence.prediction.walkforward import REPORT_DIR


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 26.8: Extended walkforward with intra-season folds")
    ap.add_argument("--fast", action="store_true", default=True,
                    help="Use fast models only (default: True)")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="Evaluate ALL models (slower, includes XGBoost/DC/Elo)")
    ap.add_argument("--n-boot", type=int, default=200,
                    help="Bootstrap iterations for paired significance (default: 200)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    args = ap.parse_args()

    print(f"[phase26_8] Running extended walkforward (fast={args.fast}, "
          f"n_boot={args.n_boot}, seed={args.seed})")

    report = run_walkforward_extended(
        seed=args.seed, n_boot=args.n_boot, fast=args.fast,
    )

    # Write text report
    txt_path = REPORT_DIR.parent / "phase26_8" / f"phase26_8_report_{report['produced_at']}.txt"
    write_extended_report_txt(report, txt_path)
    print(f"[phase26_8] Text report: {txt_path}")

    # Summary to stdout
    print("\n[phase26_8] === Pooled Results (all folds) ===")
    for nm, m in sorted(report["pooled"].items()):
        print(f"  {nm}: acc={m['accuracy']:.4f} brier={m['brier_score']:.4f} "
              f"logloss={m['log_loss']:.4f} n={m['n']}")

    print("\n[phase26_8] === Pooled — Seasonal folds (fold1+fold2) ===")
    for nm, m in sorted(report["pooled_seasonal"].items()):
        print(f"  {nm}: acc={m['accuracy']:.4f} brier={m['brier_score']:.4f} "
              f"logloss={m['log_loss']:.4f} n={m['n']}")

    print("\n[phase26_8] === Pooled — Intra-season folds (fold1+fold3+fold4) ===")
    for nm, m in sorted(report["pooled_intra"].items()):
        print(f"  {nm}: acc={m['accuracy']:.4f} brier={m['brier_score']:.4f} "
              f"logloss={m['log_loss']:.4f} n={m['n']}")

    print("\n[phase26_8] === Brier Decomposition (Poisson) ===")
    for pool_name, dec in report["brier_decomposition"].items():
        d = dec["poisson"]
        print(f"  {pool_name}: brier={d['brier']:.4f} "
              f"reliab={d['reliability']:.4f} resol={d['resolution']:.4f} "
              f"unc={d['uncertainty']:.4f}")

    print(f"\n[phase26_8] Production DB unchanged: "
          f"{report['dataset']['production_db_unchanged']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
