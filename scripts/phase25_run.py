#!/usr/bin/env python3
"""Phase 25 standalone runner.

Re-uses the isolated Phase-23 evaluation data (disposable DB) and runs the
walk-forward backtesting pipeline.  Production DB is opened read-only and
SHA-256 gated (fail-closed).

Run:  .venv\\Scripts\\python.exe scripts/phase25_run.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from football_intelligence.prediction.walkforward import (
    run_walkforward,
)  # noqa: E402


if __name__ == "__main__":
    run_walkforward()
#!/usr/bin/env python3
"""Phase 25 standalone runner.

Re-uses the isolated Phase-23 evaluation data (disposable DB) and runs the
walk-forward backtesting pipeline.  Production DB is opened read-only and
SHA-256 gated (fail-closed).

Run:  .venv\\Scripts\\python.exe scripts/phase25_run.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from football_intelligence.prediction.walkforward import (
    run_walkforward,
)  # noqa: E402


if __name__ == "__main__":
    run_walkforward()
