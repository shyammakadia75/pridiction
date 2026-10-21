"""Run slow Phase 25 integration tests in a subprocess with output to file."""
import subprocess
import sys
from pathlib import Path

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/test_phase25_walkforward.py::test_walkforward_produces_valid_report",
     "tests/test_phase25_walkforward.py::test_prod_db_unchanged_after_walkforward",
     "tests/test_phase25_walkforward.py::test_deterministic_results_under_fixed_seed",
     "-v", "--tb=short"],
    capture_output=True, text=True, timeout=180
)

output = result.stdout + result.stderr
Path("phase25_slow_tests_output.txt").write_text(output)

print(f"Exit code: {result.returncode}")
print("Output written to phase25_slow_tests_output.txt")
print("--- Last 20 lines ---")
for line in output.split("\n")[-20:]:
    print(line)
