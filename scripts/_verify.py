"""Temporary verification runner.

Exercises the CLI (init-db, health, --version) and the full pytest suite,
capturing ALL output into verify.log via Python-internal redirection to avoid
PowerShell native-command stderr handling quirks in this environment.
"""
from __future__ import annotations

import contextlib
import io
import pathlib
import subprocess
import sys
import traceback

from football_intelligence.logging_config import reset_logging

buf = io.StringIO()


def _run_cli(argv):
    reset_logging()
    buf.write(f"\n=== football-intelligence {' '.join(argv)} ===\n")
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            rc = main(argv)
            buf.write(f"(exit code {rc})\n")
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            buf.write(f"(SystemExit {code})\n")
        except Exception:
            buf.write(traceback.format_exc())


buf.write("########## CLI SMOKE TEST ##########\n")
try:
    from football_intelligence.cli import main

    for argv in (["--version"], ["init-db"], ["health"]):
        _run_cli(list(argv))
except Exception:
    buf.write(traceback.format_exc())


buf.write("\n########## PYTEST (full suite) ##########\n")
proc = subprocess.run(
    [sys.executable, "-m", "pytest", "-v", "--tb=short", "-p", "no:cacheprovider"],
    capture_output=True,
    text=True,
)
buf.write(f"pytest exit code: {proc.returncode}\n")
buf.write(proc.stdout)
buf.write("\n--- pytest stderr ---\n")
buf.write(proc.stderr)

pathlib.Path("verify.log").write_text(buf.getvalue())
print("wrote verify.log ; pytest exit code:", proc.returncode)
