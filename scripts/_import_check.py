"""Temporary: capture the full import traceback for football_intelligence.cli.

Writes all output to import_check.log so we can read it regardless of the
PowerShell stderr-handling quirks.
"""
import io
import traceback

out = io.StringIO()
modules = [
    "football_intelligence.config",
    "football_intelligence.logging_config",
    "football_intelligence.database",
    "football_intelligence.database.connection",
    "football_intelligence.database.initialize",
    "football_intelligence.data.validation.validators",
    "football_intelligence.data.ingestion.base",
    "football_intelligence.cli",
]
for modname in modules:
    try:
        __import__(modname, fromlist=["*"])
        out.write(f"OK  {modname}\n")
    except Exception:
        out.write(f"ERR {modname}\n")
        out.write(traceback.format_exc())
        out.write("\n")

path = "import_check.log"
with open(path, "w") as fh:
    fh.write(out.getvalue())
print("wrote", path)

