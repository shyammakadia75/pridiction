"""One-off fixer: restore correct indentation on the duplicated validator line."""
import pathlib

p = pathlib.Path("src/football_intelligence/data/validation/validators.py")
lines = p.read_text().splitlines(keepends=True)
target = '            result.add_error(col, f"{null_count} missing value(s) in required column.")\n'
for i, line in enumerate(lines):
    if "missing value(s) in required column." in line:
        lines[i] = target
p.write_text("".join(lines))
print("fixed line", i + 1, "->", repr(lines[i]))
