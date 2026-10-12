"""Inspect README.md characters."""
import pathlib

content = pathlib.Path("README.md").read_text(encoding="utf-8")
for i, line in enumerate(content.split("\n")):
    if "base.py" in line and "BaseIngestor" in line:
        print(f"Line {i+1} repr: {repr(line)}")
    if "test_validation.py" in line:
        print(f"Line {i+1} test repr: {repr(line)}")
    if "Verified state" in line:
        print(f"Line {i+1} verified repr: {repr(line)}")
    if "57 passed" in line:
        print(f"Line {i+1} 57passed repr: {repr(line)}")
    if "no real database and no network" in line:
        print(f"Line {i+1} required repr: {repr(line)}")
