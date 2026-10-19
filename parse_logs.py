import re
total = 0
for f in ["fx", "fy", "fz"]:
    raw = open(f + ".txt", encoding="utf-16-le", errors="replace").read()
    m = re.search(r"(\d+) passed(?:.*?in ([\d.]+)s)?", raw)
    n = int(m.group(1)) if m else 0
    tm = m.group(2) if m and m.lastindex >= 2 else "?"
    bad = "FAIL" if ("failed" in raw or "error" in raw.lower()) else "ok"
    total += n
    print(f"{f}: {n} passed in {tm}s [{bad}]")
print("TOTAL distinct (3 disjoint batches):", total)
