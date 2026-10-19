import re, sys
from pathlib import Path
targets = sys.argv[1:]
for tp in targets:
    parts = tp.split(":")
    path = parts[0]
    a = b = None
    if len(parts) >= 2 and "-" in parts[1]:
        a = int(parts[1].split("-")[0]); b = int(parts[1].split("-")[1])
    L = Path(path).read_text(encoding="utf-8").split("\n")
    if a and b:
        print("==== %s [%d..%d] (%d lines) ====" % (path, a, b, len(L)))
        print("\n".join("%4d| %s" % (i+1, L[i]) for i in range(a-1, b-1)))
    else:
        print("==== %s (%d lines) ====" % (path, len(L)))
        for i, l in enumerate(L):
            if re.match(r"\s*(def |class |@dataclass)", l):
                print("%4d| %s" % (i+1, l.rstrip()))
