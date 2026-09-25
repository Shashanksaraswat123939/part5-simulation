"""Run every tests/test_*.py; exit non-zero if any fails."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
failed = []
for t in sorted((here / "tests").glob("test_*.py")):
    print(f"\n=== {t.name}", flush=True)
    r = subprocess.run([sys.executable, str(t)], cwd=str(here))
    if r.returncode:
        failed.append(t.name)
print(f"\nTOTAL: {len(failed)} failed file(s)" + (f": {failed}" if failed else ""))
sys.exit(1 if failed else 0)
