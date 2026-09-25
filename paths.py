"""Put the sibling parts on sys.path (repo folders side by side)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARTS = {n: ROOT / n for n in ("part1-simulation", "part2-simulation", "part3-simulation",
                               "part4-simulation")}


def add_parts() -> None:
    for p in PARTS.values():
        if p.is_dir() and str(p) not in sys.path:
            sys.path.append(str(p))
    sb = PARTS["part1-simulation"] / "sandbox"
    if sb.is_dir() and str(sb) not in sys.path:
        sys.path.append(str(sb))
    os.environ.setdefault("PART2_PATH", str(PARTS["part2-simulation"]))


add_parts()
