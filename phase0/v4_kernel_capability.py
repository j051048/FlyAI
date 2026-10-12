"""Opt-in native kernel compatibility smoke; no model checkpoint required."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "engines" / "deepseek_v4"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from engines.deepseek_v4.v4_native_capability import main

if __name__ == "__main__":
    raise SystemExit(main())
