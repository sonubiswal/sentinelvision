"""Thin wrapper: annotate a webcam / video file and print a running tally.

    python scripts/detect_webcam.py --source 0 --show
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelvision.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["--source", "0", "--show", *sys.argv[1:]]))
