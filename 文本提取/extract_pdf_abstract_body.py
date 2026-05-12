"""Command-line entrypoint for batch PDF abstract/body extraction."""

from __future__ import annotations

from pathlib import Path
import sys

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from pdf_abstract_body.runner import main


if __name__ == "__main__":
    raise SystemExit(main())
