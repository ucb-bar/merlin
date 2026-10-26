#!/usr/bin/env python3
"""Pre-launch acceptance of the author workflows on a frozen framework export.

    python build_tools/scripts/preflight_author_workflows.py --framework EXPORT \
        --inputs EXPORT/out/artifacts/fresh-inputs [--keep] [--fast] [--sandbox-args FILE] \
        [--compiler-execution MODE] [--compiler-library-root DIR --compiler-library-contract FILE \
         --compiler-library-contract-sha256 SHA]

Stages an author workspace with the export's own staging owners and runs each workflow the author
depends on with only the author-visible tree importable. ``tooling-readiness`` replays the launch
preflight's tooling assessment on the sealed bundle (run it under the launch environment and pass the
launch's compiler execution/library selection). Prints a PASS/FAIL table, writes a JSON report
under ``--work-root`` (default ``$TMPDIR``) and exits non-zero on any failure. Implementation:
``merlin_experiments.phase1.author_acceptance``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO / "packages" / "merlin-experiments" / "src", _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from merlin_experiments.phase1.author_acceptance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
