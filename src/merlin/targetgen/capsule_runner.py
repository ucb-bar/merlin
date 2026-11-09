"""Compatibility identity for the independent capsule evaluator.

New code imports ``merlin.evaluation.capsule.runner`` or its algorithm owner.
The exact old module identity remains for external simulators and old pickles.
"""

from __future__ import annotations

import sys

from merlin.evaluation.capsule import runner as _runner

if __name__ == "__main__":
    raise SystemExit(_runner.main())

sys.modules[__name__] = _runner
