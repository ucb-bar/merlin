"""Legacy QA CLI adapter; RTL treatment callbacks live in the experiments package.

The historical CLI dispatches ordinary QA. The authoring treatment adds advisory
checks through merlin_experiments.phase1.feedback.rtlchecks, not this import path.
"""

from __future__ import annotations

import _common as C
from merlin_experiments.phase1.feedback import qa


def main(argv: list[str] | None = None) -> int:
    return qa.main(argv, context=C.CONTEXT)


if __name__ == "__main__":
    raise SystemExit(main())
