"""A program is as long as its commands, not as long as its data.

An emitted artifact that grows with the payload is a lowering defect the harness used to find the
hard way. Measured on the first campaign graded against a model's own layers: a candidate lowered a
convolution by unrolling its host gather at compile time, one ``getelementptr``/``store`` pair per
element, and emitted 71 MB to 768 MB of LLVM-dialect text per layer, against a median of 109 KB
and a maximum of 12.5 MB over the 97 graded capsules. The grader then parsed that text for its
trace decode and audits, inside the agent's single self-check slot, for 76 minutes; every other
self-check the agent asked for in that time was starved, and nothing told it why.

So the size is checked where the artifact arrives, before anything reads it twice, and the refusal
says what the usual cause is. The limit is a statement about what this harness will analyse, not
about any target: it sits five times above the largest artifact a passing program has needed, and
``MERLIN_MAX_ARTIFACT_BYTES`` moves it for a corpus whose honest programs are larger.
"""

from __future__ import annotations

import os

LIMIT_ENV = "MERLIN_MAX_ARTIFACT_BYTES"
#: Five times the largest artifact any capsule of the graded corpus has needed (12.5 MB).
DEFAULT_LIMIT_BYTES = 64 * 1024 * 1024


def limit_bytes() -> int:
    raw = os.environ.get(LIMIT_ENV, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_LIMIT_BYTES


def refusal(artifact_text: str, *, elements: int | None = None) -> str | None:
    """Why this emitted artifact is not analysed, or ``None`` when it is an ordinary size."""
    size, limit = len(artifact_text), limit_bytes()
    if size <= limit:
        return None
    per_element = f", about {size // elements} bytes for each of its {elements} payload elements" if elements else ""
    return (
        f"the emitted artifact is {size / 2**20:.0f} MiB{per_element}; this harness analyses at most "
        f"{limit / 2**20:.0f} MiB. A program this long has almost always been unrolled over its DATA "
        f"(one load or store per element, a gather written out at compile time). Emit a loop, or the "
        f"unit's own command for the movement, so the program grows with its commands and not with "
        f"its payload."
    )
