"""The expert baseline a result is scored against — and what it must carry to be citable.

``attainment_vs_expert`` is the number every headline claim rests on: not "we improved on our own
starting point" but "we are this close to the expert". It was a bare float passed into the beam, so
nothing recorded what was measured, on which workload, in which dtype, or on which substrate.

That is not a hypothetical gap. Measured across 14 recorded beam runs, two int8 runs were scored
against the **fp32** expert's wall time while correctly using the int8 expert's disassembly:
``bitvla_int8`` and ``openvla_int8`` reused their fp32 sibling's number to the digit. Since int8 is
substantially faster than fp32 on the same silicon, both reported BEATING the expert (1.269x, 1.859x).
The one int8 cell carrying its own number reports 0.113 -- 8.8x slower than the expert. The wins were
an artifact of the baseline, and nothing in the record could have told them apart.

So a baseline is a measurement like any other and carries its own identity. A comparison between a
result and a baseline whose declared identity disagrees with the result's is REFUSED, because a number
attributed to the wrong comparand is worse than no number: it gets cited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: A baseline supplied as a bare number, with nothing recorded about what it measured. Kept usable so
#: existing callers keep working, but stamped so no reader can mistake it for a verified comparand.
UNRECORDED = "unrecorded"


@dataclass(frozen=True)
class ExpertBaseline:
    """A measured expert wall time, with the identity of what produced it."""
    wall_ns: float
    workload: str | None = None       # the bundle it was measured on
    dtype: str | None = None          # the numeric format it was measured in
    substrate: str | None = None      # which device/simulator produced the number
    revision: str | None = None       # the expert source revision
    note: str = ""

    @staticmethod
    def of(value: Any) -> "ExpertBaseline | None":
        """Accept an ExpertBaseline, a bare number (recorded as unrecorded), or None."""
        if value is None:
            return None
        if isinstance(value, ExpertBaseline):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return ExpertBaseline(wall_ns=float(value), note=UNRECORDED)
        return None

    @property
    def provenance_recorded(self) -> bool:
        return self.note != UNRECORDED and bool(self.workload or self.dtype or self.substrate)

    def mismatches(self, *, workload: str | None, dtype: str | None) -> tuple[str, ...]:
        """Why this baseline does not describe the run being scored. Empty when it is comparable.

        Only DECLARED fields are checked. An unrecorded baseline cannot be shown to mismatch — which is
        precisely why it cannot be cited either, and why ``provenance_recorded`` is reported alongside
        any attainment computed from it.
        """
        problems: list[str] = []
        if self.dtype and dtype and _norm_dtype(self.dtype) != _norm_dtype(dtype):
            problems.append(
                f"baseline was measured in {self.dtype!r} but this run is {dtype!r} — comparing a "
                f"result against another numeric format's expert time measures the format, not us")
        if self.workload and workload and self.workload != workload:
            problems.append(
                f"baseline was measured on {self.workload!r} but this run is {workload!r}")
        return tuple(problems)


def _norm_dtype(name: str) -> str:
    """``f32`` and ``fp32`` are the same format spelled two ways, and both appear in recorded runs.
    Normalizing here stops a spelling difference from reading as a dtype mismatch — and stops the same
    difference from silently splitting a benchmark cell in two."""
    s = str(name).strip().lower()
    return ("f" + s[2:]) if s.startswith("fp") else s


def attainment(baseline: Any, wall_ns: float | None, *, workload: str | None = None,
               dtype: str | None = None) -> tuple[float | None, tuple[str, ...], bool]:
    """``(attainment, problems, provenance_recorded)`` for one measured wall against ``baseline``.

    Returns ``None`` for the ratio when there is no baseline, no measurement, or a declared mismatch —
    fail closed, because the alternative is a plausible number scored against the wrong comparand.
    """
    b = ExpertBaseline.of(baseline)
    if b is None or not wall_ns:
        return None, (), False
    problems = b.mismatches(workload=workload, dtype=dtype)
    if problems:
        return None, problems, b.provenance_recorded
    return round(b.wall_ns / wall_ns, 3), (), b.provenance_recorded
