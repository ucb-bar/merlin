"""Whether an execution engine's accelerator counters are measurements or noise.

WHY THIS EXISTS, precisely. A functional ISS can implement an accelerator's counter-read instruction
by returning a random number, and one does: the RoCC model this tree runs increments every counter
with `rand()` on every accelerator instruction. An optimization loop read those values for hours,
reported them in a receipt beside genuine cycle counts, and had no signal at all -- the giveaway was
a byte counter reading `7` after a whole-model inference, which nothing checked. The values were not
wrong in a way that looked wrong; they looked like small measurements.

So this module refuses rather than annotates. :func:`values_or_refusal` returns counter values only
when the engine that produced them is DECLARED to wire them to hardware events, with the evidence
cited in the engine's own sources (``merlin/contract/counter_trust.yaml``). An engine that is not
declared is ``UNKNOWN`` and is refused too -- fail closed, because the whole failure mode is a
plausible number arriving from an unexamined source.

**Not a simulator ranking and not a target fact.** Nothing here names a target, an opcode or a
counter; it is keyed on the ENGINE that executed the program, and the per-engine verdict is reviewed
data, not a heuristic. `merlin.perf.hw_counters` derives what a target can count; this says whether
what came back means anything.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

__all__ = ["REAL", "FABRICATED", "UNTIMED", "UNKNOWN", "Verdict", "verdict_for",
           "is_trusted", "values_or_refusal", "require_trusted", "declared_engines"]

#: Counters are wired to hardware event signals; values are measurements.
REAL = "real"
#: The engine synthesises values; any number it reports is noise.
FABRICATED = "fabricated"
#: The engine has no cycle model, so a cycle counter is meaningless even if not fabricated.
UNTIMED = "untimed"
#: The engine is not declared. Refused, like a fabricated one -- an unexamined source is not a
#: trusted one, and this is the state that let the original failure through.
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Verdict:
    """One engine's verdict plus the citation that justifies it."""

    engine: str
    verdict: str
    evidence: str = ""
    evidence_source: str = ""
    verified: str = ""

    @property
    def trusted(self) -> bool:
        return self.verdict == REAL

    def refusal(self) -> str | None:
        """Why these values must not be reported, or ``None`` when they may be."""
        if self.trusted:
            return None
        if self.verdict == UNKNOWN:
            return (f"engine {self.engine!r} is not declared in counter_trust.yaml, so whether its "
                    f"accelerator counters carry real values is UNKNOWN; refusing to report them")
        detail = f" ({self.evidence_source})" if self.evidence_source else ""
        return (f"engine {self.engine!r} accelerator counters are {self.verdict}{detail}: "
                f"{self.evidence.strip() or 'no evidence recorded'}")

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "verdict": self.verdict,
                "evidence": self.evidence, "evidence_source": self.evidence_source,
                "verified": self.verified}


@lru_cache(maxsize=1)
def _declared() -> dict[str, Verdict]:
    """The reviewed table. Absent or malformed means every engine is UNKNOWN, never trusted."""
    import yaml
    from merlin.common.paths import merlin_dir

    path = merlin_dir() / "contract" / "counter_trust.yaml"
    try:
        body = yaml.safe_load(path.read_text())
    except (OSError, ValueError):
        return {}
    engines = (body or {}).get("engines")
    if not isinstance(engines, Mapping):
        return {}
    out: dict[str, Verdict] = {}
    for name, spec in engines.items():
        if not isinstance(spec, Mapping):
            continue
        v = str(spec.get("verdict") or "")
        if v not in (REAL, FABRICATED, UNTIMED):
            continue                      # an unrecognised verdict is not a licence to trust
        out[str(name)] = Verdict(str(name), v, str(spec.get("evidence") or ""),
                                 str(spec.get("evidence_source") or ""),
                                 str(spec.get("verified") or ""))
    return out


def declared_engines() -> tuple[str, ...]:
    return tuple(sorted(_declared()))


def verdict_for(engine: Any) -> Verdict:
    """``engine``'s verdict, or an UNKNOWN verdict when it is not declared."""
    name = str(engine or "").strip().lower()
    hit = _declared().get(name)
    return hit if hit is not None else Verdict(name or "<unnamed>", UNKNOWN)


def is_trusted(engine: Any) -> bool:
    return verdict_for(engine).trusted


def values_or_refusal(engine: Any, values: Mapping[str, Any]
                      ) -> tuple[dict[str, Any] | None, str | None]:
    """``(values, None)`` when ``engine``'s counters are real; ``(None, why)`` otherwise.

    Callers should record the refusal STRING in the receipt where the numbers would have gone, so a
    reader sees why the field is absent instead of finding it silently missing -- an absent field
    reads as "not collected", which is the same shape as the original defect.
    """
    v = verdict_for(engine)
    why = v.refusal()
    return (dict(values), None) if why is None else (None, why)


def require_trusted(engine: Any) -> Verdict:
    """The verdict, raising ``ValueError`` unless ``engine``'s counters are real."""
    v = verdict_for(engine)
    why = v.refusal()
    if why is not None:
        raise ValueError(why)
    return v
