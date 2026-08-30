"""Measure a systolic array's fill/drain depth from its circuit, instead of assuming a law.

A fill/drain depth is an intercept, not a rate: it is paid once per weight reload and it dominates
small tiles. It is also the term most easily assumed. The obvious closed form for a square
weight-stationary array is ``2*DIM`` -- rows plus columns -- and a slightly better one is ``2*DIM-2``.
Both are guesses about a pipeline whose actual length is a property of the emitted circuit, and the
gap between them is two cycles on one array and much larger on another microarchitecture.

So this reads the depth out of the circuit: the length of the output-valid delay-line register chain,
counted directly in the IR. Nothing is fitted and no law is applied.

WHY A LAW IS STILL WORTH KEEPING, and why this module reports BOTH. A measured depth is the truth for
the design that was elaborated; a law is what lets the model answer for a design that has not been
elaborated yet -- a different mesh dimension in a design-space sweep, where there is no circuit to
read. Keeping the two apart, and reporting when they disagree, is what tells you whether the law may
be extrapolated at all. A law that matches the circuit on the one point it was checked against is
evidence, not proof; a law that does not match is refuted for this family and must not be swept with.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class HandshakeUnavailable(RuntimeError):
    """The circuit could not be read, so the fill/drain depth is UNKNOWN -- never a default."""


@dataclass(frozen=True)
class FillDepth:
    """A fill/drain depth, its source, and how it compares with the law it would otherwise be given."""

    dim: int
    measured_cycles: int
    #: What the named law predicts for this dimension, or None when no law was offered.
    law_cycles: int | None
    law: str | None
    #: Double-buffering the circuit carries. A second slot lets the next reload overlap the current
    #: compute, which is why it belongs beside the depth rather than in a separate report.
    weight_buffer_slots: int
    accumulator_banks: int
    source: str

    @property
    def law_agrees(self) -> bool | None:
        """True/False when a law was offered, None when none was."""
        return None if self.law_cycles is None else self.law_cycles == self.measured_cycles

    def claim(self) -> str:
        if self.law_cycles is None:
            return f"fill/drain {self.measured_cycles} cycles, measured from {self.source}"
        if self.law_agrees:
            return (f"fill/drain {self.measured_cycles} cycles, measured; the law {self.law!r} agrees "
                    f"at DIM={self.dim} -- evidence it may extrapolate, not proof")
        return (f"fill/drain {self.measured_cycles} cycles, measured; the law {self.law!r} predicts "
                f"{self.law_cycles} and is REFUTED for this design -- do not sweep with it")


def measure_fill_depth(target: str, *, law: str | None = "systolic_2d",
                       hw_mlir: Any = None) -> FillDepth:
    """Read the fill/drain depth from the target's elaborated circuit, and check any offered law.

    ``law`` names a fill law from :mod:`merlin.perf.record` to cross-check against; pass None to skip
    the comparison. Raises rather than returning a default when the circuit is unreachable: an
    unmeasurable intercept is UNKNOWN, and a pipeline silently given a depth of zero reads as an array
    that fills instantly."""
    from merlin.targetgen.rtl import mlc_bridge
    path = hw_mlir if hw_mlir is not None else mlc_bridge.core_hw_mlir(target)
    if path is None:
        raise HandshakeUnavailable(
            f"no elaborated circuit is resolvable for {target!r}; the fill/drain depth is UNKNOWN")
    try:
        with mlc_bridge._mlc_cwd():
            from mlc.passes.infer_handshake_depth import infer_handshake_depth
            facts = infer_handshake_depth(str(path))
    except Exception as exc:  # noqa: BLE001 - a circuit we cannot read is UNKNOWN, not a default
        raise HandshakeUnavailable(
            f"the circuit for {target!r} could not be read for a fill/drain depth: "
            f"{type(exc).__name__}: {exc}") from exc

    law_cycles = None
    if law is not None:
        from merlin.perf.record import fill_cycles
        law_cycles = int(fill_cycles(law, facts.dim))
    return FillDepth(dim=int(facts.dim), measured_cycles=int(facts.fill_drain_depth),
                     law_cycles=law_cycles, law=law,
                     weight_buffer_slots=int(facts.weight_buffer_slots),
                     accumulator_banks=int(facts.accumulator_banks),
                     source=f"{facts.array_module}.{facts.valid_register_family} "
                            f"({facts.pe_instances} MAC instances)")
