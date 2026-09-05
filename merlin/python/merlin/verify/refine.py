"""Translation validation for the ``interface`` plane.

This answers the question a pass-rate cannot: *does the program the compiler produced compute the
same function as the specification it was produced from* — for **all inputs** at a given shape, not
just the stimulus we happened to run.

Method, following Fehr et al. (PLDI 2025) §5.1: give both sides a semantics by lowering to the
``smt`` dialect, assert the NEGATION of the refinement relation, and solve.

    unsat   -> verified: the two agree on every input at this shape
    sat     -> refuted:  the model is a concrete counterexample
    unknown -> neither:  a timeout is NOT a pass

Both sides are encoded over the SAME symbolic inputs, in one pass — encoding them independently
would make the query trivially satisfiable and the result meaningless.

**What the specification side is, stated precisely.** It is the declared workload: each committed
output is the contraction of one activation argument with the reused weight argument, built from the
function *signature*. It is deliberately NOT read back from the interface ops, which would make the
check a tautology. It is also not (yet) an encoding of the ``linalg`` body the pass consumed;
encoding ``linalg.quantized_matmul`` with its zero-point arithmetic is the natural next step, and
until it lands this validates the interface program against the workload's declared contraction
rather than against its source IR. Recorded here rather than blurred.

Extents are concrete, taken from the IR's own types, so every query is quantifier-free (QF_BV).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import HAS_XDSL
from .smt_export import Verdict, check_module
from .smt_semantics import Encoder, UnsupportedSemantics, encode_interface


@dataclass(frozen=True)
class RefinementResult:
    """A verdict together with the shape it is about — a verdict without its shape is not citable."""
    verdict: Verdict
    m: int
    k: int
    n: int
    reuse: int
    n_outputs: int

    @property
    def status(self) -> str:
        return self.verdict.status

    @property
    def verified(self) -> bool:
        return self.verdict.verified

    def __str__(self) -> str:
        return (f"{self.status:7s} m={self.m} k={self.k} n={self.n} "
                f"reuse={self.reuse} outputs={self.n_outputs}")


def validate_interface_module(module, *, acc_width: int = 32,
                              timeout_ms: int = 60_000) -> Verdict:
    """Check every ``interface.commit`` against the workload's declared contraction."""
    if not HAS_XDSL:
        raise UnsupportedSemantics("xDSL is not installed")
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from .smt_ops import SolverOp

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()

        # TARGET: interpret the IR the compiler actually produced.
        got = encode_interface(enc, module, acc_width=acc_width)
        if not got.outputs:
            raise UnsupportedSemantics("module commits no outputs; nothing to validate")
        if len(got.inputs) < 2:
            raise UnsupportedSemantics("expected at least one activation and one weight argument")

        # SPEC: the declared workload, over the SAME symbolic inputs. Convention of the reference
        # workload is that the trailing argument is the reused weight.
        weight = got.inputs[-1]
        activations = got.inputs[:-1]
        if len(activations) < len(got.outputs):
            raise UnsupportedSemantics(
                f"{len(got.outputs)} commits but only {len(activations)} activation arguments")

        diffs = []
        for i, name in enumerate(sorted(got.outputs)):
            expected = enc.matmul(activations[i], weight, acc_width=acc_width)
            diffs.append(enc.any_differs(got.outputs[name], expected))
        term = diffs[0]
        for d in diffs[1:]:
            term = smt.OrOp(term, d).results[0]
        smt.AssertOp(term)
        smt.YieldOp()

    mod = builtin.ModuleOp([SolverOp.from_region(Region([blk]))])
    return check_module(mod, timeout_ms=timeout_ms)


def validate_workload(*, m: int = 2, k: int = 2, n: int = 2, reuse: int = 2,
                      timeout_ms: int = 60_000) -> RefinementResult:
    """Lower the reference workload at one concrete shape and validate the interface it produced.

    This exercises the real ``merlin-materialize-interface`` pass: the module under test is whatever
    the pass emitted, not a hand-written stand-in.
    """
    from merlin.xdsl_dialects.lowering import pipeline

    result = pipeline.lower_repeated_rhs_matmul(reuse=reuse, m=m, k=k, n=n)
    module = result.interface_module
    n_out = sum(1 for op in module.walk() if op.name == "interface.commit")
    verdict = validate_interface_module(module, timeout_ms=timeout_ms)
    _record(verdict, m=m, k=k, n=n, reuse=reuse)
    return RefinementResult(verdict=verdict, m=m, k=k, n=n, reuse=reuse, n_outputs=n_out)


#: The pass this validation is ABOUT, and the requirement class it discharges. Named here rather than
#: inferred, because a verdict attributed to the wrong pass is worse than no verdict.
VALIDATED_PASS = "merlin-materialize-interface"
VALIDATED_CLASS = "host-seam"


def _record(verdict, **shape) -> None:
    """Write the verdict to the shared verification log the obligation gate reads.

    A no-op unless ``MERLIN_VERIFY_LOG`` is set, exactly like invocation recording — so the layer's
    result becomes evidence without the layer having to know whether anyone is collecting it. A
    solver ``unknown`` maps to ``unmeasured``, never to a pass.
    """
    try:
        from merlin.xdsl_dialects.lowering import passes as P

        P.record_verification(
            VALIDATED_PASS,
            requirement_class=VALIDATED_CLASS,
            method="smt",
            verdict=P.solver_verdict(verdict.status),
            evidence={"shape": shape, "solver_status": verdict.status,
                      "counterexample": bool(getattr(verdict, "model_values", None))},
            provenance={"source": "merlin.verify.refine.validate_workload",
                        "relation": "every commit equals the declared contraction, all inputs"})
    except Exception:
        # Recording must never gate a verification run; a missing log is a REPORTED state upstream.
        pass
