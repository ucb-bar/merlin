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

**Two checks, and the difference between them matters.**

:func:`validate_interface_module` builds its specification from the function *signature*: each
committed output is the contraction of one activation argument with the reused weight argument. That
is deliberately not read back from the interface ops (which would make the check a tautology), but it
is still a RE-DERIVED model of what the pass should have done — target-vs-respecification, not
source-to-target translation validation. A reviewer should read its ``unsat`` as "the emitted program
computes the declared workload", not as "the pass preserved its input".

:func:`validate_pass` closes that gap: it encodes the ACTUAL ``linalg`` module the pass consumed
(:mod:`merlin.verify.linalg_semantics`) and validates ``source -> interface`` over shared leaves. Its
``unsat`` is a per-compilation theorem about the pass on that one program at that one shape.

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


def validate_pass(source_module, interface_module, *, acc_width: int = 32,
                  timeout_ms: int = 60_000) -> Verdict:
    """Does the emitted ``interface`` program compute the same function as its ``linalg`` SOURCE?

    This is source-to-target translation validation, and it is a strictly stronger statement than
    :func:`validate_interface_module` makes. There the specification is re-derived from the function
    signature — a model of what the pass *should* have done, written by the same people who wrote the
    pass. Here the specification is the pass's own INPUT IR, walked op by op
    (:func:`merlin.verify.linalg_semantics.encode_linalg`), so the only artifacts in the query are
    the two the compiler actually handled.

    Both sides are encoded over THE SAME symbolic leaves, bound by BLOCK-ARGUMENT POSITION: the pass
    preserves the function signature, so argument ``i`` of the source is argument ``i`` of the
    output, and that is a structural invariant of the pass rather than a naming convention. Encoding
    them over independent symbols would make the query trivially satisfiable and the verdict
    worthless.

    Results are paired IN ORDER — ``func.return`` operand order on the source side, ``commit`` order
    on the interface side, which is the same order because the emitted function returns its commits
    in the order it makes them. A count mismatch is an abstention, not a refutation: it means the two
    modules do not declare the same number of results, which this equality query is not entitled to
    characterise.

    Three verdicts, and the middle one is load-bearing:

        unsat   the pass is semantics-preserving on THIS program at THIS shape, for every integer
                input the shape admits
        sat     refuted, with a concrete counterexample in the model
        unknown the solver ran out of budget, or an op had no encoding — an ABSTENTION, never a pass

    A construct with no encoding raises :class:`UnsupportedSemantics` rather than being skipped: a
    skipped op silently weakens the theorem into one nobody stated.
    """
    if not HAS_XDSL:
        raise UnsupportedSemantics("xDSL is not installed")
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from .linalg_semantics import encode_linalg
    from .smt_ops import SolverOp

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()

        # SPEC side: the linalg module the pass consumed. It declares the leaves, because it is the
        # artifact whose arguments define the program's inputs.
        spec = encode_linalg(enc, source_module, acc_width=acc_width)

        # TARGET side: the interface program the pass emitted, over those same leaves.
        got = encode_interface(enc, interface_module, acc_width=acc_width, shared=spec.inputs)
        if not got.outputs:
            raise UnsupportedSemantics("the interface module commits no outputs; nothing to validate")

        spec_names, got_names = list(spec.outputs), list(got.outputs)
        if len(spec_names) != len(got_names):
            raise UnsupportedSemantics(
                f"the source module returns {len(spec_names)} value(s) but the interface program "
                f"commits {len(got_names)}; they are not the same program")

        diffs = []
        for s_name, g_name in zip(spec_names, got_names):
            diffs.append(enc.any_differs(spec.outputs[s_name], got.outputs[g_name]))
        term = diffs[0]
        for d in diffs[1:]:
            term = smt.OrOp(term, d).results[0]
        smt.AssertOp(term)
        smt.YieldOp()

    mod = builtin.ModuleOp([SolverOp.from_region(Region([blk]))])
    return check_module(mod, timeout_ms=timeout_ms)


def validate_compilation(interface_module, cb: dict, *, acc_width: int = 32,
                         timeout_ms: int = 60_000) -> Verdict:
    """Does the emitted COMMAND BUFFER compute what the ``interface`` program specified?

    This is the check that covers a compiler we did not write. ``interface`` is the input a backend
    receives; the command buffer is what it produced. Both are encoded over THE SAME symbolic leaves
    — encoding them over independent symbols would make the query trivially satisfiable — and the
    query asserts that some declared output can differ. ``unsat`` therefore means: for every integer
    input at this shape, the backend's buffer agrees with the interface program.

    Three verdicts, and the middle one matters most here. A backend may legitimately use an opcode or
    an epilogue this encoder has no definition for; that is an ABSTENTION, never a refutation.
    Refuting a correct backend because our encoder is incomplete is the one outcome that would make
    this tool worse than useless, so every gap raises :class:`UnsupportedSemantics` and is reported
    as abstained.
    """
    if not HAS_XDSL:
        raise UnsupportedSemantics("xDSL is not installed")
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from .cb_semantics import encode_command_buffer
    from .smt_ops import SolverOp

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()

        # SPEC side: the interface program the backend was handed.
        spec = encode_interface(enc, interface_module, acc_width=acc_width)
        if not spec.outputs:
            raise UnsupportedSemantics("interface module commits no outputs; nothing to validate")

        # The two sides must share leaves. The interface plane names inputs positionally (block
        # arguments); the command buffer names them (`A0`, `W`, ...). Bind by ROLE and order, which
        # is the only correspondence both artifacts actually agree on.
        shared = _bind_leaves(spec.inputs, cb)

        # TARGET side: the buffer the backend emitted, over those same symbols.
        got, _ = encode_command_buffer(enc, cb, shared=shared, acc_width=acc_width)

        pairs = _bind_outputs(spec.outputs, got, cb)
        diffs = [enc.any_differs(t, g) for t, g in pairs]
        term = diffs[0]
        for d in diffs[1:]:
            term = smt.OrOp(term, d).results[0]
        smt.AssertOp(term)
        smt.YieldOp()

    mod = builtin.ModuleOp([SolverOp.from_region(Region([blk]))])
    return check_module(mod, timeout_ms=timeout_ms)


def _bind_outputs(spec_outputs: dict, got: dict, cb: dict) -> list:
    """Pair the interface program's commits with the buffer's declared outputs, IN ORDER.

    The two artifacts do not share output names and are not expected to: the interface plane numbers
    its commits (``commit0``, ``commit1``) while the runtime lowering mints ``Y0``, ``Y1`` in commit
    order. Order is the correspondence they genuinely share. A count mismatch is an abstention — it
    means the buffer commits a different number of results than the program specified, which is a
    real disagreement but not one this equality check is entitled to characterise.
    """
    declared = list(cb.get("outputs") or ()) or sorted(got)
    spec_names = sorted(spec_outputs)
    if len(declared) != len(spec_names):
        raise UnsupportedSemantics(
            f"the interface program commits {len(spec_names)} output(s) {spec_names} but the command "
            f"buffer declares {len(declared)} ({declared}); they are not the same program")
    out = []
    for spec_name, cb_name in zip(spec_names, declared):
        a, b = spec_outputs[spec_name], got[cb_name]
        if (a.rows, a.cols) != (b.rows, b.cols):
            raise UnsupportedSemantics(
                f"output {spec_name!r} is {(a.rows, a.cols)} in the interface program but "
                f"{cb_name!r} is {(b.rows, b.cols)} in the command buffer")
        out.append((a, b))
    return out


def _bind_leaves(spec_inputs: list, cb: dict) -> dict:
    """Map command-buffer tensor names onto the interface program's symbolic inputs.

    The interface plane carries inputs as ordered block arguments with no names; the command buffer
    names them and tags each with a derived ``role``. Binding by (role, order) is the correspondence
    both artifacts genuinely share — the runtime lowering mints ``A0, A1, ...`` for activations and
    ``W, W1, ...`` for pack sources in argument order, so this is a real invariant, not a guess.

    A shape or width mismatch is an abstention: it means the two artifacts are not about the same
    program, and comparing them anyway would produce a meaningless verdict.
    """
    tensors = cb.get("tensors") or {}
    leaves = [(n, t) for n, t in tensors.items() if str(t.get("role")) in ("input", "weight")]
    # Weights last, mirroring the interface convention that the trailing argument is the reused
    # weight; within a role, the lowering's own name order is the argument order.
    inputs = sorted((n for n, t in leaves if str(t.get("role")) == "input"))
    weights = sorted((n for n, t in leaves if str(t.get("role")) == "weight"))
    ordered = inputs + weights
    if len(ordered) != len(spec_inputs):
        raise UnsupportedSemantics(
            f"the interface program has {len(spec_inputs)} leaf inputs but the command buffer "
            f"declares {len(ordered)} ({ordered}); they are not the same program")
    bound = {}
    for name, tensor in zip(ordered, spec_inputs):
        spec_shape = (tensor.rows, tensor.cols)
        cb_shape = tuple(tensors[name].get("shape") or ())
        if cb_shape != spec_shape:
            raise UnsupportedSemantics(
                f"leaf {name!r} is {cb_shape} in the command buffer but {spec_shape} in the "
                f"interface program")
        bound[name] = tensor
    return bound


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


def validate_equivalence(spec_cb: dict, agent_cb: dict, *, acc_width: int = 32,
                         timeout_ms: int = 60_000) -> Verdict:
    """Do two command buffers denote the same function, for every input at this shape?

    The reason this exists beside :func:`validate_compilation` is a parser gap, not a new idea. A
    capsule hands its backend an ``input.interface.mlir`` written in the ``merlin_iface`` contract
    grammar, which has custom assembly that xDSL cannot parse — ``cli._load_interface`` says so in its
    own docstring. But :func:`merlin.targetgen.contract.interface_emit.parse_interface_mlir` reads
    that grammar into a command-buffer dict, so the specification can be brought to the same encoder
    the submission goes through. That makes every archived submission checkable without any in-tree
    lowering.

    Example, over an archived capsule-bench submission::

        from merlin.targetgen.contract.interface_emit import parse_interface_mlir
        spec  = parse_interface_mlir((unit / "generated/input.interface.mlir").read_text())
        agent = json.loads((unit / "generated/command_buffer.json").read_text())
        validate_equivalence(spec, agent).status      # 'unsat' | 'sat' | 'unknown'

    **This is weaker than :func:`validate_compilation`, and the difference must not be blurred.**
    There, the spec side goes through ``encode_interface`` and the target side through
    ``encode_command_buffer`` — two encoders, so a bug in one does not cancel. Here both sides go
    through ``encode_command_buffer``. A defect in the shared encoder cancels wherever the two
    structures coincide, and cancels *completely* when the two buffers are identical. The theorem is
    therefore exactly: **the two buffers denote the same function under our shared bitvector
    semantics** — not "the submission is correct".

    A caller measuring coverage must exclude structurally identical pairs, or it will count ``X == X``
    as verification. Measured on the archived corpus, 2,500 of 4,111 submissions were identical to the
    program they were handed, so that exclusion is 61% of the material rather than a corner case; see
    :func:`merlin.verify.ablation.classify`.

    Abstains (raises :class:`UnsupportedSemantics`) rather than refuting whenever the two sides cannot
    be compared at all — a differing output count or a shape mismatch means they are not the same
    program, which is a real disagreement but not one an equality check is entitled to characterise.
    """
    if not HAS_XDSL:
        raise UnsupportedSemantics("xDSL is not installed")
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from .cb_semantics import encode_command_buffer
    from .smt_ops import SolverOp

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()

        # SPEC side first: it declares the leaves, because it is the artifact whose tensors define
        # the program's inputs. The submission is then encoded over those same symbols.
        spec_out, leaves = encode_command_buffer(enc, spec_cb, acc_width=acc_width)
        if not spec_out:
            raise UnsupportedSemantics(
                "the interface program commits no outputs; there is nothing to validate against")
        agent_out, _ = encode_command_buffer(enc, agent_cb, shared=leaves, acc_width=acc_width)

        spec_names, agent_names = sorted(spec_out), sorted(agent_out)
        if len(spec_names) != len(agent_names):
            raise UnsupportedSemantics(
                f"the interface program commits {len(spec_names)} output(s) {spec_names} but the "
                f"submitted buffer commits {len(agent_names)} ({agent_names}); they are not the "
                f"same program")

        diffs = []
        for s_name, a_name in zip(spec_names, agent_names):
            a, b = spec_out[s_name], agent_out[a_name]
            if (a.rows, a.cols) != (b.rows, b.cols):
                raise UnsupportedSemantics(
                    f"output {s_name!r} is {(a.rows, a.cols)} in the interface program but "
                    f"{a_name!r} is {(b.rows, b.cols)} in the submitted buffer")
            diffs.append(enc.any_differs(a, b))
        term = diffs[0]
        for d in diffs[1:]:
            term = smt.OrOp(term, d).results[0]
        smt.AssertOp(term)
        smt.YieldOp()

    mod = builtin.ModuleOp([SolverOp.from_region(Region([blk]))])
    return check_module(mod, timeout_ms=timeout_ms)
