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
(:mod:`merlin.verification.translation.dialect_semantics`) and validates ``source -> interface`` over shared leaves. Its
``unsat`` is a per-compilation theorem about the pass on that one program at that one shape.
The source encoder requires a defined contraction accumulator, such as an explicit integer
``linalg.fill``; a bare ``tensor.empty`` accumulator abstains because MLIR leaves its contents
unspecified.

Extents are concrete, taken from the IR's own types, so every query is quantifier-free (QF_BV).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from merlin.verification import HAS_XDSL
from merlin.verification.translation.dialect_semantics import ENCODABLE_OPS
from merlin.verification.translation.smt_encoding import (
    Encoder,
    SmtUnavailable,
    UnsupportedSemantics,
    Verdict,
    bounded_tool_output,
    check_module,
    encode_interface,
    module_text,
)

# ---- refine ------------------------------------------------------------------------------------


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
        return f"{self.status:7s} m={self.m} k={self.k} n={self.n} reuse={self.reuse} outputs={self.n_outputs}"


def validate_interface_module(
    module, *, acc_width: int = 32, timeout_ms: int = 60_000, translator: str | Path | None = None
) -> Verdict:
    """Check every ``interface.commit`` against the workload's declared contraction."""
    if not HAS_XDSL:
        raise UnsupportedSemantics("xDSL is not installed")
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from merlin.verification.translation.smt_encoding import SolverOp

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
            raise UnsupportedSemantics(f"{len(got.outputs)} commits but only {len(activations)} activation arguments")

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
    return check_module(mod, timeout_ms=timeout_ms, translator=translator)


def validate_pass(
    source_module,
    interface_module,
    *,
    acc_width: int = 32,
    timeout_ms: int = 60_000,
    translator: str | Path | None = None,
) -> Verdict:
    """Does the emitted ``interface`` program compute the same function as its ``linalg`` SOURCE?

    This is source-to-target translation validation, and it is a strictly stronger statement than
    :func:`validate_interface_module` makes. There the specification is re-derived from the function
    signature — a model of what the pass *should* have done, written by the same people who wrote the
    pass. Here the specification is the pass's own INPUT IR, walked op by op
    (:func:`merlin.verification.translation.dialect_semantics.encode_linalg`), so the only artifacts in the query are
    the two the compiler actually handled.

    Both sides are encoded over THE SAME symbolic leaves. In-tree ``interface.*`` binds by
    block-argument position. The frozen ``merlin_iface`` text grammar binds only explicit ``argN``
    leaf names to source argument N. Encoding independent symbols would make the verdict worthless.

    Results are paired IN ORDER — ``func.return`` operand order on the source side, ``commit`` order
    on the interface side, which is the same order because the emitted function returns its commits
    in the order it makes them. A count mismatch is an abstention, not a refutation: it means the two
    modules do not declare the same number of results, which this equality query is not entitled to
    characterise.

    Three verdicts, and the middle one is load-bearing:

        unsat   the pass is semantics-preserving on THIS supported program at THIS shape, for every
                integer input the shape admits; accumulated inits must have defined contents
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

    from merlin.verification.translation.dialect_semantics import encode_linalg
    from merlin.verification.translation.smt_encoding import SolverOp

    merlin_iface = isinstance(interface_module, str)
    if merlin_iface:
        # Parsing constructs IR operations. Do it before entering ImplicitBuilder, where those
        # operations would otherwise be inserted into the SMT block as they are constructed.
        from merlin.verification.translation.dialect_semantics import parse_merlin_iface

        interface_module = parse_merlin_iface(interface_module)

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()

        # SPEC side: the linalg module the pass consumed. It declares the leaves, because it is the
        # artifact whose arguments define the program's inputs.
        spec = encode_linalg(enc, source_module, acc_width=acc_width)

        # TARGET side: the interface program the pass emitted, over those same leaves.
        if merlin_iface:
            from merlin.verification.translation.dialect_semantics import encode_merlin_iface

            source_func = next(op for op in source_module.walk() if op.name == "func.func")
            if len(source_func.body.block.args) != len(spec.inputs):
                raise UnsupportedSemantics(
                    "merlin_iface argN binding requires every source argument to be a rank-2 tensor"
                )

            got = encode_merlin_iface(enc, interface_module, acc_width=acc_width, shared=spec.inputs)
        else:
            got = encode_interface(enc, interface_module, acc_width=acc_width, shared=spec.inputs)
        if not got.outputs:
            raise UnsupportedSemantics("the interface module commits no outputs; nothing to validate")

        spec_names, got_names = list(spec.outputs), list(got.outputs)
        if len(spec_names) != len(got_names):
            raise UnsupportedSemantics(
                f"the source module returns {len(spec_names)} value(s) but the interface program "
                f"commits {len(got_names)}; they are not the same program"
            )

        diffs = []
        for s_name, g_name in zip(spec_names, got_names):
            diffs.append(enc.any_differs(spec.outputs[s_name], got.outputs[g_name]))
        term = diffs[0]
        for d in diffs[1:]:
            term = smt.OrOp(term, d).results[0]
        smt.AssertOp(term)
        smt.YieldOp()

    mod = builtin.ModuleOp([SolverOp.from_region(Region([blk]))])
    return check_module(mod, timeout_ms=timeout_ms, translator=translator)


def validate_compilation(
    interface_module,
    cb: dict,
    *,
    acc_width: int = 32,
    timeout_ms: int = 60_000,
    translator: str | Path | None = None,
) -> Verdict:
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

    from merlin.verification.translation.dialect_semantics import encode_command_buffer
    from merlin.verification.translation.smt_encoding import SolverOp

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
    return check_module(mod, timeout_ms=timeout_ms, translator=translator)


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
            f"buffer declares {len(declared)} ({declared}); they are not the same program"
        )
    out = []
    for spec_name, cb_name in zip(spec_names, declared):
        a, b = spec_outputs[spec_name], got[cb_name]
        if (a.rows, a.cols) != (b.rows, b.cols):
            raise UnsupportedSemantics(
                f"output {spec_name!r} is {(a.rows, a.cols)} in the interface program but "
                f"{cb_name!r} is {(b.rows, b.cols)} in the command buffer"
            )
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
            f"declares {len(ordered)} ({ordered}); they are not the same program"
        )
    bound = {}
    for name, tensor in zip(ordered, spec_inputs):
        spec_shape = (tensor.rows, tensor.cols)
        cb_shape = tuple(tensors[name].get("shape") or ())
        if cb_shape != spec_shape:
            raise UnsupportedSemantics(
                f"leaf {name!r} is {cb_shape} in the command buffer but {spec_shape} in the interface program"
            )
        bound[name] = tensor
    return bound


def validate_workload(
    *, m: int = 2, k: int = 2, n: int = 2, reuse: int = 2, timeout_ms: int = 60_000
) -> RefinementResult:
    """Lower the reference workload at one concrete shape and validate the interface it produced.

    This exercises the real ``merlin-materialize-interface`` pass: the module under test is whatever
    the pass emitted, not a hand-written stand-in.
    """
    from merlin.compiler.dialects.lowering import pipeline

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
        from merlin.compiler.dialects.lowering import passes as P

        P.record_verification(
            VALIDATED_PASS,
            requirement_class=VALIDATED_CLASS,
            method="smt",
            verdict=P.solver_verdict(verdict.status),
            evidence={
                "shape": shape,
                "solver_status": verdict.status,
                "counterexample": bool(getattr(verdict, "model_values", None)),
            },
            provenance={
                "source": "merlin.verification.translation.refinement.validate_workload",
                "relation": "every commit equals the declared contraction, all inputs",
            },
        )
    except Exception:
        # Recording must never gate a verification run; a missing log is a REPORTED state upstream.
        pass


class OutputContractViolation(Exception):
    """The buffer does not deliver the outputs its specification declared.

    A REFUTATION, not an abstention, and deliberately a separate type from a numeric divergence.
    ``sat`` carries a counterexample by contract; this defect has no counterexample because no input
    is needed to see it -- the buffer simply never publishes the declared result.

    Found by reading the ablation's own contradictory cell. Thirteen archived submissions were graded
    `numeric=fail` with EVERY element mismatched ("your command buffer does not compute the declared
    operation") while this checker called them VERIFIED. All thirteen computed the right values and
    committed them under a name the specification never declared -- eleven to ``output_tensor``
    instead of ``Y0``, and two committing ``Y0`` twice. The checker paired outputs by sorted ORDER,
    so a single-output buffer matched a single-output spec no matter what either called its result.
    A checker that cannot see a whole class of real defect, and reports the strongest possible
    verdict on it, is worse than one that abstains.
    """


def validate_equivalence(spec_cb: dict, agent_cb: dict, *, acc_width: int = 32, timeout_ms: int = 60_000) -> Verdict:
    """Do two command buffers denote the same function, for every input at this shape?

    The reason this exists beside :func:`validate_compilation` is a parser gap, not a new idea. A
    capsule hands its backend an ``input.interface.mlir`` written in the ``merlin_iface`` contract
    grammar, which has custom assembly that xDSL cannot parse — ``cli._load_interface`` says so in its
    own docstring. But :func:`merlin.target.contract.interfaces.grammar.parse_interface_mlir` reads
    that grammar into a command-buffer dict, so the specification can be brought to the same encoder
    the submission goes through. That makes every archived submission checkable without any in-tree
    lowering.

    Example, over an archived capsule-bench submission::

        from merlin.target.contract.interfaces.grammar import parse_interface_mlir

        spec = parse_interface_mlir((unit / "generated/input.interface.mlir").read_text())
        agent = json.loads((unit / "generated/command_buffer.json").read_text())
        validate_equivalence(spec, agent).status  # 'unsat' | 'sat' | 'unknown'

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
    :func:`merlin.verification.ablation.classify`.

    Abstains (raises :class:`UnsupportedSemantics`) rather than refuting whenever the two sides cannot
    be compared at all — a differing output count or a shape mismatch means they are not the same
    program, which is a real disagreement but not one an equality check is entitled to characterise.
    """
    if not HAS_XDSL:
        raise UnsupportedSemantics("xDSL is not installed")
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from merlin.verification.translation.dialect_semantics import encode_command_buffer
    from merlin.verification.translation.smt_encoding import SolverOp

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()

        # SPEC side first: it declares the leaves, because it is the artifact whose tensors define
        # the program's inputs. The submission is then encoded over those same symbols.
        spec_out, leaves = encode_command_buffer(enc, spec_cb, acc_width=acc_width)
        if not spec_out:
            raise UnsupportedSemantics("the interface program commits no outputs; there is nothing to validate against")
        agent_out, _ = encode_command_buffer(enc, agent_cb, shared=leaves, acc_width=acc_width)

        # Bind outputs BY NAME. Order-binding was the defect described on OutputContractViolation:
        # it made the checker blind to a buffer publishing the right numbers under the wrong name.
        spec_names, agent_names = sorted(spec_out), sorted(agent_out)
        missing = [n for n in spec_names if n not in agent_out]
        if missing:
            raise OutputContractViolation(
                f"the interface program declares output(s) {missing} which the submitted buffer "
                f"never commits; it commits {agent_names}. The buffer may compute the right values, "
                f"but it does not deliver them under the declared name, so no consumer can read "
                f"them."
            )
        extra = [n for n in agent_names if n not in spec_out]
        if extra:
            raise OutputContractViolation(
                f"the submitted buffer commits {extra}, which the interface program never declared "
                f"(it declares {spec_names})"
            )

        diffs = []
        for s_name in spec_names:
            a, b = spec_out[s_name], agent_out[s_name]
            a_name = s_name
            if (a.rows, a.cols) != (b.rows, b.cols):
                raise UnsupportedSemantics(
                    f"output {s_name!r} is {(a.rows, a.cols)} in the interface program but "
                    f"{a_name!r} is {(b.rows, b.cols)} in the submitted buffer"
                )
            diffs.append(enc.any_differs(a, b))
        term = diffs[0]
        for d in diffs[1:]:
            term = smt.OrOp(term, d).results[0]
        smt.AssertOp(term)
        smt.YieldOp()

    mod = builtin.ModuleOp([SolverOp.from_region(Region([blk]))])
    return check_module(mod, timeout_ms=timeout_ms)

    # Preserve historical object lookup; implementation and completion state live here.


# ---- receipts ----------------------------------------------------------------------------------
# Reproducible evidence for one supported compiler transformation.
#
# The receipt identifies both *actual* artifacts, the verifier implementation, translator binary,
# solver version, SMT query, and outcome. A qualification gate reruns the obligation. Neither an IR
# syntax check nor a copied ``unsat`` string is accepted as semantic evidence.

Boundary = Literal["linalg_to_interface", "interface_to_command_buffer"]
_BOUNDARIES = frozenset(("linalg_to_interface", "interface_to_command_buffer"))
_SEMANTIC_FILES = ("dialect_semantics.py", "refinement.py", "smt_encoding.py")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _identity(value: Any, *, command_buffer: bool = False) -> dict[str, str]:
    if command_buffer:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        kind = "canonical_json"
    elif isinstance(value, str):
        data = value.encode("utf-8")
        kind = "merlin_iface_text_utf8"
    else:
        data = module_text(value).encode("utf-8")
        kind = "xdsl_generic_ir"
    return {"encoding": kind, "sha256": _sha(data)}


def _ir_signature(module: Any) -> dict[str, Any]:
    if isinstance(module, str):
        from merlin.verification.translation.dialect_semantics import merlin_iface_signature

        try:
            return merlin_iface_signature(module)
        except UnsupportedSemantics as exc:
            return {"grammar": "merlin_iface", "parse_error": str(exc)}
    funcs = [op for op in module.walk() if op.name == "func.func"]
    if len(funcs) != 1:
        return {"function_count": len(funcs)}
    block = funcs[0].body.block
    returns = [op for op in block.ops if op.name == "func.return"]
    return {
        "function_count": 1,
        "inputs": [str(arg.type) for arg in block.args],
        "return_count": len(returns),
        "outputs": [str(value.type) for value in returns[0].operands] if len(returns) == 1 else [],
    }


def _cb_signature(cb: dict) -> dict[str, Any]:
    return {
        "tensors": {
            name: {key: tensor.get(key) for key in ("dtype", "shape", "role")}
            for name, tensor in sorted((cb.get("tensors") or {}).items())
        },
        "outputs": list(cb.get("outputs") or ()),
    }


def _verifier_digest() -> str:
    root = Path(__file__).parent
    data = b"".join(name.encode() + b"\0" + (root / name).read_bytes() + b"\0" for name in _SEMANTIC_FILES)
    bridge = root.parents[1] / "development" / "targetgen" / "contract" / "interface_emit.py"
    data += b"targetgen/contract/interface_emit.py\0" + bridge.read_bytes() + b"\0"
    return _sha(data)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _toolchain(translator: str | Path) -> dict[str, str | None]:
    path = Path(translator).resolve(strict=True)
    if not path.is_file():
        raise SmtUnavailable(f"translator {path} is not a regular file")
    returncode, stdout, stderr = bounded_tool_output(
        [str(path), "--version"], timeout_s=10, max_output_bytes=1024 * 1024
    )
    if returncode != 0:
        raise SmtUnavailable(f"cannot identify translator {path}: exit {returncode}: {stderr[:1000]}")
    reported = stdout.strip()
    try:
        import z3

        solver_version = z3.get_version_string()
    except ImportError:
        solver_version = None
    return {
        "mlir_translate_path": str(path),
        "mlir_translate_sha256": _sha(path.read_bytes()),
        "mlir_translate_version": reported,
        "z3_version": solver_version,
        "xdsl_version": _package_version("xdsl"),
        "verifier_sha256": _verifier_digest(),
    }


@dataclass(frozen=True)
class TransformReceipt:
    schema: str
    method: str
    boundary: str
    source: dict[str, str]
    target: dict[str, str]
    typed_signatures: dict[str, Any]
    toolchain: dict[str, str | None]
    acc_width: int
    timeout_ms: int
    assumptions: tuple[str, ...]
    status: str
    reason: str | None
    smt2_sha256: str | None
    counterexample: dict[str, Any] | None

    @property
    def verified(self) -> bool:
        return self.method == "smt_translation_validation" and self.status == "verified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> TransformReceipt:
        """Load a JSON round trip; qualification still replays the obligation."""
        if record.get("schema") != "merlin.transform_verification.v1":
            raise ValueError(f"unrecognized transform receipt schema {record.get('schema')!r}")
        return cls(**{**record, "assumptions": tuple(record["assumptions"])})


def verify_transformation(
    boundary: Boundary,
    source: Any,
    target: Any,
    *,
    translator: str | Path,
    expected_translator_sha256: str | None = None,
    acc_width: int = 32,
    timeout_ms: int = 60_000,
) -> TransformReceipt:
    """Prove one concrete, supported source/target pair or record why no proof was obtained.

    A recorded ``verified`` means QF_BV unsatisfiability for every bit pattern of the typed input
    tensors at the concrete IR shape, under the explicitly listed semantics assumptions. It does
    not claim all shapes, arbitrary models, floating point, or equivalence to real hardware.
    """
    if boundary not in _BOUNDARIES:
        raise ValueError(f"unknown transformation boundary {boundary!r}")
    if acc_width <= 0 or timeout_ms <= 0:
        raise ValueError("acc_width and timeout_ms must be positive")

    is_cb = boundary == "interface_to_command_buffer"
    src_id = _identity(source)
    dst_id = _identity(target, command_buffer=is_cb)
    typed_signatures = {
        "source": _ir_signature(source),
        "target": _cb_signature(target) if is_cb else _ir_signature(target),
    }
    assumptions = (
        "rank-2 positive concrete integer tensor extents; signed bitvectors; modular accumulation",
        "input tensors bound by command-buffer role and order"
        if is_cb
        else (
            "merlin_iface leaves named argN bound to source argument N"
            if isinstance(target, str)
            else "input tensors bound by block-argument position"
        ),
        "contraction init must have defined contents; a bare tensor.empty init abstains"
        if not is_cb
        else "interface.resident_pack preserves values; physical layout is outside this value proof",
        *(
            ("merlin_iface.resident_pack preserves values; physical layout is outside this value proof",)
            if isinstance(target, str) and not is_cb
            else ()
        ),
    )
    tool: dict[str, str | None] = {}
    status = "unavailable"
    reason: str | None = None
    smt_hash: str | None = None
    counterexample: dict[str, Any] | None = None
    try:
        tool = _toolchain(translator)
        actual_sha = tool["mlir_translate_sha256"]
        if expected_translator_sha256 is not None and actual_sha != expected_translator_sha256:
            raise SmtUnavailable(
                f"translator SHA-256 mismatch: expected {expected_translator_sha256}, found {actual_sha}"
            )

        verdict = (
            validate_compilation(source, target, acc_width=acc_width, timeout_ms=timeout_ms, translator=translator)
            if is_cb
            else validate_pass(source, target, acc_width=acc_width, timeout_ms=timeout_ms, translator=translator)
        )
        # Reject a replacement of the selected binary between the initial hash/version read and
        # export. The receipt's tool identity must describe the executable that produced the query.
        after_hash = _sha(Path(translator).resolve(strict=True).read_bytes())
        if after_hash != actual_sha:
            raise SmtUnavailable("translator bytes changed during the verification attempt")
        smt_hash = _sha(verdict.smt2.encode("utf-8"))
        status = {"unsat": "verified", "sat": "refuted", "unknown": "unknown"}[verdict.status]
        reason = verdict.reason
        if status == "refuted":
            counterexample = {"model": verdict.model, "inputs": verdict.model_values}
    except UnsupportedSemantics as exc:
        status, reason = "unsupported", str(exc)
    except SmtUnavailable as exc:
        status, reason = "unavailable", str(exc)
    except Exception as exc:
        # A checker error is visible and cannot be promoted into a proof. The exception class is
        # included to make an unexpected translator or encoder failure diagnosable from the receipt.
        status, reason = "error", f"{type(exc).__name__}: {exc}"

    return TransformReceipt(
        schema="merlin.transform_verification.v1",
        method="smt_translation_validation",
        boundary=boundary,
        source=src_id,
        target=dst_id,
        typed_signatures=typed_signatures,
        toolchain=tool,
        acc_width=acc_width,
        timeout_ms=timeout_ms,
        assumptions=assumptions,
        status=status,
        reason=reason,
        smt2_sha256=smt_hash,
        counterexample=counterexample,
    )


def qualify_receipt(
    receipt: TransformReceipt,
    source: Any,
    target: Any,
    *,
    translator: str | Path,
) -> bool:
    """Rerun and require the *same* semantic proof for the exact current artifacts/toolchain.

    A syntax verifier may be useful separately, but it cannot produce this receipt's method. Old,
    fabricated, edited, unsupported, refuted, unknown, and tool-mismatched receipts cannot qualify.
    """
    if not receipt.verified or receipt.boundary not in _BOUNDARIES:
        return False
    replay = verify_transformation(
        receipt.boundary,
        source,
        target,
        translator=translator,
        expected_translator_sha256=receipt.toolchain.get("mlir_translate_sha256"),
        acc_width=receipt.acc_width,
        timeout_ms=receipt.timeout_ms,
    )
    return replay.verified and replay == receipt

    # Preserve historical object lookup; implementation and completion state live here.


# ---- model_coverage ----------------------------------------------------------------------------
# Conservative, machine-readable inventory of captured MLIR versus the SMT source subset.
#
# This is a textual inventory, not a parser, a proof, or a certificate. It only claims a possible
# source-side encoding when every visible op and tensor type fits the known subset; the real SMT
# encoder remains the authority and may abstain on further value-level preconditions.

_ELEMENT = frozenset(("i8", "i16", "i32", "i64"))
_STRUCTURAL = frozenset(("builtin.module", "func.func"))


def _leading_op(line: str) -> str | None:
    """Read an MLIR statement's leading operation name, without parsing its body."""
    statement = line.lstrip()
    if statement.startswith("%"):
        lhs, separator, rhs = statement.partition("=")
        names = [part.strip() for part in lhs.split(",")]
        if not separator or not all(
            name.startswith("%") and len(name) > 1 and all(c.isalnum() or c == "_" for c in name[1:]) for name in names
        ):
            return None
        statement = rhs.lstrip()
    if statement.startswith('"'):
        name, separator, _ = statement[1:].partition('"')
        return name if separator and name and all(c.isalnum() or c in "_." for c in name) else None
    if not statement or not statement[0].isalpha():
        return None
    name = statement.split(maxsplit=1)[0].split("(", 1)[0]
    name = name.rstrip("{,:;")
    return name if name and all(c.isalnum() or c in "_." for c in name) else None


def _tensor_spellings(text: str) -> Counter[str]:
    """Count simple `tensor<...>` spellings; nested syntax is left unclassified."""
    found: Counter[str] = Counter()
    start = 0
    while (marker := text.find("tensor<", start)) >= 0:
        begin = marker + len("tensor<")
        end = text.find(">", begin)
        if end < 0:
            break
        spelling = text[begin:end]
        if "<" not in spelling:
            found[spelling] += 1
        start = begin if "<" in spelling else end + 1
    return found


def _tensor_signature(spelling: str) -> tuple[int, str, bool]:
    parts = spelling.split("x")
    dtype = parts[-1]
    shape = parts[:-1]
    return len(shape), dtype, any(not dimension.isdigit() or int(dimension) <= 0 for dimension in shape)


def audit_capture(path: str | Path) -> dict:
    """Inventory one exact captured file; unknown syntax always becomes an abstention reason."""
    source = Path(path)
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    ops: Counter[str] = Counter()
    unparsed_lines = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith(("//", "#", "}")):
            continue
        name = _leading_op(line)
        if name:
            if name in _STRUCTURAL:
                continue
                # Affine region labels and MLIR block argument declarations are not operations.
            if name.startswith("^") or name.startswith("affine_map"):
                continue
            if "." in name:
                ops[name] += 1
        elif stripped.startswith("%") and " = " in stripped:
            unparsed_lines += 1

    tensor_spellings = _tensor_spellings(text)
    types: Counter[tuple[int, str, bool]] = Counter()
    for spelling, count in tensor_spellings.items():
        types[_tensor_signature(spelling)] += count
    ranks = sorted({rank for rank, _, _ in types})
    dtypes = sorted({dtype for _, dtype, _ in types})
    unsupported_ops = {name: count for name, count in sorted(ops.items()) if name not in ENCODABLE_OPS}
    unsupported_types = sorted(
        {
            f"rank={rank},dtype={dtype},dynamic={dynamic}"
            for rank, dtype, dynamic in types
            if rank != 2 or dtype not in _ELEMENT or dynamic
        }
    )
    blockers = []
    if unsupported_ops:
        blockers.append("unencoded source operations")
    if unsupported_types:
        blockers.append("tensor types outside rank-2 concrete integer domain")
    if unparsed_lines:
        blockers.append("unparsed assignment lines")
    if not ops:
        blockers.append("no operations inventoried")
    return {
        "schema": "merlin.capture_smt_coverage_inventory.v1",
        "method": "textual_inventory_not_proof",
        "capture": str(source),
        "capture_sha256": hashlib.sha256(raw).hexdigest(),
        "source_boundary": "linalg_to_interface",
        "eligible_for_whole_module_smt_attempt": not blockers,
        "blockers": blockers,
        "operation_counts": dict(sorted(ops.items())),
        "unsupported_operation_counts": unsupported_ops,
        "tensor_ranks_observed": ranks,
        "tensor_dtypes_observed": dtypes,
        "tensor_type_occurrence_counts": dict(sorted(tensor_spellings.items())),
        "unsupported_tensor_signatures": unsupported_types,
        "unparsed_assignment_lines": unparsed_lines,
        "candidate_supported_operation_count": sum(count for name, count in ops.items() if name in ENCODABLE_OPS),
        "observed_operation_count": sum(ops.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="inventory captured MLIR against the current SMT source subset")
    parser.add_argument("captures", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit_capture(path) for path in args.captures], indent=2, sort_keys=True))

    # Preserve historical object lookup; implementation and completion state live here.


if __name__ == "__main__":
    main()
