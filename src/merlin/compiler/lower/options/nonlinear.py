"""``int_softmax_table``: the integer softmax's per-element work, restructured exactly (default off).

A capture made with integer nonlinears (``targetgen/_integer_nonlinear.py``) spells softmax as a row
maximum, a fixed exponent grid index ``q = clamp(round((x - max) / S), lo, hi)``, an integer exp and a
64-bit floor division per element, an i64 row sum, ``P = p / T``, and then the next contraction's
per-row int8 quantization of ``P`` -- its row minimum and maximum, then a divide, round and clamp per
element. Measured on one SmolVLA SigLIP layer, the integer exp and division alone were 46 spike
instructions per element of a 12x1024x1024 score tensor.

This rewrite runs over the CAPTURED IR (the capture is unchanged) before the lowering pipeline, in the
compiler's Python (:data:`RUNNER_PRELUDE`, the source of ``_int_softmax_table_rt.py``). Each piece is
the same value as what it replaces:

* the numerator becomes a read of a table of every grid index, evaluated at compile time with the
  IR's own integer semantics;
* the clamp's upper bound (which never binds, ``x - max <= 0``) is dropped and its lower bound
  becomes a compare-and-select, so NaN -- a row with no finite maximum, poison in the original --
  also lands inside the table;
* the row sum accumulates in i32 when it provably cannot overflow;
* ``P``'s quantization is computed once per row on its candidate values ``k / T`` and each element
  reads its int8 at ``p`` (the row's largest ``P`` is ``max / T``, and the row minimum only enters
  clamped with 0 -- both checked on the IR);
* a constant step on the scores applied behind a reshape (the attention scale) moves before it, so it
  fuses into the op that produced them.

The match is STRUCTURAL -- the ops above, never a model, shape or target name -- and anything that
differs leaves the IR as it was, with the reason in the report. A test executes the original and the
rewritten module on the same inputs and requires bit-identical outputs.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import math
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.paths import module_source_path

# ---- int_softmax_table -------------------------------------------------------------------------

FEATURE = "int_softmax_table"
REPORT_PREFIX = "OK int_softmax_table "
#: The runner argv slot that gates the rewrite (after the data layout, so no existing slot moves).
ARGV_INDEX = 20
_RT_SOURCE = module_source_path("merlin.compiler.lower.options.int_softmax_rt")


def runtime_source() -> str:
    """The runner half, verbatim (it is only ever executed by the compiler's Python)."""
    return _RT_SOURCE.read_text(encoding="utf-8")


#: Spliced into every lowering runner. ``_INT_SOFTMAX_TABLE`` gates the call each runner makes right
#: after it parses the module, before any other pre-pipeline rewrite.
RUNNER_PRELUDE = (
    "\n"
    + "exec(compile("
    + repr(runtime_source())
    + ", "
    + repr(str(_RT_SOURCE))
    + ", 'exec'), globals())\n"
    + f"\n\n_INT_SOFTMAX_TABLE = len(sys.argv) > {ARGV_INDEX} and sys.argv[{ARGV_INDEX}] == '1'\n\n\n"
    "def _ist_run_and_report(ctx, module):\n"
    "    import json as _ist_json\n"
    "    print(_IST_TOKEN + _ist_json.dumps(_int_softmax_table(ctx, module), sort_keys=True), flush=True)\n"
)

#: What each runner variant executes after parsing the module.
RUNNER_CALL = "if _INT_SOFTMAX_TABLE:\n    _ist_run_and_report(ctx, module)\n"


def _feature():
    from merlin.compiler.selection.features import ImprFeature

    return ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "Restructure every integer softmax in the captured IR, exactly: the per-element integer "
            "exp and floor division become a table read (evaluated at compile time with the IR's own "
            "integer semantics), the never-binding upper clamp is dropped and the lower one becomes a "
            "compare-and-select, the row sum accumulates in i32 when it cannot overflow, the next "
            "contraction's per-row int8 quantization of the probabilities runs once per row on their "
            "candidate values, and the attention scale moves before the reshape that hid it from "
            "fusion. Structure-matched, never by name; default off; runner-gated."
        ),
    )


def ensure_registered() -> str:
    from merlin.compiler.selection.features import known, register

    if FEATURE not in known():
        register(_feature())
    return FEATURE


def require_report(stdout: str, work: str | Path) -> dict:
    """The runner's report for a requested rewrite, kept beside the lowered IR.

    A requested rewrite that never reported did not run (a runner variant that skipped it would emit
    a valid baseline object), so its absence is an error. Matching nothing is not: a module without an
    integer softmax is unchanged, and the report says why each softmax-shaped region was left alone.
    """
    lines = [line[len(REPORT_PREFIX) :] for line in stdout.splitlines() if line.startswith(REPORT_PREFIX)]
    if len(lines) != 1:
        raise ValueError(f"{FEATURE} was requested but the lowering runner reported {len(lines)} times, not once")
    report = json.loads(lines[0])
    (Path(work) / f"{FEATURE}_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def apply_for_test(mlir_text: str, *, timeout: int = 600) -> tuple[str, dict]:
    """Apply the shipped rewrite to ``mlir_text`` in the compiler's Python; ``(rewritten text, report)``."""
    from merlin.compiler.lower.model.toolchain import m2m_python

    work = Path(tempfile.mkdtemp(prefix="merlin_int_softmax_table_"))
    src, dst, script = work / "in.mlir", work / "out.mlir", work / "run.py"
    src.write_text(mlir_text, encoding="utf-8")
    script.write_text(
        "import json, sys\nfrom torch_mlir import ir\n" + runtime_source() + "\nctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n    module = ir.Module.parse(f.read(), ctx)\n"
        "report = _int_softmax_table(ctx, module)\n"
        "with open(sys.argv[2], 'w') as f:\n    f.write(str(module.operation))\n"
        "print(_IST_TOKEN + json.dumps(report, sort_keys=True))\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [str(m2m_python()), str(script), str(src), str(dst)], capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{FEATURE} rewrite failed:\n{proc.stdout}\n{proc.stderr}")
    return dst.read_text(encoding="utf-8"), require_report(proc.stdout, work)


# Direct imports (tests, controlled build scripts) see a resolvable feature name; the lowering also
# registers it explicitly before normalizing a feature set.
ensure_registered()


# ---- selected_expf_equivalence -----------------------------------------------------------------
# Explicit, implementation-pinned permission for one expf input equivalence class.
#
# This helper selects no compiler feature. Its caller retains the live grant through
# compilation and the actual link, uses the returned absolute-archive linker argv,
# and checks the freshly generated GNU map afterwards. No saved report restores a
# grant. The ordinary closed-region/errno/fenv guards remain separate obligations.

SELECTED_LIBM_SHA256 = "38965f2df6e94d3389596fcf414caf37040294fc0f43f3afa3e4369acb837934"
PROTOCOL = "merlin.selected_expf_finite_underflow_equivalence.v1"
_SECTIONS = {
    ".text.expf": "libm_a-wf_exp.o",
    ".text.finitef": "libm_a-sf_finite.o",
    ".text.__ieee754_expf": "libm_a-ef_exp.o",
    ".text.with_errnof": "libm_a-math_errf.o",
    ".text.xflowf": "libm_a-math_errf.o",
    ".text.__math_uflowf": "libm_a-math_errf.o",
}


class ExpfEquivalenceError(ValueError):
    """The explicitly selected callee is absent, changed, or unsupported."""


def _sha(path: Path, *, limit: int) -> str:
    if not path.is_file() or path.stat().st_size > limit:
        raise ExpfEquivalenceError(f"missing or oversized selected artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ExpfUnderflowEquivalence:
    """A live selected archive, with fixed binary32 and sticky-effect facts.

    The finite input words form an unsigned inclusive interval. The first class
    occurrence still invokes the original expf. These facts do not permit dropping
    its first UF/NX/errno effects, changing any source arithmetic, or merging -inf.
    """

    library_path: Path

    @property
    def minimum_input_bits(self) -> int:
        return 0xC2CFF1B6

    @property
    def maximum_input_bits(self) -> int:
        return 0xFF7FFFFF

    @property
    def result_bits(self) -> int:
        return 0

    @property
    def raised_fflags(self) -> int:
        return 3  # RISC-V UF | NX; all five legal FRM modes, unchanged FRM.

    @property
    def errno_value(self) -> int:
        return 34

    @property
    def library_sha256(self) -> str:
        return SELECTED_LIBM_SHA256

    def revalidate(self) -> None:
        if type(self) is not ExpfUnderflowEquivalence or not isinstance(self.library_path, Path):
            raise ExpfEquivalenceError("expected the original typed expf equivalence grant")
        if not self.library_path.is_absolute() or self.library_path.resolve() != self.library_path:
            raise ExpfEquivalenceError("selected archive path must remain canonical and absolute")
        if _sha(self.library_path, limit=64 << 20) != SELECTED_LIBM_SHA256:
            raise ExpfEquivalenceError("selected expf library bytes changed or are unsupported")


def select_expf_underflow_equivalence(library: Path) -> ExpfUnderflowEquivalence:
    """Explicitly select the reviewed archive; there is no library discovery/default."""
    selected = ExpfUnderflowEquivalence(Path(library).resolve())
    selected.revalidate()
    return selected


_CURRENT: contextvars.ContextVar[ExpfUnderflowEquivalence | None] = contextvars.ContextVar(
    "merlin_selected_expf_underflow_equivalence", default=None
)


def _checked(grant: ExpfUnderflowEquivalence) -> ExpfUnderflowEquivalence:
    if type(grant) is not ExpfUnderflowEquivalence:
        raise ExpfEquivalenceError("expected the original typed expf equivalence grant")
    grant.revalidate()
    return grant


@contextlib.contextmanager
def selected_expf_underflow_scope(grant: ExpfUnderflowEquivalence) -> Iterator[None]:
    """Compiler-only explicit scope; the caller must retain the grant through link."""
    token = _CURRENT.set(_checked(grant))
    try:
        yield
        _checked(grant)
    finally:
        _CURRENT.reset(token)


def current_expf_underflow_equivalence() -> ExpfUnderflowEquivalence | None:
    """Reopen the current archive at pipeline construction, or return None."""
    grant = _CURRENT.get()
    return _checked(grant) if grant is not None else None


def bind_expf_underflow_link(
    grant: ExpfUnderflowEquivalence, argv: Sequence[str], *, map_path: Path
) -> tuple[str, ...]:
    """Replace one ordinary -lm with the exact archive and request a fresh map.

    Caller executes this argv immediately. Library search paths may remain for
    unrelated libraries; alternate libm entries, symbol wrapping/definitions,
    response files, shared links and existing map overrides are unsupported.
    """
    _checked(grant)
    args = tuple(str(a) for a in argv)
    if not args or args.count("-lm") != 1:
        raise ExpfEquivalenceError("expected exactly one ordinary -lm in the actual link")
    forbidden = ("--wrap", "--defsym", "--allow-multiple-definition", "-Map", "--print-map")
    for arg in args:
        if arg.startswith("@") or arg in {"-shared", "-pie", "-M"}:
            raise ExpfEquivalenceError("unsupported indirect or dynamic selected-callee link")
        if any(part in arg for part in forbidden):
            raise ExpfEquivalenceError("unsupported symbol or linker-map override")
        if arg != "-lm" and (arg.startswith("-lm") or Path(arg).name in {"libm.a", "libm.so"}):
            raise ExpfEquivalenceError("duplicate or alternate libm selection")
    if [a for a in args if a.startswith("-march=")] != ["-march=rv64gc"] or [
        a for a in args if a.startswith("-mabi=")
    ] != ["-mabi=lp64d"]:
        raise ExpfEquivalenceError("selected expf equivalence requires rv64gc/lp64d")
    output_map = Path(map_path).resolve()
    if "," in str(output_map) or any(c.isspace() for c in str(output_map)):
        raise ExpfEquivalenceError("unsupported linker-map path")
    if output_map.exists():
        raise ExpfEquivalenceError("selected-callee link requires a fresh map path")
    return tuple(str(grant.library_path) if a == "-lm" else a for a in args) + ("-Wl,-Map=" + str(output_map),)


def verify_expf_underflow_link(grant: ExpfUnderflowEquivalence, *, elf: Path, map_path: Path) -> dict[str, object]:
    """Check fresh link evidence; returns diagnostics, never qualification authority.

    GNU map section ownership detects an object/library overriding any function
    in the reviewed path. The caller supplies the map produced by its actual
    returned link invocation, and retains its ordinary invocation/output custody.
    """
    _checked(grant)
    elf = Path(elf)
    with elf.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\xf3\x00":
        raise ExpfEquivalenceError("selected-callee output is not little-endian ELF64 RISC-V")
    map_path = Path(map_path)
    map_sha = _sha(map_path, limit=16 << 20)
    owners: dict[str, list[str]] = {section: [] for section in _SECTIONS}
    pending = None
    for line in map_path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0].startswith(".text"):
            pending = words[0] if words[0] in _SECTIONS else None
            words = words[1:]
        if pending and len(words) >= 3 and words[0].startswith("0x") and words[1].startswith("0x"):
            if int(words[1], 16):
                owners[pending].append(" ".join(words[2:]))
            pending = None
    for section, member in _SECTIONS.items():
        expected = f"{grant.library_path}({member})"
        if owners[section] != [expected]:
            raise ExpfEquivalenceError(f"selected expf path has missing/foreign owner for {section}")
    _checked(grant)
    return {
        "protocol": PROTOCOL,
        "qualification": "UNQUALIFIED",
        "library_path": str(grant.library_path),
        "library_sha256": SELECTED_LIBM_SHA256,
        "elf_sha256": _sha(elf, limit=512 << 20),
        "linker_map_sha256": map_sha,
        "sections": {key: value[0] for key, value in owners.items()},
    }


# ---- cpu_softmax -------------------------------------------------------------------------------
# Explicit source-compatible float32 CPU softmax; never enabled by default.
#
# PyTorch2.10 AVX2 uses eight-lane ordered sums, SLEEF exp_u10 and reciprocal
# multiplication. This scalar IR preserves that selected numerical policy without
# requiring the destination to implement the source vector ISA.
# The exponential schedule adapts SLEEF xexpf, revision5a1d179; its Boost
# Software License and copyright notice are preserved in NOTICE.

CPU_SOFTMAX_F32_AVX2_POLICY = "torch_2_10_cpu_softmax_f32_avx2"


def cpu_softmax_sum_order(backend_policy):
    """The pinned source lane count and ordered binary denominator tree."""
    if backend_policy != CPU_SOFTMAX_F32_AVX2_POLICY:
        raise ValueError("explicit pinned f32 softmax policy required")
    return 8, (((0, 4), (2, 6)), ((1, 5), (3, 7)))


def cpu_softmax_row_numerics(*, backend_policy, assume_finite_inputs=False):
    """Expose the existing source algorithm to shared counted-row scheduling.

    This is source numerical data, not target support or qualification. The
    selected exp implementation must realize build_cpu_softmax_exp_nonpositive.
    """
    from merlin.compiler.schedule.elementwise.row_softmax_checks import RowSoftmaxNumerics

    lanes, tree = cpu_softmax_sum_order(backend_policy)
    return RowSoftmaxNumerics(
        backend_policy,
        "sleef_xexpf_5a1d179_nonpositive_f32",
        "f32",
        lanes,
        tree,
        "reciprocal_multiply",
        assume_finite_inputs,
        True,
    )


def build_cpu_softmax_exp_nonpositive(value, *, backend_policy: str, domain: str):
    """Scalar SLEEF exp_u10 for nonpositive f32, including negative infinity.

    NaN/positive arguments are outside the caller-proven domain. Clamp before
    integer conversion so underflow inputs cannot introduce conversion poison.
    Float status flags are outside this unconstrained floating-point policy.
    """
    from xdsl.dialects import arith, math
    from xdsl.dialects.builtin import FloatAttr, f32, i32

    if backend_policy != CPU_SOFTMAX_F32_AVX2_POLICY:
        raise ValueError("explicit pinned f32 softmax policy required")
    if domain != "nonpositive":
        raise ValueError("nonpositive domain required")
    if value.type != f32:
        raise TypeError("scalar f32 required")
    ops = []

    def emit(op):
        ops.append(op)
        return op.results[0]

    def c(v):
        return emit(arith.ConstantOp(FloatAttr(v, f32)))

    def ci(v):
        return emit(arith.ConstantOp.from_int_and_width(v, 32))

    def add(a, b):
        return emit(arith.AddfOp(a, b))

    def mul(a, b):
        return emit(arith.MulfOp(a, b))

    def fma(a, b, d):
        return emit(math.FmaOp(a, b, d))

    threshold = c(-104.0)
    under = emit(arith.CmpfOp(value, threshold, "olt"))
    safe = emit(arith.SelectOp(under, threshold, value))
    scaled = mul(safe, c(1.44269504088896340736))
    magic = c(12582912.0)
    rounded = emit(arith.SubfOp(add(scaled, magic), magic))
    q = emit(arith.FPToSIOp(rounded, i32))
    reduced = fma(rounded, c(-0.693145751953125), safe)
    reduced = fma(rounded, c(-1.428606765330187045e-6), reduced)
    u = c(0.000198527617612853646278381)
    for coefficient in [
        0.00139304355252534151077271,
        0.00833336077630519866943359,
        0.0416664853692054748535156,
        0.166666671633720397949219,
        0.5,
    ]:
        u = fma(u, reduced, c(coefficient))
    u = add(c(1.0), fma(mul(reduced, reduced), u, reduced))
    half = emit(arith.ShRSIOp(q, ci(1)))
    other = emit(arith.SubiOp(q, half))

    def power(exponent):
        biased = emit(arith.AddiOp(exponent, ci(127)))
        bits = emit(arith.ShLIOp(biased, ci(23)))
        return emit(arith.BitcastOp(bits, f32))

    result = mul(mul(u, power(half)), power(other))
    result = emit(arith.SelectOp(under, c(0.0), result))
    return ops, result


def build_cpu_softmax_lastdim(value, *, backend_policy: str, assume_finite_inputs: bool = False):
    """Return detached ops and source-compatible softmax along the last axis."""
    from xdsl.dialects import arith, scf, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        FloatAttr,
        IndexType,
        TensorType,
        f32,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if backend_policy != CPU_SOFTMAX_F32_AVX2_POLICY:
        raise ValueError("explicit pinned f32 softmax policy required")
    if not assume_finite_inputs:
        raise ValueError("finite-input contract required")
    if not isinstance(value.type, TensorType) or value.type.element_type != f32:
        raise TypeError("f32 tensor required")
    shape = list(value.type.get_shape())
    rank = len(shape)
    if not shape or any(n <= 0 for n in shape) or shape[-1] < 8:
        raise ValueError("static positive shape and last axis at least eight required")
    n = shape[-1]
    prefix = shape[:-1]
    ops = []

    def maps(*values):
        return ArrayAttr([AffineMapAttr(v) for v in values])

    def generic(inputs, output, body, indexing, reduction=False):
        its = (
            [L.IteratorType.PARALLEL] * (len(shape) - 1) + [L.IteratorType.REDUCTION]
            if reduction
            else [L.IteratorType.PARALLEL] * len(output.type.get_shape())
        )
        return L.GenericOp(
            inputs=inputs,
            outputs=(output,),
            body=Region(body),
            indexing_maps=indexing,
            iterator_types=ArrayAttr([L.IteratorTypeAttr(x) for x in its]),
            result_types=(output.type,),
        )

    projection = AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in range(rank - 1)))
    max_type = TensorType(f32, prefix)
    init = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(max_type, [float("-inf")]))
    bb = Block(arg_types=[f32, f32])
    mx = arith.MaximumfOp(*bb.args)
    bb.add_ops([mx, L.YieldOp(mx.result)])
    maximum = generic((value,), init.result, bb, maps(AffineMap.identity(rank), projection), True)
    ops.extend([init, maximum])
    empty = tensor.EmptyOp((), value.type)
    bb = Block(arg_types=[f32, f32, f32])
    shift = arith.SubfOp(bb.args[0], bb.args[1])
    exp_ops, ex = build_cpu_softmax_exp_nonpositive(shift.result, backend_policy=backend_policy, domain="nonpositive")
    bb.add_ops([shift, *exp_ops, L.YieldOp(ex)])
    exps = generic(
        (value, maximum.results[0]),
        empty.tensor,
        bb,
        maps(AffineMap.identity(rank), projection, AffineMap.identity(rank)),
    )
    ops.extend([empty, exps])
    lane_count, sum_tree = cpu_softmax_sum_order(backend_policy)
    lane_type = TensorType(f32, [*prefix, lane_count])
    empty = tensor.EmptyOp((), lane_type)
    bb = Block(arg_types=[f32])
    indices = [L.IndexOp(i) for i in range(rank)]
    eight = arith.ConstantOp.from_int_and_width(lane_count, IndexType())
    end = arith.ConstantOp.from_int_and_width(n, IndexType())
    start = arith.AddiOp(indices[-1].result, eight.result)
    first = tensor.ExtractOp(exps.results[0], [*[x.result for x in indices[:-1]], indices[-1].result], f32)
    loop_body = Block(arg_types=[IndexType(), f32])
    item = tensor.ExtractOp(exps.results[0], [*[x.result for x in indices[:-1]], loop_body.args[0]], f32)
    total = arith.AddfOp(loop_body.args[1], item.result)
    loop_body.add_ops([item, total, scf.YieldOp(total.result)])
    loop = scf.ForOp(start.result, end.result, eight.result, [first.result], Region(loop_body))
    bb.add_ops([*indices, eight, end, start, first, loop, L.YieldOp(loop.results[0])])
    lanes = generic((), empty.tensor, bb, maps(AffineMap.identity(rank)))
    ops.extend([empty, lanes])
    empty = tensor.EmptyOp((), max_type)
    bb = Block(arg_types=[f32])
    indices = [L.IndexOp(i) for i in range(rank - 1)]
    bb.add_ops(indices)
    xs = []
    for j in range(lane_count):
        index = arith.ConstantOp.from_int_and_width(j, IndexType())
        item = tensor.ExtractOp(lanes.results[0], [*[x.result for x in indices], index.result], f32)
        bb.add_ops([index, item])
        xs.append(item.result)

    def plus(a, b):
        op = arith.AddfOp(a, b)
        bb.add_op(op)
        return op.result

    def combine(tree):
        return xs[tree] if type(tree) is int else plus(combine(tree[0]), combine(tree[1]))

    total = combine(sum_tree)
    one = arith.ConstantOp(FloatAttr(1.0, f32))
    reciprocal = arith.DivfOp(one.result, total)
    bb.add_ops([one, reciprocal, L.YieldOp(reciprocal.result)])
    denom = generic((), empty.tensor, bb, maps(AffineMap.identity(rank - 1)))
    ops.extend([empty, denom])
    empty = tensor.EmptyOp((), value.type)
    bb = Block(arg_types=[f32, f32, f32])
    normalized = arith.MulfOp(bb.args[0], bb.args[1])
    bb.add_ops([normalized, L.YieldOp(normalized.result)])
    result = generic(
        (exps.results[0], denom.results[0]),
        empty.tensor,
        bb,
        maps(AffineMap.identity(rank), projection, AffineMap.identity(rank)),
    )
    ops.extend([empty, result])
    return ops, result.results[0]


# ---- normalization_reassociation ---------------------------------------------------------------
# ``layer-norm-chunked-sums``: sum a LayerNorm row in contiguous chunks, then sum the chunks.
#
# NUMERICS-CHANGING and default off. The rewrite changes the order of floating-point additions, so the
# result is not bit-identical to the source reduction; it is a different numerical model and is graded
# against its own reference. Selecting it (``--pass layer-norm-chunked-sums``) is the caller's explicit
# permission for two things the source does not grant:
#
# * reassociation of the row sum's f32 additions;
# * finite intermediates. Overflow, infinities, NaN payloads and exception order are not preserved
#   as an unrestricted IEEE contract. A partial sum that overflows in one order may stay finite in
#   another.
#
# Only a ``linalg.reduce`` whose source provenance says it computes a LayerNorm (``prov.op =
# "layer_norm"``) is eligible: the operation kind, never a model, shape or target name. It must also be
# an f32 ``arith.addf`` over the innermost axis with a static extent divisible by the chunk size, and a
# ``+0.0`` splat initial value. Anything else is left exactly as it was. The rewrite expands the reduced
# axis into ``[groups, chunk]``, sums each chunk into a fresh ``+0.0``-filled partial tensor, and points
# the original reduction at the partial sums. Every loop extent is an operand dimension, so the result
# re-parses and re-applies to nothing.
#
# Half-precision inputs are never accumulated here. A source LayerNorm already computes its sums in
# f32, as its opmath decomposition requires, so an eligible sum is an f32 sum.

#: The attribute recording that a reduction was chunked, with the permissions it was chunked under.
MARKER = "merlin.normalization_reassociation"
#: The provenance op kind that makes a reduction eligible.
SOURCE_OP = "layer_norm"
DEFAULT_CHUNK = 32


def _positive_zero_splat(init) -> bool:
    """The reduction's initial value is a ``tensor.splat`` of the constant ``+0.0``."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr

    owner = init.owner
    if getattr(owner, "name", None) != "tensor.splat" or len(owner.operands) != 1:
        return False
    constant = owner.operands[0].owner
    if not isinstance(constant, arith.ConstantOp) or not isinstance(constant.value, FloatAttr):
        return False
    value = constant.value.value.data
    return value == 0.0 and math.copysign(1.0, value) > 0


def _eligible_shape(op, chunk_size: int) -> list[int] | None:
    """The reduced operand's static shape when ``op`` is an eligible f32 innermost-axis add."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import TensorType, f32
    from xdsl.dialects.linalg import ops as linalg

    if getattr(op.attributes.get("prov.op"), "data", None) != SOURCE_OP or MARKER in op.attributes:
        return None
    if len(op.operands) != 2 or len(op.results) != 1:
        return None
    value, init = op.operands
    input_type, output_type = value.type, init.type
    if not isinstance(input_type, TensorType) or not isinstance(output_type, TensorType):
        return None
    if input_type.element_type != f32 or output_type.element_type != f32:
        return None
    shape = list(input_type.get_shape())
    rank = len(shape)
    if not rank or any(size <= 0 for size in shape):
        return None
    if shape[-1] < 2 * chunk_size or shape[-1] % chunk_size:
        return None
    if list(output_type.get_shape()) != shape[:-1] or op.results[0].type != output_type:
        return None
    if tuple(op.dimensions.iter_values()) != (rank - 1,):
        return None
    body = list(op.regions[0].block.ops)
    if len(body) != 2 or not isinstance(body[0], arith.AddfOp) or not isinstance(body[1], linalg.YieldOp):
        return None
    if list(body[0].operands) != list(op.regions[0].block.args) or list(body[1].operands) != [body[0].result]:
        return None
    # The original init takes part only in the final reduction; requiring +0 keeps a nonzero initial
    # accumulator from being added once per chunk.
    if not _positive_zero_splat(init):
        return None
    return shape


def chunk_layer_norm_sums(
    module,
    *,
    allow_reassociation: bool = False,
    assume_finite_intermediates: bool = False,
    chunk_size: int = DEFAULT_CHUNK,
    report_out: dict | None = None,
    on_rewrite: Callable[[Any, tuple[Any, ...], Any], None] | None = None,
) -> int:
    """Chunk every eligible LayerNorm row sum; returns how many were rewritten.

    Both numerical permissions must be given, or nothing changes. ``on_rewrite(source, generated,
    result_owner)`` reports each rewrite's new operations to a caller that accounts for them (the
    source transform map); the reduction keeps its results, so it is its own result owner.
    """
    if type(chunk_size) is not int or chunk_size < 2:
        raise ValueError("chunk_size must be an integer of at least two")
    report = report_out if report_out is not None else {}
    report.update(rewritten=0, opted_in=bool(allow_reassociation and assume_finite_intermediates))
    if not report["opted_in"]:
        return 0

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr, StringAttr, TensorType, f32, i64
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import InsertPoint, Rewriter

    from merlin.compiler.ir.provenance import carry_provenance

    contract = StringAttr(f"chunk={chunk_size}; explicit reassociation; finite intermediates assumed")
    for op in list(module.walk()):
        if not isinstance(op, linalg.ReduceOp):
            continue
        shape = _eligible_shape(op, chunk_size)
        if shape is None:
            continue
        value, init = op.operands
        rank = len(shape)
        groups = shape[-1] // chunk_size
        partial_type = TensorType(f32, [*shape[:-1], groups])
        expanded_shape = [*shape[:-1], groups, chunk_size]
        reassociation = ArrayAttr(
            [ArrayAttr([IntegerAttr(i, i64)]) for i in range(rank - 1)]
            + [ArrayAttr([IntegerAttr(rank - 1, i64), IntegerAttr(rank, i64)])]
        )
        expanded = tensor.ExpandShapeOp(value, (), reassociation, expanded_shape, TensorType(f32, expanded_shape))
        zero = arith.ConstantOp(FloatAttr(0.0, f32))
        empty = tensor.EmptyOp((), partial_type)
        fill = linalg.FillOp((zero.result,), (empty.tensor,), res=(partial_type,))
        body = Block(arg_types=[f32, f32])
        add = arith.AddfOp(body.args[0], body.args[1])
        body.add_ops([add, linalg.YieldOp(add.result)])
        partial = linalg.GenericOp(
            inputs=(expanded.results[0],),
            outputs=(fill.results[0],),
            body=Region(body),
            indexing_maps=ArrayAttr(
                [
                    AffineMapAttr(AffineMap.identity(rank + 1)),
                    AffineMapAttr(AffineMap(rank + 1, 0, tuple(AffineDimExpr(i) for i in range(rank)))),
                ]
            ),
            iterator_types=ArrayAttr(
                [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * rank
                + [linalg.IteratorTypeAttr(linalg.IteratorType.REDUCTION)]
            ),
            result_types=(partial_type,),
        )
        generated = (expanded, zero, empty, fill, partial)
        for new in generated:
            carry_provenance(new, op, "layer_norm_chunked_sum")
        partial.attributes[MARKER] = contract
        op.attributes[MARKER] = contract
        Rewriter.insert_op(list(generated), InsertPoint.before(op))
        op.operands = (partial.results[0], init)
        if on_rewrite is not None:
            on_rewrite(op, generated, op)
        report["rewritten"] += 1
    return report["rewritten"]


# ---- act_poly ----------------------------------------------------------------------------------
# Vectorizable polynomial approximation of transcendental activations (default-off feature).
#
# The baseline RVV pipeline lowers ``math.erf``/``math.exp``/``math.tanh`` through
# ``convert-math-to-libm`` -> a SCALAR libm call (``erff``/``expf``/``tanhf``) in a loop. The
# elementwise activation ``linalg.generic`` is never vectorized (the baseline transform schedule
# only matches matmul/batch_matmul), so a GELU/sigmoid/SiLU activation runs as a scalar libm loop —
# the ~11-18x gap vs XNNPACK's vectorized-polynomial RVV kernels measured in
# ``output/kernels/ceiling/cross_framework_ops_k1.md``.
#
# This module is the GENERAL compiler capability that closes that gap: a transcendental -> inline
# arith-polynomial rewrite over the math ops, run BEFORE vectorization so the resulting arith chain
# (mul/add/sub/div/select + bitcast/shift for the 2^n exponent) vectorizes to ``vfmacc`` chains
# instead of a scalar call. It is NOT a hand-written kernel and NOT op/shape-specific: it rewrites
# every ``math.exp``/``math.erf``/``math.tanh`` in the module, so GELU (erf), sigmoid/SiLU (exp), and
# tanh all benefit from one transform.
#
# Accuracy: these are APPROXIMATIONS (f32 minimax), so the gate is cosine / relative-error vs the
# libm reference, not bit-exact (see the feature's tests + the report). The polynomials:
#   * exp(x): range-reduce x = n*ln2 + r, exp = 2^n * P5(r) (degree-6 Horner of the Taylor series,
#     f32-refined), 2^n assembled via (int(n)+127)<<23 bitcast. max rel err ~6.6e-7 over |x|<~10.
#   * erf(x): Abramowitz-Stegun 7.1.26 rational-in-t poly with t = 1/(1+p|x|) and an exp(-x^2) tail,
#     sign restored via copysign. max abs err ~5.8e-7.
#   * tanh(x) = 2/(1+exp(-2x)) - 1 (reuses the exp poly).
# The exp coefficients/structure mirror XNNPACK's rr2-p5 range-reduction (the CEILING REFERENCE);
# the erf uses the standard A&S 7.1.26 rational form. We emit the MLIR; XNNPACK's kernel is only the
# coefficient/structure reference, never linked.
#
# This file runs inside the model2MLIR venv (it needs the upstream MLIR Python bindings). It is
# imported by the lowering runner ONLY when ``approximate_transcendental_activation`` or the
# vectorized activation feature is enabled. The approximation is independent of scheduling;
# with both features off it is never imported and the pipeline is byte-identical.


def rewrite_source(*, fused: bool = False) -> str:
    """Return the self-contained Python source of the rewriter, to be prepended to the lowering
    runner (which executes in the m2m venv). Kept as a source string so the runner stays a single
    self-contained script and no extra import path wiring into the m2m venv is needed."""
    body = (
        "return _math.FmaOp(a, b, c).result" if fused else "return _arith.AddFOp(_arith.MulFOp(a, b).result, c).result"
    )
    return _REWRITER_SRC.replace("__AP_FMA_BODY__", body)


# The rewriter body. Self-contained (only torch_mlir.{ir,dialects.arith,dialects.math}); defines
# `apply_activation_polynomial(module, ctx)` which the runner calls after Module.parse and before
# the PassManager runs. Emitted as a string so it can be prepended to the in-venv runner verbatim.
_REWRITER_SRC = r'''
# --- vectorizable transcendental-activation polynomial rewriter (default-off feature) ---
from torch_mlir.dialects import arith as _arith
from torch_mlir.dialects import math as _math

def _ap_is_vec(t):
    try:
        _ir.VectorType(t); return True
    except (ValueError, TypeError):
        return False

def _ap_i32():
    return _ir.IntegerType.get_signless(32)

def _ap_supported_float(t):
    if _ap_is_vec(t):
        t = _ir.VectorType(t).element_type
    return t == _ir.F32Type.get()

def _ap_int_ty(t):
    if _ap_is_vec(t):
        vt = _ir.VectorType(t)
        return _ir.VectorType.get(vt.shape, _ap_i32())
    return _ap_i32()

def _ap_fconst(t, v):
    if _ap_is_vec(t):
        et = _ir.VectorType(t).element_type
        attr = _ir.DenseElementsAttr.get_splat(t, _ir.FloatAttr.get(et, float(v)))
    else:
        attr = _ir.FloatAttr.get(t, float(v))
    return _arith.ConstantOp(t, attr).result

def _ap_iconst(ity, v):
    if _ap_is_vec(ity):
        attr = _ir.DenseElementsAttr.get_splat(ity, _ir.IntegerAttr.get(_ap_i32(), int(v)))
    else:
        attr = _ir.IntegerAttr.get(_ap_i32(), int(v))
    return _arith.ConstantOp(ity, attr).result

def _ap_fma(a, b, c):
    # By default a*b + c is arith mul+add. The whole activation body is then PURE ARITH — it
    # contains NO `math.*` op at all. That is deliberate: the un-rewritten softmax `math.exp` must
    # reach the baseline `convert-math-to-libm` -> scalar `expf` call (the exact, crash-free path).
    # The old design emitted `math.fma`, which forced a `convert-math-to-llvm` BEFORE
    # `convert-math-to-libm` (so the vector fma became `llvm.intr.fma` instead of being scalarized) —
    # but that same pass also converted the softmax `math.exp` to `llvm.intr.exp` (`llvm.exp.f32`),
    # which the freestanding spike/RVV runtime cannot legalize -> a wild instruction -> the openvla
    # "bad syscall" CRASH (status=fail, no OUT/DONE). With pure arith the activation chain vectorizes
    # to vfmul.vv + vfadd.vv (the RISC-V backend contracts adjacent mul+add into vfmacc under the
    # pipeline's fast-math where it can), and crucially the softmax exp lowers to `expf` exactly as in
    # the baseline — so no extra pipeline pass, no `llvm.intr.exp`, no crash.
    # Explicit fused evaluation instead emits math.fma and selects the narrow FMA-only
    # intrinsic lowering. It does not convert normalization exp to llvm.intr.exp.
    __AP_FMA_BODY__

def _ap_absf(x, t):
    # |x| in pure arith (no math.absf): max(x, -x). See _ap_fma for why the activation body avoids any
    # math.* op (so the softmax exp can take the exact libm path without a math-to-llvm pass).
    return _arith.MaximumFOp(x, _arith.NegFOp(x).result).result

def _ap_roundeven(x, t):
    # round-to-nearest-even in pure arith via the classic add/sub-magic trick: for f32, adding then
    # subtracting MAGIC = 1.5 * 2^23 (= 2^23 + 2^22) forces the FP rounding hardware to round to the
    # nearest integer (ties-to-even, the default rounding mode), valid for |x| < 2^23. Used only for
    # the exp range-reduction n = round(x*log2e); x is clamped to [-87,88] so x*log2e in [-126,127],
    # far inside 2^23 — exact. Avoids math.roundeven (see _ap_absf for why math ops are avoided).
    MAGIC = _ap_fconst(t, 12582912.0)   # 1.5 * 2**23
    biased = _arith.AddFOp(x, MAGIC).result
    return _arith.SubFOp(biased, MAGIC).result

def _ap_horner(coeffs, r, t):
    # Estrin/Horner P(r) = (((c0*r + c1)*r + c2)... evaluated with fused multiply-adds.
    p = _ap_fconst(t, coeffs[0])
    for c in coeffs[1:]:
        p = _ap_fma(p, r, _ap_fconst(t, c))
    return p

def _ap_exp(x, t):
    # exp(x): x = n*ln2 + r ; exp = 2^n * polyexp(r). 2^n via (int(n)+127)<<23 bitcast.
    LOG2E = _ap_fconst(t, 1.4426950408889634)
    NEG_LN2HI = _ap_fconst(t, -0.6931471824645996)
    NEG_LN2LO = _ap_fconst(t, 1.904654323148236e-09)
    LO = _ap_fconst(t, -87.0); HI = _ap_fconst(t, 88.0)
    x = _arith.MaximumFOp(x, LO).result
    x = _arith.MinimumFOp(x, HI).result
    nf = _ap_roundeven(_arith.MulFOp(x, LOG2E).result, t)
    r = _ap_fma(nf, NEG_LN2HI, x)            # x - n*ln2_hi  (fused)
    r = _ap_fma(nf, NEG_LN2LO, r)            # r - n*ln2_lo  (fused)
    p = _ap_horner([0.0013888889, 0.008333334, 0.041666668, 0.16666667, 0.5, 1.0, 1.0], r, t)
    ity = _ap_int_ty(t)
    ni = _arith.FPToSIOp(ity, nf).result
    ni = _arith.AddIOp(ni, _ap_iconst(ity, 127)).result
    ni = _arith.ShLIOp(ni, _ap_iconst(ity, 23)).result
    scale = _arith.BitcastOp(t, ni).result
    return _arith.MulFOp(p, scale).result

def _ap_erf(x, t):
    # erf(x): Abramowitz-Stegun 7.1.26, sign via copysign, exp(-x^2) tail.
    a = [_ap_fconst(t, c) for c in (1.061405429, -1.453152027, 1.421413741,
                                    -0.284496736, 0.254829592)]
    pc = _ap_fconst(t, 0.3275911); one = _ap_fconst(t, 1.0); zero = _ap_fconst(t, 0.0)
    neg_one = _ap_fconst(t, -1.0)
    ax = _ap_absf(x, t)
    # sign(x) as +-1 via a select (math.copysign has no vector LLVM legalization in this build).
    # OGE (predicate 3) so x>=0 -> +1, else -1 (erf(0)=0 either way, so the sign of 0 is irrelevant).
    pred = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(64), int(_arith.CmpFPredicate.OGE))
    pos = _arith.CmpFOp(pred, x, zero).result
    s = _arith.SelectOp(pos, one, neg_one).result
    tt = _arith.DivFOp(one, _ap_fma(pc, ax, one)).result          # 1/(1 + p*|x|)
    poly = a[0]
    for c in a[1:]:
        poly = _ap_fma(poly, tt, c)                               # fused Horner
    poly = _arith.MulFOp(poly, tt).result
    negx2 = _arith.NegFOp(_arith.MulFOp(ax, ax).result).result
    e = _ap_exp(negx2, t)
    y = _arith.SubFOp(one, _arith.MulFOp(poly, e).result).result  # 1 - poly*exp(-x^2)
    return _arith.MulFOp(s, y).result

def _ap_tanh(x, t):
    # tanh(x) = 2*sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1.
    two = _ap_fconst(t, 2.0); one = _ap_fconst(t, 1.0)
    e = _ap_exp(_arith.NegFOp(_arith.MulFOp(two, x).result).result, t)
    return _arith.SubFOp(_arith.DivFOp(two, _arith.AddFOp(one, e).result).result, one).result

_AP_BUILDERS = {"math.exp": _ap_exp, "math.erf": _ap_erf, "math.tanh": _ap_tanh}

# Provenance-identified ACTIVATION ops whose transcendental we may replace with the f32 minimax
# polynomial. These are the elementwise activations whose input range is bounded enough that the
# minimax poly is valid (gelu/silu/sigmoid/tanh inputs are pre-activation logits, O(1)-O(10), and the
# exp poly range-reduces+clamps to |x|<~87). DELIBERATELY EXCLUDES `softmax` (prov.family =
# "normalization"): a softmax computes exp(x - rowmax) and then DIVIDES by the row sum; the minimax
# exp differs from libm by ~1e-7 per element and that error is AMPLIFIED through the normalization
# (and the argument x-rowmax can be a large-magnitude logit gap), so blanket-rewriting the softmax exp
# drove openvla whole-model cos to 0.541 (the regression this fix closes). Softmax exp stays on libm.
_AP_ACTIVATION_OPS = frozenset({"gelu", "silu", "sigmoid", "tanh", "hardsigmoid", "hardswish"})


def _ap_prov(generic_op):
    """(prov.op, prov.family) of a linalg.generic from its `attrs` dict, or (None, None)."""
    op = generic_op.operation
    pop = pfam = None
    try:
        pop = _ir.StringAttr(op.attributes["prov.op"]).value
    except (KeyError, ValueError, TypeError):
        pass
    try:
        pfam = _ir.StringAttr(op.attributes["prov.family"]).value
    except (KeyError, ValueError, TypeError):
        pass
    return pop, pfam


def _ap_all_parallel(generic_op):
    """True iff the generic's iterator_types are ALL "parallel" (a pure elementwise map — NO
    reduction). A softmax/normalization carries a reduction iterator on its max/sum generics; a
    genuine activation is pure-parallel. Used only for the no-provenance structural fallback."""
    op = generic_op.operation
    try:
        its = op.attributes["iterator_types"]
    except KeyError:
        return False
    s = str(its)
    return "reduction" not in s and "parallel" in s


def _ap_is_activation_generic(generic_op):
    """True iff `generic_op` is a linalg.generic to which the minimax activation polynomial may be
    safely applied — an ELEMENTWISE ACTIVATION (gelu/silu/sigmoid/tanh), NOT a softmax/normalization
    (whose exp must stay on the exact libm/llvm.exp path; see _AP_ACTIVATION_OPS).

    Provenance is the PRIMARY signal (m2m tags every op): prov.op in the activation set, or
    prov.family == "elementwise" with a non-softmax op. When the generic carries NO provenance at all
    (hand-written / synthetic modules, e.g. the isolated workloads), fall back to the STRUCTURAL
    activation signature: all-parallel iterators (pure elementwise, no reduction). A normalization
    that DOES carry provenance (prov.family == "normalization"/"softmax", or prov.op == "softmax") is
    rejected even if structurally parallel — the provenance wins."""
    if generic_op.operation.name != "linalg.generic":
        return False
    pop, pfam = _ap_prov(generic_op)
    # provenance explicitly marks a normalization/softmax -> NEVER approximate its exp.
    if pop == "softmax" or pfam in ("normalization", "softmax"):
        return False
    if pop in _AP_ACTIVATION_OPS:
        return True
    if pfam == "elementwise" and pop is not None:
        return True
    # no provenance at all -> structural fallback: a pure-parallel (no-reduction) elementwise generic.
    if pop is None and pfam is None:
        return _ap_all_parallel(generic_op)
    return False


def apply_activation_polynomial(module, ctx):
    """Replace math.exp/erf/tanh with the inline arith minimax polynomial ONLY inside the
    linalg.generic the PROVENANCE identifies as an elementwise activation (gelu/silu/sigmoid/tanh).

    PRECISE TARGETING (the whole-model-correctness fix): the previous version rewrote EVERY
    math.exp/erf/tanh in the module — including the exp inside a SOFTMAX (prov.family =
    "normalization"). A minimax exp differs from libm by ~1e-7 and softmax AMPLIFIES that error
    through its row-sum normalization, so the blanket rewrite drove openvla whole-model cos to 0.541
    while the synthetic isolated tests (benign small ranges, no softmax) stayed cos=1.0. The
    abstraction already KNOWS which generic is an activation (prov.op / prov.family from m2m); we use
    that identity instead of op-class alone. A transcendental whose enclosing generic is a softmax /
    normalization / unknown context is LEFT as math.exp -> the baseline libm call (correct), so only
    the genuinely in-range activation transcendentals are approximated. Returns the count rewritten.
    """
    targets = []

    def _walk(op, in_activation):
        for region in op.regions:
            for block in region.blocks:
                for o in block.operations:
                    here = in_activation
                    if o.operation.name == "linalg.generic":
                        here = _ap_is_activation_generic(o)
                    if (here and o.operation.name in _AP_BUILDERS
                            and len(o.operands) == 1
                            and _ap_supported_float(o.operands[0].type)):
                        targets.append(o)
                    _walk(o.operation, here)

    _walk(module.operation, False)
    n = 0
    touched_generics = []
    for o in targets:
        op = o.operation
        # tag the enclosing activation generic so the SCHEDULE can vectorize PRECISELY these (and
        # only these) by `transform.structured.match attributes{merlin.act_vectorize}` — no blanket
        # foreach over every generic, no failures(suppress) masking a miscompile.
        par = op.parent
        while par is not None and par.name != "linalg.generic":
            par = par.parent
        if par is not None and par not in touched_generics:
            par.attributes["merlin.act_vectorize"] = _ir.UnitAttr.get()
            touched_generics.append(par)
        x = op.operands[0]; t = x.type
        with _ir.InsertionPoint(op), _ir.Location.unknown():
            new = _AP_BUILDERS[op.name](x, t)
        op.result.replace_all_uses_with(new)
        op.erase()
        n += 1
    return n
# --- end activation polynomial rewriter ---
'''
