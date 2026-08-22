"""Run a matmul interface on a self-hosted-ISA (``external_backend``) target's program oracle with the REAL
operands INJECTED — a TARGET-AGNOSTIC tool.

Nothing here is target-specific: the kernel is emitted by the target's GENERATED OOT package (the same
``run_entrypoints`` the grader uses), the operand layout + golden come from the target's ``model_ext``, and
the RTL cosim is DERIVED from the target by mlc's registry (``program_oracle``). This module only threads the
interface, injects the operands onto the command buffer's leaf tensors (by the harness-stamped DRAM layout),
and reads the output back. It carries no target-name literal and fails closed (returns ``None``) on any gap.
"""
from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _encode_operand(values, dtype: str) -> bytes | None:
    """Encode a nested-list operand to raw device bytes for the cb's DECLARED dtype.

    DERIVED FROM THE FORMAT REGISTRY, not from the spelling: the registry knows each format's kind,
    element width, exponent and mantissa bits, so one dispatch covers every integer width, every IEEE
    float, and every OCP fp code -- including formats nobody has written a branch for.

    It used to be four hardcoded name comparisons with a fallback that imported
    ``merlin.targetgen.rtl.fp8_codec.encode_bytes``. That module path does not exist and neither does that
    function, so the fallback raised ModuleNotFoundError into a bare ``except`` and returned None for
    EVERY float format. Fail-closed reporting then turned that into "no reachable oracle", so the whole
    float-datapath mesh path looked unavailable rather than unimplemented, on every target that uses it.

    Returns ``None`` (fail closed) for a format with no derivable byte encoding -- a sub-byte format,
    whose packing is a layout decision this function must not guess at.
    """
    import numpy as np

    from merlin.common import quant_formats as qf

    a = np.array(values, dtype=np.float64)
    if not qf.has(dtype):
        return None
    f = qf.get(dtype)
    bits = int(f.element_bits or 0)
    if bits % 8 or bits == 0:
        return None                                  # sub-byte: packing is not this function's call

    if f.kind == "int_affine":
        lo, hi = ((-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if f.signed else (0, 2 ** bits - 1))
        code = "<i" if f.signed else "<u"
        return np.clip(np.rint(a), lo, hi).astype(f"{code}{bits // 8}").tobytes()

    if f.kind == "float_ieee":
        if bits == 32:
            return a.astype("<f4").tobytes()
        if bits == 64:
            return a.astype("<f8").tobytes()
        if bits == 16 and int(f.exp_bits or 0) == 5:
            return a.astype("<f2").tobytes()         # IEEE half
        if bits == 16 and int(f.exp_bits or 0) == 8:
            # bfloat16 is the top half of the fp32 word, round-to-nearest-even on the dropped half.
            u = a.astype("<f4").view("<u4").astype(np.uint32)
            rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
            return rounded.astype("<u2").tobytes()
        return None

    if f.kind == "fp_ocp":
        from merlin.targetgen.fp8_codec import ocp_encode
        eb, mb = int(f.exp_bits or 0), int(f.mant_bits or 0)
        if not eb or bits != 8:
            return None
        flat = [ocp_encode(float(v), eb, mb) for v in a.reshape(-1)]
        return bytes(bytearray(flat))

    return None


def matmul_on_program_oracle(target: str, interface_mlir: str, A, W, *, model_ext: str,
                             package: str | None, timeout: int = 900,
                             dtype_hint: str | None = None,
                             observed: dict | None = None) -> list | None:
    """Emit the target's matmul kernel from ``interface_mlir`` via its generated package, inject ``A``/``W``
    onto the command buffer, run the mlc-derived program oracle, and return the output tensor (nested list)
    or ``None`` (fail closed). Target-agnostic — the target only enters as the parameter that selects its own
    generated package + mlc cosim."""
    if package is None:
        return None
    from . import capsule_common as CC
    from . import program_oracle as PO
    from ..benchharness import runs_root
    from .capsule_common import make_run_paths

    with tempfile.TemporaryDirectory(prefix="mesh_prog_") as td:
        tdp = Path(td)
        # a minimal capsule dir the shared entrypoint runner accepts (interface only; op = matmul).
        cdir = tdp / "cap"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "capsule.interface.mlir").write_text(interface_mlir, encoding="utf-8")
        capsule = {"name": "mesh_layer", "kind": "op", "interface_mlir": "capsule.interface.mlir",
                   "operation": {"op": "matmul", "attributes": {}}, "__dir__": str(cdir),
                   "required_oracle_tiers": ["L3"]}
        paths = make_run_paths(runs_root(target, "mesh_prog"), "mesh_layer", suite="mesh",
                               target=target, dtype="prog", benchmark="mesh_layer")
        # The emitted-artifact name is the TARGET's, not a constant: a self-hosted-ISA target emits an
        # assembly kernel where a RoCC target emits LLVM-dialect MLIR. Read it from the same runner config
        # the grader uses rather than naming one.
        from .capsule_runner import _config_for_target
        try:
            _fourth = _config_for_target(target, None, dtype_hint or "int8").fourth_output_name
        except Exception:                            # noqa: BLE001 — no runner config: cannot emit here
            return None
        try:
            _pkg, cb, kernel_text = CC.run_entrypoints(None, package, capsule, paths, contract=None,
                                                       timeout=timeout, fourth_output_name=_fourth)
        except (TypeError, AttributeError, ImportError, NameError):
            # OUR bug, not the target's. These were swallowed into the same `None` as a genuine
            # unavailability, and the caller reports None as "no reachable oracle in this env" -- so a
            # wrong call signature here presented as a missing oracle and this path never ran at all
            # on any target that uses it. Re-raised so a defect in the runner is attributed to the runner.
            raise
        except Exception:                            # noqa: BLE001 — package can't emit this kernel: honest None
            return None
        if cb is None or not kernel_text:
            return None

        # inject the REAL operands onto the cb's leaf tensors (the harness-stamped DRAM layout carries the
        # base; we only supply the bytes, encoded for each tensor's declared dtype).
        operands = {"A0": A, "W": W}
        for tname, tspec in (cb.get("tensors") or {}).items():
            if tspec.get("role") in ("input", "weight", "bias") and tname in operands:
                raw = _encode_operand(operands[tname], tspec.get("dtype", "i8"))
                if raw is None:
                    return None
                tspec["preload_b64"] = base64.b64encode(raw).decode()

        run = PO.program_oracle_adapter(target, model_ext=model_ext)
        odir = tdp / "oracle"
        odir.mkdir(parents=True, exist_ok=True)      # the adapter writes the emitted kernel into it
        try:
            res = run(cb, kernel_text, odir, timeout)
        except PO.ProgramDidNotHalt:
            # A VERDICT about the program, not an absent oracle. ``ProgramDidNotHalt`` subclasses
            # ``OracleUnavailable``, so catching the parent first reported "no reachable oracle" for a
            # program that ran and hung -- and the caller turned that into a skip. Re-raise so it is
            # attributed to the artifact; the cycle budget itself now scales with the declared work
            # (``derive_cycle_budget``), so reaching this means a genuine hang.
            raise
        except PO.OracleUnavailable:
            return None
        if observed is not None and res.get("oracle"):
            observed["oracle"] = res["oracle"]
        outs = res.get("outputs") or {}
        return next(iter(outs.values()), None)


# ---------------------------------------------------------------------------
# Whole-model image splice: a single co-scheduled program that walks a routing
# plan and dispatches each op to its lane (mesh for the systolic/spatial/simt
# contractions, scalar for norms/activations/elementwise), wiring every op's
# output tensor to the next op's input. Target-agnostic: the ``lane`` of a step
# is READ from the routing plan (whether the router placed the op on a mesh unit),
# never assumed; the extents come from the op's ``OpDemand``; the tensor identities
# come from the linalg module's own def-use edges. This proves the splice
# ORCHESTRATION (ordering + activation handoff + mesh/scalar co-scheduling) on the
# engine; the single-binary arena runtime is a later slice.
# ---------------------------------------------------------------------------

# linalg op class -> the (lane-independent) op family the splice executes. Contraction ops are the
# mesh candidates; the rest are the scalar/vector lane. Derived structurally from the op class (no
# regex, no target literal).
_ELEMENTWISE_FAMILY = {"linalg.add": "add", "linalg.mul": "mul"}

# The op families the co-scheduled splice can execute end to end. A transformer's non-matmul ops are
# carried in the linalg module as ``linalg.generic`` ops tagged with a ``library_call`` naming the family
# (the live-module analog of the ``prov.op`` tags m2m stamps on captured models). The splice reads that
# tag STRUCTURALLY (no regex, no target literal) and dispatches each family to a small numpy executor on
# the scalar/vector lane. Two classes:
#   * SCALAR — norms / activations / positional embeddings that run inline on the vector lane;
#   * FUSED  — ops that DECOMPOSE into mesh matmuls + a scalar glue step (attention = Q@Kᵀ on the mesh,
#     softmax on the scalar lane, @V on the mesh; geglu = two mesh matmuls + a gelu gate). Their matmul
#     sub-ops still route to the MESH oracle; the splice threads the activation between the lanes.
_SCALAR_LIB_FAMILIES = frozenset({"softmax", "rmsnorm", "layernorm", "rope", "silu", "gelu"})
_FUSED_MESH_FAMILIES = frozenset({"attention", "attention_full", "attention_qk", "geglu"})

# The op families the toy xDSL engine can lower + evaluate (its command-buffer vocabulary). A module built
# only from these can be graded against the engine reference; a module carrying any transcendental/fused
# family (softmax/rmsnorm/…, whose exp/rsqrt/div the engine cannot evaluate) is graded against a
# host-eager numpy recomputation instead — see :func:`_engine_can_lower` / :func:`_host_eager_final`.
_ENGINE_FAMILIES = frozenset({"matmul", "relu", "add", "mul"})

# rmsnorm / layernorm numerical epsilon (a fixed, shared constant so the spliced lane and the host-eager
# reference use the SAME arithmetic — the gate proves the splice, not a mismatched epsilon).
_NORM_EPS = 1e-6


def _library_call(op) -> str | None:
    """The ``library_call`` tag on a ``linalg.generic`` (the op-family name), or ``None`` if untagged.
    Structural read of the op's own property — no text matching."""
    props = getattr(op, "properties", None)
    if not props:
        return None
    lc = props.get("library_call")
    return getattr(lc, "data", None)


def _is_block_arg(value) -> bool:
    from xdsl.ir import Block
    return isinstance(getattr(value, "owner", None), Block)


def _is_zero_fill(value) -> bool:
    """``value`` is a ``linalg.fill`` of the arith-constant scalar 0 (the relu ``max(x, 0)`` idiom)."""
    owner = getattr(value, "owner", None)
    if getattr(owner, "name", None) != "linalg.fill":
        return False
    scalar = owner.inputs[0]
    c = getattr(scalar, "owner", None)
    if getattr(c, "name", None) != "arith.constant":
        return False
    v = getattr(getattr(c, "value", None), "value", None)
    return getattr(v, "data", None) == 0


def _op_family(op):
    """The splice op family for a linalg op, or ``None`` if it is not a routable compute op (an
    ``EmptyOp``/``FillOp``/``ConstantOp`` init, or the terminator). Structural — matches on the op class,
    then on a ``linalg.generic``'s ``library_call`` tag (softmax/rmsnorm/rope/attention/…)."""
    from xdsl.dialects.linalg import ops as lo
    if isinstance(op, (lo.QuantizedMatmulOp, lo.MatmulOp)):
        return "matmul"
    if isinstance(op, lo.MaxOp) and _is_zero_fill(op.inputs[1]):
        return "relu"
    if isinstance(op, lo.GenericOp):
        # a tagged generic is a routable op even if its family is unknown here — return the tag so it is
        # ROUTED (not silently dropped); its executor fails closed on an unsupported family.
        return _library_call(op)
    return _ELEMENTWISE_FAMILY.get(getattr(op, "name", None))


def _shape(value) -> list[int]:
    return [int(d) for d in value.type.get_shape()]


def _ordered_compute_ops(module) -> list:
    """The module's routable compute ops in program order, each as
    ``{"op", "family", "inputs" (SSA values), "result" (SSA value)}``. Inits/consts/terminator are
    dropped — exactly the ops that :func:`model_op_demands`/the router see, in the same order."""
    from xdsl.dialects.linalg import ops as lo
    out = []
    for op in module.walk():
        fam = _op_family(op)
        if fam is None:
            continue
        if fam == "relu":
            inputs = [op.inputs[0]]                 # the zero-fill operand is not an activation
        elif isinstance(op, (lo.QuantizedMatmulOp, lo.MatmulOp)):
            inputs = [op.inputs[0], op.inputs[1]]   # (lhs activation, rhs weight)
        else:
            inputs = list(op.inputs)                # elementwise: two real tensor operands
        out.append({"op": op, "family": fam, "inputs": inputs, "result": op.results[0]})
    return out


def demands_from_module(module, in_fmt: str, weight_fmt: str | None = None) -> list:
    """Per-op routing :class:`~merlin.targetgen.routing.OpDemand`\\ s read STRUCTURALLY from a live linalg
    module (no prov annotations required, unlike ``model_op_demands`` which parses captured-model MLIR text).
    A contraction carries its real (M, K, N) extents and a weight format; every other op is unary. The order
    is the module's program order — the same order :func:`build_whole_model_program` walks the plan in."""
    from merlin.targetgen.routing import OpDemand
    wf = weight_fmt or in_fmt
    demands = []
    for info in _ordered_compute_ops(module):
        if info["family"] == "matmul":
            m, k = _shape(info["inputs"][0])
            n = _shape(info["inputs"][1])[1]
            demands.append(OpDemand(op="matmul", in_fmt=in_fmt, weight_fmt=wf, site="matmul",
                                    m=m, k=k, n=n))
        else:
            demands.append(OpDemand(op=info["family"], in_fmt=in_fmt, weight_fmt=None,
                                    site=info["family"]))
    return demands


@dataclass(frozen=True)
class WholeModelStep:
    """One op of the spliced program: which lane runs it, its extents, and the tensor-id handoff."""

    index: int
    op: str
    family: str
    lane: str                       # "mesh" (systolic/spatial/simt) or "scalar" (vector/RVV lane)
    unit: str | None                # the compute unit the router chose (None -> scalar/RVV fallback)
    inputs: tuple[str, ...]         # tensor ids this step consumes (leaf ids or earlier steps' outputs)
    output: str                     # tensor id this step produces
    m: int | None = None
    k: int | None = None
    n: int | None = None


@dataclass
class WholeModelProgram:
    """A structured, executable whole-model program: an ordered list of lane-tagged steps whose tensor
    ids wire each op's output to the next op's input, plus the model's leaf tensors and final output."""

    target: str
    steps: list = field(default_factory=list)
    leaves: dict = field(default_factory=dict)     # leaf tensor id -> {"role", "shape", "arg_index"}
    output: str = ""

    def n_mesh(self) -> int:
        return sum(1 for s in self.steps if s.lane == "mesh")

    def n_scalar(self) -> int:
        return sum(1 for s in self.steps if s.lane == "scalar")


def build_whole_model_program(plan: dict, target: str, module) -> WholeModelProgram:
    """Walk the ORDERED ``plan["results"]`` (from :func:`routing.route_plan`) alongside ``module``'s own
    compute ops and emit a co-scheduled whole-model program. Each op becomes a :class:`WholeModelStep`
    tagged ``lane="mesh"`` when the router placed it on a mesh unit (``plan["mesh"]``) or ``lane="scalar"``
    otherwise, carrying the op's real extents and the tensor ids that hand its inputs/output between steps.

    TARGET-AGNOSTIC: the lane is READ from the plan (never assumed from an op name), extents come from the
    op's ``OpDemand``, and the tensor identities come from the module's def-use edges. ``plan["results"]``
    must be in the module's program order (it is, when built from :func:`demands_from_module`)."""
    results = plan["results"]
    compute = _ordered_compute_ops(module)
    if len(results) != len(compute):
        raise ValueError(
            f"routing plan has {len(results)} ops but the module has {len(compute)} compute ops — the plan "
            f"must be routed from this module's demands (demands_from_module) so the two walks align")
    mesh_ids = {id(r) for r in plan.get("mesh", [])}

    # Leaf tensor ids: each block argument is a model leaf (an activation input or a resident weight).
    prog = WholeModelProgram(target=target)
    leaf_id: dict = {}                                       # SSA block-arg value -> leaf id
    from xdsl.ir import Block
    for info in compute:                                     # discover roles from how args are used
        for pos, v in enumerate(info["inputs"]):
            if _is_block_arg(v) and v not in leaf_id:
                # a matmul's operand[1] is its weight; everything else consumed here is an activation.
                role = "weight" if (info["family"] == "matmul" and pos == 1) else "activation"
                lid = f"L{v.index}"
                leaf_id[v] = lid
                prog.leaves[lid] = {"role": role, "shape": _shape(v), "arg_index": v.index}

    value_id: dict = dict(leaf_id)                           # SSA value -> tensor id (leaves + step outputs)
    for i, (info, r) in enumerate(zip(compute, results)):
        out_id = f"t{i}"
        in_ids = tuple(value_id.get(v, "?") for v in info["inputs"])
        lane = "mesh" if id(r) in mesh_ids else "scalar"
        d = r.demand
        prog.steps.append(WholeModelStep(
            index=i, op=d.op, family=info["family"], lane=lane, unit=r.unit,
            inputs=in_ids, output=out_id, m=d.m, k=d.k, n=d.n))
        value_id[info["result"]] = out_id
    prog.output = prog.steps[-1].output if prog.steps else ""
    return prog


def _mesh_matmul_on_engine(lhs, rhs, target: str):
    """Execute one matmul LAYER on the mesh lane by lowering a single-contraction module through the SAME
    staged pipeline (``lower_module``) and running it on the engine — the compiled path, per op. Returns the
    output as a numpy array. (The RTL-cosim mesh execution is ``run_matmul_on_mesh``; this engine path is
    what the whole-model splice is gated against, numerically exact.)"""
    import numpy as np
    from merlin.xdsl_dialects.lowering import execute, lower_module
    from merlin.xdsl_dialects.lowering.input_workload import build_matmul_chain

    a = np.asarray(lhs)
    w = np.asarray(rhs)
    m, k = a.shape
    n = w.shape[1]
    res = lower_module(build_matmul_chain(dims=(m, k, n), elem="f32"), target=target)
    # Feed the operands through UNTRUNCATED (the engine computes in f64); truncating intermediates to
    # f32 between layers would diverge from the single-module reference, which keeps them in f64.
    run = execute(res, {"A0": a.tolist(), "W": w.tolist()})
    return np.asarray(next(iter(run["outputs"].values())))


def _erf(x):
    """Elementwise error function in f64 (numpy has no ufunc for it); small tensors, exactness over speed."""
    import math

    import numpy as np
    return np.vectorize(math.erf, otypes=[np.float64])(np.asarray(x, dtype=np.float64))


def _rope(x, base: float = 10000.0):
    """LLaMA-style rotary position embedding on a ``(seq, dim)`` activation (dim even): rotate each
    (first-half, second-half) pair of channels by the position-dependent angle ``pos * base**(-2i/dim)``.
    Target-agnostic — the standard rotary transform, computed in f64."""
    import numpy as np
    a = np.asarray(x, dtype=np.float64)
    seq, dim = a.shape[-2], a.shape[-1]
    half = dim // 2
    inv = base ** (-2.0 * np.arange(half) / dim)          # (half,)
    ang = np.outer(np.arange(seq), inv)                    # (seq, half)
    cos = np.concatenate([np.cos(ang), np.cos(ang)], axis=-1)
    sin = np.concatenate([np.sin(ang), np.sin(ang)], axis=-1)
    rot = np.concatenate([-a[..., half:], a[..., :half]], axis=-1)
    return a * cos + rot * sin


def _scalar_op(family: str, operands: list):
    """Execute one norm/activation/elementwise op on the scalar/vector (RVV) lane, in the engine's working
    precision (f64) so the lane handoff is numerically exact vs the reference. Each op is a small numpy
    implementation, derived generically from the op's math (no target special-casing):

    * ``relu`` = max(x, 0); ``add``/``mul`` = elementwise;
    * ``softmax`` = row (last-axis) softmax with the max-subtraction stability shift;
    * ``rmsnorm`` = x · rsqrt(mean(x², last axis) + eps) · gamma;
    * ``layernorm`` = (x − mean) / sqrt(var + eps) · gamma + beta;
    * ``silu`` = x · sigmoid(x); ``gelu`` = 0.5·x·(1 + erf(x/√2)); ``rope`` = rotary embedding.
    """
    import numpy as np
    ops = [np.asarray(o, dtype=np.float64) for o in operands]
    x = ops[0]
    if family == "relu":
        return np.maximum(x, 0.0)
    if family == "add":
        return ops[0] + ops[1]
    if family == "mul":
        return ops[0] * ops[1]
    if family == "softmax":
        z = x - x.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)
    if family == "rmsnorm":
        gamma = ops[1] if len(ops) > 1 else 1.0
        ms = np.mean(x * x, axis=-1, keepdims=True)
        return x / np.sqrt(ms + _NORM_EPS) * gamma
    if family == "layernorm":
        gamma = ops[1] if len(ops) > 1 else 1.0
        beta = ops[2] if len(ops) > 2 else 0.0
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + _NORM_EPS) * gamma + beta
    if family == "silu":
        return x / (1.0 + np.exp(-x))
    if family == "gelu":
        return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))
    if family == "rope":
        return _rope(x)
    raise ValueError(f"no scalar-lane executor for op family {family!r}")


def _fused_op(family: str, operands: list, mesh):
    """Execute one FUSED op that DECOMPOSES into mesh matmuls + a scalar glue step. ``mesh(lhs, rhs)`` runs
    a matmul on the mesh lane (the compiled engine path, or the real oracle when the splice injects one) so
    the contraction sub-ops of attention/geglu stay on the accelerator; the softmax / gelu / gate run inline
    on the scalar lane. Target-agnostic — the decomposition is the op's math, not a per-target route.

    * ``attention``/``attention_full`` = softmax(Q·Kᵀ / √d) · V — Q·Kᵀ and P·V on the mesh, softmax scalar;
    * ``attention_qk`` = Q·Kᵀ / √d — the pre-softmax scores (a mesh matmul with a transposed rhs);
    * ``geglu`` = gelu(X·Wg) ⊙ (X·Wu) — two mesh matmuls, a gelu gate and an elementwise product.
    """
    import numpy as np
    ops = [np.asarray(o, dtype=np.float64) for o in operands]
    if family in ("attention", "attention_full"):
        q, k, v = ops[0], ops[1], ops[2]
        scores = np.asarray(mesh(q, k.T), dtype=np.float64) / np.sqrt(q.shape[-1])
        probs = _scalar_op("softmax", [scores])
        return np.asarray(mesh(probs, v), dtype=np.float64)
    if family == "attention_qk":
        q, k = ops[0], ops[1]
        return np.asarray(mesh(q, k.T), dtype=np.float64) / np.sqrt(q.shape[-1])
    if family == "geglu":
        x, wg, wu = ops[0], ops[1], ops[2]
        gate = _scalar_op("gelu", [np.asarray(mesh(x, wg), dtype=np.float64)])
        return gate * np.asarray(mesh(x, wu), dtype=np.float64)
    raise ValueError(f"no fused-op executor for op family {family!r}")


def run_whole_model_program(program: WholeModelProgram, leaf_values: dict, mesh_exec=None):
    """Execute the spliced program: seed the leaves, then walk the steps IN ORDER, dispatching each to its
    lane (mesh matmuls through the compiled engine path, scalar ops through the vector lane) and threading
    every step's output tensor into the steps that consume it. ``leaf_values`` maps leaf id -> array. Returns
    ``{"outputs": {output_id: array}, "env": {tensor id -> array}}`` — the whole-model result plus every
    intermediate, so a caller can inspect the handoff.

    ``mesh_exec`` overrides the mesh-lane executor: a callable ``(lhs_array, rhs_array, step) -> array`` (or
    ``None`` to signal an unavailable oracle for that layer). When ``None`` (default) the mesh lane runs on the
    engine (``_mesh_matmul_on_engine``) — the numerically-exact reference path. A real driver passes an
    executor that dispatches each mesh matmul onto the target's REAL oracle (``run_matmul_on_mesh``), which
    threads that layer's on-device output into the scalar lane. A mesh_exec returning ``None`` raises
    ``MeshLayerUnavailable`` so the whole run fails closed (never fabricates the layer's output)."""
    import numpy as np
    env: dict = {}
    for lid, arr in leaf_values.items():
        env[lid] = np.asarray(arr, dtype=np.float32)
    missing = [lid for lid in program.leaves if lid not in env]
    if missing:
        raise ValueError(f"missing leaf inputs for {missing} (have {sorted(env)})")

    def _run_mesh(lhs, rhs, step):
        """One matmul on the mesh lane: the compiled engine path (``mesh_exec is None``), else the injected
        oracle executor. A ``None`` from the oracle fails closed (never fabricates the layer's output)."""
        if mesh_exec is None:
            return _mesh_matmul_on_engine(lhs, rhs, program.target)
        got = mesh_exec(lhs, rhs, step)
        if got is None:                                  # oracle could not run this layer — fail closed
            raise MeshLayerUnavailable(step.index, step.m, step.k, step.n)
        return np.asarray(got, dtype=np.float32)

    for step in program.steps:
        operands = [env[i] for i in step.inputs]
        if step.lane == "mesh":
            if step.family != "matmul":
                raise ValueError(f"mesh lane got a non-matmul op {step.op!r} at step {step.index}")
            env[step.output] = _run_mesh(operands[0], operands[1], step)
        elif step.family in _FUSED_MESH_FAMILIES:
            # a fused op: its matmul sub-ops run on the MESH lane (through the same executor), its softmax /
            # gelu / gate run inline on the scalar lane, threaded here by the splice.
            env[step.output] = _fused_op(step.family, operands,
                                         lambda lhs, rhs, _s=step: _run_mesh(lhs, rhs, _s))
        else:
            env[step.output] = _scalar_op(step.family, operands)
    return {"outputs": {program.output: env[program.output]}, "env": env}


class MeshLayerUnavailable(RuntimeError):
    """A mesh matmul layer could not execute on the real oracle (fail closed — never fabricate the output)."""

    def __init__(self, index: int, m, k, n):
        self.index, self.m, self.k, self.n = index, m, k, n
        super().__init__(f"mesh layer {index} ({m}x{k}x{n}) has no reachable oracle in this env")


def _reference_leaf_names(module) -> dict:
    """Reproduce the runtime lowering's leaf-tensor naming so the single-module reference can be injected by
    the SAME arrays the splice seeds: matmul-RHS block args are weights (``W``, ``W1``, …, in first-use
    order); every other tensor block arg is an activation (``A0``, ``A1``, …, in arg order). Returns
    ``{block-arg SSA value -> cb leaf name}``."""
    weights: list = []
    for info in _ordered_compute_ops(module):
        if info["family"] == "matmul":
            rhs = info["inputs"][1]
            if _is_block_arg(rhs) and rhs not in weights:
                weights.append(rhs)
    names: dict = {}
    for i, w in enumerate(weights):
        names[w] = "W" if i == 0 else f"W{i}"
    fn = next(op for op in module.walk() if op.name == "func.func")
    n_a = 0
    for arg in fn.body.blocks[0].args:
        if arg in names:
            continue
        names[arg] = f"A{n_a}"
        n_a += 1
    return names


def _engine_can_lower(module) -> bool:
    """Can the toy xDSL engine lower + evaluate this whole module? True iff every compute op is in the
    engine's command-buffer vocabulary (:data:`_ENGINE_FAMILIES`). A module carrying a transcendental /
    fused family (softmax/rmsnorm/rope/attention/…) is NOT engine-gradeable — it must be graded against the
    host-eager numpy reference (:func:`_host_eager_final`)."""
    return all(info["family"] in _ENGINE_FAMILIES for info in _ordered_compute_ops(module))


def _host_eager_final(program: "WholeModelProgram", leaf_values: dict):
    """A host-eager (numpy, f64) recomputation of the WHOLE spliced program — the reference a module with
    transcendental/fused ops is gated against (the toy engine cannot evaluate exp/rsqrt/div). Walks the
    same steps in the same order, but computes EVERY op in numpy: matmuls as ``@`` in f64, scalar ops and
    fused ops via the same executors the splice uses (so the only thing under test is that the spliced
    on-mesh path reproduces this independent numpy whole-model result)."""
    import numpy as np
    env: dict = {lid: np.asarray(arr, dtype=np.float64) for lid, arr in leaf_values.items()}

    def _np_mesh(lhs, rhs, _step=None):
        return np.asarray(lhs, dtype=np.float64) @ np.asarray(rhs, dtype=np.float64)

    for step in program.steps:
        operands = [env[i] for i in step.inputs]
        if step.family == "matmul":
            env[step.output] = _np_mesh(operands[0], operands[1])
        elif step.family in _FUSED_MESH_FAMILIES:
            env[step.output] = _fused_op(step.family, operands, _np_mesh)
        else:
            env[step.output] = _scalar_op(step.family, operands)
    return env[program.output]


def verify_whole_model_program(module, target: str, in_fmt: str = "f32",
                               weight_fmt: str | None = None, seed: int = 0, units=None,
                               int_operands: bool = False, rtol: float = 1e-5,
                               atol: float = 1e-5) -> dict:
    """End-to-end proof that the SPLICE orchestration is correct: route ``module``'s ops across ``target``'s
    compute units, build the co-scheduled whole-model program, run it per-op (mesh matmuls on the engine,
    scalar ops on the vector lane, activations handed between steps), and compare the final tensor to the
    single-module ``lower_module`` result of the WHOLE module. Both paths use the same engine for matmuls,
    so a correct splice reproduces the reference exactly. Returns a structured result (never asserts) with
    ``exact``/``match`` and the two finals so the caller can gate it.

    Fully verifiable and target-agnostic: the only target-specific input is the routing contract.
    ``units`` overrides the routing contract's compute units (so a caller can prove the splice against a
    format the target's own mesh does not accept, e.g. an f32 mesh) — the engine reference still lowers
    through ``target``.

    REFERENCE SELECTION: a module built only from the engine's op vocabulary (matmul/relu/add/mul) is gated
    against the single-module ``lower_module`` engine result (both paths run matmuls on the same engine, so
    a correct splice reproduces it EXACTLY). A module carrying a transcendental / fused op the engine cannot
    evaluate (softmax/rmsnorm/rope/attention/…) is gated against a HOST-EAGER numpy recomputation of the
    whole module instead (:func:`_host_eager_final`) — this is what makes softmax/rmsnorm/attention chains
    gradeable. ``int_operands`` seeds small integers so an integer-exact whole model matches bit-for-bit."""
    import numpy as np
    from merlin.targetgen import routing as _routing
    from merlin.xdsl_dialects.lowering import execute, lower_module

    # leaf operands, keyed once and reused for BOTH paths (same arrays, two injections). Small integers
    # (``int_operands``) keep matmuls exact so the whole model — even with softmax/rmsnorm downstream —
    # matches its numpy reference bit-for-bit; otherwise standard-normal floats gated within tolerance.
    rng = np.random.default_rng(seed)
    fn = next(op for op in module.walk() if op.name == "func.func")
    args = list(fn.body.blocks[0].args)
    if int_operands:
        arrays = {a: np.rint(rng.standard_normal(tuple(_shape(a))) * 3).clip(-8, 7).astype(np.float32)
                  for a in args}
    else:
        arrays = {a: rng.standard_normal(tuple(_shape(a))).astype(np.float32) for a in args}

    # spliced per-op orchestration over the routing plan (mesh matmuls on the engine, scalar/fused lane
    # inline, activations handed between steps).
    demands = demands_from_module(module, in_fmt, weight_fmt)
    plan = _routing.route_plan_on(demands, units if units is not None else _cu_units(target))
    program = build_whole_model_program(plan, target, module)
    leaf_values = {f"L{a.index}": arrays[a] for a in args if f"L{a.index}" in program.leaves}
    spliced = run_whole_model_program(program, leaf_values)
    spliced_final = spliced["outputs"][program.output]

    if _engine_can_lower(module):
        # single-module engine reference (the whole model compiled + run as ONE module).
        ref_names = _reference_leaf_names(module)
        ref_inj = {ref_names[a]: arrays[a].tolist() for a in args}
        ref_run = execute(lower_module(module, target=target), ref_inj)
        ref_final = np.asarray(next(iter(ref_run["outputs"].values())))
        ref_kind, ref_correct = "engine", bool(ref_run.get("correct"))
    else:
        # host-eager numpy reference (the engine cannot evaluate this module's transcendental/fused ops).
        ref_final = _host_eager_final(program, leaf_values)
        ref_kind, ref_correct = "host_eager", None

    exact = bool(np.array_equal(spliced_final, ref_final))
    match = bool(np.allclose(spliced_final, ref_final, rtol=rtol, atol=atol))
    return {
        "target": target,
        "exact": exact,
        "match": match,
        "ref_kind": ref_kind,
        "ref_correct": ref_correct,
        "n_steps": len(program.steps),
        "n_mesh": program.n_mesh(),
        "n_scalar": program.n_scalar(),
        "output_id": program.output,
        "ref_final": ref_final,
        "spliced_final": spliced_final,
        "program": program,
    }


def _cu_units(target: str):
    """The target's contract compute units (loaded via the registry), for routing the splice plan."""
    from merlin.targetgen import compute_units as _cu
    from merlin.targetgen import target_registry as tr
    return _cu.compute_units(tr.load_contract(target))
