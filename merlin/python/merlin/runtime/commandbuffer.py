"""Load/validate a Merlin command buffer and materialize its declared tensors.

The execution-oriented command buffer carries a ``tensors`` table (name -> shape/dtype/role)
in addition to the opcode list. Input/weight/bias tensors are materialized deterministically
(see :func:`Tensor.deterministic`) so a run is reproducible without external input files; an
explicit inputs mapping can override them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tensor import Tensor, pool_out_dims  # noqa: F401  (pool_out_dims re-exported: see its docstring)

#: The attribute names a POOLING epilogue stage reads, and the arity each one must have.
#:
#: They ride on the COMMAND's attributes because that is where the hardware carries them: this target
#: family fuses pooling into the STORE path (``config_st(..., pool_stride, pool_size, pool_out_dim,
#: porows, pocols, orows, ocols, upad, lpad)``) and into the fused conv loop, so the pool parameters are
#: configured on the same instruction that reads the accumulator out. By the time an engine reaches the
#: readout the tensor is a 2-D ``[rows, channels]`` and no conv geometry is in scope, so the extent the
#: rows unflatten to (``pool_in_dims`` = the ABI's ``orows``/``ocols``) must be declared, not guessed.
POOL_ATTR_ARITY = {"pool_in_dims": 2, "pool_size": 2, "pool_stride": 2, "pool_padding": 4}
#: Optional companion: the value a padded cell contributes to the max. No default on purpose -- see
#: :meth:`Tensor.maxpool2d_rows`.
POOL_PAD_VALUE_ATTR = "pool_pad_value"


def pool_params(attrs: dict[str, Any], *, op: str) -> dict[str, Any]:
    """Read a pooling epilogue's geometry out of a command's attributes, FAILING CLOSED.

    Every engine that applies a pooling stage (the capsule golden, the reference recomputation and the
    simulator) parses it through this one function, so they cannot disagree about which attribute names
    carry the geometry or about what an absent one means. An absent or wrong-arity attribute RAISES and
    names itself: a pooling stage whose window silently defaulted would return a correctly-shaped tensor
    of numbers nobody computed, and the integer gate would then enforce it.
    """
    out: dict[str, Any] = {}
    for key, arity in POOL_ATTR_ARITY.items():
        v = attrs.get(key)
        if v is None:
            if key == "pool_padding":                 # the one parameter with a meaningful "none"
                out[key] = (0, 0, 0, 0)
                continue
            raise ValueError(
                f"{op}: a pooling epilogue stage needs attribute {key!r} "
                f"({arity} int(s)) and the command does not declare it")
        if not isinstance(v, (list, tuple)) or len(v) != arity:
            raise ValueError(
                f"{op}: pooling attribute {key!r} must be a list of {arity} int(s), got {v!r}")
        try:
            out[key] = tuple(int(x) for x in v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{op}: pooling attribute {key!r} has a non-integer entry: {v!r}") from e
    pv = attrs.get(POOL_PAD_VALUE_ATTR)
    out["pad_value"] = None if pv is None else int(pv)
    return out


def apply_pool_stage(t: Tensor, stage: str, attrs: dict[str, Any], *, op: str) -> Tensor:
    """Apply the named pooling stage to ``t``. The single definition all three engines call."""
    if stage != "maxpool":                            # pragma: no cover - callers dispatch on the name
        raise ValueError(f"{op}: {stage!r} is not a pooling stage this runtime defines")
    p = pool_params(attrs, op=op)
    return t.maxpool2d_rows(in_dims=p["pool_in_dims"], pool_size=p["pool_size"],
                            pool_stride=p["pool_stride"], pool_padding=p["pool_padding"],
                            pad_value=p["pad_value"])


#: The epilogue stages the command-buffer ABI admits, in the order a COMMIT applies them. This is the
#: single definition the ABI, its JSON schema and the interface dialect all read, so a stage exists here
#: or it does not exist. It is NOT a claim that a given target implements one: an engine that does not
#: RAISES by name, and a target that cannot execute it fails at its oracle. Widening this tuple is a
#: contract change and needs both an engine that implements the stage and a target rung evidencing it.
EPILOGUE_STAGES: tuple[str, ...] = ("bias_add", "bias", "requant", "acc_scale", "relu", "maxpool")

#: Set form, for membership tests.
EPILOGUE_STAGE_SET = frozenset(EPILOGUE_STAGES)

#: The subset of the epilogue that adds a per-column bias, which is the one stage needing an operand
#: NAME resolved rather than a flag. Kept as its own tuple because a bias dropped in silence leaves
#: every element off by exactly its column's bias, which reads as a plausible answer.
BIAS_STAGES = ("bias_add", "bias")


def bias_tensor_name(operands: dict[str, Any], attrs: dict[str, Any], *, op: str) -> str:
    """Name of the bias tensor a ``bias_add``/``bias`` stage consumes. One definition, all engines.

    The name is carried in TWO places in this tree, and both are legitimate:

    * ``attributes["bias"]`` -- what a buffer that came through the interface grammar carries.
      ``merlin_iface.commit`` declares ``bias`` as a property, so the ingest lands it beside
      ``epilogue`` in the attributes (see ``interface_emit``/``ir_ingest``).
    * ``operands["bias"]`` -- what a buffer an emitter built directly carries, because there the
      bias is a named operand slot like any other (``linalg_lower``, ``runtime_lowering``, and the
      whole-op ``BIAS_ADD`` whose operand keys the grammar table itself spells ``["src", "bias"]``).

    Reading only one of them is how a bias gets DROPPED IN SILENCE. The reference/simulate engines
    read only the operands while the golden engine read only the attributes, so on a fused
    matmul+bias capsule the golden added the bias and the reference did not -- and the capsule
    failed L0 with every element off by exactly its column's bias, which reads like a rounding or
    ordering defect rather than a stage that never ran.

    FAIL CLOSED when neither carries a name: a declared stage with no operand is not a no-op, it is
    an unanswerable command. Skipping it would make two engines that both skip it AGREE on a value
    neither computed -- the same silent-pass hazard the unknown-stage branches guard against.
    """
    for source in (attrs, operands):
        name = (source or {}).get("bias")
        if name:
            return str(name)
    raise ValueError(
        f"{op} declares a bias epilogue stage but names no bias tensor in either its operands or "
        f"its attributes. The stage cannot be executed and is NOT skipped: a dropped bias makes "
        f"every output element differ from the golden by its own column's bias, which is "
        f"indistinguishable from an arithmetic defect")


def load_command_buffer(path: str | Path) -> dict[str, Any]:
    """Load a command-buffer JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


REQUIRED_KEYS = ("abi_version", "target", "commands")


def validate_command_buffer(cb: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty == valid).

    Beyond the required keys, this catches the one structural error the JSON schema cannot express: an
    operand slot holds the NAME of a buffer, and the schema types it as a bare string, so a value that
    names nothing at all still validates. A submission that emitted commands referencing shape strings
    (``dst: "16x16"``) with no ``tensors`` declared passed schema validation, was told it was valid, and
    was then rejected downstream for a constraint the contract never stated -- so it spent its session
    guessing spellings instead of writing a compiler.

    The check is deliberately the weakest one that is certainly true: commands that reference operands
    need SOMETHING to reference. Names produced by earlier commands (an accumulator, a committed
    intermediate) are legitimately absent from ``tensors``, so per-name resolution is NOT asserted here --
    only that a computing command buffer declares at least one buffer to compute over."""
    problems: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in cb:
            problems.append(f"missing key '{k}'")
    for i, cmd in enumerate(cb.get("commands", [])):
        if "opcode" not in cmd:
            problems.append(f"command {i} missing 'opcode'")
    cmds = cb.get("commands") or []
    referenced = sorted({str(v) for c in cmds for v in (c.get("operands") or {}).values()
                         if isinstance(v, str) and v})
    if referenced and not (cb.get("tensors") or {}) and not cb.get("declined"):
        problems.append(
            f"commands reference operand name(s) {referenced[:6]}"
            f"{' ...' if len(referenced) > 6 else ''} but the command buffer declares no 'tensors'. "
            f"An operand slot holds the NAME of a tensor declared in 'tensors' (e.g. \"Y0\"), not a "
            f"shape, a type, or a dimension list")
    return problems


def conv_out_dims(H: int, W: int, kh: int, kw: int, stride, padding, dilation) -> tuple[int, int]:
    sh, sw = stride
    pt, pl, pb, pr = padding
    dh, dw = dilation
    Ho = (H + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    Wo = (W + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    return Ho, Wo


def conv_im2col(ifm: Tensor, *, kh: int, kw: int, ci: int, stride, padding, dilation,
                layout: str = "nhwc") -> Tensor:
    """Build the [N*Ho*Wo, Kh*Kw*Ci] im2col matrix from an NHWC activation (zero-pad OOB taps).

    This is the single source of truth shared by the runner harness/reference/simulate (via
    :func:`materialize_inputs`) and the capsule golden. Column order = (kh, kw, ci), matching the
    weight packing [Kh*Kw*Ci, Co].
    """
    if layout != "nhwc":
        raise ValueError(f"conv_im2col layout {layout!r} unsupported (nhwc only)")
    N, H, W, C = ifm.shape
    if C != ci:
        raise ValueError(f"conv_im2col channel mismatch: ifm C={C} != ci={ci}")
    sh, sw = stride
    pt, pl, _, _ = padding
    dh, dw = dilation
    Ho, Wo = conv_out_dims(H, W, kh, kw, stride, padding, dilation)
    a = ifm.data
    rows: list[int] = []

    def at(n, y, x, c):
        if 0 <= y < H and 0 <= x < W:
            return a[((n * H + y) * W + x) * C + c]
        return 0

    for n in range(N):
        for oy in range(Ho):
            for ox in range(Wo):
                by, bx = oy * sh - pt, ox * sw - pl
                for ky in range(kh):
                    for kx in range(kw):
                        for c in range(ci):
                            rows.append(at(n, by + ky * dh, bx + kx * dw, c))
    return Tensor((N * Ho * Wo, kh * kw * ci), rows, ifm.dtype)


def materialize_inputs(cb: dict[str, Any], inputs: dict[str, Any] | None = None) -> dict[str, Tensor]:
    """Create the leaf input tensors declared in the command buffer's ``tensors`` table.

    A tensor is a *leaf* (materialized here) when its role is input/weight/bias, i.e. it is
    not produced by a command. ``inputs`` may supply explicit nested-list data per name.

    If the command buffer declares ``params.im2col_recipes``, each derived activation is built by
    gathering conv windows from its source leaf (so a compiler-lowered conv2d's im2col activation is
    materialized identically for the reference, the simulator, and the device harness).
    """
    inputs = inputs or {}
    produced = set()
    for cmd in cb.get("commands", []):
        ops = cmd.get("operands", {})
        for key in ("dst",):
            if key in ops:
                produced.add(ops[key])
    env: dict[str, Tensor] = {}
    for name, spec in cb.get("tensors", {}).items():
        if name in produced:
            continue
        shape = tuple(spec["shape"])
        dtype = spec.get("dtype", "i8")
        if name in inputs:
            flat = _flatten(inputs[name])
            env[name] = Tensor(shape, flat, dtype)
        else:
            env[name] = Tensor.deterministic(name, shape, dtype)
    # additive: overwrite derived im2col activations from their source leaf
    for r in cb.get("params", {}).get("im2col_recipes", []):
        src = env[r["source"]]
        env[r["target"]] = conv_im2col(
            src, kh=int(r["kh"]), kw=int(r["kw"]), ci=int(r["ci"]),
            stride=tuple(r.get("stride", [1, 1])), padding=tuple(r.get("padding", [0, 0, 0, 0])),
            dilation=tuple(r.get("dilation", [1, 1])), layout=r.get("layout", "nhwc"))
    return env


#: Operand keys that PRODUCE a tensor. This is the command-buffer ABI's own vocabulary (the buffer
#: declares its ``abi_version``), not a fact about any target — every command-buffer endpoint names its
#: result with one of these, whatever its opcodes are called. Kept next to :func:`materialize_inputs`,
#: which uses the same notion of "produced" to decide what is a leaf.
#: Operand keys that name a WRITE by spelling. Not exhaustive by construction -- the schema lets a
#: buffer spell its destination `result`, `y`, or anything else -- so :func:`dataflow_operands` also
#: accepts a key whose tensor DECLARES `role: output`. See `_produces` there.
PRODUCING_KEYS = ("dst", "out", "output")


def _produces(key: str, name: str, tensors: dict) -> bool:
    """Is this operand position a WRITE?

    By key spelling first, then by the named tensor's declared ``role``. Keying on spelling alone made
    the binder's reach depend on which words someone had listed: a buffer spelling its destination
    ``result`` or ``y`` -- both allowed by the schema -- had its output counted as a READ, so no
    produced tensor was declared and the whole buffer bound to nothing.

    This does NOT reintroduce role as the way to CHOOSE the output; that stays dataflow, for the reason
    in :func:`dataflow_operands` (a fused buffer declares three ``role: output`` tensors and two are
    intermediates). Role decides only whether a POSITION writes; dataflow still decides which write is
    the result.
    """
    if key in PRODUCING_KEYS:
        return True
    return str((tensors.get(name) or {}).get("role", "")).lower() == "output"


def dataflow_operands(cb: dict[str, Any]) -> tuple[list[str], str] | None:
    """``(leaf_input_names, final_output_name)`` of a command buffer, derived by DATAFLOW alone.

    Target-agnostic and opcode-agnostic by construction: a declared tensor that some command CONSUMES and
    no command PRODUCES is a leaf input; the last produced tensor that is itself declared is the output;
    a tensor that is both produced and consumed is an intermediate and is neither. Nothing here reads a
    target fact, an opcode meaning, or an op-specific shape rule, so it works for any endpoint that
    speaks this ABI — SIMT, systolic/NPU, or otherwise.

    Why dataflow rather than the declared ``role``: role cannot decide it. Measured on a real fused
    flash-attention buffer, THREE tensors declare ``role: output`` (``S``, ``P``, ``Y0``) because each is
    produced by some command, and two of them are intermediates feeding the next stage. Reading roles
    alone yields three candidate outputs and no way to choose; the dataflow yields exactly ``Q, K, V`` in
    and ``Y0`` out.

    This is the general case the single-op binders cannot express. A binder that pattern-matches "one
    matmul" returns ``None`` on a CHAIN, which is what left every fused capsule (attention_qk -> softmax
    -> matmul -> commit, rmsnorm -> matmul, chained matmuls) unbindable and therefore ungradeable.

    Returns ``None`` when the buffer carries no commands, no declared ``tensors``, or no produced tensor
    that is declared — i.e. when dataflow genuinely cannot answer, never a guess.

    NOTE ON ORDER: the returned inputs are in first-consumption order (the order the command stream
    first reads them), which is deterministic and reproducible. It is NOT authoritative for a kernel ABI
    — the emitted kernel's own signature is. A caller that must match a kernel signature should reorder
    these by matching declared shapes to that signature rather than trusting this order.
    """
    tensors = cb.get("tensors") or {}
    commands = cb.get("commands") or []
    if not tensors or not commands:
        return None

    produced: list[str] = []
    consumed: list[str] = []
    for cmd in commands:
        ops = cmd.get("operands") or {}
        for key, name in ops.items():
            if not isinstance(name, str):
                continue
            (produced if _produces(key, name, tensors) else consumed).append(name)

    produced_set = set(produced)
    # leaves, in first-consumption order, de-duplicated
    seen: set[str] = set()
    leaves: list[str] = []
    for name in consumed:
        if name in produced_set or name in seen or name not in tensors:
            continue
        seen.add(name)
        leaves.append(name)

    # The output is the LAST produced tensor that is DECLARED. An internal accumulator is produced but
    # not declared (measured: `acc_Y0` is a dst and absent from `tensors`), so walking backwards over the
    # produced list and taking the first declared hit lands on the committed result rather than on it.
    out = next((n for n in reversed(produced) if n in tensors), None)
    if out is None or not leaves:
        return None
    return leaves, out


def _flatten(nested) -> list[int]:
    """Flatten a nested list of ANY rank to its scalars, row-major.

    This peeled exactly one level, which is right for a rank-2 operand and silently wrong for a
    rank-3 one: a batched activation came back as a list of LISTS, and ``Tensor`` then rejected it for
    having 4 elements where its shape needs 12. That reads as a malformed input rather than as an
    unsupported rank, which is how a rank the contract admits stays unreachable -- the target contract
    declares rank-3 contractions legal and nothing downstream could materialize one.

    Recursing costs nothing here and is rank-agnostic, so the next rank does not need another edit.
    """
    out: list[int] = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out
