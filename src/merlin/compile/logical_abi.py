"""Public declaration-only logical kernel ABI and tensor boundary metadata.

This owner reads shapes, roles, selected argument-order tokens, output dtypes
and declared derivation/epilogue geometry. It never reads tensor values, runs a
reference implementation, renders a harness, or grants target execution. Both
the ordinary runner and authored builders consume these same declarations.

Pass ``logical_abi(selected_public_path)`` or
``logical_abi_from_document(selected_document)`` in an isolated SDK. An omitted
path uses the ordinary explicitly selected contract directory; this module does
not invent a default argument order when that declaration is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any


class HarnessRenderError(RuntimeError):
    """The buffer cannot be called through the logical ABI, or the declared contract data is incomplete."""


LogicalAbiError = HarnessRenderError


@dataclass(frozen=True)
class LogicalAbi:
    """``logical_kernel_abi`` from the OOT backend contract."""

    version: int
    symbol_pattern: str
    whole_program_order: tuple[str, ...]
    default_order: tuple[str, ...]
    published_by: frozenset[str]
    alignment_bytes: int
    poison_byte: int
    blob_min_elements: int | None


def logical_abi(path=None) -> LogicalAbi:
    import yaml

    from merlin.common.paths import contract_dir

    source = path if path is not None else contract_dir() / "mlir_oot_backend_contract.yaml"
    with open(source, encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    return logical_abi_from_document(document, source=source)


def logical_abi_from_document(document, *, source="selected logical ABI declaration") -> LogicalAbi:
    """Read the explicit public declaration; never infer an ABI from semantic roles."""
    block = document.get("logical_kernel_abi") if isinstance(document, dict) else None
    if not isinstance(block, dict):
        raise HarnessRenderError(f"{source}: no logical_kernel_abi block; the harness ABI is undeclared")
    order = block.get("argument_order") or {}
    pointee = block.get("pointee") or {}
    outputs = block.get("outputs") or {}
    try:
        abi = LogicalAbi(
            version=int(block["version"]),
            symbol_pattern=str(block["symbol"]),
            whole_program_order=tuple(order["whole_program"]),
            default_order=tuple(order["default"]),
            published_by=frozenset(str(op) for op in block["published_by"]),
            alignment_bytes=int(pointee["alignment_bytes"]),
            poison_byte=int(outputs["poison_byte"]),
            blob_min_elements=(int(block["blob_min_elements"]) if block.get("blob_min_elements") else None),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessRenderError(f"{source}: logical_kernel_abi is incomplete ({type(exc).__name__}: {exc})") from exc
    known = {"declared_whole_program_args", "logical_inputs_in_declaration_order", "logical_outputs_in_result_order"}
    unknown = sorted(set(abi.whole_program_order + abi.default_order) - known)
    if unknown:
        raise HarnessRenderError(f"logical_kernel_abi orders by token(s) {unknown} this renderer does not resolve")
    if abi.alignment_bytes <= 0 or abi.alignment_bytes & (abi.alignment_bytes - 1):
        raise HarnessRenderError("logical_kernel_abi.pointee.alignment_bytes must be a positive power of two")
    if not 0 <= abi.poison_byte <= 255:
        raise HarnessRenderError("logical_kernel_abi.outputs.poison_byte must be one byte")
    return abi


POOL_ATTR_ARITY = {"pool_in_dims": 2, "pool_size": 2, "pool_stride": 2, "pool_padding": 4}


POOL_PAD_VALUE_ATTR = "pool_pad_value"


IM2COL_RECIPE_KEY = "im2col_recipes"


DERIVATION_RECIPE_KEYS: tuple[str, ...] = (IM2COL_RECIPE_KEY,)


PRODUCING_KEYS = ("dst", "out", "output")


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
            if key == "pool_padding":  # the one parameter with a meaningful "none"
                out[key] = (0, 0, 0, 0)
                continue
            raise ValueError(
                f"{op}: a pooling epilogue stage needs attribute {key!r} "
                f"({arity} int(s)) and the command does not declare it"
            )
        if not isinstance(v, (list, tuple)) or len(v) != arity:
            raise ValueError(f"{op}: pooling attribute {key!r} must be a list of {arity} int(s), got {v!r}")
        try:
            out[key] = tuple(int(x) for x in v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{op}: pooling attribute {key!r} has a non-integer entry: {v!r}") from e
    pv = attrs.get(POOL_PAD_VALUE_ATTR)
    out["pad_value"] = None if pv is None else int(pv)
    return out


def pool_out_dims(H: int, W: int, pool_size, pool_stride, pool_padding) -> tuple[int, int]:
    """Declared pooled extent, shared by ABI shapes and the runtime tensor owner.

    This computes dimensions only; it never reads a tensor or executes pooling.
    """
    ph, pw = int(pool_size[0]), int(pool_size[1])
    sh, sw = int(pool_stride[0]), int(pool_stride[1])
    pt, pl, pb, pr = (int(x) for x in pool_padding)
    return (H + pt + pb - ph) // sh + 1, (W + pl + pr - pw) // sw + 1


def harness_derived_tensors(cb: dict[str, Any]) -> dict[str, str]:
    """``{derived tensor name: the params key that derives it}`` for one command buffer.

    Reads only the ABI's own vocabulary (``params.<recipe key>[].target``), so it is target-agnostic
    and opcode-agnostic: it answers "which of this buffer's tensors did the HARNESS build, rather
    than the emitted program", for any endpoint that speaks this ABI. An empty result means every
    tensor in the buffer is either a declared leaf or produced by a command.
    """
    out: dict[str, str] = {}
    params = cb.get("params") or {}
    if not isinstance(params, dict):
        return out
    for key in DERIVATION_RECIPE_KEYS:
        for recipe in params.get(key) or ():
            if not isinstance(recipe, dict):
                continue
            name = recipe.get("target")
            if isinstance(name, str) and name:
                out[name] = key
    return out


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


def declared_output_dtypes(cb: dict[str, Any]) -> dict[str, str]:
    """``{tensor name: declared element dtype}`` for every tensor a readback can land in.

    ONE fact, read the way the harness reads it, so the writer and the reader of a buffer cannot
    disagree about what is in it. A command that names a destination may DECLARE the container its
    result lands in (``attributes.output_dtype`` — a movement is precisely a container widening, and
    a commit's readout dtype is not the accumulator's), and that declaration is what a harness sizes
    the destination buffer from; a tensor that no command re-declares carries its own ``dtype``. The
    command's declaration wins for exactly the destination it names, which is the same resolution
    order a target harness applies when it allocates the buffer.

    Target-agnostic: only the buffer's own vocabulary is read (``dst``/``out``/``output`` keys, the
    ``role`` a tensor declares, the ``output_dtype`` attribute the ABI defines). A buffer with no
    commands still answers — its declared outputs are its ``role: output`` tensors — which is the
    case a program that runs entirely on the host lane presents.
    """
    tensors = cb.get("tensors") or {}
    dtypes: dict[str, str] = {}
    for name, spec in tensors.items():
        if not isinstance(spec, dict):
            continue
        dtype = spec.get("dtype")
        if dtype:
            dtypes[str(name)] = str(dtype)
    for cmd in cb.get("commands") or []:
        ops = cmd.get("operands") or {}
        declared = (cmd.get("attributes") or {}).get("output_dtype")
        if not declared:
            continue
        for key, name in ops.items():
            if isinstance(name, str) and _produces(key, name, tensors) and name in dtypes:
                dtypes[name] = str(declared)
    return dtypes


def commit_output_shape(cb: dict[str, Any], acc_name: str, attrs: dict[str, Any] | None = None) -> tuple[int, int]:
    """(m, n) for a commit: m from the matmul lhs rows, n from the resident weight cols.

    A POOLING epilogue reduces m: the accumulator's rows unflatten to a ``pool_in_dims`` plane and the
    window walks it, so the committed tensor has fewer rows than the matmul produced. Printing the
    matmul's row count for a pooled commit would emit a module whose declared result type disagrees
    with what any engine executing it computes -- a type error the grammar cannot catch because the
    grammar is what would be lying."""
    tensors = cb.get("tensors", {})
    res_src: dict[str, str] = {}
    for cmd in cb.get("commands", []):
        if cmd["opcode"] == "RES_PACK":
            res_src[cmd["operands"]["dst"]] = cmd["operands"]["src"]
    for cmd in cb.get("commands", []):
        if cmd["opcode"] in ("MATMUL_RESIDENT", "MATMUL") and cmd["operands"]["dst"] == acc_name:
            lhs = cmd["operands"]["lhs"]
            rhs = cmd["operands"]["rhs"]
            w = res_src.get(rhs, rhs)
            m = int(tensors[lhs]["shape"][0])
            n = int(tensors[w]["shape"][1])
            if "maxpool" in ((attrs or {}).get("epilogue") or []):
                p = pool_params(attrs or {}, op=f"COMMIT of {acc_name!r}")
                H, W = p["pool_in_dims"]
                Ho, Wo = pool_out_dims(H, W, p["pool_size"], p["pool_stride"], p["pool_padding"])
                m = (m // (H * W)) * Ho * Wo
            return m, n
    raise ValueError(f"no matmul produces accumulator {acc_name!r}")


@dataclass(frozen=True)
class Buffer:
    name: str
    kind: str  # "input" | "output" | "scratch" | "inout"
    shape: tuple[int, ...]
    dtype: str

    @property
    def elements(self) -> int:
        return prod(self.shape) if self.shape else 1

    @property
    def matrix(self) -> tuple[int, int]:
        """The ``OUT`` line's (rows, cols): leading logical extents multiply into rows."""
        if not self.shape:
            return 1, 1
        return prod(self.shape[:-1]), self.shape[-1]


def _shape(value, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in value
    ):
        raise HarnessRenderError(f"tensor {name!r} needs a shape of positive integer extents, got {value!r}")
    return tuple(value)


def _produced(cb) -> set[str]:
    return {
        name
        for cmd in cb.get("commands") or []
        for key, name in (cmd.get("operands") or {}).items()
        if key == "dst" and isinstance(name, str)
    }


def logical_inputs(cb) -> list[str]:
    """Every declared leaf the reference reads, in declaration order: not produced by a command, not
    derived by a harness recipe (a derived matrix is a compiler's lowering detail, never a pointer),
    and not an output or intermediate."""
    produced, derived = _produced(cb), harness_derived_tensors(cb)
    return [
        name
        for name, spec in (cb.get("tensors") or {}).items()
        if name not in produced
        and name not in derived
        and (spec or {}).get("role", "input") not in ("output", "intermediate")
    ]


def logical_outputs(cb, published_by) -> list[str]:
    """The results, in order: every destination Merlin's reference publishes (program order), then every
    declared role=output tensor not yet listed -- including one no command produces, which is how a
    program routed entirely onto the host lane declares its results."""
    tensors = cb.get("tensors") or {}
    names: list[str] = []
    for cmd in cb.get("commands") or []:
        if cmd.get("opcode") in published_by:
            dst = (cmd.get("operands") or {}).get("dst")
            if isinstance(dst, str) and dst not in names:
                names.append(dst)
    for name, spec in tensors.items():
        if (spec or {}).get("role") == "output" and name not in names:
            names.append(name)
    declared = cb.get("outputs")
    if declared:
        names = [name for name in names if name in set(declared)]
    return names


def _output_buffer(cb, name: str) -> Buffer:
    """An output's logical shape/dtype: its declaration, else Merlin's own derivation for the command
    that produces it. Anything else is refused -- an output nobody sized cannot be read back."""
    tensors = cb.get("tensors") or {}
    dtypes = declared_output_dtypes(cb)
    spec = tensors.get(name)
    if isinstance(spec, dict) and spec.get("shape") is not None:
        return Buffer(name, "output", _shape(spec["shape"], name=name), dtypes.get(name) or str(spec.get("dtype")))
    for cmd in cb.get("commands") or []:
        ops, attrs = cmd.get("operands") or {}, cmd.get("attributes") or {}
        if ops.get("dst") != name:
            continue
        if cmd.get("opcode") == "COMMIT":
            dtype = attrs.get("output_dtype")
            if not dtype:
                raise HarnessRenderError(f"COMMIT {name!r} declares no output_dtype")
            return Buffer(name, "output", tuple(commit_output_shape(cb, ops["src"], attrs)), str(dtype))
        if cmd.get("opcode") == "MOVEMENT":
            src = tensors.get(ops.get("src")) or {}
            dtype = attrs.get("output_dtype") or src.get("dtype")
            return Buffer(name, "output", _shape(src.get("shape"), name=name), str(dtype))
    raise HarnessRenderError(f"result {name!r} has no declared tensor and no derivable logical shape")


def logical_interface(cb, abi: LogicalAbi) -> list[Buffer]:
    """The ordered pointer arguments of ``cb`` under the logical ABI."""
    tensors = cb.get("tensors") or {}
    kabi = cb.get("kernel_abi") or {}
    if kabi.get("kind") == "whole_program":
        args = kabi.get("args")
        outputs = kabi.get("outputs") or []
        if not isinstance(args, list) or not args:
            raise HarnessRenderError("a whole-program buffer must declare kernel_abi.args")
        buffers = []
        for arg in args:
            name, access = (arg or {}).get("tensor"), (arg or {}).get("access")
            spec = tensors.get(name)
            if not isinstance(spec, dict) or access not in ("read", "write", "readwrite"):
                raise HarnessRenderError(f"whole-program argument {arg!r} is not a typed, declared tensor")
            if name in outputs and access != "write":
                raise HarnessRenderError(f"whole-program output {name!r} must be write-only under the logical ABI")
            kind = (
                "input"
                if access == "read"
                else ("output" if name in outputs else ("inout" if access == "readwrite" else "scratch"))
            )
            buffers.append(Buffer(name, kind, _shape(spec.get("shape"), name=name), str(spec.get("dtype") or "")))
        if len({b.name for b in buffers}) != len(buffers):
            raise HarnessRenderError("whole-program arguments must be distinct tensors")
        if set(outputs) - {b.name for b in buffers if b.kind == "output"}:
            raise HarnessRenderError("every whole-program output must be a write/readwrite argument")
        return _ordered(abi.whole_program_order, {"declared_whole_program_args": buffers})
    inputs = [
        Buffer(n, "input", _shape(tensors[n].get("shape"), name=n), str(tensors[n].get("dtype") or "i8"))
        for n in logical_inputs(cb)
    ]
    outputs = [_output_buffer(cb, n) for n in logical_outputs(cb, abi.published_by)]
    if not outputs:
        raise HarnessRenderError("the buffer declares no result the reference reports; nothing to read back")
    return _ordered(
        abi.default_order,
        {"logical_inputs_in_declaration_order": inputs, "logical_outputs_in_result_order": outputs},
    )


def _ordered(order, resolved) -> list[Buffer]:
    out: list[Buffer] = []
    for token in order:
        if token not in resolved:
            raise HarnessRenderError(f"logical_kernel_abi token {token!r} does not apply to this buffer")
        out.extend(resolved[token])
    return out


def kernel_arg_order(cb, abi: LogicalAbi | None = None) -> list[str]:
    """The pointer argument names, in call order (no values, no rendering)."""
    return [b.name for b in logical_interface(cb, abi or logical_abi())]


def kernel_abi_from_commands(cb) -> dict:
    """The logical pointer ABI as a ``kernel_abi`` record (an explicit declaration is returned unchanged)."""
    declared = cb.get("kernel_abi")
    if isinstance(declared, dict) and declared.get("args"):
        return declared
    buffers = logical_interface(cb, logical_abi())
    return {
        "kind": "logical_v2",
        "args": [{"tensor": b.name, "access": "write" if b.kind == "output" else "read"} for b in buffers],
        "outputs": [b.name for b in buffers if b.kind == "output"],
    }
