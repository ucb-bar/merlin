"""The C adapter between compiled host code and a device kernel.

The host side of an offloaded contraction is an MLIR function call. Its lowered private declaration
passes a memref as ``(allocated, aligned, offset, sizes..., strides...)`` -- seven scalars for a rank-2
operand -- but calls the external ``_mlir_ciface_*`` as a result-descriptor pointer followed by one
pointer per operand descriptor. The device side is the target's own kernel. Under the logical
kernel ABI (the default, ``logical_kernel_abi`` in the OOT backend contract) it takes dense row-major
logical tensors in the order the group's interface declares them; a support that selects the
version-1 resident ABI (``harness_abi.kernel_abi_version: 1``) gets
``void {target}_kernel(weight, lhs_0.., out_0..)`` with edge tiles zero-padded to the tile edge.
Neither side can call the other directly, so something has to unpack one convention into the other.
That is all this emits.

Deliberately NOT emitted here: the kernel itself. The device kernel is the target's own backend
artifact -- the thing its OOT package builds and the thing the capsule ladder certifies -- and
regenerating a transcription of it here would mean a model executes code no oracle ever graded. The
shim declares the kernel ``extern`` and the archive supplies it.

One structural choice worth stating: a single ``void*``-based descriptor serves every element type.
The shim only passes pointers through -- it never reads an element -- and pointer fields have the same
size and alignment whatever they point at, so the struct is ABI-identical for i8, i32, f32 or a
microscaled format alike. The alternative, a typedef per dtype, is what forces the existing
matrix-unit shim to know which dtypes exist.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

# ---- device_shim -------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalBuffer:
    """One pointer of a kernel's logical interface: its name, direction, host role, shape and dtype.

    ``role`` is how the host call supplies it: ``input_<k>`` / ``weight_<k>`` / ``bias_<k>`` (the
    interface's own tensor roles, numbered in order) and ``out_<k>`` (the k-th result). A role the host
    call does not carry is a value with no host source."""

    name: str
    kind: str
    role: str
    shape: tuple[int, ...]
    dtype: str


def logical_buffers(
    interface: str, *, observed_outputs: tuple[str, ...] | None = None
) -> tuple[tuple[LogicalBuffer, ...] | None, str]:
    """``(buffers in kernel_arg_order, "")`` for one interface, or ``(None, reason)``.

    The order, shapes and dtypes are :func:`merlin.target.contract.harness.harness_render.logical_interface`'s
    -- what a runner-owned capsule harness passes. An explicit original caller
    may supply source-bound observed outputs for a closed connected graph; no
    observation is suppressed by inference from candidate code. The caller owes
    that graph closure, separately from this pointer transport. Inputs are numbered within their declared role;
    activations (role ``input``) in the order the commands read them (a contraction's left operand
    first), weights and biases in declaration order; results in result order.
    """
    from merlin.compiler.lower.device.shim import _LHS_KEYS, _RHS_KEYS, _elem_bytes
    from merlin.target.contract.harness import harness_render as HR
    from merlin.target.contract.interfaces.grammar import parse_interface_mlir

    try:
        cb = parse_interface_mlir(interface)
        if observed_outputs is not None:
            produced = {c.get("operands", {}).get("dst") for c in cb["commands"]}
            if (
                type(observed_outputs) is not tuple
                or not observed_outputs
                or len(set(observed_outputs)) != len(observed_outputs)
                or any(type(name) is not str or name not in produced for name in observed_outputs)
            ):
                raise ValueError("original caller must supply distinct actually produced observed outputs")
            cb["outputs"] = list(observed_outputs)
        ordered = HR.logical_interface(cb, HR.logical_abi())
    except Exception as exc:  # noqa: BLE001 -- an interface the logical ABI cannot describe is a refusal
        return None, f"the logical kernel ABI does not resolve this interface ({type(exc).__name__}: {exc})"
    tensors = cb.get("tensors") or {}
    resident = {
        (cmd.get("operands") or {}).get("dst"): (cmd.get("operands") or {}).get("src")
        for cmd in cb.get("commands") or []
        if cmd.get("opcode") == "RES_PACK"
    }
    read_order: list[str] = []
    for cmd in cb.get("commands") or []:
        ops = cmd.get("operands") or {}
        for key in (*_LHS_KEYS, *_RHS_KEYS):
            value = ops.get(key)
            value = resident.get(value, value) if isinstance(value, str) else None
            if value is not None and value not in read_order:
                read_order.append(value)
    counters: dict[str, int] = {}
    position = {name: index for index, name in enumerate(read_order)}
    inputs = sorted(
        (b.name for b in ordered if b.kind == "input" and (tensors.get(b.name) or {}).get("role", "input") == "input"),
        key=lambda name: position.get(name, len(position)),
    )
    role_of = {name: f"input_{index}" for index, name in enumerate(inputs)}
    out: list[LogicalBuffer] = []
    for buffer in ordered:
        if buffer.kind not in ("input", "output"):
            return None, f"interface value {buffer.name!r} is {buffer.kind}; a group call passes inputs and results"
        if buffer.kind == "output":
            role = f"out_{counters.get('out', 0)}"
            counters["out"] = counters.get("out", 0) + 1
        elif buffer.name in role_of:
            role = role_of[buffer.name]
        else:
            declared = str((tensors.get(buffer.name) or {}).get("role") or "input")
            role = f"{declared}_{counters.get(declared, 0)}"
            counters[declared] = counters.get(declared, 0) + 1
        if _elem_bytes(buffer.dtype) is None:
            return None, f"interface value {buffer.name!r} has a sub-byte or unknown element type {buffer.dtype!r}"
        out.append(
            LogicalBuffer(
                name=buffer.name,
                kind=buffer.kind,
                role=role,
                shape=tuple(int(v) for v in buffer.shape),
                dtype=buffer.dtype,
            )
        )
    return tuple(out), ""


def _memref_typedef(rank: int) -> str:
    if rank == 0:
        return "typedef struct { void *allocated; void *aligned; intptr_t offset; } merlin_lmemref_0d;"
    return (
        f"typedef struct {{ void *allocated; void *aligned; intptr_t offset; intptr_t sizes[{rank}]; "
        f"intptr_t strides[{rank}]; }} merlin_lmemref_{rank}d;"
    )


def _logical_entry(
    symbol: str,
    kernel: str,
    call: Sequence[LogicalBuffer],
    order: Sequence[LogicalBuffer],
    batch: int | None = None,
) -> str:
    """The expanded-memref entry for one call, plus its C-interface wrapper.

    Parameters arrive in the HOST CALL's order (``call`` carries the host shapes); the kernel is called
    in the logical ABI's order. Every buffer is checked against its shape (a contradiction returns an
    empty result rather than reaching the kernel) and must be dense row-major (otherwise trap), and no
    result may overlap another buffer. With ``batch`` every host buffer is ``batch`` disjoint slices of
    the kernel's logical tensor, and the kernel is called once per slice."""
    from merlin.compiler.lower.device.shim import _elem_bytes

    params: list[str] = []
    shape_checks: list[str] = []
    stride_checks: list[str] = []
    spans: list[str] = []
    voids: list[str] = []
    for index, buf in enumerate(call):
        p = f"p{index}"
        rank = len(buf.shape)
        params += [f"void *{p}_alloc", f"void *{p}_aligned", f"intptr_t {p}_off"]
        params += [f"intptr_t {p}_s{d}" for d in range(rank)] + [f"intptr_t {p}_st{d}" for d in range(rank)]
        pitch = 1
        for d in reversed(range(rank)):
            shape_checks.append(f"{p}_s{d} != {buf.shape[d]}")
            stride_checks.append(f"{p}_st{d} != {pitch}")
            pitch *= buf.shape[d]
        rows = 1
        for extent in buf.shape[:-1]:
            rows *= extent
        cols = buf.shape[-1] if rank else 1
        spans.append(
            f"!merlin_span({p}_aligned, {p}_off, {rows}, {cols}, {_elem_bytes(buf.dtype)}u, &{p}_addr, &{p}_bytes)"
        )
        voids.append(f"(void){p}_alloc;")
    index_of = {buf.role: index for index, buf in enumerate(call)}
    outputs = [index for index, buf in enumerate(call) if buf.kind == "output"]
    disjoint = [
        f"!merlin_disjoint((void *)p{o}_addr, p{o}_bytes, (void *)p{other}_addr, p{other}_bytes)"
        for o in outputs
        for other in range(len(call))
        if other != o and not (call[other].kind == "output" and other < o)
    ]
    single = len(outputs) == 1
    result_t = f"merlin_lmemref_{len(call[outputs[0]].shape)}d" if single else f"{symbol}_results"
    lines: list[str] = []
    if not single:
        fields = " ".join(f"merlin_lmemref_{len(call[o].shape)}d r{k};" for k, o in enumerate(outputs))
        lines.append(f"typedef struct {{ {fields} }} {symbol}_results;")
    addr_decl = ", ".join(f"p{i}_addr, p{i}_bytes" for i in range(len(call)))
    lines.append(f"/* {symbol}: logical kernel ABI, pointer order {', '.join(b.name for b in order)}. */")
    lines.append(f"{result_t} {symbol}(\n    " + ",\n    ".join(params) + ")\n{")
    lines.append(f"  {result_t} r;\n  memset(&r, 0, sizeof r);")
    if shape_checks:
        lines.append(f"  if ({' || '.join(shape_checks)}) return r;")
        lines.append(f"  if ({' || '.join(stride_checks)}) __builtin_trap();")
    lines.append("  " + " ".join(voids))
    lines.append(f"  uintptr_t {addr_decl};")
    guard = " ||\n      ".join(spans + disjoint)
    lines.append(f"  if ({guard})\n    __builtin_trap();")
    if batch is None:
        args = ", ".join(f"(void *)p{index_of[buf.role]}_addr" for buf in order)
        lines.append(f"  {kernel}({args});")
    else:
        args = ", ".join(
            f"(void *)(p{index_of[buf.role]}_addr + (uintptr_t)slice * (p{index_of[buf.role]}_bytes / {batch}u))"
            for buf in order
        )
        lines.append(f"  for (long slice = 0; slice < {batch}; ++slice)\n    {kernel}({args});")
    for k, o in enumerate(outputs):
        target = "r" if single else f"r.r{k}"
        lines.append(f"  {target}.allocated = p{o}_alloc; {target}.aligned = p{o}_aligned; {target}.offset = p{o}_off;")
        for d in range(len(call[o].shape)):
            lines.append(f"  {target}.sizes[{d}] = p{o}_s{d}; {target}.strides[{d}] = p{o}_st{d};")
    lines.append("  return r;\n}")
    descriptors = ", ".join(f"const merlin_lmemref_{len(buf.shape)}d *d{i}" for i, buf in enumerate(call))
    fields: list[str] = []
    for i, buf in enumerate(call):
        fields += [f"d{i}->allocated", f"d{i}->aligned", f"d{i}->offset"]
        fields += [f"d{i}->sizes[{d}]" for d in range(len(buf.shape))]
        fields += [f"d{i}->strides[{d}]" for d in range(len(buf.shape))]
    lines.append(
        f"void _mlir_ciface_{symbol}({result_t} *result, {descriptors})\n"
        f"{{\n  *result = {symbol}(\n      " + ",\n      ".join(fields) + ");\n}"
    )
    return "\n".join(lines) + "\n"


def emit_logical_translation_unit(
    device: str,
    symbols: Mapping[str, tuple[Sequence[LogicalBuffer], Sequence[Mapping[str, Any]]]],
    *,
    kernel_symbol_for: Callable[[str], str],
    descriptor_results: bool = False,
) -> ShimUnit:
    """One entry per symbol, passing EVERY pointer of the kernel's logical interface.

    ``symbols[s] = (buffers, call)``: the kernel's logical buffers (:func:`logical_buffers`, kernel
    order) and the host call's operands in call order, each ``{"role", "shape", "dtype"}`` -- what the
    rewrite that minted the call supplies (activation, stored weight, folded bias, destinations). Each
    buffer is matched to the call operand with its role; an interface value no operand supplies, or one
    whose host operand has another shape or dtype, is DECLINED by name, never called with a guess.
    """
    from merlin.compiler.lower.device.shim import _RANK2_POINTER_GUARD, ShimUnit, descriptor_result_marker

    if type(descriptor_results) is not bool:
        raise ValueError("descriptor results require an explicit Boolean choice")
    entries: list[str] = []
    identities = []
    bridges = []
    emitted: list[str] = []
    skipped: list[tuple[str, str]] = []
    ranks: set[int] = set()
    kernels: dict[str, int] = {}
    for sym in sorted(symbols):
        buffers, call = symbols[sym]
        by_role = {buf.role: buf for buf in buffers}
        supplied = {str(op.get("role")): op for op in call}
        missing = [buf for buf in buffers if buf.role not in supplied]
        if missing:
            skipped.append(
                (sym, "; ".join(f"interface value {b.name!r} ({b.role}) has no host source" for b in missing))
            )
            continue
        extra = [role for role in supplied if role not in by_role]
        if extra or len(supplied) != len(call):
            skipped.append(
                (sym, f"the host call passes {sorted(extra) or 'duplicate'} roles the interface does not take")
            )
            continue
        host = {r: tuple(int(v) for v in op.get("shape") or ()) for r, op in supplied.items()}
        # A BATCH IS A LOOP over disjoint slices: every host buffer is the same count of the kernel's
        # logical tensor, so the shim calls the one per-slice kernel once per slice.
        leading = {shape[0] for r, shape in host.items() if len(shape) == len(by_role[r].shape) + 1}
        batch = (
            next(iter(leading))
            if len(leading) == 1
            and all(
                len(shape) == len(by_role[r].shape) + 1 and shape[1:] == by_role[r].shape for r, shape in host.items()
            )
            else None
        )
        disagree = [
            f"interface value {by_role[r].name!r} is {list(by_role[r].shape)} {by_role[r].dtype}, "
            f"the host supplies {list(host[r])} {op.get('dtype')}"
            for r, op in supplied.items()
            if (batch is None and host[r] != by_role[r].shape) or str(op.get("dtype")) != by_role[r].dtype
        ]
        if disagree:
            skipped.append((sym, "; ".join(disagree)))
            continue
        if not any(buf.kind == "output" for buf in buffers):
            skipped.append((sym, "the interface declares no result"))
            continue
        call_buffers = [replace(by_role[str(op["role"])], shape=host[str(op["role"])]) for op in call]
        ranks.update(len(buf.shape) for buf in call_buffers)
        kernel = kernel_symbol_for(sym)
        kernels[kernel] = len(buffers)
        entries.append(_logical_entry(sym, kernel, call_buffers, buffers, batch))
        emitted.append(sym)
        if descriptor_results:
            from xdsl.dialects.builtin import IntegerType, TensorType

            from merlin.compiler.lower.options.descriptor_result_identity import DescriptorResultIdentity

            outputs = [i for i, buf in enumerate(call_buffers) if buf.kind == "output"]
            if (
                len(outputs) == 1
                and 1 <= len(call_buffers) <= 128
                and all(
                    len(buf.shape) <= 8 and all(d > 0 for d in buf.shape) and buf.dtype in ("i8", "i16", "i32", "i64")
                    for buf in call_buffers
                )
            ):
                identity = DescriptorResultIdentity(
                    sym,
                    sym + "_borrowed_result",
                    sym + "_result_identity",
                    outputs[0],
                    tuple(TensorType(IntegerType(int(buf.dtype[1:])), buf.shape) for buf in call_buffers),
                    tuple("write" if i == outputs[0] else "read" for i in range(len(call_buffers))),
                )
                identities.append(identity)
                bridges.append(_logical_borrowed_result_wrapper(identity))
    if not emitted:
        return ShimUnit(text="", skipped=tuple(skipped))
    typedefs = "\n".join(_memref_typedef(rank) for rank in sorted(ranks))
    externs = "\n".join(
        f"extern void {k}({', '.join(f'void *arg_{i}' for i in range(n))});" for k, n in sorted(kernels.items())
    )
    text = (
        f"/* Generated by merlin.compiler.lower.device.shim for device {device!r} (logical kernel ABI). Do not edit.\n"
        " * Each entry hands the kernel every pointer of its logical interface, dense row-major, in the\n"
        " * interface's own order; the kernels are supplied by the archive. */\n"
        "#include <stdint.h>\n#include <string.h>\n\n"
        f"{typedefs}\n{_RANK2_POINTER_GUARD}\n{externs}\n\n"
        + "\n".join(entries)
        + f"/* {len(emitted)} entry point(s) emitted. */\n"
    )
    if identities:
        reserved = set(emitted) | set(kernels) | {"_mlir_ciface_" + name for name in emitted}
        for value in identities:
            names = (value.borrowed_symbol, value.wrapper_symbol, "_mlir_ciface_" + value.borrowed_symbol)
            if any(name in reserved for name in names):
                raise ValueError("generated descriptor-result symbol collision")
            reserved.update(names)
        text += descriptor_result_marker(tuple(identities)) + "".join(bridges)
    return ShimUnit(
        text=text,
        symbols=tuple(emitted),
        kernels=tuple(sorted(kernels)),
        skipped=tuple(skipped),
        result_identities=tuple(identities),
    )


def _logical_borrowed_result_wrapper(identity: DescriptorResultIdentity) -> str:
    """Check every field of the actual single output, for mixed ranked arguments."""
    from merlin.compiler.lower.device.shim import _borrowed_result_wrapper

    symbol, borrowed = identity.symbol, identity.borrowed_symbol
    # Reuse the portable symbol validation of the existing descriptor bridge.
    _borrowed_result_wrapper(symbol, 2, borrowed)
    ranks = [len(typ.get_shape()) for typ in identity.argument_types]
    output = identity.result_argument
    descriptors = ", ".join(f"const merlin_lmemref_{rank}d *d{i}" for i, rank in enumerate(ranks))
    fields = []
    for i, rank in enumerate(ranks):
        fields += [f"d{i}->{name}" for name in ("allocated", "aligned", "offset")]
        fields += [f"d{i}->sizes[{axis}]" for axis in range(rank)]
        fields += [f"d{i}->strides[{axis}]" for axis in range(rank)]
    tests = [f"r.{name} != d{output}->{name}" for name in ("allocated", "aligned", "offset")]
    tests += [f"r.sizes[{axis}] != d{output}->sizes[{axis}]" for axis in range(ranks[output])]
    tests += [f"r.strides[{axis}] != d{output}->strides[{axis}]" for axis in range(ranks[output])]
    return (
        f"\n/* Checked logical borrowed result; independent effects remain required. */\n"
        f"void _mlir_ciface_{borrowed}({descriptors})\n{{\n"
        "  if (" + " || ".join(f"!d{i}" for i in range(len(ranks))) + ") __builtin_trap();\n"
        f"  merlin_lmemref_{ranks[output]}d r = {symbol}(\n      "
        + ",\n      ".join(fields)
        + ");\n  if ("
        + " ||\n      ".join(tests)
        + ") __builtin_trap();\n}\n"
    )


def rebind_logical_translation_unit(
    device: str,
    *,
    interfaces: Mapping[str, str],
    call_buffers: Mapping[str, Sequence[Mapping[str, Any]]],
    kernels: Mapping[str, str],
    replacements: Mapping[str, str],
    shim_source,
    shim_object,
    compilation: Mapping[str, Any],
) -> ShimUnit:
    """Reconstruct an actually compiled logical shim with selected interfaces.

    The complete old logical prefix must re-emit byte-for-byte. Any appended
    bridge source is retained only from the freshly verified original C/object
    producer, without reconstructing or granting its source/effect ownership.
    Original host roles, element types, input/output kinds and descriptor order
    stay fixed; target pointer order follows each replacement interface.

    This is a pure source reconstruction, not source semantics, storage effects,
    target support, numerical qualification or a new completed production.
    Callers must independently prove replacement applicability, compile/admit
    its target and shim, relink normally, and qualify the resulting execution.
    No caller callback runs between the original producer checks.
    """
    from pathlib import Path

    from merlin.common.digest import is_sha256, sha256_bytes, sha256_file
    from merlin.common.process import invocation_records as invocation_record
    from merlin.compiler.lower.device.shim import ShimUnit

    def direct(value):
        path = Path(value)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or not path.is_file()
            or any(part.is_symlink() for part in (path, *path.parents))
        ):
            raise ValueError("logical shim replay requires direct original files")
        return path

    if (
        any(type(value) is not dict for value in (interfaces, call_buffers, kernels, replacements, compilation))
        or type(device) is not str
        or not device
        or any(
            type(name) is not str or not name
            for mapping in (interfaces, call_buffers, kernels, replacements)
            for name in mapping
        )
        or not interfaces
        or set(interfaces) != set(call_buffers)
        or set(interfaces) != set(kernels)
        or not set(replacements) <= set(interfaces)
        or any(
            type(value) is not str or not value
            for value in (*interfaces.values(), *kernels.values(), *replacements.values())
        )
        or not {"path", "sha256"} <= set(compilation)
        or set(compilation) - {"path", "sha256", "size_bytes", "bytes"}
        or not is_sha256(compilation["sha256"])
    ):
        raise ValueError("logical shim replay lost its complete original symbol/interface roster")
    calls = {}
    for symbol, buffers in call_buffers.items():
        if (
            type(buffers) not in (tuple, list)
            or not buffers
            or any(
                type(b) is not dict
                or set(b) != {"role", "shape", "dtype"}
                or type(b["role"]) is not str
                or type(b["dtype"]) is not str
                or type(b["shape"]) not in (tuple, list)
                or any(type(d) is not int or d < 1 for d in b["shape"])
                for b in buffers
            )
        ):
            raise ValueError("logical shim replay needs original plain typed call buffers")
        calls[symbol] = tuple({**b, "shape": tuple(b["shape"])} for b in buffers)
    source, obj, producer = direct(shim_source), direct(shim_object), direct(compilation["path"])
    original_bytes = source.read_bytes()

    def require_producer():
        if sha256_file(producer) != compilation["sha256"] or any(
            compilation[key] != producer.stat().st_size for key in ("size_bytes", "bytes") if key in compilation
        ):
            raise ValueError("logical shim original producer changed")
        observed = invocation_record.verify(producer)
        if (
            observed.get("stage") != "device_shim_compilation"
            or observed.get("returncode") != 0
            or observed.get("inputs") != [{"path": str(source), "sha256": sha256_bytes(original_bytes)}]
            or observed.get("outputs") != [{"path": str(obj), "sha256": sha256_file(obj)}]
            or source.read_bytes() != original_bytes
        ):
            raise ValueError("logical shim lost its original compiled source/object custody")

    require_producer()
    groups, next_groups = {}, {}
    for symbol, interface in interfaces.items():
        old, why = logical_buffers(interface)
        if old is None:
            raise ValueError("logical shim original interface declined: " + why)
        new, why = logical_buffers(replacements.get(symbol, interface))
        if new is None:
            raise ValueError("logical shim replacement interface declined: " + why)
        if {b.role: (b.dtype, b.kind) for b in old} != {b.role: (b.dtype, b.kind) for b in new}:
            raise ValueError("logical shim replacement changes original host roles or typed input/output ABI")
        groups[symbol] = (old, calls[symbol])
        next_groups[symbol] = (new, calls[symbol])
    # Only an internal immutable string lookup is passed to the ordinary emitter.
    names = dict(kernels)
    old = emit_logical_translation_unit(device, groups, kernel_symbol_for=names.get)
    new = emit_logical_translation_unit(device, next_groups, kernel_symbol_for=names.get)
    if (
        old.skipped
        or new.skipped
        or set(old.symbols) != set(interfaces)
        or old.symbols != new.symbols
        or old.kernels != new.kernels
    ):
        raise ValueError("logical shim original or replacement layout/descriptor ABI declined")
    prefix = old.text.encode("utf-8")
    if not original_bytes.startswith(prefix):
        raise ValueError("logical shim original complete prefix does not re-emit exactly")
    suffix = original_bytes[len(prefix) :].decode("utf-8")
    require_producer()
    return replace(new, text=new.text + suffix)
