"""Solution-neutral runner-owned harness renderer (the logical kernel ABI, version 2).

A graded package emits a kernel FUNCTION; the runner owns the C caller that embeds the leaf tensors,
calls that function once per measured invocation, and prints the ``OUT``/``METRIC``/``DONE`` protocol
the console parser reads. This module writes that caller for ANY target, from declared data only:

* ``logical_kernel_abi`` in ``merlin/contract/mlir_oot_backend_contract.yaml`` -- the pointer order,
  the pointee layout (dense row-major LOGICAL tensors), the output poison, the readout format;
* the target contract's ``harness_abi.entry_symbol`` / ``cycle_window_metric`` and its
  ``logical_harness`` block -- the host instructions that read the cycle counter and wait for
  outstanding accelerator work (conventions of the host ISA, not of any compiler);
* the command buffer's own tensor declarations, and Merlin's reference semantics for which
  destinations are results (:mod:`merlin.runtime.reference`).

The caller does not pad, gather, repack or reorder anything, and includes no target library header.
Tiling, padding, packing and layout transforms are the candidate compiler's job; the harness hands it
the tensors exactly as the capsule states them, in one declared pointer order.

Every output buffer is filled with a declared poison byte before EVERY invocation (outside the cycle
window), so a kernel that does not write an output -- or wrote it only during the warm-up -- cannot
pass on stale or zero-initialised memory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from typing import Any


class HarnessRenderError(RuntimeError):
    """The buffer cannot be called through the logical ABI, or the declared contract data is incomplete."""


#: Backend modules conventionally expose their renderer's refusal as ``CodegenError``.
CodegenError = HarnessRenderError

_HOST_STORAGE_BYTES = 256 * 1024 * 1024  # host allocation safety cap, not a target capacity
_COHERENT_OUTPUT_DTYPES = frozenset({"i8", "i16", "i32", "i64", "f32"})


# --------------------------------------------------------------------------------------------------
# declared data
# --------------------------------------------------------------------------------------------------
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


@dataclass(frozen=True)
class HostHooks:
    """How the harness enters the kernel and brackets it, for one target."""

    entry_symbol: str
    cycle_window_metric: str | None
    cycle_counter_asm: str
    completion_asm: str
    #: The target this hook set was resolved for, and its optional ``logical_harness.counter_bracket``
    #: declaration (used only when ``MERLIN_HW_COUNTERS`` requests a counter bracket).
    target: str = ""
    counter_bracket: Mapping[str, Any] | None = None

    @property
    def completion(self) -> str:
        return f'__asm__ __volatile__("{self.completion_asm}" ::: "memory");'


def hooks_from_contract(contract: Mapping[str, Any], *, target: str) -> HostHooks:
    from merlin.targetgen.contract.harness_abi import HarnessAbiError, from_contract

    try:
        abi = from_contract(dict(contract or {}), target=target)
    except HarnessAbiError as exc:
        raise HarnessRenderError(str(exc)) from exc
    block = (contract or {}).get("logical_harness")
    if not isinstance(block, dict):
        raise HarnessRenderError(
            f"target {target!r} declares no `logical_harness` block (the host cycle-counter read and "
            f"completion instruction); the runner will not guess either"
        )
    values = {}
    for key in ("cycle_counter_asm", "completion_asm"):
        value = block.get(key)
        if not isinstance(value, str) or not value or '"' in value or "\\" in value or "\n" in value:
            raise HarnessRenderError(f"target {target!r}: logical_harness.{key} must be one plain instruction")
        values[key] = value
    bracket = block.get("counter_bracket")
    return HostHooks(
        abi.entry_symbol,
        abi.cycle_window_metric,
        values["cycle_counter_asm"],
        values["completion_asm"],
        target=target,
        counter_bracket=bracket if isinstance(bracket, Mapping) else None,
    )


_CACHE: dict[tuple, tuple[LogicalAbi, HostHooks]] = {}


def resolve(target: str) -> tuple[LogicalAbi, HostHooks]:
    key = (target, os.environ.get("MERLIN_TARGET_PATH", ""), os.environ.get("MERLIN_TARGET_CONTRACT", ""))
    if key not in _CACHE:
        from merlin.targetgen.target_registry import resolve as resolve_target

        _CACHE[key] = (logical_abi(), hooks_from_contract(resolve_target(target).load_contract(), target=target))
    return _CACHE[key]


# --------------------------------------------------------------------------------------------------
# element containers (the dtype's own storage width; a console transport, not a layout choice)
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Container:
    ctype: str
    cast: str
    conv: str
    word_bytes: int
    signed: bool

    def printf_element(self, expr: str) -> str:
        return f'printf(" {self.conv}", ({self.cast}){expr});'


def container_for(dtype: str) -> Container:
    """Integers in ``int<bits>_t``; floats as their stored bit pattern in ``uint<bits>_t`` (decoded
    from the same declared dtype by :func:`merlin.runtime.backends.base.decode_float_readback`)."""
    from merlin.common.quant_formats import storage_bits
    from merlin.runtime.fp8_formats import float_format_of

    try:
        bits = storage_bits(dtype)
    except Exception as exc:  # noqa: BLE001 -- an unregistered dtype is refused by name
        raise HarnessRenderError(f"tensor dtype {dtype!r} has no registered storage width") from exc
    if bits % 8 or bits not in (8, 16, 32, 64):
        from merlin.llvmlower.c_runtime import DT_BYTES

        declared = DT_BYTES.get(dtype)
        if not declared or declared * 8 not in (8, 16, 32, 64):
            raise HarnessRenderError(
                f"dtype {dtype!r} stores {bits} bits per element and the compiler declares no whole-byte "
                f"storage width for it; refusing to guess a stride"
            )
        bits = declared * 8
    if float_format_of(dtype) is not None:
        return Container(
            f"uint{bits}_t",
            "unsigned long long" if bits > 32 else "unsigned",
            "%llu" if bits > 32 else "%u",
            bits // 8,
            False,
        )
    unsigned = dtype.startswith("u")
    if unsigned:
        return Container(
            f"uint{bits}_t",
            "unsigned long long" if bits > 32 else "unsigned",
            "%llu" if bits > 32 else "%u",
            bits // 8,
            False,
        )
    return Container(
        f"int{bits}_t", "long long" if bits > 32 else "int", "%lld" if bits > 32 else "%d", bits // 8, True
    )


def container_words(values, dtype: str) -> list[int]:
    from merlin.runtime.fp8_formats import float_format_of

    fmt = float_format_of(dtype)
    if fmt is None:
        return [int(v) for v in values]
    from merlin.runtime import fp8_formats

    return [int(c) for c in fp8_formats.float_to_codes(list(values), fmt)]


# --------------------------------------------------------------------------------------------------
# the logical interface of a command buffer
# --------------------------------------------------------------------------------------------------
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
    from merlin.runtime.commandbuffer import harness_derived_tensors

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
    from merlin.runtime.commandbuffer import declared_output_dtypes

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
            from merlin.targetgen.contract.interface_emit import _commit_out_shape

            dtype = attrs.get("output_dtype")
            if not dtype:
                raise HarnessRenderError(f"COMMIT {name!r} declares no output_dtype")
            return Buffer(name, "output", tuple(_commit_out_shape(cb, ops["src"], attrs)), str(dtype))
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


# --------------------------------------------------------------------------------------------------
# C fragments
# --------------------------------------------------------------------------------------------------
def _aligned(abi: LogicalAbi) -> str:
    return f"__attribute__((aligned({abi.alignment_bytes})))"


def _initializer(words) -> str:
    return ",".join(str(int(w)) for w in words)


def _input_decl(abi: LogicalAbi, buf: Buffer, words: list[int], blobs: dict | None) -> str:
    container = container_for(buf.dtype)
    symbol = f"T_{buf.name}"
    if blobs is None or abi.blob_min_elements is None or len(words) < abi.blob_min_elements:
        return f"static const {container.ctype} {symbol}[{len(words)}] {_aligned(abi)} = {{{_initializer(words)}}};"
    blobs[symbol] = {
        "bytes": b"".join(int(w).to_bytes(container.word_bytes, "little", signed=container.signed) for w in words),
        "align": abi.alignment_bytes,
        "elems": len(words),
    }
    return f"extern const {container.ctype} {symbol}[{len(words)}];"


def _preamble(hooks: HostHooks, nargs: int, extra_includes: str = "", *, symbol: str | None = None) -> str:
    params = ", ".join("void *" for _ in range(nargs)) or "void"
    return (
        "#include <stdint.h>\n#include <stdio.h>\n"
        + extra_includes
        + "static inline uint64_t merlin_read_cycles(void) {\n"
        + f'  uint64_t value;\n  __asm__ __volatile__("{hooks.cycle_counter_asm}" : "=r"(value));\n  return value;\n'
        + "}\n"
        + f"extern void {symbol or hooks.entry_symbol}({params});\n"
    )


def _cache_state_requested() -> str:
    state = str(os.environ.get("MERLIN_CACHE_STATE", "cold")).strip().lower()
    if state not in ("cold", "warm"):
        raise HarnessRenderError(f"unsupported cache-state measurement condition {state!r}")
    return state


def _counter_fragments(hooks: HostHooks, *, supported: bool = True) -> dict | None:
    """The header-free counter bracket when ``MERLIN_HW_COUNTERS`` requests one, else None.

    Built from the target's RTL facts and contract data (:mod:`merlin.targetgen.contract.counter_bracket`).
    A requested bracket that cannot be derived REFUSES the render: an uninstrumented run must never be
    read as an instrumented one."""
    from merlin.targetgen.contract import counter_bracket as CB

    if not CB.counters_requested():
        return None
    if not supported:
        raise HarnessRenderError("MERLIN_HW_COUNTERS is supported only by the standard measured caller")
    try:
        return CB.bracket_for_target(hooks.target, hooks.counter_bracket, unit=CB.unit_requested())
    except CB.CounterBracketError as exc:
        from merlin.common.path_scrub import scrub_host_paths

        raise HarnessRenderError(
            f"requested counter instrumentation unavailable: {scrub_host_paths(str(exc))}"
        ) from exc


# --------------------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------------------
def render_harness(
    cb: dict,
    *,
    target: str,
    inputs: dict | None = None,
    prepack_authorizations=None,
    compact_caller=None,
    warm_profile=None,
    blobs: dict | None = None,
    source_owned_mutables=None,
    readback_policy=None,
) -> str:
    """Render the runner-owned harness C for ``cb`` on ``target`` (the ``harness_renderer`` capability).

    ``inputs`` (name -> nested list) INJECTS operand values; absent, each leaf is materialized from its
    name exactly as the reference does. ``blobs`` receives large constant inputs as bytes when the
    caller can link them. The remaining keywords are whole-program opt-ins and are refused elsewhere.
    """
    abi, hooks = resolve(target)
    return render_with(
        cb,
        abi=abi,
        hooks=hooks,
        inputs=inputs,
        prepack_authorizations=prepack_authorizations,
        compact_caller=compact_caller,
        warm_profile=warm_profile,
        blobs=blobs,
        source_owned_mutables=source_owned_mutables,
        readback_policy=readback_policy,
    )


def explicit_whole_program(cb: dict, abi: LogicalAbi) -> dict:
    """``cb`` with its default logical boundary written out as the equivalent whole-program ABI.

    The default ABI passes dense input then output buffers in ``abi.default_order``; this names exactly
    those pointers, in that order, as ``kernel_abi.args`` (inputs ``read``, outputs ``write``) and
    declares each output tensor, so the rendered harness is the same program. It exists for opt-ins
    defined only on the explicit boundary (a fixed-layout memory readback). A buffer that already
    declares a kernel ABI is returned unchanged.
    """
    import copy

    if cb.get("kernel_abi"):
        return cb
    buffers = logical_interface(cb, abi)
    out = copy.deepcopy(cb)
    tensors = out.setdefault("tensors", {})
    for buf in buffers:
        if buf.kind == "output":
            tensors[buf.name] = {"shape": list(buf.shape), "dtype": buf.dtype, "role": "output"}
        else:
            tensors[buf.name] = {**tensors[buf.name], "dtype": buf.dtype}
    out["kernel_abi"] = {
        "kind": "whole_program",
        "args": [{"tensor": b.name, "access": "write" if b.kind == "output" else "read"} for b in buffers],
        "outputs": [b.name for b in buffers if b.kind == "output"],
    }
    return out


def _memory_transport(policy) -> bool:
    from merlin.targetgen.contract.readback_policy import MEMORY_TRANSPORTS, selected

    return selected(policy).transport in MEMORY_TRANSPORTS


def render_with(
    cb: dict,
    *,
    abi: LogicalAbi,
    hooks: HostHooks,
    inputs: dict | None = None,
    prepack_authorizations=None,
    compact_caller=None,
    warm_profile=None,
    blobs: dict | None = None,
    source_owned_mutables=None,
    readback_policy=None,
    cache_state: str | None = None,
) -> str:
    """:func:`render_harness` against explicitly supplied contract data (no registry resolution).

    ``cache_state`` (``cold``/``warm``) overrides the ``MERLIN_CACHE_STATE`` environment selection."""
    whole_program = (cb.get("kernel_abi") or {}).get("kind") == "whole_program"
    if source_owned_mutables is not None and not whole_program:
        raise HarnessRenderError("source-owned scratch requires an explicit whole-program ABI")
    if readback_policy is not None and not whole_program and _memory_transport(readback_policy):
        # The console transports (B64/BIN) read back the logical interface's own output buffers, so the
        # default logical ABI carries them; a memory dump admits a fixed whole-program output layout.
        raise HarnessRenderError("memory readback requires an explicit whole-program kernel ABI")
    if compact_caller is not None:
        if readback_policy is not None:
            raise HarnessRenderError("full-value readback cannot use a prepared compact caller")
        if inputs is not None or prepack_authorizations is not None or source_owned_mutables is not None:
            raise HarnessRenderError("compact caller accepts only its already validated explicit byte inputs")
        _counter_fragments(hooks, supported=False)
        return _compact_caller(cb, compact_caller, abi=abi, hooks=hooks, warm_profile=warm_profile)
    if prepack_authorizations is not None and not whole_program:
        raise HarnessRenderError("host prepack authorization requires the explicit whole-program caller")

    from merlin.runtime.commandbuffer import CONSOLE_VALUE_CAP_PARAM, materialize_inputs
    from merlin.runtime.storage_binding import resolve_storage_bindings

    buffers = logical_interface(cb, abi)
    scratch = frozenset()
    if source_owned_mutables is not None:
        scratch = _check_scratch(cb, buffers, inputs, source_owned_mutables)
    storage = None
    if whole_program:
        try:
            storage = resolve_storage_bindings(
                cb, inputs, max_storage_bytes=_HOST_STORAGE_BYTES, prepack_authorizations=prepack_authorizations
            )
        except ValueError as exc:
            error = HarnessRenderError(str(exc))
            if hasattr(exc, "storage_obligations"):
                error.storage_obligations = exc.storage_obligations
            raise error from exc
    leaves = materialize_inputs(cb, inputs) if storage is None else None

    decls: list[str] = []
    prepare: list[str] = []  # emitted before EVERY invocation, outside the cycle window
    for buf in buffers:
        symbol = f"T_{buf.name}"
        if storage is not None:
            binding = storage[buf.name]
            container = container_for(binding.encoding.dtype)
            elements = binding.encoding.storage_elements
            values = None
            if buf.kind in ("input", "inout") and buf.name not in scratch and binding.logical_values is not None:
                values = binding.pack_words(container_words(binding.logical_values, binding.encoding.dtype))
        else:
            container = container_for(buf.dtype)
            elements = buf.elements
            values = None
            if buf.kind in ("input", "inout") and buf.name not in scratch:
                if buf.name not in leaves:
                    raise HarnessRenderError(f"input {buf.name!r} was not materialized")
                data = list(leaves[buf.name].data)
                if len(data) != elements:
                    raise HarnessRenderError(f"input {buf.name!r} has {len(data)} values for shape {list(buf.shape)}")
                values = container_words(data, buf.dtype)
        if buf.kind == "input" and values is not None:
            if storage is None:
                decls.append(_input_decl(abi, buf, values, blobs))
            else:
                decls.append(
                    f"static const {container.ctype} {symbol}[{elements}] {_aligned(abi)} = {{{_initializer(values)}}};"
                )
            continue
        decls.append(f"static {container.ctype} {symbol}[{elements}] {_aligned(abi)};")
        if values is not None:  # mutable input: restored from its initial value before every call
            decls.append(f"static const {container.ctype} {symbol}__initial[{elements}] = {{{_initializer(values)}}};")
            prepare.append(f"  __builtin_memcpy({symbol}, {symbol}__initial, sizeof({symbol}));")
        elif buf.kind == "output":
            prepare.append(f"  __builtin_memset({symbol}, {abi.poison_byte}, sizeof({symbol}));")

    call_args = ", ".join(f"(void*){f'T_{b.name}'}" for b in buffers)
    invoke = f"{hooks.entry_symbol}({call_args});"
    value_cap = (cb.get("params") or {}).get(CONSOLE_VALUE_CAP_PARAM)
    if value_cap is not None and (isinstance(value_cap, bool) or not isinstance(value_cap, int) or value_cap < 1):
        raise HarnessRenderError(f"params.{CONSOLE_VALUE_CAP_PARAM} must be a positive integer, got {value_cap!r}")
    readout = _Readout(cb, buffers, storage, readback_policy, value_cap)
    prints = readout.lines()
    decls.extend(readout.declarations())
    includes = readout.includes()

    if warm_profile is not None:
        _counter_fragments(hooks, supported=False)
        if readout.packet:
            raise HarnessRenderError("coherent packet requires the declared standard measurement")
        from merlin.perf.warm_profile_harness import (
            TargetInvocationHooks,
            render_warm_then_measure_main,
            require_strict_final_warm_profile,
        )

        reset = "\n".join(line.strip() for line in prepare) or None
        main = render_warm_then_measure_main(
            prepare_input=reset or "/* ABI inputs are statically initialized before main. */",
            invocation=TargetInvocationHooks(invoke=invoke, complete=hooks.completion),
            reset_after_warm=reset,
            validate_outputs="0",  # OUT parsing + golden validation are runner-owned.
            contract=require_strict_final_warm_profile(warm_profile),
            cycle_reader="merlin_read_cycles",
            success_body="\n".join(prints) + ("\n" if prints else "") + 'printf("DONE\\n");',
        )
        return _preamble(hooks, len(buffers), includes) + "\n".join(decls) + "\n" + main

    if cache_state not in (None, "cold", "warm"):
        raise HarnessRenderError(f"unsupported cache-state measurement condition {cache_state!r}")
    warm = (cache_state or _cache_state_requested()) == "warm"
    counters = _counter_fragments(hooks)
    if counters is not None:
        includes += counters["helper"]
    body = ["int main(void) {"]
    body += readout.setup()
    if warm:
        body += prepare + [
            f"  {invoke}",
            f"  {hooks.completion}",
            "  // merlin: warmup completed outside the measured window.",
        ]
    body += prepare
    if counters is not None:
        body += counters["prologue"]
    body += [
        "  uint64_t c0 = merlin_read_cycles();",
        f"  {invoke}",
        f"  {hooks.completion}",
        "  uint64_t c1 = merlin_read_cycles();",
        '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
    ]
    if counters is not None:
        body += counters["epilogue"]
    if hooks.cycle_window_metric:
        body.append(f'  printf("METRIC {hooks.cycle_window_metric} 1\\n");')
    body += prints
    body += readout.finish()
    body += ['  printf("DONE\\n");', "  return 0;", "}"]
    return _preamble(hooks, len(buffers), includes) + "\n".join(decls) + "\n" + "\n".join(body) + "\n"


def _check_scratch(cb, buffers, inputs, names) -> frozenset[str]:
    if (
        type(names) is not tuple
        or any(type(n) is not str or not n for n in names)
        or len(set(names)) != len(names)
        or not isinstance(inputs, dict)
    ):
        raise HarnessRenderError("source-owned scratch requires a distinct whole-program binding roster")
    tensors = cb.get("tensors") or {}
    scratch = frozenset(names)
    expected = {
        b.name
        for b in buffers
        if b.kind in ("scratch", "inout") and (tensors.get(b.name) or {}).get("role") == "intermediate"
    }
    if set(inputs) != {b.name for b in buffers if b.kind == "input"} or scratch != expected:
        raise HarnessRenderError("source-owned scratch differs from the exact pointer and entry ABI")
    return scratch


class _Readout:
    """The output transport: ``OUT`` lines (default), an ``OUTSUM`` digest above the console cap, or an
    explicitly selected full-value policy. Every buffer is dense, so element ``i*cols+j`` is simply
    index ``i*cols+j`` unless the candidate declared an explicit storage encoding."""

    def __init__(self, cb, buffers, storage, policy, value_cap):
        from merlin.targetgen.contract.readback_policy import (
            COHERENT_DUMP_V1,
            COHERENT_PACKET_V1,
            FULL_VALUES_B64,
            FULL_VALUES_BIN,
            OUT_DIGEST_V1,
            selected,
        )

        policy = selected(policy)
        self.digest = policy is not None and policy.transport == OUT_DIGEST_V1
        self.cb, self.storage, self.value_cap = cb, storage, value_cap
        self.outputs = [b for b in buffers if b.kind == "output"]
        self.b64 = policy is not None and policy.transport == FULL_VALUES_B64
        self.binary = policy is not None and policy.transport == FULL_VALUES_BIN
        self.packet = policy is not None and policy.transport == COHERENT_PACKET_V1
        self.coherent = policy is not None and policy.transport in (COHERENT_DUMP_V1, COHERENT_PACKET_V1)
        self.digested = False
        self.packet_expected: list = []
        self.packet_capacity = None
        if self.coherent:
            self._check_coherent(buffers)
        self._lines = []
        for buf in self.outputs:
            rows, cols = buf.matrix
            if storage is not None:
                binding = storage[buf.name]
                shape = binding.encoding.logical_shape
                rows, cols = (prod(shape[:-1]), shape[-1]) if shape else (1, 1)
                terms = [str(binding.encoding.offset_elements)]
                for divisor, extent, stride in binding.logical_offset_terms():
                    terms.append(f"(((i * {cols} + j) / {divisor}) % {extent}) * {stride}")
                element = f"T_{buf.name}[{' + '.join(terms)}]"
                container = container_for(binding.encoding.dtype)
                contiguous = _contiguous_output_pointer(buf.name, binding)
            else:
                element = f"T_{buf.name}[i * {cols} + j]"
                container = container_for(buf.dtype)
                contiguous = f"T_{buf.name}"
            self._lines += self._one(buf, rows, cols, container, element, contiguous)
        if self.packet:
            from merlin.runtime.out_packet import out_bin_packet_capacity

            try:
                self.packet_capacity = out_bin_packet_capacity(self.packet_expected)
            except ValueError as exc:
                raise HarnessRenderError(str(exc)) from exc

    def _check_coherent(self, buffers):
        kabi = self.cb.get("kernel_abi") or {}
        names = [b.name for b in self.outputs]
        tensors = self.cb.get("tensors") or {}
        if (
            kabi.get("kind") != "whole_program"
            or not names
            or len(names) > 1024
            or any(not n.isascii() or not all(c.isalnum() or c == "_" for c in n) for n in names)
        ):
            raise HarnessRenderError("coherent dump requires a closed whole-program output roster")
        if any(
            tensors[n].get("role") != "output" or tensors[n].get("dtype") not in _COHERENT_OUTPUT_DTYPES for n in names
        ):
            raise HarnessRenderError("coherent dump requires declared writable outputs of supported physical dtype")
        if set(names) != {n for n, s in tensors.items() if isinstance(s, dict) and s.get("role") == "output"}:
            raise HarnessRenderError("coherent dump output roster differs from declared output tensors")
        total = 0
        for buf in self.outputs:
            elements = self.storage[buf.name].encoding.storage_elements if self.storage is not None else buf.elements
            total += elements * container_for(buf.dtype).word_bytes
            if total > _HOST_STORAGE_BYTES:
                raise HarnessRenderError("coherent dump exceeds the bounded output region total")

    def _one(self, buf, rows, cols, container, element, contiguous):
        name = buf.name
        loop = f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++)"
        sign = 1 if container.signed else 0
        scan = "signed" if container.signed else "unsigned"
        scan_type = "int64_t" if container.signed else "uint64_t"
        count = rows * cols
        if self.coherent and not self.packet:
            return []
        if self.digest:
            # One XXH64 of the output's dense bytes, computed by this grader-rendered harness over the
            # buffer the candidate wrote (merlin/runtime/baremetal/out_digest.h).
            if contiguous is None or not name.isascii() or not all(c.isalnum() or c == "_" for c in name):
                raise HarnessRenderError("an output digest needs a dense, closed-identifier output buffer")
            nbytes = count * container.word_bytes
            return [
                f'  printf("OUT_DIGEST {name} %lu %016lx\\n", (unsigned long){nbytes}UL, '
                f"(unsigned long)merlin_out_digest({contiguous}, {nbytes}ULL));"
            ]
        if (self.b64 or self.binary or self.packet) and (
            not name.isascii() or not all(c.isalnum() or c == "_" for c in name)
        ):
            raise HarnessRenderError("packed output name is not a closed ASCII identifier")
        if self.packet:
            from merlin.runtime.out_packet import ExpectedOutput

            self.packet_expected.append(
                ExpectedOutput(
                    name=name,
                    rows=rows,
                    cols=cols,
                    logical_dtype=buf.dtype,
                    container_signed=container.signed,
                    container_max_bytes=container.word_bytes,
                    logical_shape=tuple(buf.shape),
                )
            )
            pack = (
                f"  if (!merlin_out_bin_words_i32(&merlin_out, {contiguous}, {count}ULL)) return 2;"
                if container.ctype == "int32_t" and contiguous is not None
                else f"{loop} if (!merlin_out_bin_word(&merlin_out, (uint64_t)({element}))) return 2;"
            )
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {count}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "OUT_BIN_BEGIN v1 {name} ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, {rows}ULL)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, " ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, {cols}ULL)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, " ")) return 2;',
                "  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, merlin_wire_width)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, merlin_wire_signed ? " s " : " u ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, {count}ULL * merlin_wire_width)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "\\n")) return 2;',
                f"  merlin_out_bin_init(&merlin_out, {count}ULL, merlin_wire_width, "
                + "merlin_wire_signed, merlin_packet_write);",
                pack,
                "  if (!merlin_out_bin_finish(&merlin_out)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "OUT_BIN_END v1 ")) return 2;',
                "  if (!merlin_out_bin_memory_hex16(&merlin_packet_sink, merlin_out.checksum)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "\\n")) return 2;',
                "  }",
            ]
        if self.b64:
            literal = f"OUT_B64_BEGIN v1 {name} {rows} {cols} 8 s\\n"
            header = f"OUT_B64_BEGIN v1 {name} {rows} {cols} %u %c\\n"
            pack = (
                f"  if (!merlin_out_b64_words_i32(&merlin_out, {contiguous}, {count}ULL)) return 2;"
                if container.ctype == "int32_t" and contiguous is not None
                else f"{loop} if (!merlin_out_b64_word(&merlin_out, (uint64_t)({element}))) return 2;"
            )
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {count}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  char merlin_begin[sizeof("{literal}")];',
                f"  sprintf(merlin_begin, \"{header}\", merlin_wire_width, merlin_wire_signed ? 's' : 'u');",
                "  printstr(merlin_begin);",
                f"  merlin_out_b64_init(&merlin_out, {count}ULL, merlin_wire_width, merlin_wire_signed, printstr);",
                pack,
                "  if (!merlin_out_b64_finish(&merlin_out)) return 2;",
                '  printstr("OUT_B64_END\\n");',
                "  }",
            ]
        if self.binary:
            biggest = f"OUT_BIN_BEGIN v1 {name} {rows} {cols} 8 s {count * 8}\\n"
            header = f"OUT_BIN_BEGIN v1 {name} {rows} {cols} %u %c %llu\\n"
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {count}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  char merlin_begin[sizeof("{biggest}")];',
                f'  sprintf(merlin_begin, "{header}", merlin_wire_width, '
                f"merlin_wire_signed ? 's' : 'u', (unsigned long long)({count}ULL * merlin_wire_width));",
                "  printstr(merlin_begin);",
                f"  merlin_out_bin_init(&merlin_out, {count}ULL, merlin_wire_width, merlin_wire_signed, printbuf);",
                f"{loop} if (!merlin_out_bin_word(&merlin_out, (uint64_t)({element}))) return 2;",
                "  if (!merlin_out_bin_finish(&merlin_out)) return 2;",
                '  char merlin_end[sizeof("OUT_BIN_END v1 0123456789abcdef\\n")];',
                '  sprintf(merlin_end, "OUT_BIN_END v1 %016llx\\n", (unsigned long long)merlin_out.checksum);',
                "  printstr(merlin_end);",
                "  }",
            ]
        from merlin.runtime.commandbuffer import OUTPUT_DIGEST_LINE

        print_element = container.printf_element(element)
        if self.value_cap is None or count <= self.value_cap:
            return [f'  printf("OUT {name} {rows} {cols}");', f"{loop} {print_element}", '  printf("\\n");']
        self.digested = True
        return [
            "  merlin_outsum = 1469598103934665603ULL;",
            f"{loop} MERLIN_OUTSUM_ADD({print_element[len('printf(') :]}",
            f'  printf("{OUTPUT_DIGEST_LINE} {name} {rows} {cols} %016llx\\n", merlin_outsum);',
        ]

    def lines(self) -> list[str]:
        return list(self._lines)

    def includes(self) -> str:
        if self.digest:
            return '#include "out_digest.h"\n'
        if self.b64:
            return '#include "out_b64.h"\nextern void printstr(const char *);\n'
        if self.binary:
            return (
                '#include "out_b64.h"\n#include "out_bin.h"\n'
                "extern void printstr(const char *);\nextern int printbuf(const void *, size_t);\n"
            )
        if self.packet:
            return '#include "out_b64.h"\n#include "out_bin.h"\n#include "out_bin_memory.h"\n'
        return ""

    def declarations(self) -> list[str]:
        decls: list[str] = []
        if self.coherent:
            # Propose (never assert) the span the outputs occupy; the trusted post-link reader must
            # prove the output symbols tile it exactly, or refuse.
            names = [b.name for b in self.outputs]
            first, last = names[-1], names[0]
            buf = self.outputs[0]
            elements = self.storage[last].encoding.storage_elements if self.storage is not None else buf.elements
            decls.append(
                '__asm__(".globl begin_signature\\n"\n'
                f'        ".set begin_signature, T_{first}\\n"\n'
                '        ".globl end_signature\\n"\n'
                f'        ".set end_signature, T_{last}+{elements * container_for(buf.dtype).word_bytes}\\n");'
            )
        if self.b64:
            decls += ["static merlin_out_b64 merlin_out;", "static merlin_out_b64_range merlin_out_range;"]
        if self.binary:
            decls += ["static merlin_out_bin merlin_out;", "static merlin_out_b64_range merlin_out_range;"]
        if self.packet:
            decls += [
                f"unsigned char merlin_readback_packet[{self.packet_capacity}] __attribute__((used));",
                "volatile uint64_t merlin_readback_packet_used __attribute__((used));",
                "static merlin_out_bin_memory merlin_packet_sink;",
                "static merlin_out_bin merlin_out;",
                "static merlin_out_b64_range merlin_out_range;",
                "static int merlin_packet_write(const void *bytes, size_t count) {",
                "  return merlin_out_bin_memory_append(&merlin_packet_sink, bytes, count);",
                "}",
            ]
        if self.digested:
            decls.append(
                "static unsigned long long merlin_outsum;\n"
                "#define MERLIN_OUTSUM_ADD(...) do { char b_[48]; int n_ = sprintf(b_, __VA_ARGS__); "
                "for (int k_ = 0; k_ < n_; k_++) merlin_outsum = (merlin_outsum ^ (unsigned char)b_[k_]) "
                "* 1099511628211ULL; } while (0)"
            )
        return decls

    def setup(self) -> list[str]:
        if not self.packet:
            return []
        return [
            f"  merlin_out_bin_memory_init(&merlin_packet_sink, merlin_readback_packet, {self.packet_capacity}u, "
            "&merlin_readback_packet_used);",
            "  if (!merlin_packet_sink.valid) return 2;",
        ]

    def finish(self) -> list[str]:
        if not self.packet:
            return []
        return [
            '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "DONE\\n")) return 2;',
            '  __asm__ __volatile__("fence rw,rw" ::: "memory");',
            "  if (!merlin_out_bin_memory_finish(&merlin_packet_sink)) return 2;",
            '  __asm__ __volatile__("fence rw,rw" ::: "memory");',
        ]


def _contiguous_output_pointer(name: str, binding) -> str | None:
    encoding = binding.encoding
    shape = tuple(encoding.logical_shape)
    strides = tuple(encoding.logical_strides_elements)
    expected = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
    offset = encoding.offset_elements
    if (
        any(type(d) is not int or d <= 0 for d in shape)
        or strides != expected
        or type(offset) is not int
        or offset < 0
        or prod(shape) <= 0
        or prod(shape) > encoding.storage_elements - offset
    ):
        return None
    return f"T_{name} + {offset}"


# --------------------------------------------------------------------------------------------------
# prepared compact caller (an input mode, not a command shape)
# --------------------------------------------------------------------------------------------------
def _identifier(name) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and name.isascii()
        and (name[0].isalpha() or name[0] == "_")
        and all(c.isalnum() or c == "_" for c in name)
    )


def _compact_caller(cb, prepared, *, abi: LogicalAbi, hooks: HostHooks, warm_profile) -> str:
    """One warm call, restore mutable initial bytes, one measured call, typed readback.

    ``prepared`` is the selected backend's trusted host preparation (``prepare_compact_caller``); it
    re-verifies its identity against ``cb`` and is read only through ``compact_contract_json``,
    ``binding.base_views()`` and ``derived.target_abi.base_access`` / ``.base_alignments``.
    """
    from merlin.perf.storage_encoding import GroupedAxesStorage

    verify = getattr(prepared, "verify_identity", None)
    if not callable(verify) or not hasattr(prepared, "compact_contract_json"):
        raise HarnessRenderError("exact host-prepared compact caller required")
    verify(cb)
    contract = json.loads(prepared.compact_contract_json)
    symbol = contract["compact_symbol"]
    if not _identifier(symbol):
        raise HarnessRenderError("compact symbol is not a C identifier")
    declarations, restore = [], []
    arenas = prepared.binding.base_views()
    target_abi = prepared.derived.target_abi
    for i, (arena, access, alignment) in enumerate(
        zip(arenas, target_abi.base_access, target_abi.base_alignments, strict=True)
    ):
        name = f"merlin_compact_base_{i}"
        initializer = ",".join(str(v) for v in arena) if access != "write" else "0"
        declarations.append(
            f"{'const ' if access == 'read' else ''}unsigned char {name}[{len(arena)}] "
            f"__attribute__((aligned({alignment}), used)) = {{{initializer}}};"
        )
        if access == "readwrite":
            declarations.append(
                f"static const unsigned char merlin_compact_initial_{i}[{len(arena)}] = {{{initializer}}};"
            )
            restore.append(f"  __builtin_memcpy({name}, merlin_compact_initial_{i}, {len(arena)});")
        elif access == "write":
            restore.append(f"  __builtin_memset({name}, {abi.poison_byte}, {len(arena)});")
    bindings = sorted(contract["bindings"], key=lambda row: row["argument_index"])
    outputs = set(cb["kernel_abi"]["outputs"])
    prints = []
    for i, arg in enumerate(cb["kernel_abi"]["args"]):
        name = arg["tensor"]
        if not _identifier(name):
            raise HarnessRenderError("compact output names must be safe identifiers")
        if name not in outputs:
            continue
        encoding = GroupedAxesStorage.from_dict(cb["params"]["storage_encodings"][name])
        container = container_for(encoding.dtype)
        rows, cols = (
            (prod(encoding.logical_shape[:-1]), encoding.logical_shape[-1]) if encoding.logical_shape else (1, 1)
        )
        terms = [str(encoding.offset_elements)]
        divisor = prod(encoding.logical_shape)
        for extent, stride in zip(encoding.logical_shape, encoding.logical_strides_elements, strict=True):
            divisor //= extent
            terms.append(f"(((i * {cols} + j) / {divisor}) % {extent}) * {stride}")
        row = bindings[i]
        address = (
            f"merlin_compact_base_{row['base_index']} + {row['byte_offset']} + "
            f"({' + '.join(terms)}) * sizeof({container.ctype})"
        )
        prints += [
            f'  printf("OUT {name} {rows} {cols}");',
            f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++) {{",
            f"    {container.ctype} value;",
            f"    __builtin_memcpy(&value, {address}, sizeof(value));",
            "    " + container.printf_element("value"),
            "  }",
            '  printf("\\n");',
        ]
    arguments = ", ".join(f"(void*)merlin_compact_base_{i}" for i in range(len(arenas)))
    preamble = _preamble(hooks, len(arenas), symbol=symbol)
    invoke = f"{symbol}({arguments});"
    if warm_profile is not None:
        from merlin.perf.warm_profile_harness import (
            TargetInvocationHooks,
            render_warm_then_measure_main,
            require_strict_final_warm_profile,
        )

        main = render_warm_then_measure_main(
            prepare_input="/* Compact arenas are initialized before main. */",
            invocation=TargetInvocationHooks(invoke=invoke, complete=hooks.completion),
            reset_after_warm=("\n".join(r.strip() for r in restore) if restore else None),
            validate_outputs="0",
            contract=require_strict_final_warm_profile(warm_profile),
            cycle_reader="merlin_read_cycles",
            success_body="\n".join(prints) + ("\n" if prints else "") + 'printf("DONE\\n");',
        )
        return preamble + "\n".join(declarations) + "\n" + main
    body = [
        "int main(void) {",
        f"  {invoke}",
        f"  {hooks.completion}",
        "  // Restore caller initial bytes after the unmeasured warm invocation.",
        *restore,
        "  uint64_t c0 = merlin_read_cycles();",
        f"  {invoke}",
        f"  {hooks.completion}",
        "  uint64_t c1 = merlin_read_cycles();",
        '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
    ]
    if hooks.cycle_window_metric:
        body.append(f'  printf("METRIC {hooks.cycle_window_metric} 1\\n");')
    body += prints + ['  printf("DONE\\n");', "  return 0;", "}"]
    return preamble + "\n".join(declarations) + "\n" + "\n".join(body) + "\n"
