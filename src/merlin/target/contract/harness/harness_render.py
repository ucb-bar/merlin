"""Solution-neutral runner-owned harness renderer (the logical kernel ABI, version 2).

A graded package emits a kernel FUNCTION; the runner owns the C caller that embeds the leaf tensors,
calls that function once per measured invocation, and prints the ``OUT``/``METRIC``/``DONE`` protocol
the console parser reads. This module writes that caller for ANY target, from declared data only:

* ``logical_kernel_abi`` in ``src/merlin/resources/contracts/mlir_oot_backend_contract.yaml`` -- the pointer order,
  the pointee layout (dense row-major LOGICAL tensors), the output poison, the readout format;
* the target contract's ``harness_abi.entry_symbol`` / ``cycle_window_metric`` and its
  ``logical_harness`` block -- the host instructions that read the cycle counter and wait for
  outstanding accelerator work (conventions of the host ISA, not of any compiler);
* the command buffer's own tensor declarations, and Merlin's reference semantics for which
  destinations are results (:mod:`merlin.runtime.engine.reference`).

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

from merlin.compiler.ir.logical_abi import (
    Buffer,
    HarnessRenderError,
    LogicalAbi,
    _ordered,
    _output_buffer,
    _produced,
    _shape,
    kernel_abi_from_commands,
    kernel_arg_order,
    logical_abi,
    logical_inputs,
    logical_interface,
    logical_outputs,
)

# ---- harness_render ----------------------------------------------------------------------------

CodegenError = HarnessRenderError


_HOST_STORAGE_BYTES = 256 * 1024 * 1024  # host allocation safety cap, not a target capacity


_COHERENT_OUTPUT_DTYPES = frozenset({"i8", "i16", "i32", "i64", "f32"})


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
    complete_call_timing: object | None = None

    @property
    def completion(self) -> str:
        return f'__asm__ __volatile__("{self.completion_asm}" ::: "memory");'


def hooks_from_contract(contract: Mapping[str, Any], *, target: str) -> HostHooks:
    from merlin.target.contract.interfaces.abi import HarnessAbiError, from_contract

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
    from merlin.runtime.harness.call_timing import complete_call_plan as complete_timing_from_contract

    timing = complete_timing_from_contract(contract)
    return HostHooks(
        abi.entry_symbol,
        abi.cycle_window_metric,
        values["cycle_counter_asm"],
        values["completion_asm"],
        target=target,
        counter_bracket=bracket if isinstance(bracket, Mapping) else None,
        complete_call_timing=timing,
    )


_CACHE: dict[tuple, tuple[LogicalAbi, HostHooks]] = {}


def resolve(target: str) -> tuple[LogicalAbi, HostHooks]:
    key = (target, os.environ.get("MERLIN_TARGET_PATH", ""), os.environ.get("MERLIN_TARGET_CONTRACT", ""))
    from merlin.runtime.harness.call_timing import _NUMERICS_ONLY

    if _NUMERICS_ONLY.get():
        from merlin.target.registry.identity import resolve as resolve_target

        return logical_abi(), hooks_from_contract(resolve_target(target).load_contract(), target=target)
    if key not in _CACHE:
        from merlin.target.registry.identity import resolve as resolve_target

        _CACHE[key] = (logical_abi(), hooks_from_contract(resolve_target(target).load_contract(), target=target))
    return _CACHE[key]


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
    from the same declared dtype by :func:`merlin.runtime.backends.registry.decode_float_readback`)."""
    from merlin.common.fp8_formats import float_format_of
    from merlin.common.target_data import storage_bits

    try:
        bits = storage_bits(dtype)
    except Exception as exc:  # noqa: BLE001 -- an unregistered dtype is refused by name
        raise HarnessRenderError(f"tensor dtype {dtype!r} has no registered storage width") from exc
    if bits % 8 or bits not in (8, 16, 32, 64):
        from merlin.compiler.lower.c.runtime import DT_BYTES

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
    from merlin.common.fp8_formats import float_format_of

    fmt = float_format_of(dtype)
    if fmt is None:
        return [int(v) for v in values]
    from merlin.common import fp8_formats

    return [int(c) for c in fp8_formats.float_to_codes(list(values), fmt)]


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

    Built from the target's RTL facts and contract data (:mod:`merlin.target.contract.harness.counter_bracket`).
    A requested bracket that cannot be derived REFUSES the render: an uninstrumented run must never be
    read as an instrumented one."""
    from merlin.target.contract.harness import counter_bracket as CB

    if not CB.counters_requested():
        return None
    if not supported:
        raise HarnessRenderError("MERLIN_HW_COUNTERS is supported only by the standard measured caller")
    try:
        return CB.bracket_for_target(hooks.target, hooks.counter_bracket, unit=CB.unit_requested())
    except CB.CounterBracketError as exc:
        from merlin.common.access import scrub_host_paths

        raise HarnessRenderError(
            f"requested counter instrumentation unavailable: {scrub_host_paths(str(exc))}"
        ) from exc


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
    from merlin.target.contract.harness.readback_policy import MEMORY_TRANSPORTS, selected

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
    from merlin.target.contract.harness.readout import _compact_caller, _Readout

    whole_program = (cb.get("kernel_abi") or {}).get("kind") == "whole_program"
    if source_owned_mutables is not None and not whole_program:
        raise HarnessRenderError("source-owned scratch requires an explicit whole-program ABI")
    if readback_policy is not None and not whole_program and _memory_transport(readback_policy):
        # The console transports (B64/BIN) read back the logical interface's own output buffers, so the
        # default logical ABI carries them; a memory dump admits a fixed whole-program output layout.
        raise HarnessRenderError("memory readback requires an explicit whole-program kernel ABI")
    if compact_caller is not None:
        if hooks.complete_call_timing is not None:
            raise HarnessRenderError("compact caller lacks the selected complete-call timing producer")
        if readback_policy is not None:
            raise HarnessRenderError("full-value readback cannot use a prepared compact caller")
        if inputs is not None or prepack_authorizations is not None or source_owned_mutables is not None:
            raise HarnessRenderError("compact caller accepts only its already validated explicit byte inputs")
        _counter_fragments(hooks, supported=False)
        return _compact_caller(cb, compact_caller, abi=abi, hooks=hooks, warm_profile=warm_profile)
    if prepack_authorizations is not None and not whole_program:
        raise HarnessRenderError("host prepack authorization requires the explicit whole-program caller")

    from merlin.runtime.engine.command_buffer import CONSOLE_VALUE_CAP_PARAM, materialize_inputs
    from merlin.runtime.engine.storage_binding import resolve_storage_bindings

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
        if hooks.complete_call_timing is not None:
            raise HarnessRenderError("complete-call timing requires the original single-call boundary")
        _counter_fragments(hooks, supported=False)
        if readout.packet:
            raise HarnessRenderError("coherent packet requires the declared standard measurement")
        from merlin.analysis.perf.measurement.warm_profile_harness import (
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
    timing = hooks.complete_call_timing
    if timing is not None:
        from merlin.runtime.harness.call_timing import CompleteCallTiming

        if type(timing) is not CompleteCallTiming or warm or counters is not None:
            raise HarnessRenderError(
                "complete-call timing requires its original single cold call without nested instrumentation"
            )
        timing.verify()
        decls.extend(timing.declarations(console="stdio"))
    body = ["int main(void) {"]
    if timing is not None:
        body += timing.begin()
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
    if timing is None:
        body += [
            "  uint64_t c0 = merlin_read_cycles();",
            f"  {invoke}",
            f"  {hooks.completion}",
            "  uint64_t c1 = merlin_read_cycles();",
            '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
        ]
    else:
        body += [f"  {invoke}", f"  {hooks.completion}"]
    if counters is not None:
        body += counters["epilogue"]
    if hooks.cycle_window_metric:
        body.append(f'  printf("METRIC {hooks.cycle_window_metric} 1\\n");')
    body += prints
    body += readout.finish()
    if timing is not None:
        body += timing.end(console="stdio")
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
