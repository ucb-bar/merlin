"""Neutral logical-pointer storage and complete readback for a selected contract.

The harness transports semantic tensors in dense row-major order. It does not
lower operations, prepare device layouts, compute answers or select a schedule.
Calling an entry/completion/counter supplied by a contract grants no hardware,
effect, numerical or timing authority.
"""

from __future__ import annotations

import array
import base64
import math
import struct
import sys
from dataclasses import dataclass

from merlin.common.quant_formats import get
from merlin.targetgen.contract.readback_policy import (
    COHERENT_DUMP_V1,
    COHERENT_PACKET_V1,
    COHERENT_PACKET_V2,
    FULL_VALUES_B64,
    ReadbackPolicy,
)

from .direct_kernel_harness import DirectKernelAbi, _identifier, render_direct_kernel


@dataclass(frozen=True)
class LogicalHarnessAbi:
    pointer_abi: DirectKernelAbi
    readback_policy: ReadbackPolicy
    prelude_symbol: str | None


def validate_contract(contract) -> LogicalHarnessAbi:
    """Read only the explicit logical ABI v2; never inherit legacy C snippets."""
    block = contract.get("harness_abi") if type(contract) is dict else None
    required = {
        "version",
        "kind",
        "entry_symbol",
        "fence_symbol",
        "tensor_alignment",
        "byte_order",
        "main_convention",
        "readback_transport",
    }
    if (
        type(block) is not dict
        or not required <= set(block)
        or set(block) - required - {"prelude_symbol"}
        or type(block["version"]) is not int
        or block["version"] != 2
        or block["kind"] != "logical_pointer"
    ):
        raise ValueError("neutral harness requires the complete logical-pointer ABI v2")
    abi = DirectKernelAbi(
        block["entry_symbol"],
        block["fence_symbol"],
        block["tensor_alignment"],
        block["byte_order"],
        block["main_convention"],
    )
    abi.verify()
    prelude = block.get("prelude_symbol")
    if prelude is not None:
        _identifier(prelude)
        if prelude in {abi.entry_symbol, abi.completion_symbol, "main", "console_init", "htif_puts", "htif_exit"}:
            raise ValueError("neutral harness prelude overlaps an entry or console symbol")
    if block["readback_transport"] not in (FULL_VALUES_B64, COHERENT_DUMP_V1):
        raise ValueError("neutral harness requires complete B64 or coherent memory readback")
    return LogicalHarnessAbi(abi, ReadbackPolicy(block["readback_transport"]), prelude)


def _tensor(spec):
    allowed = {"shape", "dtype", "role", "layout", "strides", "offset", "storage_shape", "data", "preload_b64", "base"}
    if type(spec) is not dict or set(spec) - allowed:
        raise ValueError("neutral harness does not accept transformed or ambiguous tensor storage")
    # The ordinary runner records canonical DRAM placement separately from the
    # logical pointer ABI. This annotation never selects an address or transforms
    # harness storage; only the declared logical extent and values are transported.
    if "base" in spec and (type(spec["base"]) is not int or not 0 <= spec["base"] < 2**64):
        raise ValueError("neutral harness has an invalid DRAM placement annotation")
    shape = spec.get("shape")
    if type(shape) is not list or len(shape) > 32 or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError("neutral harness requires positive static logical tensor dimensions")
    if spec.get("layout", "dense_row_major") not in ("dense_row_major", "row_major"):
        raise ValueError("neutral harness requires dense row-major logical storage")
    strides, stride = [], 1
    for dimension in reversed(shape):
        strides.insert(0, stride)
        stride *= dimension
    if "strides" in spec and (
        type(spec["strides"]) is not list
        or any(type(s) is not int for s in spec["strides"])
        or spec["strides"] != strides
    ):
        raise ValueError("neutral harness refuses changed or non-dense logical strides")
    if "offset" in spec and (type(spec["offset"]) is not int or spec["offset"] != 0):
        raise ValueError("neutral harness requires a zero logical storage offset")
    if "storage_shape" in spec and (
        type(spec["storage_shape"]) is not list
        or any(type(s) is not int for s in spec["storage_shape"])
        or spec["storage_shape"] != shape
    ):
        raise ValueError("neutral harness refuses tensor padding")
    if type(spec.get("dtype")) is not str or spec["dtype"].startswith("q"):
        raise ValueError("neutral harness requires an unambiguous scalar storage dtype")
    try:
        dtype = get(spec["dtype"])
    except KeyError as error:
        raise ValueError("neutral harness dtype is not registered") from error
    if dtype.kind not in ("int_affine", "float_ieee") or dtype.element_bits not in (8, 16, 32, 64) or dtype.pack_bits:
        raise ValueError("neutral harness refuses packed or non-scalar storage")
    return tuple(shape), dtype, stride * (dtype.element_bits // 8)


def _roster(cb, *, source_owned_mutables=None):
    if type(cb) is not dict or type(cb.get("tensors")) is not dict or not cb["tensors"]:
        raise ValueError("neutral harness requires a complete logical tensor roster")
    tensors = cb["tensors"]
    params = cb.get("params", {})
    from .commandbuffer import DERIVATION_RECIPE_KEYS, whole_program_entry_bindings

    if type(params) is not dict or any(params.get(key) for key in DERIVATION_RECIPE_KEYS):
        raise ValueError("neutral harness cannot derive operands or prepare a device layout")
    for name in tensors:
        _identifier(name)
    kernel = cb.get("kernel_abi")
    if "kernel_abi" not in cb:
        reads, writes = [], []
        for name, spec in tensors.items():
            role = spec.get("role") if type(spec) is dict else None
            if role in ("input", "weight", "bias", "scale"):
                reads.append(name)
            elif role == "output":
                writes.append(name)
            else:
                raise ValueError("neutral harness cannot infer ambiguous logical pointer roles")
        kernel = {
            "kind": "whole_program",
            "args": [{"tensor": name, "access": "read"} for name in reads]
            + [{"tensor": name, "access": "write"} for name in writes],
            "outputs": writes,
        }
    if type(kernel) is not dict or set(kernel) != {"kind", "args", "outputs"} or kernel["kind"] != "whole_program":
        raise ValueError("neutral harness requires the explicit whole-program logical pointer ABI")
    args, outputs = kernel["args"], kernel["outputs"]
    if type(args) is not list or not args or type(outputs) is not list or not outputs:
        raise ValueError("neutral harness has an incomplete logical pointer/output roster")
    slots = {}
    mutable = (
        {name for name, spec in tensors.items() if type(spec) is dict and spec.get("role") == "intermediate"}
        if source_owned_mutables is None
        else set(source_owned_mutables)
    )
    for arg in args:
        if (
            type(arg) is not dict
            or set(arg) != {"tensor", "access"}
            or arg["access"] not in ("read", "write", "readwrite")
        ):
            raise ValueError("neutral harness argument has no exact tensor/access declaration")
        name = _identifier(arg["tensor"])
        if name in slots or name not in tensors:
            raise ValueError("neutral harness repeats or substitutes a logical pointer")
        slots[name] = arg["access"]
    if not mutable <= set(slots) or any(type(name) is not str for name in mutable):
        raise ValueError("neutral source-owned scratch is outside the original pointer roster")
    if set(slots) != set(tensors) or any(type(name) is not str for name in outputs):
        raise ValueError("neutral harness omits or adds a declared logical tensor")
    if len(outputs) != len(set(outputs)) or set(outputs) != {
        name for name, access in slots.items() if access != "read" and name not in mutable
    }:
        raise ValueError("neutral harness omits, repeats or substitutes a complete output")
    for name, spec in tensors.items():
        role = spec.get("role") if type(spec) is dict else None
        if role is not None and (
            role not in ("input", "weight", "bias", "scale", "output", "intermediate")
            or (role == "output" and name not in outputs)
            or (role != "output" and name not in mutable and slots[name] == "write")
            or (role == "intermediate" and (name not in mutable or slots[name] == "read"))
        ):
            raise ValueError("neutral harness pointer access conflicts with its semantic role")
    normalized = {**cb, "kernel_abi": kernel}
    leaves = whole_program_entry_bindings(normalized)
    if leaves is not None and leaves != [
        name for name, access in slots.items() if access != "write" and name not in mutable
    ]:
        raise ValueError("neutral harness changes the ordered logical input bindings")
    return normalized, slots


def logical_output_names(cb) -> tuple[str, ...]:
    """Validate the closed logical roster without allocating or changing source."""
    normalized, _slots = _roster(cb)
    for spec in normalized["tensors"].values():
        _tensor(spec)
    return tuple(normalized["kernel_abi"]["outputs"])


def _values(value, shape):
    if not shape:
        if type(value) in (list, tuple):
            raise ValueError("neutral input has excess tensor dimensions")
        yield value
        return
    if type(value) not in (list, tuple) or len(value) != shape[0]:
        raise ValueError("neutral input differs from its complete logical shape")
    for child in value:
        yield from _values(child, shape[1:])


def _payload(value, spec, shape, dtype, extent):
    width = dtype.element_bits // 8
    if type(value) is bytes:
        raw = value  # Explicit canonical little-endian scalar bytes.
    else:
        if type(value) is dict:
            if (
                set(value) - {"shape", "dtype", "values"}
                or type(value.get("shape")) is not list
                or any(type(d) is not int for d in value["shape"])
                or value["shape"] != list(shape)
            ):
                raise ValueError("neutral input has an incomplete or changed typed value declaration")
            if "dtype" in value:
                try:
                    same_dtype = type(value["dtype"]) is str and get(value["dtype"]).name == dtype.name
                except KeyError:
                    same_dtype = False
                if not same_dtype:
                    raise ValueError("neutral input changes the declared storage dtype")
            value = value.get("values")
        if value is None:
            raise ValueError("neutral harness requires every exact logical input value")
        packed = None
        if (not dtype.is_float and type(value) in (list, tuple)
                and len(value) == extent // width
                and all(type(scalar) is int for scalar in value)):
            code = {1: "b", 2: "h", 4: "i", 8: "q"}.get(width)
            if code is not None:
                code = code if dtype.signed else code.upper()
                if array.array(code).itemsize == width:
                    try:
                        packed = array.array(code, value)
                    except OverflowError as error:
                        raise ValueError("neutral integer input is outside its storage range") from error
                    if sys.byteorder != "little":
                        packed.byteswap()
        if packed is not None:
            raw = packed.tobytes()
        else:
            flat_values = (
                type(value) in (list, tuple)
                and len(value) == extent // width
                and all(type(scalar) not in (list, tuple) for scalar in value)
            )
            raw = bytearray()
            for scalar in value if flat_values else _values(value, shape):
                if not dtype.is_float:
                    if type(scalar) is not int:
                        raise ValueError("neutral integer inputs require exact integers")
                    try:
                        word = scalar.to_bytes(width, "little", signed=dtype.signed)
                    except OverflowError as error:
                        raise ValueError("neutral integer input is outside its storage range") from error
                else:
                    formats = {(5, 10): "e", (8, 23): "f", (11, 52): "d"}
                    code = formats.get((dtype.exp_bits, dtype.mant_bits))
                    if code is None or type(scalar) is not float or not math.isfinite(scalar):
                        raise ValueError("neutral floating inputs need exact source bytes for this value/format")
                    try:
                        word = struct.pack("<" + code, scalar)
                    except (OverflowError, struct.error) as error:
                        raise ValueError("neutral floating input is outside its storage range") from error
                    restored = struct.unpack("<" + code, word)[0]
                    if restored != scalar or math.copysign(1, restored) != math.copysign(1, scalar):
                        raise ValueError("neutral floating input would require numerical rounding")
                raw.extend(word)
            raw = bytes(raw)
    if len(raw) != extent:
        raise ValueError("neutral input bytes omit or add logical tensor elements")
    encoded = base64.b64encode(raw).decode("ascii")
    if "preload_b64" in spec and spec["preload_b64"] != encoded:
        raise ValueError("neutral explicit input differs from declared source bytes")
    return encoded


def _check_streamed_source(payload, spec, shape, dtype, extent):
    """Compare every lazy input byte with any explicit source declaration.

    Source scalar validation uses the same canonical encoder as the ordinary
    literal path. The selected finite storage budget bounds its temporary
    bytes; payload consumption stays chunked and never becomes C literals.
    """
    if "data" not in spec and "preload_b64" not in spec:
        return
    if "data" in spec:
        declared = base64.b64decode(_payload(spec["data"], spec, shape, dtype, extent), validate=True)
    else:
        try:
            declared = base64.b64decode(spec["preload_b64"], validate=True)
        except (TypeError, ValueError) as error:
            raise ValueError("neutral explicit source bytes are not canonical base64") from error
        _payload(declared, spec, shape, dtype, extent)
    offset = 0
    for chunk in payload.chunks():
        end = offset + len(chunk)
        if chunk != memoryview(declared)[offset:end]:
            raise ValueError("neutral streamed input differs from declared source values or bytes")
        offset = end
    if offset != extent:
        raise ValueError("neutral streamed input omits or adds declared source bytes")
    return declared


def selected_storage_budget(contract):
    """Finite software allocation policy, independent of candidate tensors."""
    budget = (contract.get("logical_harness") or {}).get("storage_budget_bytes", 16 << 20)
    if type(budget) is not int or not 0 < budget <= 256 << 20:
        raise ValueError("logical harness storage policy exceeds its fixed software ceiling")
    return budget


def render_harness(
    cb,
    *,
    target,
    inputs,
    contract,
    readback_policy=None,
    original_abi=None,
    invocation_plan=None,
    counter_plan=None,
    phase_plan=None,
    max_storage_bytes=16 * 1024 * 1024,
    source_owned_mutables=(),
    blobs=None,
):
    """Ordinary renderer: storage/calls only, with complete output-only poisoning.

    Missing ``kernel_abi`` uses declared semantic tensor-role order, inputs then
    outputs. An explicit roster preserves its exact pointer and output order.
    Input bytes are canonical little endian; numeric values must be exactly
    representable. Missing inputs never select deterministic fills or answers.
    The caller's existing grader owns original-source/numerical correspondence.
    """
    if type(target) is not str or not target:
        raise ValueError("neutral harness needs an explicit target selection")
    selected = validate_contract(contract)
    if readback_policy is None:
        readback_policy = selected.readback_policy
    engine_policies = {
        ReadbackPolicy(block["readback_transport"])
        for engine, block in (contract.get("runner", {}).get("chipyard_rocc", {}).get("engines", {})).items()
        if engine == "gsim" and type(block) is dict
        and block.get("readback_transport") in {COHERENT_DUMP_V1, COHERENT_PACKET_V1, COHERENT_PACKET_V2}
    }
    if (
        type(readback_policy) is not ReadbackPolicy
        or readback_policy not in {selected.readback_policy, *engine_policies}
    ):
        raise ValueError("neutral harness readback differs from its selected contract")
    if type(max_storage_bytes) is not int or not 0 < max_storage_bytes < 1 << 64:
        raise ValueError("neutral harness requires a positive complete storage budget")
    if type(source_owned_mutables) is not tuple or len(set(source_owned_mutables)) != len(source_owned_mutables):
        raise ValueError("neutral source-owned scratch requires its original exact tuple")
    normalized, slots = _roster(cb, source_owned_mutables=source_owned_mutables)
    layouts = {name: _tensor(spec) for name, spec in normalized["tensors"].items()}
    total = sum(row[2] for row in layouts.values())
    if readback_policy.transport == COHERENT_PACKET_V1:
        from .direct_kernel_harness import packet_output_blocks
        from .out_packet import out_bin_packet_capacity

        total += 8 + out_bin_packet_capacity(tuple(frame for _, _, frame in packet_output_blocks(normalized)))
    elif readback_policy.transport == COHERENT_PACKET_V2:
        from .direct_kernel_harness import packet_output_roster_v2
        from .out_packet_v2 import out_packet_v2_capacity

        total += 8 + out_packet_v2_capacity(packet_output_roster_v2(normalized))
    if original_abi is not None:
        from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi

        if type(original_abi) is not CompileOnlySourceAbi:
            raise ValueError("neutral harness original ABI must be an exact typed declaration")
        original_abi.bind(normalized)
    abi = selected.pointer_abi
    if invocation_plan is not None:
        from .direct_kernel_invocation import DirectKernelInvocationPlan

        if type(invocation_plan) is not DirectKernelInvocationPlan:
            raise ValueError("neutral harness requires an explicit typed invocation plan")
        histories = invocation_plan.bind(
            normalized, entry_symbol=abi.entry_symbol, completion_symbol=abi.completion_symbol
        )
        total += 8 + sum(row.byte_extent for row in histories)
    if counter_plan is not None:
        from .direct_kernel_counter import DirectKernelCounterPlan

        if type(counter_plan) is not DirectKernelCounterPlan:
            raise ValueError("neutral harness requires an explicit typed counter plan")
        total += sum(counter_plan.bind(normalized, abi=abi, invocation_plan=invocation_plan).values())
    if phase_plan is not None:
        from .direct_kernel_phases import DirectKernelPhasePlan

        if type(phase_plan) is not DirectKernelPhasePlan or phase_plan.counter_plan is not counter_plan:
            raise ValueError("neutral harness phases require the same selected counter plan")
        total += sum(phase_plan.bind(normalized, abi=abi, invocation_plan=invocation_plan).values())
    if total > max_storage_bytes:
        raise ValueError("neutral harness exceeds its complete storage budget before input rendering")
    expected = {name for name, access in slots.items() if access != "write" and name not in source_owned_mutables}
    if type(inputs) is not dict or set(inputs) != expected:
        raise ValueError("neutral harness needs the exact complete logical input roster")
    from .tensor_payload import TensorPayload

    tensors, streamed = {}, {}
    for name, spec in normalized["tensors"].items():
        shape, dtype, extent = layouts[name]
        tensors[name] = {
            "shape": list(shape),
            "dtype": spec["dtype"],
            **({"role": spec["role"]} if "role" in spec else {}),
        }
        if name in expected:
            if type(inputs[name]) is TensorPayload:
                if blobs is None:
                    raise ValueError("streamed logical input needs explicit sidecars")
                payload = inputs[name].bind(shape=shape, dtype=spec["dtype"], byte_order=abi.byte_order)
                declared = _check_streamed_source(payload, spec, shape, dtype, extent)
                if declared is None:
                    streamed[name] = inputs[name]
                else:
                    import numpy as np

                    from merlin.llvmlower.weight_payload import WeightPayload

                    # Freeze the exact bytes just compared, rather than read
                    # a caller's lazy provider again after source validation.
                    frozen = WeightPayload.array(extent, lambda data=declared: np.frombuffer(data, dtype=np.uint8))
                    streamed[name] = TensorPayload(shape, spec["dtype"], frozen)
            else:
                tensors[name]["preload_b64"] = _payload(inputs[name], spec, shape, dtype, extent)
                if ("data" in spec
                        and _payload(spec["data"], spec, shape, dtype, extent) != tensors[name]["preload_b64"]):
                    raise ValueError("neutral explicit input differs from declared source values")
        elif "data" in spec or "preload_b64" in spec:
            raise ValueError("neutral output-only storage cannot carry precomputed data")
    normalized = {**normalized, "tensors": tensors}
    from .complete_call_timing import from_contract
    from .kernel_call_timing import from_contract as kernel_timing_from_contract

    complete_call_timing = from_contract(contract)
    kernel_call_timing = kernel_timing_from_contract(contract)
    return render_direct_kernel(
        normalized,
        inputs={name: streamed.get(name, {}) for name in expected},
        abi=abi,
        readback_policy=readback_policy,
        invocation_plan=invocation_plan,
        counter_plan=counter_plan,
        phase_plan=phase_plan,
        output_poison=0xA5,
        prelude_symbol=selected.prelude_symbol,
        complete_call_timing=complete_call_timing,
        kernel_call_timing=kernel_call_timing,
        source_owned_mutables=source_owned_mutables,
        blobs=blobs,
    )
