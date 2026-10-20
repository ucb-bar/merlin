"""Bind optional input-copy auditing to a capsule's original logical convolution ABI.

This supplies no kernel or numerical reference. The independently selected harness determines the
pointer order; the original program determines types and extents. The emitted code must still
prove its actual copies, bounds and accelerator consumption.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def source_convolution(capsule, *, target):
    from merlin.llvmlower.device_shim import kernel_abi_for
    from merlin.runtime.backends.base import harness_build_recipe
    from merlin.targetgen.contract.harness_render import logical_abi, logical_interface
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    if (capsule.get("operation") or {}).get("op") != "conv2d":
        return None
    kernel_abi = kernel_abi_for(target)
    if kernel_abi is None or kernel_abi.version != 2:
        return None
    name = capsule.get("interface_mlir")
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("input preparation requires a contained original interface")
    source = Path(capsule["__dir__"]) / name
    if any(path.is_symlink() for path in (source, *source.parents)):
        raise ValueError("input preparation original source is indirect")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    screen = capsule.get("software_screen") or {}
    if screen.get("status") != "admitted" or screen.get("program_sha256") != digest:
        raise ValueError("input preparation lacks original source admission")
    original = parse_interface_mlir(raw.decode("utf-8"))
    commands = original["commands"]
    if original["target"] != target or [row["opcode"] for row in commands] != ["RES_PACK", "CONV2D", "EVICT"]:
        return None
    pack, convolution, evict = commands
    attrs = convolution.get("attributes") or {}
    if attrs.get("epilogue") or attrs.get("output_dtype") != "i32":
        return None
    weights, handle = pack["operands"]["src"], pack["operands"]["dst"]
    if convolution["operands"]["weight"] != handle or evict["operands"]["handle"] != handle:
        raise ValueError("original convolution has an unbound resident weight")
    roles = {convolution["operands"]["ifm"]: "input_0", weights: "weight_0", convolution["operands"]["dst"]: "out_0"}
    abi = logical_abi()
    buffers = logical_interface(original, abi)
    if len(roles) != 3 or len(buffers) != 3 or {buffer.name for buffer in buffers} != set(roles):
        raise ValueError("original convolution lacks an exact three-buffer ABI")
    context = [{"role": roles[buffer.name], "shape": list(buffer.shape), "dtype": buffer.dtype} for buffer in buffers]
    if [buffer["role"] for buffer in context] != ["input_0", "weight_0", "out_0"]:
        return None
    # The logical ABI fixes buffer order, not the selected executable's entry.
    # Use the same independently selected recipe as the native compiler and
    # stack preflight. A candidate's own function spelling grants no selection.
    try:
        entry = harness_build_recipe(target).require_kernel_stack_frame().entry_symbol
    except (KeyError, NotImplementedError) as error:
        raise ValueError("input preparation has no selected harness entry") from error
    return {
        "source_sha256": digest,
        "entry_symbol": entry,
        "call_buffers": context,
        "argument_alignment": abi.alignment_bytes,
    }
