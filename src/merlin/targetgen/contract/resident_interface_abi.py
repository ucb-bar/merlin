"""Resolve a single resident matmul's pointers from its selected interface contract.

The target name never chooses wiring.  This small supported subset of the shared
``kernel_abi.arg_order_by_command_shape`` (the version-1 kernel ABI) resolves command operands to
external pointer names and refuses any interface it cannot describe exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from merlin.common.digest import sha256_bytes
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir
from merlin.targetgen.contract.schemas import legacy_kernel_abi_path


@dataclass(frozen=True)
class ResidentInterfaceAbi:
    target: str
    contract_sha256: str
    kernel_symbol: str
    pointer_order: tuple[str, ...]
    weight_tensor: str
    lhs_tensor: str
    output_tensor: str
    m: int
    n: int
    k: int
    dtypes: tuple[str, str, str]


def bind_single_resident_matmul(
    interface: str, *, target: str, abi_contract: bytes | None = None
) -> ResidentInterfaceAbi:
    """Validate exact resident dataflow and compare declaration to contract order.

    Only one pack, matmul and commit with no fused epilogue is supported by the
    current rank-2 shim.  Other command shapes are explicit refusals, not an
    assumption that their pointer ABI happens to look like this one.
    """
    raw = abi_contract if abi_contract is not None else legacy_kernel_abi_path().read_bytes()
    if not isinstance(raw, bytes):
        raise ValueError("selected kernel ABI contract must be exact bytes")
    document = yaml.safe_load(raw)
    kernel_abi = document.get("kernel_abi") if isinstance(document, dict) else None
    rows = kernel_abi.get("arg_order_by_command_shape") if isinstance(kernel_abi, dict) else None
    tokens = kernel_abi.get("arg_order_tokens") if isinstance(kernel_abi, dict) else None
    matching = (
        [row for row in rows if isinstance(row, dict) and row.get("shape") == "resident_matmul"]
        if isinstance(rows, list) else []
    )
    supported_order = (
        "resident_weights_in_resident_pack_order",
        "matmul_lhs_group_major",
        "commit_outputs_group_major",
        "commit_biases_group_major",
    )
    if len(matching) != 1 or not isinstance(tokens, dict):
        raise ValueError("selected ABI contract has no unique resident-matmul pointer rule")
    order = matching[0].get("order")
    if order != list(supported_order) or not set(supported_order) <= set(tokens):
        raise ValueError("selected ABI contract has a resident pointer order unsupported by the rank-2 shim")
    symbol_pattern = kernel_abi.get("symbol")
    if not isinstance(symbol_pattern, str) or "{target}" not in symbol_pattern:
        raise ValueError("selected ABI contract has no target-parameterized kernel symbol")

    parsed = parse_interface_mlir(interface)
    if emit_interface_mlir(parsed) != interface:
        raise ValueError("selected resident interface is not a canonical, fully typed round trip")
    if parsed["target"] != target:
        raise ValueError("selected interface target differs from requested target")
    commands = parsed["commands"]
    if [command.get("opcode") for command in commands] != ["RES_PACK", "MATMUL_RESIDENT", "COMMIT"]:
        raise ValueError("selected interface is not one exact resident-matmul command group")
    pack, mm, commit = (command.get("operands") or {} for command in commands)
    weight, handle = pack.get("src"), pack.get("dst")
    lhs, acc = mm.get("lhs"), mm.get("dst")
    output = commit.get("dst")
    tensors = parsed["tensors"]
    if (
        not all(isinstance(name, str) and name for name in (weight, handle, lhs, acc, output))
        or mm.get("rhs") != handle or commit.get("src") != acc
        or weight not in tensors or lhs not in tensors or weight == lhs
        or output in tensors or len(tensors) != 2
        or (commands[0].get("attributes") or {}).get("layout") != "packed_rhs"
        or (commands[2].get("attributes") or {}).get("epilogue") != []
    ):
        raise ValueError("selected resident commands do not bind one weight, lhs and output")
    a, b = tensors[lhs], tensors[weight]
    if a.get("role") not in {"input", "weight"} or b.get("role") not in {"input", "weight"}:
        raise ValueError("selected resident pointers are not external input tensors")
    left, right = a.get("shape"), b.get("shape")
    if (
        not isinstance(left, list) or not isinstance(right, list)
        or len(left) != 2 or len(right) != 2
        or any(type(extent) is not int or extent <= 0 for extent in [*left, *right])
        or left[1] != right[0]
    ):
        raise ValueError("selected resident tensor shapes disagree")
    dtypes = (a.get("dtype"), b.get("dtype"), (commands[2].get("attributes") or {}).get("output_dtype"))
    if any(not isinstance(dtype, str) or not dtype for dtype in dtypes):
        raise ValueError("selected resident pointer precision is incomplete")
    resolved = {
        "resident_weights_in_resident_pack_order": [weight],
        "matmul_lhs_group_major": [lhs],
        "commit_outputs_group_major": [output],
        "commit_biases_group_major": [],
    }
    pointer_order = tuple(name for token in order for name in resolved[token])
    # Input declarations precede commands; the commit defines the output last.
    declared_order = (*tensors, output)
    if pointer_order != declared_order:
        raise ValueError("selected interface declaration order disagrees with pointer ABI contract")
    return ResidentInterfaceAbi(
        target, sha256_bytes(raw), symbol_pattern.replace("{target}", target), pointer_order,
        weight, lhs, output, left[0], right[1], left[1], dtypes,
    )
