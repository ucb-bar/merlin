"""Opt-in schema-CONV2D selection and native LLVM artifact emission.

The default Gemmini compiler remains unchanged. The opt-in requires an explicit
source-bound contract, retains refusal reasons, and does not certify runtime
numerics. Its dense NHWC/HWIO pointer ABI is NOT the legacy padded im2col ABI.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .gemmini_loop_conv import UnsupportedNativeConv, _require, emit_native_conv, native_entry_instructions


def emit_selected_native_conv(cb, *, contract):
    """Return (target LLVM MLIR, argument names, selection receipt), or refuse."""
    commands = cb.get("commands", [])
    convolutions = [command for command in commands if command.get("opcode") == "CONV2D"]
    _require(len(convolutions) == 1, "native route requires one schema convolution region")
    conv = deepcopy(convolutions[0])
    weight = conv.get("operands", {}).get("weight")
    packs = [command for command in commands if command.get("opcode") == "RES_PACK"]
    evicts = [command for command in commands if command.get("opcode") == "EVICT"]
    _require(len(commands) == 1 + len(packs) + len(evicts) and len(packs) <= 1 and len(evicts) <= 1,
             "native route cannot discard other commands")
    if packs:
        pack = packs[0]
        _require(pack.get("operands", {}).get("dst") == weight
                 and pack.get("attributes", {}) == {"layout": "packed_rhs"}
                 and commands.index(pack) < commands.index(convolutions[0]),
                 "unsupported resident packing or ordering")
        weight = pack["operands"]["src"]
        conv["operands"]["weight"] = weight
    if evicts:
        _require(packs and evicts[0].get("operands") == {"handle": packs[0]["operands"]["dst"]}
                 and not evicts[0].get("attributes")
                 and commands.index(evicts[0]) > commands.index(convolutions[0]), "unsupported eviction")
    tensors = cb.get("tensors", {})
    operands = conv.get("operands", {})
    _require(set(operands) == {"ifm", "weight", "dst"}, "unsupported convolution operands")
    _require(all(name in tensors for name in operands.values()), "missing convolution tensor ABI")
    ci, co = tensors[operands["ifm"]]["shape"][-1], tensors[weight]["shape"][-1]
    arg_order = [weight, operands["ifm"], operands["dst"]]
    _require(len(set(arg_order)) == len(arg_order), "aliased convolution tensor ABI")
    pointers = {"weight": "weight_ptr", "ifm": "input_ptr", "dst": "output_ptr"}
    receipt = emit_native_conv(conv, tensors, contract=contract, pointers=pointers,
                               row_strides={"ifm": ci, "weight": co, "dst": co})
    entry = native_entry_instructions(contract, output_channels=co, activation=receipt["parameters"]["activation"])
    completion = contract.header.macro("gemmini_fence")
    _require(completion is not None and completion.body.startswith('asm volatile("')
             and completion.body.endswith('")'), "unsupported target completion ABI")
    assembly = completion.body[len('asm volatile("'):-len('")')]
    _require(assembly and '"' not in assembly and "\\" not in assembly, "unsupported completion assembly")
    fence = f'    llvm.inline_asm has_side_effects "{assembly}", "~{{memory}}" : () -> ()'
    lines = ["module {", "  llvm.func @gemmini_kernel(%a0: !llvm.ptr, %a1: !llvm.ptr, %a2: !llvm.ptr) {", fence]
    names = {pointers["weight"]: "%p0", pointers["ifm"]: "%p1", pointers["dst"]: "%p2"}
    for index in range(len(arg_order)):
        lines.append(f"    %p{index} = llvm.ptrtoint %a{index} : !llvm.ptr to i64")
    counter = 0
    for instruction in entry + receipt["instructions"]:
        operands_ssa = []
        for register in ("rs1", "rs2"):
            value = instruction[register]
            if isinstance(value, str):
                operands_ssa.append(names[value])
            else:
                counter += 1
                ssa = f"%c{counter}"
                lines.append(f"    {ssa} = llvm.mlir.constant({value} : i64) : i64")
                operands_ssa.append(ssa)
        lines.append(f'    llvm.inline_asm has_side_effects ".insn r {contract.custom_opcode}, '
                     f'{contract.funct3}, {instruction["funct"]}, x0, $0, $1", "r,r,~{{memory}}" '
                     f'{operands_ssa[0]}, {operands_ssa[1]} : (i64, i64) -> ()')
    lines.extend([fence, "    llvm.return", "  }", "}"])
    text = "\n".join(lines) + "\n"
    receipt.update({"selection": "selected_explicit_opt_in", "default_enabled": False,
                    "entry_instructions": entry, "arg_order": arg_order,
                    "physical_abi": "dense_NHWC_flattened_HWIO_no_legacy_padding",
                    "command_buffer_sha256": hashlib.sha256(json.dumps(cb, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest(),
                    "target_artifact_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "compiler_path": "emit_kernel_mlir_before_CONV2D_normalization",
                    "full_model_native_selection": False})
    return text, arg_order, receipt
