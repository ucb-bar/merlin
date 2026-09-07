"""Opt-in, host-prepared compact caller; format setup only, never model arithmetic."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import prod
from pathlib import Path
import struct
import subprocess

from merlin.runtime.compact_binding import CompactBinding, resolve_compact_binding
from merlin.perf.storage_encoding import GroupedAxesStorage
from .gemmini_compact_abi import DerivedCompactABI, derive_compact_target_abi, _file_sha


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identifier(name):
    return (isinstance(name, str) and bool(name) and name.isascii()
            and (name[0].isalpha() or name[0] == "_")
            and all(char.isalnum() or char == "_" for char in name))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _signature(text, contract):
    from xdsl.context import Context
    from xdsl.dialects import builtin, llvm
    from xdsl.parser import Parser

    _require(type(text) is str and len(text.encode()) <= 32 * 1024 * 1024, "bounded emitted LLVM text required")
    context = Context()
    context.load_dialect(builtin.Builtin)
    context.load_dialect(llvm.LLVM)
    module = Parser(context, text).parse_module()
    module.verify()
    symbol = contract.get("compact_symbol")
    _require(_identifier(symbol), "compact symbol is not a C identifier")
    functions = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp) and op.sym_name.data == symbol]
    _require(len(functions) == 1 and functions[0].body.blocks, "unique defined compact entry required")
    fn = functions[0]
    _require(not fn.function_type.is_variadic and isinstance(fn.function_type.output, llvm.LLVMVoidType),
             "compact caller requires nonvariadic void entry")
    _require(not any(key in fn.properties or key in fn.attributes for key in ("arg_attrs", "res_attrs")),
             "compact function ABI attributes require explicit handling")
    arguments = tuple(fn.body.blocks.first.args)
    _require(len(arguments) == contract.get("compact_argument_count"), "emitted compact argument count mismatch")
    for arg in arguments:
        _require(isinstance(arg.type, llvm.LLVMPointerType), "compact entry must use only data pointers")
        space = arg.type.addr_space
        _require(isinstance(space, builtin.NoneAttr) or getattr(getattr(space, "value", None), "data", None) == 0,
                 "compact caller supports only verified address-space-zero pointers")
    return {"lowered_mlir_sha256": hashlib.sha256(text.encode()).hexdigest(), "symbol": symbol,
            "argument_count": len(arguments), "address_space": 0,
            "signature_verified": True, "pointer_substitution_equivalence": "UNPROVEN",
            "stronger_emitted_access_alignment_verified": False}


@dataclass(frozen=True)
class PreparedCompactCaller:
    binding: CompactBinding
    derived: DerivedCompactABI
    command_buffer_json: str
    compact_contract_json: str
    signature_json: str
    producer_pins_json: str

    def verify_identity(self, cb):
        _require(_digest(cb) == self.binding.setup_evidence()["command_buffer_sha256"],
                 "prepared compact caller belongs to another command buffer")
        evidence = self.derived.to_evidence()
        _require(all(_file_sha(path) == sha for path, sha in evidence["source_pins"].items()),
                 "compact ABI source/tool pins changed after preparation")
        _require(all(_file_sha(path) == sha for path, sha in json.loads(self.producer_pins_json).items()),
                 "compact caller producer changed after preparation")


def prepare_compact_caller(cb, compact_contract, logical_payloads, *, lowered_mlir_text,
                           workdir, prepack_authorizations=None, max_storage_bytes=64 * 1024):
    """Trusted host API. The explicit byte budget is not target memory capacity."""
    _require(logical_payloads is not None and compact_contract is not None, "explicit compact contract and logical bytes required")
    _require(type(max_storage_bytes) is int and 0 < max_storage_bytes <= 256 * 1024 * 1024,
             "compact caller storage budget outside host limits")
    signature = _signature(lowered_mlir_text, compact_contract)
    derived = derive_compact_target_abi(cb, compact_contract, workdir=workdir)
    binding = resolve_compact_binding(cb, compact_contract, logical_payloads,
        target_abi=derived.target_abi, max_storage_bytes=max_storage_bytes,
        prepack_authorizations=prepack_authorizations)
    from merlin.runtime import compact_binding, storage_binding
    from merlin.targetgen.contract import compile as compile_module
    producer_paths = (Path(__file__).resolve(), Path(compact_binding.__file__).resolve(),
                      Path(storage_binding.__file__).resolve(), Path(compile_module.__file__).resolve())
    producer_pins = {str(path): _file_sha(path) for path in producer_paths}
    result = PreparedCompactCaller(binding, derived, json.dumps(cb, sort_keys=True),
        json.dumps(compact_contract, sort_keys=True), json.dumps(signature, sort_keys=True),
        json.dumps(producer_pins, sort_keys=True))
    result.verify_identity(cb)
    return result


def render_compact_caller(cb, prepared):
    """One warm call, restore mutable initial bytes, one compute+completion window."""
    from .gemmini_codegen_mlir import container_for, _measurement_c_fragments

    _require(type(prepared) is PreparedCompactCaller, "exact host-prepared compact caller required")
    prepared.verify_identity(cb)
    contract = json.loads(prepared.compact_contract_json)
    symbol = contract["compact_symbol"]
    declarations, restore = [], []
    arenas = prepared.binding.base_views()
    for i, (arena, access, alignment) in enumerate(zip(arenas, prepared.derived.target_abi.base_access,
                                                     prepared.derived.target_abi.base_alignments, strict=True)):
        name = f"merlin_compact_base_{i}"
        initializer = ",".join(str(value) for value in arena) if access != "write" else "0"
        declarations.append(f"{'const ' if access == 'read' else ''}unsigned char {name}[{len(arena)}] "
                            f"__attribute__((aligned({alignment}), used)) = {{{initializer}}};")
        if access == "readwrite":
            declarations.append(f"static const unsigned char merlin_compact_initial_{i}[{len(arena)}] = {{{initializer}}};")
            restore.append(f"  __builtin_memcpy({name}, merlin_compact_initial_{i}, {len(arena)});")
        elif access == "write":
            restore.append(f"  __builtin_memset({name}, 0, {len(arena)});")
    bindings = sorted(contract["bindings"], key=lambda row: row["argument_index"])
    outputs = set(cb["kernel_abi"]["outputs"])
    prints = []
    for i, arg in enumerate(cb["kernel_abi"]["args"]):
        name = arg["tensor"]
        _require(_identifier(name), "compact output names must be safe identifiers")
        if name not in outputs:
            continue
        encoding = GroupedAxesStorage.from_dict(cb["params"]["storage_encodings"][name])
        container = container_for(encoding.dtype)
        rows, cols = (prod(encoding.logical_shape[:-1]), encoding.logical_shape[-1]) if encoding.logical_shape else (1, 1)
        stride_terms = [str(encoding.offset_elements)]
        divisor = prod(encoding.logical_shape)
        for extent, stride in zip(encoding.logical_shape, encoding.logical_strides_elements, strict=True):
            divisor //= extent
            stride_terms.append(f"(((i * {cols} + j) / {divisor}) % {extent}) * {stride}")
        row = bindings[i]
        byte_address = (f"merlin_compact_base_{row['base_index']} + {row['byte_offset']} + "
                        f"({' + '.join(stride_terms)}) * sizeof({container.ctype})")
        prints += [f'  printf("OUT {name} {rows} {cols}");',
            f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++) {{",
            f"    {container.ctype} value;", f"    __builtin_memcpy(&value, {byte_address}, sizeof(value));",
            "    " + container.printf_element("value"), "  }", '  printf("\\n");']
    call = f"  {symbol}({', '.join(f'(void*)merlin_compact_base_{i}' for i in range(len(arenas)))});\n  gemmini_fence();\n"
    fragments = _measurement_c_fragments("")
    # Explicit compact protocol, independent of the legacy cold/warm environment default.
    return ('#include <stdint.h>\n#include <stdio.h>\n#include "include/gemmini_testutils.h"\n'
        + fragments["include"] + f"extern void {symbol}({', '.join('void*' for _ in arenas)});\n"
        + "\n".join(declarations) + "\nint main() {\n" + call
        + "  // Restore caller initial bytes after the unmeasured warm invocation.\n"
        + "\n".join(restore) + "\n  gemmini_fence();\n" + fragments["prologue"]
        + "  uint64_t c0 = read_cycles();\n" + call + "  uint64_t c1 = read_cycles();\n"
        + fragments["epilogue"] + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
        + '  printf("METRIC cycle_window_gemmini_region 1\\n");\n'
        + "\n".join(prints) + '\n  printf("DONE\\n");\n  return 0;\n}\n')


def _load_segments(path):
    """Read ELF load mappings; constants below are ELF format, not target geometry."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(64)
        _require(len(header) >= 52 and header[:4] == b"\x7fELF", "invalid linked ELF header")
        _require(header[4] in (1, 2) and header[5] in (1, 2), "unsupported ELF class/byte order")
        endian = "<" if header[5] == 1 else ">"
        _require(struct.unpack_from(endian + "H", header, 16)[0] == 2, "compact caller must be a linked executable")
        if header[4] == 2:
            offset = struct.unpack_from(endian + "Q", header, 32)[0]
            entry_size, count = struct.unpack_from(endian + "HH", header, 54)
            fmt = endian + "IIQQQQQQ"
        else:
            offset = struct.unpack_from(endian + "I", header, 28)[0]
            entry_size, count = struct.unpack_from(endian + "HH", header, 42)
            fmt = endian + "IIIIIIII"
        _require(entry_size == struct.calcsize(fmt) and 0 < count <= 1024
                 and offset + count * entry_size <= size, "invalid ELF program-header bounds")
        segments = []
        for i in range(count):
            stream.seek(offset + i * entry_size)
            row = struct.unpack(fmt, stream.read(entry_size))
            if row[0] != 1:
                continue
            if header[4] == 2:
                _, flags, file_offset, address, _, file_size, memory_size, _ = row
            else:
                _, file_offset, address, _, file_size, memory_size, flags, _ = row
            _require(file_size <= memory_size and file_offset + file_size <= size, "invalid ELF load segment")
            segments.append((address, address + memory_size, flags))
    return segments, "little" if header[5] == 1 else "big"


def verify_compact_caller_link(cb, prepared, *, object_path, elf_path, workdir, expected_object_sha256):
    """Static linked-allocation proof, not invocation or pointer-substitution proof."""
    _require(type(prepared) is PreparedCompactCaller, "exact host-prepared compact caller required")
    prepared.verify_identity(cb)
    _require(_file_sha(object_path) == expected_object_sha256, "compact kernel object changed during caller build")
    evidence = prepared.derived.to_evidence()
    nm = Path(evidence["commands"][-1]["argv"][0])
    _require(_file_sha(nm) == evidence["source_pins"][str(nm)], "target symbol tool changed")
    elf = Path(elf_path)
    before = _file_sha(elf)
    proc = subprocess.run([str(nm), "-S", "--radix=d", "--defined-only", str(elf)],
                          capture_output=True, text=True, timeout=15)
    _require(proc.returncode == 0 and len(proc.stdout) <= 4 * 1024 * 1024, "linked symbol extraction failed")
    symbols = {}
    compact_symbol = json.loads(prepared.signature_json)["symbol"]
    wanted = {f"merlin_compact_base_{i}" for i in range(len(prepared.binding.base_views()))} | {compact_symbol}
    for line in proc.stdout.splitlines():
        fields = line.split()
        if fields and fields[-1] in wanted:
            _require(len(fields) == 4 and fields[0].isdecimal() and fields[1].isdecimal()
                     and fields[-1] not in symbols, "ambiguous linked compact symbol")
            symbols[fields[-1]] = (int(fields[0]), int(fields[1]), fields[2])
    _require(set(symbols) == wanted, "linked compact function or arena symbol missing")
    _require(symbols[compact_symbol][2] in ("T", "t"), "compact entry is not linked code")
    segments, byte_order = _load_segments(elf)
    _require(byte_order == evidence["kernel_layout"]["byte_order"], "linked ELF byte order differs from target facts")
    entry_address, entry_extent, _ = symbols[compact_symbol]
    _require(entry_extent > 0 and any(start <= entry_address and entry_address + entry_extent <= end and flags & 1
                                    for start, end, flags in segments), "compact entry lacks executable load mapping")
    addresses = []
    for i, (arena, access) in enumerate(zip(prepared.binding.base_views(), prepared.derived.target_abi.base_access, strict=True)):
        address, extent, kind = symbols[f"merlin_compact_base_{i}"]
        _require(extent == len(arena), "linked compact arena extent differs from prepared bytes")
        _require(address + extent <= 1 << evidence["kernel_layout"]["pointer_bits"], "linked arena exceeds pointer representation")
        _require(kind in (("R", "r") if access == "read" else ("B", "b", "D", "d")),
                 "linked arena storage class differs from homogeneous access")
        required_flags = 4 if access == "read" else 6  # ELF PF_R / PF_R|PF_W.
        _require(any(start <= address and address + extent <= end and flags & required_flags == required_flags
                     for start, end, flags in segments), "compact arena lacks a matching loaded segment")
        addresses.append(address)
    address_receipt = prepared.binding.runtime_address_evidence(addresses)
    _require(_file_sha(elf) == before, "linked ELF changed during allocation check")
    prepared.verify_identity(cb)
    result = {"schema": "compact_caller_link_v1", "status": "static_build_validated",
        "scope": "actual linked caller allocations and signature only", "elf_sha256": before,
        "object_sha256": _file_sha(object_path), "harness_sha256": _file_sha(Path(workdir) / "harness.c"),
        "signature": json.loads(prepared.signature_json), "binding": prepared.binding.setup_evidence(),
        "producer_pins": json.loads(prepared.producer_pins_json),
        "target_abi": evidence, "linked_symbols": symbols, "address_check": address_receipt,
        "linked_allocations_verified": True, "runtime_allocation_observed": False,
        "warm_invocations_emitted": 1, "measured_invocations_emitted": 1,
        "mutable_initial_bytes_restored_before_measurement": True,
        "target_executed": False, "numerical_equivalence": "UNPROVEN", "cache_used": False}
    (Path(workdir) / "compact_caller_link.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
