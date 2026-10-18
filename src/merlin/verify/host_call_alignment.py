"""Low-bit alignment from the actual host LLVM passed to its selected compiler.

This observes descriptor call arguments and integer address expressions. It grants
no bounds, nonaliasing, mandatory execution, numerical or final-image authority.
Unknown origins retain one-byte alignment. A candidate's own assumptions are
never evidence about a caller's addresses.
"""

import os
import subprocess
import tempfile
from collections import Counter
from contextlib import nullcontext
from math import gcd
from pathlib import Path

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Operation
from xdsl.parser import Parser

from merlin.compile.model_execution_inputs import file_sha256
from merlin.llvmlower import toolchain
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context
from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer


def _power(value, width):
    value %= 1 << width
    return value & -value if value else 1 << width


def value_alignment(value, active=frozenset()):
    """Prove a power-of-two divisor of the value's address bits, otherwise one."""
    if value in active or len(active) >= 128:
        return 1
    active = active | {value}
    typ = value.type
    op = value.owner
    if not isinstance(typ, IntegerType):
        if (isinstance(op, Operation) and op.name in {"llvm.inttoptr", "llvm.bitcast"}
                and len(op.operands) == 1):
            return value_alignment(op.operands[0], active)
        return 1
    bits = typ.width.data
    if not 1 <= bits <= 128:
        return 1
    constant = constant_integer(value)
    if constant is not None:
        return _power(constant, bits)
    if not isinstance(op, Operation):
        return 1
    name, operands = op.name, op.operands
    if name in {"llvm.inttoptr", "llvm.ptrtoint", "llvm.bitcast"} and len(operands) == 1:
        return min(value_alignment(operands[0], active), 1 << bits)
    if name == "llvm.sub" and len(operands) == 2:
        remainder = operands[1].owner
        if (isinstance(remainder, Operation) and remainder.name == "llvm.urem"
                and len(remainder.operands) == 2 and remainder.operands[0] is operands[0]):
            divisor = constant_integer(remainder.operands[1])
            if type(divisor) is int and 0 < divisor <= 1 << bits and not divisor & (divisor - 1):
                # x - (x urem 2**k) clears exactly those low bits, including
                # unsigned wrap. The remainder must refer to this same SSA x.
                return divisor
    if len(operands) != 2:
        return 1
    left, right = (value_alignment(operand, active) for operand in operands)
    if name in {"llvm.add", "llvm.sub", "llvm.or", "llvm.xor"}:
        return min(left, right)
    if name == "llvm.and":
        return max(left, right)
    if name == "llvm.mul":
        return min(left * right, 1 << bits)
    if name == "llvm.shl":
        shift = constant_integer(operands[1])
        if type(shift) is int and 0 <= shift < bits:
            return min(left << shift, 1 << bits)
    return 1


def _element_bytes(dtype):
    from merlin.common.quant_formats import get

    width = get(dtype).element_bits
    if width % 8:
        raise ValueError("host call alignment needs an exact byte element width")
    return width // 8


def call_alignments(module, assigned, preparations, *, entry):
    functions = [op for op in module.body.block.ops
                 if op.name == "llvm.func" and op.sym_name.data == entry]
    if len(functions) != 1:
        raise ValueError("host alignment needs one exact original model entry")
    expected = Counter(row["symbol"] for row in assigned if row["symbol"] in preparations)
    observed = Counter()
    result = {}
    for op in functions[0].walk():
        reference = op.properties.get("callee") if op.name == "llvm.call" else None
        symbol = getattr(getattr(reference, "root_reference", None), "data", None)
        if symbol not in expected:
            continue
        if reference.nested_references.data:
            raise ValueError("host kernel call has an indirect symbol identity")
        buffers = preparations[symbol]["call_buffers"]
        observed[symbol] += 1
        start, values = 0, []
        for buffer in buffers:
            rank = len(buffer["shape"])
            if start + 3 + 2 * rank > len(op.operands):
                raise ValueError("host kernel call lacks its original memref descriptor")
            pointer, offset = op.operands[start + 1], op.operands[start + 2]
            values.append(gcd(value_alignment(pointer), value_alignment(offset) * _element_bytes(buffer["dtype"])))
            start += 3 + 2 * rank
        if start != len(op.operands):
            raise ValueError("host kernel call argument count differs from logical source ABI")
        if symbol in result:
            values = [gcd(a, b) for a, b in zip(result[symbol], values, strict=True)]
        result[symbol] = values
    if observed != expected:
        raise ValueError("host alignment call roster differs from source-bound kernel calls")
    return result


def inspect(host_artifacts, assigned, preparations, *, diagnostics=None, entry="forward"):
    """Reopen exact host inputs and use only selected stock LLVM analysis tools."""
    originals = {}
    for role in ("llvm_ir", "object"):
        row = host_artifacts.get(role) or {}
        path = Path(str(row.get("path", "")))
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != row.get("size_bytes")
                or file_sha256(path) != row.get("sha256")):
            raise ValueError("host alignment input differs from the actual compiled host artifact")
        originals[role] = (path, row["sha256"])
    translator = toolchain.mlir_translate().resolve(strict=True)
    tools = [translator.with_name("llvm-extract").resolve(strict=True), translator,
             translator.with_name("mlir-opt").resolve(strict=True)]
    pins = {str(tool): file_sha256(tool) for tool in tools}
    if diagnostics is None:
        selected_tmp = os.environ.get("TMPDIR")
        if not selected_tmp:
            raise ValueError("host alignment requires an explicitly selected scratch directory")
        workspace = tempfile.TemporaryDirectory(prefix="host-call-alignment-", dir=selected_tmp)
    else:
        diagnostics = Path(diagnostics)
        diagnostics.mkdir(parents=True, exist_ok=True)
        workspace = nullcontext(tempfile.mkdtemp(prefix="host-call-alignment-", dir=diagnostics))
    with workspace as temporary:
        directory = Path(temporary)
        sliced, imported, generic = (directory / name for name in ("entry.ll", "entry.mlir", "entry.generic.mlir"))
        commands = [
            [str(tools[0]), f"--func={entry}", "-S", str(originals["llvm_ir"][0]), "-o", str(sliced)],
            [str(tools[1]), "--import-llvm", str(sliced), "-o", str(imported)],
            [str(tools[2]), str(imported), "--mlir-print-op-generic", "-o", str(generic)],
        ]
        for command in commands:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if completed.returncode:
                raise ValueError("selected stock LLVM cannot inspect original host alignment: "
                                 + completed.stderr[-500:])
        module = Parser(make_llvm_context(), generic.read_text()).parse_module()
        result = call_alignments(module, assigned, preparations, entry=entry)
        observations = {
            name: file_sha256(directory / name) for name in ("entry.ll", "entry.mlir", "entry.generic.mlir")
        }
    if any(file_sha256(path) != digest for path, digest in originals.values()) or any(
        file_sha256(Path(path)) != digest for path, digest in pins.items()
    ):
        raise ValueError("host alignment source or selected observer changed during inspection")
    return result, {
        "host_artifacts": {role: digest for role, (_, digest) in originals.items()},
        "tools": pins, "observations": observations,
        "scope": "actual host descriptor address alignment only; "
        "no mandatory execution, bounds, alias or numeric proof",
    }
