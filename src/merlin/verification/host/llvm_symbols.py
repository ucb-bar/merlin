"""Resolve a defined LLVM function through structurally proved symbol aliases."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections import Counter
from contextlib import nullcontext
from math import gcd
from pathlib import Path

from xdsl.dialects.builtin import IntegerType, SymbolRefAttr
from xdsl.ir import Operation
from xdsl.parser import Parser

from merlin.common.digest import sha256_file as file_sha256
from merlin.compiler.dialects.lowering.analyses import constant_integer
from merlin.compiler.ir.llvm_context import make_llvm_context
from merlin.compiler.lower.model import toolchain
from merlin.compiler.transforms.analyses import operation_name

# ---- llvm_symbols ------------------------------------------------------------------------------


def defined_function(module, symbol):
    """Return the original body; opaque, cyclic or type-changing aliases refuse.

    A supported alias consists solely of taking one flat symbol's address and
    returning that exact SSA value. This resolves names without rewriting the
    artifact, trusting an arbitrary pointer expression or skipping helper audits.
    """
    symbols = {}
    for op in module.body.block.ops:
        name = op.properties.get("sym_name") or op.attributes.get("sym_name")
        if isinstance(getattr(name, "data", None), str):
            symbols.setdefault(name.data, []).append(op)
    visited, required_types = set(), []
    while symbol not in visited:
        visited.add(symbol)
        matches = symbols.get(symbol, [])
        if len(matches) != 1:
            break
        op = matches[0]
        if operation_name(op) == "llvm.func":
            typ = op.properties.get("function_type")
            if op.body.blocks and all(typ == expected for expected in required_types):
                return op
            break
        if operation_name(op) != "llvm.mlir.alias" or len(op.regions) != 1:
            break
        region = op.regions[0]
        if len(region.blocks) != 1 or region.block.args:
            break
        body = list(region.block.ops)
        if (
            len(body) != 2
            or operation_name(body[0]) != "llvm.mlir.addressof"
            or operation_name(body[1]) != "llvm.return"
            or len(body[0].results) != 1
            or list(body[1].operands) != [body[0].results[0]]
        ):
            break
        target = body[0].properties.get("global_name")
        if not isinstance(target, SymbolRefAttr) or target.nested_references:
            break
        required_types.append(op.properties.get("alias_type"))
        symbol = target.root_reference.data
    raise ValueError("LLVM symbol has no uniquely defined, type-preserving function body")

    # Preserve historical object lookup; implementation and completion state live here.


# ---- host_call_alignment -----------------------------------------------------------------------
# Low-bit alignment from the actual host LLVM passed to its selected compiler.
#
# This observes descriptor call arguments and integer address expressions. It grants
# no bounds, nonaliasing, mandatory execution, numerical or final-image authority.
# Unknown origins retain one-byte alignment. A candidate's own assumptions are
# never evidence about a caller's addresses.


def _power(value, width):
    value %= 1 << width
    return value & -value if value else 1 << width


_ALIGNMENT_STATE_LIMIT = 4096


class _AlignmentBudgetExceeded(Exception):
    pass


def value_alignment(value, active=frozenset()):
    """Prove an address divisor with bounded, per-call SSA exploration.

    Memoization includes the active path because cycles and the depth limit can
    change a subproof. Exhaustion discards the whole proof, retaining one-byte
    alignment. Nothing is cached across calls or source mutations.
    """
    memo, constants = {}, {}
    remaining = _ALIGNMENT_STATE_LIMIT

    def read_constant(item):
        if item not in constants:
            constants[item] = constant_integer(item)
        return constants[item]

    def visit(item, path):
        nonlocal remaining
        if item in path or len(path) >= 128:
            return 1
        key = (item, path)
        if key in memo:
            return memo[key]
        remaining -= 1
        if remaining < 0:
            raise _AlignmentBudgetExceeded
        result = evaluate(item, path | {item})
        memo[key] = result
        return result

    def evaluate(value, active):
        typ = value.type
        op = value.owner
        if not isinstance(typ, IntegerType):
            if isinstance(op, Operation) and op.name in {"llvm.inttoptr", "llvm.bitcast"} and len(op.operands) == 1:
                return visit(op.operands[0], active)
            return 1
        bits = typ.width.data
        if not 1 <= bits <= 128:
            return 1
        constant = read_constant(value)
        if constant is not None:
            return _power(constant, bits)
        if not isinstance(op, Operation):
            return 1
        name, operands = op.name, op.operands
        if name in {"llvm.inttoptr", "llvm.ptrtoint", "llvm.bitcast"} and len(operands) == 1:
            return min(visit(operands[0], active), 1 << bits)
        if name == "llvm.sub" and len(operands) == 2:
            remainder = operands[1].owner
            if (
                isinstance(remainder, Operation)
                and remainder.name == "llvm.urem"
                and len(remainder.operands) == 2
                and remainder.operands[0] is operands[0]
            ):
                divisor = read_constant(remainder.operands[1])
                if type(divisor) is int and 0 < divisor <= 1 << bits and not divisor & (divisor - 1):
                    # x - (x urem 2**k) clears exactly those low bits, including
                    # unsigned wrap. The remainder must refer to this same SSA x.
                    return divisor
        if len(operands) != 2:
            return 1
        left, right = (visit(operand, active) for operand in operands)
        if name in {"llvm.add", "llvm.sub", "llvm.or", "llvm.xor"}:
            return min(left, right)
        if name == "llvm.and":
            return max(left, right)
        if name == "llvm.mul":
            return min(left * right, 1 << bits)
        if name == "llvm.shl":
            shift = read_constant(operands[1])
            if type(shift) is int and 0 <= shift < bits:
                return min(left << shift, 1 << bits)
        return 1

    try:
        return visit(value, frozenset(active))
    except (_AlignmentBudgetExceeded, RecursionError):
        return 1


def _element_bytes(dtype):
    from merlin.common.target_data import get

    width = get(dtype).element_bits
    if width % 8:
        raise ValueError("host call alignment needs an exact byte element width")
    return width // 8


def call_alignments(module, assigned, preparations, *, entry):
    functions = [op for op in module.body.block.ops if op.name == "llvm.func" and op.sym_name.data == entry]
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


def import_host_entry(host_artifacts, *, diagnostics=None, entry="forward", include_functions=(), max_ir_bytes=None):
    """Reopen actual compiled LLVM through stock tools; grant no storage/effect facts."""
    if (
        type(entry) is not str
        or not entry
        or type(include_functions) is not tuple
        or any(type(name) is not str or not name for name in include_functions)
        or len(include_functions) > 1024
        or len(set(include_functions)) != len(include_functions)
        or max_ir_bytes is not None
        and (type(max_ir_bytes) is not int or max_ir_bytes <= 0)
    ):
        raise ValueError("host import requires an explicit bounded original function roster")
    originals = {}
    for role in ("llvm_ir", "object"):
        row = host_artifacts.get(role) or {}
        path = Path(str(row.get("path", "")))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row.get("size_bytes")
            or file_sha256(path) != row.get("sha256")
        ):
            raise ValueError("host alignment input differs from the actual compiled host artifact")
        originals[role] = (path, row["sha256"])
    if max_ir_bytes is not None and originals["llvm_ir"][0].stat().st_size > max_ir_bytes:
        raise ValueError("host LLVM import exceeds its explicit source byte budget")
    translator = toolchain.mlir_translate().resolve(strict=True)
    tools = [
        translator.with_name("llvm-extract").resolve(strict=True),
        translator,
        translator.with_name("mlir-opt").resolve(strict=True),
    ]
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
            [
                str(tools[0]),
                *(f"--func={name}" for name in dict.fromkeys((entry, *include_functions))),
                "-S",
                str(originals["llvm_ir"][0]),
                "-o",
                str(sliced),
            ],
            [str(tools[1]), "--import-llvm", str(sliced), "-o", str(imported)],
            [str(tools[2]), str(imported), "--mlir-print-op-generic", "-o", str(generic)],
        ]
        for command in commands:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if completed.returncode:
                raise ValueError(
                    "selected stock LLVM cannot inspect original host alignment: " + completed.stderr[-500:]
                )
        if max_ir_bytes is not None and any(path.stat().st_size > max_ir_bytes for path in (sliced, imported, generic)):
            raise ValueError("host LLVM import exceeds its explicit product byte budget")
        module = Parser(make_llvm_context(), generic.read_text()).parse_module()
        observations = {
            name: file_sha256(directory / name) for name in ("entry.ll", "entry.mlir", "entry.generic.mlir")
        }
        products = (
            {str((directory / name).resolve()): digest for name, digest in observations.items()}
            if diagnostics is not None
            else {}
        )
    if any(file_sha256(path) != digest for path, digest in originals.values()) or any(
        file_sha256(Path(path)) != digest for path, digest in pins.items()
    ):
        raise ValueError("host alignment source or selected observer changed during inspection")
    return module, {
        "host_artifacts": {role: digest for role, (_, digest) in originals.items()},
        "inputs": {str(path.resolve()): digest for path, digest in originals.values()},
        "tools": pins,
        "observations": observations,
        "products": products,
        "scope": "stock-verified import of actual compiled host inputs; no bounds, alias, effect or numeric proof",
    }


def revalidate_host_entry(observation):
    for kind in ("inputs", "tools", "products"):
        for filename, digest in observation[kind].items():
            path = Path(filename)
            if not path.is_file() or path.is_symlink() or file_sha256(path) != digest:
                raise ValueError("host import source, selected observer or retained product changed")


def inspect(host_artifacts, assigned, preparations, *, diagnostics=None, entry="forward"):
    """Reopen exact host inputs and use only selected stock LLVM analysis tools."""
    module, observation = import_host_entry(host_artifacts, diagnostics=diagnostics, entry=entry)
    result = call_alignments(module, assigned, preparations, entry=entry)
    revalidate_host_entry(observation)
    observation["scope"] = (
        "actual host descriptor address alignment only; no mandatory execution, bounds, alias or numeric proof"
    )
    return result, observation
