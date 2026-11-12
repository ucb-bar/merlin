"""Byte-bound structural translation validation for a narrow real outlining pass.

For pure operations, beta-expanding an outlined call must reconstruct the exact
source SSA result graph (including the matmul region). Identical primitive
applications to identical operands have identical behavior under their unchanged
MLIR semantics. This is *not* an independent proof of those primitive semantics,
tensor arithmetic, physical memory, or generated code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


class UnsupportedOutlineIR(ValueError):
    """The IR is outside the pure structural equivalence domain."""


@dataclass(frozen=True)
class OutlineTransformReceipt:
    schema: str
    method: str
    entry: str
    source_sha256: str
    target_sha256: str
    verifier_sha256: str
    parser_sha256: str
    xdsl_version: str | None
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]
    source_graph_sha256: str | None
    target_graph_sha256: str | None
    status: str  # verified | mismatch | unsupported | unavailable | error
    reason: str | None
    mismatch_witness: dict[str, str] | None

    @property
    def verified(self) -> bool:
        return self.method == "syntactic_beta_equivalence_actual_outline_ir" and self.status == "verified"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict) -> OutlineTransformReceipt:
        if record.get("schema") != "merlin.outline_transform_verification.v1":
            raise ValueError(f"unrecognized outline receipt schema {record.get('schema')!r}")
        return cls(
            **{**record, "input_types": tuple(record["input_types"]), "output_types": tuple(record["output_types"])}
        )


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _xdsl_version() -> str | None:
    try:
        return version("xdsl")
    except PackageNotFoundError:
        return None


def _parser_digest() -> str:
    root = Path(__file__).parent.parent
    # mlir_query delegates to linalg_mlir.parse_mlir_text, whose context also
    # registers fp8 types. Bind all repository-owned parser components used by
    # this route, not just its thin query wrapper.
    paths = (
        root / "common" / "mlir_query.py",
        root / "compiler" / "frontend" / "linalg_mlir.py",
        root / "compiler" / "dialects" / "fp8.py",
    )
    return _digest(b"".join(path.read_bytes() for path in paths))


def _attrs(op) -> tuple:
    """Canonicalize both xDSL attribute stores; unknown attributes are still compared."""
    return (
        tuple(sorted((str(k), str(v)) for k, v in op.attributes.items())),
        tuple(sorted((str(k), str(v)) for k, v in op.properties.items())),
    )


def _integer_tensor(typ) -> bool:
    from xdsl.dialects.builtin import IntegerType, NoneAttr, TensorType

    return (
        isinstance(typ, TensorType)
        and isinstance(typ.element_type, IntegerType)
        and isinstance(typ.encoding, NoneAttr)
        and all(int(dim.data) >= 0 for dim in typ.shape.data)
    )


def _scalar_integer(typ) -> bool:
    from xdsl.dialects.builtin import IntegerType

    return isinstance(typ, IntegerType)


def _functions(module) -> dict[str, Any]:
    from xdsl.dialects.func import FuncOp

    if module.attributes or module.properties:
        raise UnsupportedOutlineIR("module attributes or properties are outside this proof domain")
    funcs = {}
    for fn in module.body.block.ops:
        if not isinstance(fn, FuncOp):
            raise UnsupportedOutlineIR(f"top-level operation {fn.name} is not a function")
        if fn.attributes or set(fn.properties) - {"sym_name", "function_type", "sym_visibility"}:
            raise UnsupportedOutlineIR(f"@{fn.sym_name.data} carries unmodeled function metadata")
        name = fn.sym_name.data
        if name in funcs:
            raise UnsupportedOutlineIR(f"duplicate function @{name}")
        if len(fn.body.blocks) != 1:
            raise UnsupportedOutlineIR(f"@{name} has no single-block definition")
        funcs[name] = fn
    return funcs


def _encode_block(block, inputs: tuple, funcs: dict, active: frozenset[str], *, region: bool):
    """Build a canonical expression DAG from *all* reachable pure operations."""
    if len(block.args) != len(inputs):
        raise UnsupportedOutlineIR("block argument count differs from its declaration")
    env = dict(zip(block.args, inputs, strict=True))
    terminator = "linalg.yield" if region else "func.return"
    output = None
    calls = set()
    for op in block.ops:
        if output is not None:
            raise UnsupportedOutlineIR("operation follows a terminator")
        operands = []
        for value in op.operands:
            if value not in env:
                raise UnsupportedOutlineIR(f"{op.name} reads an unbound SSA value")
            operands.append(env[value])
        operand_types = tuple(str(value.type) for value in op.operands)
        result_types = tuple(str(value.type) for value in op.results)
        if op.name == terminator:
            if op.attributes or op.properties or op.regions or op.results:
                raise UnsupportedOutlineIR(f"{terminator} has unmodeled metadata")
            output = tuple(operands)
            continue
        if op.name == "func.call" and not region:
            if op.attributes or set(op.properties) != {"callee"} or op.regions:
                raise UnsupportedOutlineIR("call has unmodeled metadata or regions")
            callee = op.properties["callee"]
            if callee.nested_references.data:
                raise UnsupportedOutlineIR("nested callee symbols are unsupported")
            name = callee.root_reference.data
            called = funcs.get(name)
            if called is None:
                raise UnsupportedOutlineIR(f"external call @{name} is unsupported")
            if tuple(str(t) for t in called.function_type.inputs) != operand_types:
                raise UnsupportedOutlineIR(f"call to @{name} has mismatched operand types")
            if tuple(str(t) for t in called.function_type.outputs) != result_types:
                raise UnsupportedOutlineIR(f"call to @{name} has mismatched result types")
            values, nested_calls = _encode_function(funcs, name, tuple(operands), active)
            calls.add(name)
            calls.update(nested_calls)
        else:
            if op.name == "linalg.matmul":
                if region or len(op.results) != 1 or len(op.operands) != 3:
                    raise UnsupportedOutlineIR("only top-level single-result tensor matmul is supported")
                if not all(_integer_tensor(value.type) for value in (*op.operands, *op.results)):
                    raise UnsupportedOutlineIR("matmul requires static, unencoded integer tensors")
                if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
                    raise UnsupportedOutlineIR("matmul region must contain one block")
            elif op.name == "arith.constant":
                if op.regions or len(op.results) != 1 or not _scalar_integer(op.results[0].type):
                    raise UnsupportedOutlineIR("only scalar integer constants are supported")
            elif op.name in {"arith.addi", "arith.subi", "arith.muli"}:
                if op.regions or len(op.operands) != 2 or len(op.results) != 1:
                    raise UnsupportedOutlineIR(f"{op.name} has unsupported shape")
                if not all(_scalar_integer(value.type) for value in (*op.operands, *op.results)):
                    raise UnsupportedOutlineIR(f"{op.name} requires scalar integers")
            else:
                raise UnsupportedOutlineIR(f"unmodeled operation {op.name}")
            regions = []
            for subregion in op.regions:
                if len(subregion.blocks) != 1:
                    raise UnsupportedOutlineIR("multiple region blocks are unsupported")
                subblock = subregion.blocks[0]
                # Region captures are deliberately not modeled. Pure matmul's
                # canonical region uses only its block arguments.
                region_args = tuple(("region_arg", i, str(arg.type)) for i, arg in enumerate(subblock.args))
                region_result, region_calls = _encode_block(subblock, region_args, funcs, active, region=True)
                if region_calls:
                    raise UnsupportedOutlineIR("calls inside matmul regions are unsupported")
                regions.append((tuple(str(arg.type) for arg in subblock.args), region_result))
            primitive = ("op", op.name, operand_types, result_types, _attrs(op), tuple(operands), tuple(regions))
            values = tuple(("result", primitive, i) for i in range(len(op.results)))
        if len(values) != len(op.results):
            raise UnsupportedOutlineIR(f"{op.name} returned an unexpected number of values")
        env.update(zip(op.results, values, strict=True))
    if output is None:
        raise UnsupportedOutlineIR(f"block has no {terminator}")
    return output, calls


def _encode_function(funcs: dict, name: str, inputs: tuple, active: frozenset[str]):
    if name in active:
        raise UnsupportedOutlineIR(f"recursive call to @{name} is unsupported")
    fn = funcs[name]
    block = fn.body.blocks[0]
    if tuple(str(arg.type) for arg in block.args) != tuple(str(t) for t in fn.function_type.inputs):
        raise UnsupportedOutlineIR(f"@{name} block arguments differ from function type")
    if len(inputs) != len(block.args):
        raise UnsupportedOutlineIR(f"@{name} call arity differs from function type")
    values, calls = _encode_block(block, inputs, funcs, active | {name}, region=False)
    if len(values) != len(fn.function_type.outputs):
        raise UnsupportedOutlineIR(f"@{name} return arity differs from function type")
    # The xDSL verifier checks return value types. Repeat explicitly because
    # equality of this graph depends on a typed call/result interface.
    return values, calls


def _graph_digest(graph: tuple) -> str:
    return _digest(json.dumps(graph, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _first_difference(left, right, path: str = "outputs") -> dict[str, str]:
    if isinstance(left, tuple) and isinstance(right, tuple):
        for i, (a, b) in enumerate(zip(left, right)):
            if a != b:
                return _first_difference(a, b, f"{path}[{i}]")
        if len(left) != len(right):
            return {"path": path, "source": f"length {len(left)}", "target": f"length {len(right)}"}
    return {"path": path, "source": repr(left)[:160], "target": repr(right)[:160]}


def verify_outline_transform(source: bytes, target: bytes, *, entry: str = "forward") -> OutlineTransformReceipt:
    """Check exact emitted IR pair by expanding calls and comparing pure SSA graphs.

    A verified result assumes unchanged MLIR primitive semantics, deterministic
    pure tensor matmul and scalar arithmetic, and well-defined input tensors.
    It does not prove those assumptions, quantify over dynamic shapes, or cover
    effects, aliasing, bufferization, quantization, LLVM, or device code.
    """
    if type(source) is not bytes or type(target) is not bytes or not entry:
        raise ValueError("source and target must be bytes and entry must be nonempty")
    base = dict(
        schema="merlin.outline_transform_verification.v1",
        method="syntactic_beta_equivalence_actual_outline_ir",
        entry=entry,
        source_sha256=_digest(source),
        target_sha256=_digest(target),
        verifier_sha256=_digest(Path(__file__).read_bytes()),
        parser_sha256=_parser_digest(),
        xdsl_version=_xdsl_version(),
        input_types=(),
        output_types=(),
        source_graph_sha256=None,
        target_graph_sha256=None,
        status="unsupported",
        reason=None,
        mismatch_witness=None,
    )
    if base["xdsl_version"] is None:
        return OutlineTransformReceipt(**{**base, "status": "unavailable", "reason": "xdsl is not installed"})
    try:
        from merlin.common import mlir_query as mq

        src = mq.parse(source.decode("utf-8"))
        dst = mq.parse(target.decode("utf-8"))
        src.verify()
        dst.verify()
        src_funcs, dst_funcs = _functions(src), _functions(dst)
        if set(src_funcs) != {entry} or entry not in dst_funcs:
            raise UnsupportedOutlineIR("source must contain only the defined entry; target must retain it")
        src_fn, dst_fn = src_funcs[entry], dst_funcs[entry]
        src_inputs = tuple(str(t) for t in src_fn.function_type.inputs)
        src_outputs = tuple(str(t) for t in src_fn.function_type.outputs)
        base["input_types"], base["output_types"] = src_inputs, src_outputs
        if not src_outputs or src_inputs != tuple(str(t) for t in dst_fn.function_type.inputs):
            raise UnsupportedOutlineIR("source/target entry inputs differ or no output is observable")
        if src_outputs != tuple(str(t) for t in dst_fn.function_type.outputs):
            raise UnsupportedOutlineIR("source/target entry results differ")
        if _attrs(src_fn) != _attrs(dst_fn):
            raise UnsupportedOutlineIR("entry function metadata differs")
        for name, fn in dst_funcs.items():
            if name != entry:
                if not name.startswith(f"{entry}$kernel_"):
                    raise UnsupportedOutlineIR(f"helper @{name} is not an emitted outline kernel")
                if getattr(fn.sym_visibility, "data", None) != "private":
                    raise UnsupportedOutlineIR(f"outlined helper @{name} is not private")
        args = tuple(("arg", i, typ) for i, typ in enumerate(src_inputs))
        expected, src_calls = _encode_function(src_funcs, entry, args, frozenset())
        actual, dst_calls = _encode_function(dst_funcs, entry, args, frozenset())
        if src_calls or not dst_calls or set(dst_funcs) != {entry, *dst_calls}:
            raise UnsupportedOutlineIR("target is not a closed, reachable outline of the call-free source")
        base["source_graph_sha256"] = _graph_digest(expected)
        base["target_graph_sha256"] = _graph_digest(actual)
        if expected == actual:
            base["status"] = "verified"
        else:
            base["status"] = "mismatch"
            base["reason"] = "beta-expanded result graphs differ; no numeric counterexample is claimed"
            base["mismatch_witness"] = _first_difference(expected, actual)
    except UnsupportedOutlineIR as exc:
        base["status"], base["reason"] = "unsupported", str(exc)
    except Exception as exc:
        base["status"], base["reason"] = "error", f"{type(exc).__name__}: {exc}"
    return OutlineTransformReceipt(**base)


def qualify_outline_receipt(receipt: OutlineTransformReceipt, source: bytes, target: bytes) -> bool:
    """Rerun the exact byte-bound obligation with the current verifier/parser."""
    return (
        receipt.verified
        and receipt.schema == "merlin.outline_transform_verification.v1"
        and verify_outline_transform(source, target, entry=receipt.entry) == receipt
    )


# Preserve published exception/record identities through the exact legacy alias.
UnsupportedOutlineIR.__module__ = "merlin.verification.outline_ir"
OutlineTransformReceipt.__module__ = "merlin.verification.outline_ir"
