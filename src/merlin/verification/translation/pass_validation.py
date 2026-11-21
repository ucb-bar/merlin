"""Byte-bound QF_BV translation validation for pure scalar xDSL functions.

The first consumer is ``perf.whole_model_chunks.chunk_forward``. This interpreter deliberately
supports single-block integer functions with constants, wrapping arithmetic,
width conversions, signed/unsigned min/max and comparisons, select, defined
calls and returns. Every other reachable operation abstains.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from xdsl.dialects import builtin, func

from merlin.compiler.transforms.certificates import ir_digest

# ---- scalar_ir ---------------------------------------------------------------------------------


class UnsupportedScalarIR(ValueError):
    """The selected before/after pair is outside this semantics."""


@dataclass(frozen=True)
class ScalarTransformReceipt:
    schema: str
    method: str
    entry: str
    source_sha256: str
    target_sha256: str
    verifier_sha256: str
    parser_sha256: str
    xdsl_version: str | None
    z3_version: str | None
    input_types: tuple[str, ...]
    output_types: tuple[str, ...]
    timeout_ms: int
    status: str  # verified | refuted | unknown | unsupported | unavailable | error
    reason: str | None
    query_sha256: str | None
    counterexample: tuple[int, ...] | None

    @property
    def verified(self) -> bool:
        return self.method == "qf_bv_actual_ir_translation_validation" and self.status == "verified"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, record: dict) -> ScalarTransformReceipt:
        if record.get("schema") != "merlin.scalar_transform_verification.v1":
            raise ValueError(f"unrecognized scalar transformation receipt schema {record.get('schema')!r}")
        return cls(
            **{
                **record,
                "input_types": tuple(record["input_types"]),
                "output_types": tuple(record["output_types"]),
                "counterexample": (tuple(record["counterexample"]) if record["counterexample"] is not None else None),
            }
        )


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _xdsl_version() -> str | None:
    try:
        return version("xdsl")
    except PackageNotFoundError:
        return None


def _width(typ) -> int:
    from xdsl.dialects.builtin import IntegerType

    if not isinstance(typ, IntegerType):
        raise UnsupportedScalarIR(f"only scalar integer types are modeled, not {typ}")
    width = int(typ.width.data)
    if width < 1:
        raise UnsupportedScalarIR(f"invalid integer width {width}")
    return width


def _functions(module) -> dict[str, object]:
    from xdsl.dialects.func import FuncOp

    found = {}
    for op in module.body.block.ops:
        if not isinstance(op, FuncOp):
            raise UnsupportedScalarIR(f"module contains unsupported top-level operation {op.name}")
        if op.attributes or set(op.properties) - {"sym_name", "function_type", "sym_visibility"}:
            raise UnsupportedScalarIR(f"function @{op.sym_name.data} carries unmodeled attributes or properties")
        name = op.sym_name.data
        if name in found:
            raise UnsupportedScalarIR(f"duplicate function @{name}")
        found[name] = op
    return found


def _encode_function(funcs, name, inputs, active):
    import z3
    from xdsl.dialects.builtin import IntegerAttr

    if name in active:
        raise UnsupportedScalarIR(f"recursive call to @{name} is not modeled")
    func = funcs.get(name)
    if func is None or len(func.body.blocks) != 1:
        raise UnsupportedScalarIR(f"@{name} has no single-block definition")
    block = func.body.blocks[0]
    declared_inputs = tuple(_width(typ) for typ in func.function_type.inputs)
    block_inputs = tuple(_width(arg.type) for arg in block.args)
    if declared_inputs != block_inputs:
        raise UnsupportedScalarIR(f"@{name} block arguments differ from the function declaration")
    if len(block.args) != len(inputs):
        raise UnsupportedScalarIR(f"@{name} call arity differs from definition")
    env = {}
    for arg, value in zip(block.args, inputs, strict=True):
        if _width(arg.type) != value.size():
            raise UnsupportedScalarIR(f"@{name} input width differs from call")
        env[arg] = value
    returned = None
    for op in block.ops:
        values = []
        for operand in op.operands:
            if operand not in env:
                raise UnsupportedScalarIR(f"@{name} {op.name} reads an unbound SSA value")
            values.append(env[operand])
        if op.name == "func.return":
            if returned is not None:
                raise UnsupportedScalarIR(f"@{name} has multiple returns")
            if op.attributes or op.properties:
                raise UnsupportedScalarIR(f"@{name} return carries unmodeled properties")
            returned = tuple(values)
            continue
        if returned is not None:
            raise UnsupportedScalarIR(f"@{name} has instructions after return")
        if op.attributes:
            raise UnsupportedScalarIR(f"@{name} {op.name} carries unmodeled attributes")
        if op.name == "arith.constant":
            if set(op.properties) != {"value"}:
                raise UnsupportedScalarIR(f"@{name} constant carries unmodeled properties")
            attr = op.properties.get("value")
            if not isinstance(attr, IntegerAttr) or len(op.results) != 1:
                raise UnsupportedScalarIR("only scalar integer constants are modeled")
            result = (z3.BitVecVal(int(attr.value.data), _width(op.results[0].type)),)
        elif op.name in ("arith.addi", "arith.subi", "arith.muli"):
            if set(op.properties) - {"overflowFlags"}:
                raise UnsupportedScalarIR(f"@{name} {op.name} carries unmodeled properties")
            flags = op.properties.get("overflowFlags")
            if flags is not None and getattr(flags, "data", None) != frozenset():
                raise UnsupportedScalarIR(f"@{name} {op.name} overflow flags have unmodeled poison semantics")
            if len(values) != 2 or len(op.results) != 1:
                raise UnsupportedScalarIR(f"@{name} {op.name} has unexpected arity")
            width = _width(op.results[0].type)
            if any(value.size() != width for value in values):
                raise UnsupportedScalarIR(f"@{name} {op.name} operand widths differ")
            result = (
                (values[0] + values[1])
                if op.name == "arith.addi"
                else (values[0] - values[1])
                if op.name == "arith.subi"
                else (values[0] * values[1]),
            )
        elif op.name in ("arith.extsi", "arith.extui", "arith.trunci"):
            if op.properties or len(values) != 1 or len(op.results) != 1:
                raise UnsupportedScalarIR("integer conversion has unsupported properties or arity")
            width, old = _width(op.results[0].type), values[0].size()
            if op.name == "arith.trunci":
                if width >= old:
                    raise UnsupportedScalarIR("truncate must decrease integer width")
                result = (z3.Extract(width - 1, 0, values[0]),)
            else:
                if width <= old:
                    raise UnsupportedScalarIR("extension must increase integer width")
                extend = z3.SignExt if op.name == "arith.extsi" else z3.ZeroExt
                result = (extend(width - old, values[0]),)
        elif op.name in ("arith.minsi", "arith.maxsi", "arith.minui", "arith.maxui", "arith.cmpi"):
            allowed = {"predicate"} if op.name == "arith.cmpi" else set()
            if set(op.properties) != allowed or len(values) != 2 or len(op.results) != 1:
                raise UnsupportedScalarIR("integer comparison has unsupported properties or arity")
            left, right = values
            if left.size() != right.size():
                raise UnsupportedScalarIR("integer comparison operand widths differ")
            if op.name == "arith.cmpi":
                predicate = op.properties["predicate"]
                if not isinstance(predicate, IntegerAttr) or _width(op.results[0].type) != 1:
                    raise UnsupportedScalarIR("comparison requires an integer predicate and i1 result")
                predicates = (
                    left == right,
                    left != right,
                    left < right,
                    left <= right,
                    left > right,
                    left >= right,
                    z3.ULT(left, right),
                    z3.ULE(left, right),
                    z3.UGT(left, right),
                    z3.UGE(left, right),
                )
                code = predicate.value.data
                if not 0 <= code < len(predicates):
                    raise UnsupportedScalarIR("unknown integer comparison predicate")
                result = (z3.If(predicates[code], z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),)
            else:
                if _width(op.results[0].type) != left.size():
                    raise UnsupportedScalarIR("integer min/max changes result width")
                minimum, unsigned = "min" in op.name, op.name.endswith("ui")
                condition = (
                    (z3.ULT(left, right) if minimum else z3.UGT(left, right))
                    if unsigned
                    else (left < right if minimum else left > right)
                )
                result = (z3.If(condition, left, right),)
        elif op.name == "arith.select":
            if op.properties or len(values) != 3 or len(op.results) != 1:
                raise UnsupportedScalarIR("integer select has unsupported properties or arity")
            cond, yes, no = values
            if cond.size() != 1 or yes.size() != no.size() or _width(op.results[0].type) != yes.size():
                raise UnsupportedScalarIR("integer select operand widths differ")
            result = (z3.If(cond != 0, yes, no),)
        elif op.name == "func.call":
            if set(op.properties) != {"callee"}:
                raise UnsupportedScalarIR(f"@{name} call carries unmodeled properties")
            callee = op.properties["callee"]
            if callee.nested_references.data:
                raise UnsupportedScalarIR("nested call symbol is not modeled")
            result = _encode_function(funcs, callee.root_reference.data, values, active | {name})
        else:
            raise UnsupportedScalarIR(f"@{name} contains unmodeled operation {op.name}")
        if len(result) != len(op.results):
            raise UnsupportedScalarIR(f"@{name} {op.name} result count differs from definition")
        for ssa, value in zip(op.results, result, strict=True):
            if _width(ssa.type) != value.size():
                raise UnsupportedScalarIR(f"@{name} {op.name} result width differs from definition")
            env[ssa] = value
    if returned is None:
        raise UnsupportedScalarIR(f"@{name} has no return")
    declared = tuple(str(typ) for typ in func.function_type.outputs)
    if tuple(f"i{value.size()}" for value in returned) != declared:
        raise UnsupportedScalarIR(f"@{name} return types differ from declaration")
    return returned


def verify_scalar_transform(
    source: bytes,
    target: bytes,
    *,
    entry: str = "forward",
    timeout_ms: int = 30_000,
) -> ScalarTransformReceipt:
    """Prove actual byte-pinned IR pair or record why no proof was obtained.

    The source and target are parsed independently, then encoded over identical
    symbolic input leaves. This checks values only; no memory/effects/FP/ABI,
    no actual LLVM/object semantics, and no claim about all shapes.
    """
    if type(source) is not bytes or type(target) is not bytes or timeout_ms < 1:
        raise ValueError("source/target must be bytes and timeout positive")
    base = dict(
        schema="merlin.scalar_transform_verification.v1",
        method="qf_bv_actual_ir_translation_validation",
        entry=entry,
        source_sha256=_digest(source),
        target_sha256=_digest(target),
        verifier_sha256=_digest(Path(__file__).read_bytes()),
        parser_sha256=_digest((Path(__file__).parents[2] / "compiler" / "ir" / "mlir_query.py").read_bytes()),
        xdsl_version=_xdsl_version(),
        z3_version=None,
        input_types=(),
        output_types=(),
        timeout_ms=timeout_ms,
        status="unsupported",
        reason=None,
        query_sha256=None,
        counterexample=None,
    )
    if base["xdsl_version"] is None:
        return ScalarTransformReceipt(**{**base, "status": "unavailable", "reason": "xdsl is not installed"})
    try:
        import z3
    except ImportError:
        return ScalarTransformReceipt(**{**base, "status": "unavailable", "reason": "z3 is not installed"})
    base["z3_version"] = z3.get_version_string()
    try:
        from merlin.compiler.ir import mlir_query as mq

        src_mod, dst_mod = mq.parse(source.decode("utf-8")), mq.parse(target.decode("utf-8"))
        if src_mod.attributes or dst_mod.attributes:
            raise UnsupportedScalarIR("module attributes are not modeled")
        src_mod.verify()
        dst_mod.verify()
        src_funcs, dst_funcs = _functions(src_mod), _functions(dst_mod)
        src_func, dst_func = src_funcs.get(entry), dst_funcs.get(entry)
        if src_func is None or dst_func is None:
            raise UnsupportedScalarIR(f"entry @{entry} missing from source or target")
        src_types = tuple(str(arg.type) for arg in src_func.body.block.args)
        dst_types = tuple(str(arg.type) for arg in dst_func.body.block.args)
        src_outputs = tuple(str(typ) for typ in src_func.function_type.outputs)
        dst_outputs = tuple(str(typ) for typ in dst_func.function_type.outputs)
        base["input_types"], base["output_types"] = src_types, src_outputs
        if src_types != dst_types or src_outputs != dst_outputs:
            raise UnsupportedScalarIR("entry source/target signatures differ")
        if not src_outputs:
            raise UnsupportedScalarIR("entry has no observable return value")
        symbols = tuple(z3.BitVec(f"arg_{i}", _width(arg.type)) for i, arg in enumerate(src_func.body.block.args))
        expected = _encode_function(src_funcs, entry, symbols, frozenset())
        actual = _encode_function(dst_funcs, entry, symbols, frozenset())
        if len(expected) != len(actual):
            raise UnsupportedScalarIR("entry return counts differ")
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        solver.add(z3.Or(*[a != b for a, b in zip(expected, actual, strict=True)]))
        base["query_sha256"] = _digest(solver.sexpr().encode("utf-8"))
        verdict = solver.check()
        if verdict == z3.unsat:
            base["status"] = "verified"
        elif verdict == z3.sat:
            model = solver.model()
            base["status"] = "refuted"
            base["counterexample"] = tuple(model.eval(sym, model_completion=True).as_long() for sym in symbols)
        else:
            base["status"], base["reason"] = "unknown", solver.reason_unknown()
    except UnsupportedScalarIR as exc:
        base["status"], base["reason"] = "unsupported", str(exc)
    except Exception as exc:
        base["status"], base["reason"] = "error", f"{type(exc).__name__}: {exc}"
    return ScalarTransformReceipt(**base)


def qualify_scalar_receipt(
    receipt: ScalarTransformReceipt,
    source: bytes,
    target: bytes,
) -> bool:
    """Rerun the same obligation on the same bytes/toolchain and require equality."""
    if not receipt.verified or receipt.schema != "merlin.scalar_transform_verification.v1":
        return False
    return verify_scalar_transform(source, target, entry=receipt.entry, timeout_ms=receipt.timeout_ms) == receipt

    # Preserve published exception/record identities through the exact legacy alias.


UnsupportedScalarIR.__module__ = "merlin.verification.translation.pass_validation"
ScalarTransformReceipt.__module__ = "merlin.verification.translation.pass_validation"


# ---- outline_ir --------------------------------------------------------------------------------
# Byte-bound structural translation validation for a narrow real outlining pass.
#
# For pure operations, beta-expanding an outlined call must reconstruct the exact
# source SSA result graph (including the matmul region). Identical primitive
# applications to identical operands have identical behavior under their unchanged
# MLIR semantics. This is *not* an independent proof of those primitive semantics,
# tensor arithmetic, physical memory, or generated code.


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
    root = Path(__file__).parents[2]
    # mlir_query delegates to linalg_mlir.parse_mlir_text, whose context also
    # registers fp8 types. Bind all repository-owned parser components used by
    # this route, not just its thin query wrapper.
    paths = (
        root / "compiler" / "ir" / "mlir_query.py",
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


def _outline_functions(module) -> dict[str, Any]:
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
            values, nested_calls = _encode_outline_function(funcs, name, tuple(operands), active)
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


def _encode_outline_function(funcs: dict, name: str, inputs: tuple, active: frozenset[str]):
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
        from merlin.compiler.ir import mlir_query as mq

        src = mq.parse(source.decode("utf-8"))
        dst = mq.parse(target.decode("utf-8"))
        src.verify()
        dst.verify()
        src_funcs, dst_funcs = _outline_functions(src), _outline_functions(dst)
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
        expected, src_calls = _encode_outline_function(src_funcs, entry, args, frozenset())
        actual, dst_calls = _encode_outline_function(dst_funcs, entry, args, frozenset())
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


UnsupportedOutlineIR.__module__ = "merlin.verification.translation.pass_validation"
OutlineTransformReceipt.__module__ = "merlin.verification.translation.pass_validation"


# ---- split_reduction ---------------------------------------------------------------------------
# Bounded translation validation for an exact signed split reduction.
#
# This is an algebraic pilot, not a verifier for an emitted kernel. The caller must
# separately establish that its source and target IR implement these two expressions,
# and that the declared widths agree with the selected hardware facts.


@dataclass(frozen=True)
class SplitReductionVerdict:
    status: str  # verified | refuted | unknown
    k: int
    operand_width: int
    partial_width: int
    output_width: int
    chunks: tuple[int, ...]
    counterexample: tuple[tuple[int, int], ...] | None = None
    reason: str | None = None


def maximum_safe_chunk(operand_width: int, partial_width: int) -> int:
    """Conservative full-domain K bound for a signed product sum.

    The maximum absolute product of two signed w-bit values is 2**(2*w-2).
    Bounding absolute sums by K times that product covers both signs.
    """
    if operand_width < 2 or partial_width < operand_width * 2:
        raise ValueError("partial width must hold a full signed operand product")
    return ((1 << (partial_width - 1)) - 1) // (1 << (2 * operand_width - 2))


def verify_split_reduction(
    *,
    k: int,
    operand_width: int,
    partial_width: int,
    output_width: int,
    chunks: tuple[int, ...],
    timeout_ms: int = 30_000,
) -> SplitReductionVerdict:
    """Prove or refute equality for all signed input pairs at this concrete K.

    The source multiplies and accumulates at ``output_width`` modulo 2**width.
    Each target chunk multiplies and accumulates at ``partial_width`` modulo
    2**width, then sign-extends its partial sum into the output accumulator.
    Missing/duplicated reduction terms are rejected before invoking the solver.
    ``unknown`` is never a proof.
    """
    if k < 1 or timeout_ms < 1 or operand_width < 2:
        raise ValueError("k, timeout and operand width must be positive and nontrivial")
    if partial_width < operand_width * 2 or output_width < partial_width:
        raise ValueError("widths must satisfy output >= partial >= twice operand width")
    if not chunks or any(size < 1 for size in chunks) or sum(chunks) != k:
        raise ValueError("positive chunks must cover every reduction term exactly once")

    import z3

    lhs = [z3.BitVec(f"lhs_{i}", operand_width) for i in range(k)]
    rhs = [z3.BitVec(f"rhs_{i}", operand_width) for i in range(k)]

    def product(i: int, width: int):
        return z3.SignExt(width - operand_width, lhs[i]) * z3.SignExt(width - operand_width, rhs[i])

    source = z3.BitVecVal(0, output_width)
    for i in range(k):
        source += product(i, output_width)

    target = z3.BitVecVal(0, output_width)
    offset = 0
    for count in chunks:
        partial = z3.BitVecVal(0, partial_width)
        for i in range(offset, offset + count):
            partial += product(i, partial_width)
        target += z3.SignExt(output_width - partial_width, partial)
        offset += count

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(source != target)
    result = solver.check()
    base = dict(
        k=k,
        operand_width=operand_width,
        partial_width=partial_width,
        output_width=output_width,
        chunks=chunks,
    )
    if result == z3.unsat:
        return SplitReductionVerdict(status="verified", **base)
    if result == z3.sat:
        model = solver.model()
        sign_bit = 1 << (operand_width - 1)
        modulus = 1 << operand_width

        def signed(value):
            raw = model.eval(value, model_completion=True).as_long()
            return raw - modulus if raw & sign_bit else raw

        return SplitReductionVerdict(
            status="refuted",
            counterexample=tuple((signed(lhs[i]), signed(rhs[i])) for i in range(k)),
            **base,
        )
    return SplitReductionVerdict(status="unknown", reason=solver.reason_unknown(), **base)

    # Preserve published exception/record identities through the exact legacy alias.


SplitReductionVerdict.__module__ = "merlin.verification.translation.pass_validation"


# ---- partition ---------------------------------------------------------------------------------
# Check actual outlined IR by reconstructing its complete original SSA program.
#
# This strict beta-expansion check preserves operation order and every operation,
# including dead computations. It establishes partitioning fidelity under existing
# MLIR call semantics, not correctness of subsequent backend lowering or execution.


@dataclass(frozen=True)
class PartitionVerification:
    source_sha256: str
    target_sha256: str
    verifier_sha256: str
    method: str = "complete_ssa_beta_expansion"

    # Preserve published pickle references while importing the actual checker owner.


PartitionVerification.__module__ = "merlin.verification.translation.pass_validation"


def verify_partition(source, partitioned, *, entry: str, symbols: tuple[str, ...]):
    if type(source) is not builtin.ModuleOp or type(partitioned) is not builtin.ModuleOp:
        raise ValueError("partition verification requires actual source and target modules")
    if type(symbols) is not tuple or len(set(symbols)) != len(symbols):
        raise ValueError("partition verification requires unique outlined symbols")
    source.verify()
    partitioned.verify()
    restored = partitioned.clone()
    originals = {op.sym_name.data for op in source.body.block.ops if type(op) is func.FuncOp}
    functions = [op for op in restored.body.block.ops if type(op) is func.FuncOp]
    by_name = {op.sym_name.data: op for op in functions}
    if len(by_name) != len(functions) or set(by_name) != originals | set(symbols) or originals & set(symbols):
        raise ValueError("partition changed original function ownership or added undeclared functions")
    driver = by_name.get(entry)
    if driver is None or len(driver.body.blocks) != 1:
        raise ValueError("partition requires the original single-block entry")
    observed = []
    for call in tuple(restored.walk()):
        if type(call) is not func.CallOp or call.callee.root_reference.data not in symbols:
            continue
        name = call.callee.root_reference.data
        if (
            call.parent is not driver.body.block
            or call.attributes
            or set(call.properties) != {"callee"}
            or call.callee.nested_references.data
        ):
            raise ValueError("outlined invocation has unmodeled effects, metadata or control")
        helper = by_name[name]
        if (
            helper.attributes
            or set(helper.properties) - {"sym_name", "function_type", "sym_visibility"}
            or getattr(helper.sym_visibility, "data", None) != "private"
            or len(helper.body.blocks) != 1
        ):
            raise ValueError("outlined function must be a plain private definition")
        block = helper.body.block
        terminator = block.last_op
        if type(terminator) is not func.ReturnOp or terminator.attributes or terminator.properties:
            raise ValueError("outlined function has no ordinary return")
        if tuple(v.type for v in call.operands) != tuple(a.type for a in block.args) or tuple(
            v.type for v in call.results
        ) != tuple(v.type for v in terminator.operands):
            raise ValueError("outlined invocation changed its ordered SSA boundary")
        mapping = dict(zip(block.args, call.operands, strict=True))
        for operation in tuple(block.ops)[:-1]:
            if any(
                type(nested) is func.CallOp and nested.callee.root_reference.data in symbols
                for nested in operation.walk()
            ):
                raise ValueError("nested outlined invocations are outside this partition contract")
            driver.body.block.insert_op_before(operation.clone(value_mapper=mapping), call)
        for result, value in zip(call.results, terminator.operands, strict=True):
            result.replace_all_uses_with(mapping[value])
        driver.body.block.erase_op(call)
        observed.append(name)
    if len(observed) != len(symbols) or set(observed) != set(symbols):
        raise ValueError("partition must invoke every outlined definition exactly once")
    for name in symbols:
        restored.body.block.erase_op(by_name[name])
    restored.verify()
    if not source.is_structurally_equivalent(restored):
        raise ValueError("partition beta-expansion differs from the complete original source")
    return PartitionVerification(
        ir_digest(source), ir_digest(partitioned), hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )


def verify_region_preview(source, preview, *, entry, operations, inputs, outputs):
    """Check an actual helper by plugging it into full original SSA and expanding.

    The supplied helper body is consumed as-is. This checker does not rebuild it
    from the original operations, and retains the complete partition verifier's
    existing attribute, boundary, dead-operation and exact reconstruction rules.
    """
    if type(source) is not builtin.ModuleOp or type(preview) is not builtin.ModuleOp:
        raise ValueError("region preview verification requires actual typed modules")
    definitions = tuple(preview.body.block.ops)
    if len(definitions) != 1 or type(definitions[0]) is not func.FuncOp:
        raise ValueError("region preview must contain one actual private helper")
    helper = definitions[0]
    symbol = helper.sym_name.data
    if any(getattr(op, "sym_name", None) is not None and op.sym_name.data == symbol for op in source.body.block.ops):
        raise ValueError("region preview symbol conflicts with original function ownership")
    restored = source.clone()
    originals, copies = tuple(source.walk()), tuple(restored.walk())
    operation_map = dict(zip(originals, copies, strict=True))
    values = {}
    for left, right in zip(originals, copies, strict=True):
        values.update(zip(left.results, right.results, strict=True))
        for a, b in zip(left.regions, right.regions, strict=True):
            for old, new in zip(a.blocks, b.blocks, strict=True):
                values.update(zip(old.args, new.args, strict=True))
    cloned = tuple(operation_map[operation] for operation in operations)
    call = func.CallOp(symbol, [values[value] for value in inputs], [value.type for value in outputs])
    cloned[0].parent.insert_op_before(call, cloned[0])
    for old, new in zip(outputs, call.results, strict=True):
        values[old].replace_all_uses_with(new)
    for operation in reversed(cloned):
        operation.parent.erase_op(operation)
    restored.body.block.add_op(helper.clone())
    return verify_partition(source, restored, entry=entry, symbols=(symbol,))
