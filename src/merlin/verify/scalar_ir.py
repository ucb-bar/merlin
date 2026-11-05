"""Byte-bound QF_BV translation validation for pure scalar xDSL functions.

The first consumer is ``perf.whole_model_chunks.chunk_forward``. This interpreter deliberately
supports single-block integer functions with constants, wrapping arithmetic,
width conversions, signed/unsigned min/max and comparisons, select, defined
calls and returns. Every other reachable operation abstains.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


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
                    left == right, left != right, left < right, left <= right, left > right, left >= right,
                    z3.ULT(left, right), z3.ULE(left, right), z3.UGT(left, right), z3.UGE(left, right),
                )
                code = predicate.value.data
                if not 0 <= code < len(predicates):
                    raise UnsupportedScalarIR("unknown integer comparison predicate")
                result = (z3.If(predicates[code], z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)),)
            else:
                if _width(op.results[0].type) != left.size():
                    raise UnsupportedScalarIR("integer min/max changes result width")
                minimum, unsigned = "min" in op.name, op.name.endswith("ui")
                condition = (z3.ULT(left, right) if minimum else z3.UGT(left, right)) if unsigned else (
                    left < right if minimum else left > right
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
        parser_sha256=_digest((Path(__file__).parent.parent / "common" / "mlir_query.py").read_bytes()),
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
        from merlin.common import mlir_query as mq

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
