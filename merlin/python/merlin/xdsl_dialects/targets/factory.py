"""Parametric builder for a tensor-resident target dialect (xDSL).

Every tensor-resident target (toy_npu, saturn, gemmini, …) shares ONE dialect shape — four ops
(pack / matmul / commit / evict) over two types (a resident/packed tensor + an accumulator) — that
differ only in op/type NAMES and two small structural knobs. Those used to be hand-copied per target
(`targets/toynpu.py`, `targets/saturn.py` were ~90% identical). This factory synthesizes the IRDL op
and type classes dynamically from the target's committed ``dialect_plan.yaml`` (op/type names + the
interface→target lowering roles), so a reference target module is now data + one ``build_dialect``
call, and a generated target can be built the same way from its plan.

The two knobs the plan does not (yet) encode are passed explicitly by the caller:
``matmul_rhs_typed`` (NPU requires the RHS to be resident; a CPU accepts a plain tensor) and
``matmul_vl_policy`` (vector targets carry an optional vl-policy prop). Neither changes the emitted
MLIR for valid inputs — only op verification — so the built dialects reproduce the hand-written ones
byte-for-byte (golden-checked in tests/ir/test_target_factory.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                       operand_def, opt_prop_def, prop_def, result_def)
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

# Must stay aligned with interface.KNOWN_EPILOGUE and the runtime engine.
KNOWN_EPILOGUE = {"bias", "bias_add", "requant", "relu"}


@dataclass(frozen=True)
class BuiltDialect:
    """A dynamically-built tensor-resident dialect + the handles the lowering loop needs."""

    dialect: Dialect
    spec: Any                 # a target_lowering.TargetSpec
    pack_op: type
    matmul_op: type
    commit_op: type
    evict_op: type
    resident_type: type
    accumulator_type: type


def _roles(plan: dict[str, Any]) -> dict[str, str]:
    """interface role (resident_pack/matmul/commit/resident_evict) -> target op name."""
    out = {}
    for rule in plan.get("lowering", []):
        out[rule["from"].split(".", 1)[1]] = rule["to"].split(".", 1)[1]
    return out


def build_dialect(target: str, *, matmul_rhs_typed: bool = False,
                  matmul_vl_policy: bool = False) -> BuiltDialect:
    """Build the tensor-resident dialect for ``target`` from its committed dialect_plan."""
    from merlin.targetgen.target_registry import load_dialect_plan
    from ..lowering.target_lowering import TargetSpec

    plan = load_dialect_plan(target)
    dname = plan["dialect_name"]
    roles = _roles(plan)
    pack_n, mm_n = roles["resident_pack"], roles["matmul"]
    commit_n, evict_n = roles["commit"], roles["resident_evict"]
    resident_tn, acc_tn = (plan["types"][0]["name"], plan["types"][1]["name"])

    def _mk_type(tn: str) -> type:
        cls = type(f"{dname}_{tn}", (ParametrizedAttribute, TypeAttribute),
                   {"name": f"{dname}.{tn}", "__annotations__": {"element_type": Attribute},
                    "__module__": __name__, "__doc__": f"!{dname}.{tn}<element_type>"})
        return irdl_attr_definition(cls)

    resident_type = _mk_type(resident_tn)
    accumulator_type = _mk_type(acc_tn)

    def _op(clsname: str, ns: dict) -> type:
        ns = {**ns, "__module__": __name__}
        return irdl_op_definition(type(clsname, (IRDLOperation,), ns))

    pack_op = _op(f"{dname}_{pack_n}", {
        "name": f"{dname}.{pack_n}", "src": operand_def(),
        "layout": prop_def(StringAttr), "res": result_def(resident_type)})

    mm_ns = {"name": f"{dname}.{mm_n}", "lhs": operand_def(),
             "rhs": operand_def(resident_type) if matmul_rhs_typed else operand_def(),
             "acc": result_def(accumulator_type)}
    if matmul_vl_policy:
        mm_ns["vl_policy"] = opt_prop_def(StringAttr)
    matmul_op = _op(f"{dname}_{mm_n}", mm_ns)

    def _commit_verify(self) -> None:
        stages = []
        for entry in self.epilogue:
            stage = entry.data if isinstance(entry, StringAttr) else None
            if stage not in KNOWN_EPILOGUE:
                raise VerifyException(
                    f"{dname}.{commit_n} epilogue stage {stage!r} not in {sorted(KNOWN_EPILOGUE)}")
            stages.append(stage)
        if ("bias" in stages or "bias_add" in stages) and self.bias is None:
            raise VerifyException(
                f"{dname}.{commit_n} epilogue has a bias stage but no `bias` tensor name")

    commit_op = _op(f"{dname}_{commit_n}", {
        "name": f"{dname}.{commit_n}", "acc": operand_def(accumulator_type),
        "epilogue": prop_def(ArrayAttr), "bias": opt_prop_def(StringAttr),
        "requant_shift": opt_prop_def(IntegerAttr), "output_dtype": opt_prop_def(StringAttr),
        "out": result_def(), "verify_": _commit_verify})

    evict_op = _op(f"{dname}_{evict_n}", {
        "name": f"{dname}.{evict_n}", "handle": operand_def(resident_type)})

    dialect = Dialect(dname, [pack_op, matmul_op, commit_op, evict_op],
                      [resident_type, accumulator_type])
    spec = TargetSpec(target, None, pack_op, matmul_op, commit_op, evict_op,
                      resident_type, accumulator_type)
    return BuiltDialect(dialect, spec, pack_op, matmul_op, commit_op, evict_op,
                        resident_type, accumulator_type)
