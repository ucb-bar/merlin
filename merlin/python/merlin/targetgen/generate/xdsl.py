"""Generate a real xDSL prototype dialect from a dialect_plan.

For the ToyNPU op/type set this emits a fully-working IRDL dialect: parametrized types
(`resident_tensor`, `accumulator`), ops (`res_pack`, `matmul`, `commit`, `evict`) with real
operand/result/property definitions and a custom verifier, a registered `Dialect`, and a
`build_example`/round-trip helper. The companion test builds a module, verifies it, and
round-trips it through the xDSL parser/printer.

Every other reviewed plan gets a concrete IRDL operation and type class for each declaration.
The generic classes accept variadic operands/results until their signatures are refined during
human review. Either way the module imports as plain Python; if xDSL is not installed,
`HAS_XDSL` is False and the helpers no-op safely.
"""
from __future__ import annotations

from typing import Any

from ...common.artifacts import Artifact

KNOWN_TYPES = {"resident_tensor", "accumulator"}
KNOWN_OPS = {"res_pack", "matmul", "commit", "evict"}


def _class_token(name: str) -> str:
    words: list[str] = []
    current = ""
    for char in name:
        if char.isalnum():
            current += char
        elif current:
            words.append(current)
            current = ""
    if current:
        words.append(current)
    token = "".join(word[:1].upper() + word[1:] for word in words) or "Declared"
    return f"N{token}" if token[0].isdigit() else token

_REAL_DIALECT = '''"""Generated xDSL prototype dialect for `{dialect}` (real IRDL).

Run the round-trip test with `pytest test_{dialect}.py`. Promote stable pieces to the MLIR/C++
scaffold under include/ + lib/ once the syntax settles. Each op maps to a Merlin `interface`
interface abstraction (res_pack<-resident_pack, matmul<-matmul, commit<-commit, evict<-resident_evict).
"""
from __future__ import annotations

DIALECT_NAME = "{dialect}"
OPS = {ops!r}
TYPES = {types!r}
EXTRA_OP_SPECS = {extra_op_specs!r}
EXTRA_TYPE_SPECS = {extra_type_specs!r}

try:
    from xdsl.ir import Dialect, TypeAttribute, Attribute, Block, Region
    from xdsl.irdl import (irdl_op_definition, IRDLOperation, irdl_attr_definition,
                           operand_def, result_def, prop_def, ParametrizedAttribute,
                           var_operand_def, var_result_def)
    from xdsl.dialects.builtin import StringAttr, ArrayAttr, TensorType, i8, i32, ModuleOp, FunctionType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.utils.exceptions import VerifyException
    HAS_XDSL = True
except Exception:  # noqa: BLE001 - xDSL is an optional prototyping dependency
    HAS_XDSL = False


_KNOWN_EPILOGUE = {{"bias_add", "bias", "requant", "acc_scale", "relu", "maxpool"}}

if HAS_XDSL:

    @irdl_attr_definition
    class ResidentTensorType(ParametrizedAttribute, TypeAttribute):
        """!{dialect}.resident_tensor<element_type> — a tensor resident in target storage."""
        name = "{dialect}.resident_tensor"
        element_type: Attribute

    @irdl_attr_definition
    class AccumulatorType(ParametrizedAttribute, TypeAttribute):
        """!{dialect}.accumulator<element_type> — uncommitted accumulation state."""
        name = "{dialect}.accumulator"
        element_type: Attribute

    @irdl_op_definition
    class ResPackOp(IRDLOperation):
        """{dialect}.res_pack — pack + make an (immutable) RHS resident."""
        name = "{dialect}.res_pack"
        src = operand_def()
        layout = prop_def(StringAttr)
        res = result_def(ResidentTensorType)

    @irdl_op_definition
    class MatmulOp(IRDLOperation):
        """{dialect}.matmul — matmul against a resident tensor -> accumulator."""
        name = "{dialect}.matmul"
        lhs = operand_def()
        rhs = operand_def(ResidentTensorType)
        acc = result_def(AccumulatorType)

    @irdl_op_definition
    class CommitOp(IRDLOperation):
        """{dialect}.commit — apply epilogue and commit an accumulator to a tensor."""
        name = "{dialect}.commit"
        acc = operand_def(AccumulatorType)
        epilogue = prop_def(ArrayAttr)
        out = result_def()

        def verify_(self) -> None:
            for entry in self.epilogue:
                stage = entry.data if isinstance(entry, StringAttr) else None
                if stage not in _KNOWN_EPILOGUE:
                    raise VerifyException(
                        "commit epilogue stage %r not in %s" % (stage, sorted(_KNOWN_EPILOGUE)))

    @irdl_op_definition
    class EvictOp(IRDLOperation):
        """{dialect}.evict — free resident storage."""
        name = "{dialect}.evict"
        handle = operand_def(ResidentTensorType)

    _OP_CLASSES = [ResPackOp, MatmulOp, CommitOp, EvictOp]
    for _op_name, _class_name in EXTRA_OP_SPECS:
        _OP_CLASSES.append(irdl_op_definition(type(
            _class_name,
            (IRDLOperation,),
            {{"name": f"{dialect}.{{_op_name}}",
             "inputs": var_operand_def(),
             "outputs": var_result_def()}},
        )))

    _TYPE_CLASSES = [ResidentTensorType, AccumulatorType]
    for _type_name, _class_name in EXTRA_TYPE_SPECS:
        _TYPE_CLASSES.append(irdl_attr_definition(type(
            _class_name,
            (ParametrizedAttribute, TypeAttribute),
            {{"name": f"{dialect}.{{_type_name}}"}},
        )))
    {DIALECT_CONST} = Dialect("{dialect}", _OP_CLASSES, _TYPE_CLASSES)

    def get_dialect():
        return {DIALECT_CONST}

    def build_example() -> "ModuleOp":
        """Build a small, verifiable module exercising every op."""
        rt = ResidentTensorType(i8)
        acct = AccumulatorType(i32)
        At = TensorType(i8, [64, 128]); Wt = TensorType(i8, [128, 64]); Yt = TensorType(i8, [64, 64])
        blk = Block(arg_types=[At, Wt])
        a, w = blk.args
        rp = ResPackOp(operands=[w], result_types=[rt],
                       properties={{"layout": StringAttr("packed_rhs")}})
        mm = MatmulOp(operands=[a, rp.res], result_types=[acct])
        cm = CommitOp(operands=[mm.acc], result_types=[Yt],
                      properties={{"epilogue": ArrayAttr([StringAttr("requant"), StringAttr("relu")])}})
        ev = EvictOp(operands=[rp.res])
        ret = ReturnOp(cm.out)
        blk.add_ops([rp, mm, cm, ev, ret])
        fn = FuncOp("main", FunctionType.from_lists([At, Wt], [Yt]), Region([blk]))
        return ModuleOp([fn])

    def roundtrip(module: "ModuleOp") -> "ModuleOp":
        """Print and re-parse a module through xDSL; returns the parsed module."""
        import io
        from xdsl.context import Context
        from xdsl.parser import Parser
        from xdsl.printer import Printer
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func
        s = io.StringIO(); Printer(stream=s).print_op(module)
        ctx = Context(); ctx.load_dialect(Builtin); ctx.load_dialect(Func); ctx.load_dialect({DIALECT_CONST})
        return Parser(ctx, s.getvalue()).parse_module()

else:  # pragma: no cover - exercised only when xDSL is absent
    def get_dialect():
        return None

    def build_example():
        return None

    def roundtrip(module):
        return module
'''

_DECLARED_DIALECT = '''"""Generated concrete xDSL dialect for `{dialect}`.

Every operation and type below comes directly from the reviewed dialect plan. Operation
signatures stay variadic until target-specific verification is authored, but the declarations
are registered IRDL classes and can be parsed, constructed, walked, and rewritten.
"""
from __future__ import annotations

DIALECT_NAME = "{dialect}"
OPS = {ops!r}
TYPES = {types!r}
OP_SPECS = {op_specs!r}
TYPE_SPECS = {type_specs!r}

try:
    from xdsl.ir import Dialect, TypeAttribute
    from xdsl.irdl import (IRDLOperation, ParametrizedAttribute,
                           irdl_attr_definition, irdl_op_definition,
                           var_operand_def, var_result_def)
    HAS_XDSL = True

    OP_CLASSES = [
        irdl_op_definition(type(
            class_name,
            (IRDLOperation,),
            {{"name": f"{dialect}.{{op_name}}",
             "inputs": var_operand_def(),
             "outputs": var_result_def()}},
        ))
        for op_name, class_name in OP_SPECS
    ]
    TYPE_CLASSES = [
        irdl_attr_definition(type(
            class_name,
            (ParametrizedAttribute, TypeAttribute),
            {{"name": f"{dialect}.{{type_name}}"}},
        ))
        for type_name, class_name in TYPE_SPECS
    ]
    {DIALECT_CONST} = Dialect("{dialect}", OP_CLASSES, TYPE_CLASSES)

    def get_dialect():
        return {DIALECT_CONST}
except Exception:  # noqa: BLE001
    HAS_XDSL = False

    OP_CLASSES = []
    TYPE_CLASSES = []

    def get_dialect():
        return None
'''

_REAL_TEST = '''"""Round-trip + verifier test for the generated xDSL dialect `{dialect}`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import {dialect}_dialect as d  # noqa: E402


pytestmark = pytest.mark.skipif(not d.HAS_XDSL, reason="xDSL not installed")


def test_module_metadata():
    assert d.DIALECT_NAME == "{dialect}"
    assert d.OPS == {ops!r}
    assert d.TYPES == {types!r}
    registered_ops = {{op.name for op in d.get_dialect().operations}}
    registered_types = {{typ.name for typ in d.get_dialect().attributes}}
    assert registered_ops == {{f"{dialect}.{{name}}" for name in d.OPS}}
    assert registered_types == {{f"{dialect}.{{name}}" for name in d.TYPES}}


def test_build_verifies():
    mod = d.build_example()
    mod.verify()  # raises on failure


def test_roundtrip_is_stable():
    mod = d.build_example()
    mod.verify()
    parsed = d.roundtrip(mod)
    parsed.verify()
    import io
    from xdsl.printer import Printer
    def text(m):
        s = io.StringIO(); Printer(stream=s).print_op(m); return s.getvalue()
    assert text(mod) == text(parsed)
'''

_DECLARED_TEST = '''"""Import/registration test for the generated xDSL dialect `{dialect}`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402
import {dialect}_dialect as d  # noqa: E402


def test_module_metadata_and_registration():
    assert d.DIALECT_NAME == "{dialect}"
    assert d.OPS == {ops!r}
    assert d.TYPES == {types!r}
    if not d.HAS_XDSL:
        pytest.skip("xDSL not installed")
    dialect = d.get_dialect()
    registered_ops = {{op.name for op in dialect.operations}}
    registered_types = {{typ.name for typ in dialect.attributes}}
    assert registered_ops == {{f"{dialect}.{{name}}" for name in d.OPS}}
    assert registered_types == {{f"{dialect}.{{name}}" for name in d.TYPES}}
'''


def generate(dialect_plan: dict[str, Any]) -> list[Artifact]:
    """Return xdsl/ artifacts for the given dialect_plan."""
    dialect = dialect_plan.get("dialect_name", dialect_plan.get("target", "target"))
    ops = [o["name"] for o in dialect_plan.get("ops", []) if isinstance(o, dict) and "name" in o]
    types = [t["name"] for t in dialect_plan.get("types", []) if isinstance(t, dict) and "name" in t]
    const = dialect.upper() + "_DIALECT"

    real = set(ops) >= KNOWN_OPS and set(types) >= KNOWN_TYPES
    op_specs = [(name, f"{_class_token(name)}Op") for name in ops]
    type_specs = [(name, f"{_class_token(name)}Type") for name in types]
    extra_op_specs = [(name, class_name) for name, class_name in op_specs if name not in KNOWN_OPS]
    extra_type_specs = [
        (name, class_name) for name, class_name in type_specs if name not in KNOWN_TYPES
    ]
    dia_tmpl, test_tmpl = (
        (_REAL_DIALECT, _REAL_TEST) if real else (_DECLARED_DIALECT, _DECLARED_TEST)
    )
    return [
        Artifact(f"xdsl/{dialect}_dialect.py",
                 dia_tmpl.format(
                     dialect=dialect,
                     ops=ops,
                     types=types,
                     op_specs=op_specs,
                     type_specs=type_specs,
                     extra_op_specs=extra_op_specs,
                     extra_type_specs=extra_type_specs,
                     DIALECT_CONST=const,
                 )),
        Artifact(f"xdsl/test_{dialect}.py",
                 test_tmpl.format(dialect=dialect, ops=ops, types=types)),
    ]
