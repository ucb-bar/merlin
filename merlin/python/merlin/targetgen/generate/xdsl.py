"""Generate a real xDSL prototype dialect from a dialect_plan.

For the ToyNPU op/type set this emits a fully-working IRDL dialect: parametrized types
(`resident_tensor`, `accumulator`), ops (`res_pack`, `matmul`, `commit`, `evict`) with real
operand/result/property definitions and a custom verifier, a registered `Dialect`, and a
`build_example`/round-trip helper. The companion test builds a module, verifies it, and
round-trips it through the xDSL parser/printer.

If the plan's ops/types are not the known ToyNPU set (e.g. a conservative non-toy plan), a
minimal but real registered dialect is emitted instead. Either way the module imports as
plain Python; if xDSL is not installed, `HAS_XDSL` is False and the helpers no-op safely.
"""
from __future__ import annotations

from typing import Any

from ...common.artifacts import Artifact
from ...runtime.commandbuffer import EPILOGUE_STAGES

KNOWN_TYPES = {"resident_tensor", "accumulator"}
KNOWN_OPS = {"res_pack", "matmul", "commit", "evict"}

_REAL_DIALECT = '''"""Generated xDSL prototype dialect for `{dialect}` (real IRDL).

Run the round-trip test with `pytest test_{dialect}.py`. Promote stable pieces to the MLIR/C++
scaffold under include/ + lib/ once the syntax settles. Each op maps to a Merlin `interface`
interface abstraction (res_pack<-resident_pack, matmul<-matmul, commit<-commit, evict<-resident_evict).
"""
from __future__ import annotations

DIALECT_NAME = "{dialect}"
OPS = {ops!r}
TYPES = {types!r}

try:
    from xdsl.ir import Dialect, TypeAttribute, Attribute, Block, Region
    from xdsl.irdl import (irdl_op_definition, IRDLOperation, irdl_attr_definition,
                           operand_def, result_def, prop_def, ParametrizedAttribute)
    from xdsl.dialects.builtin import StringAttr, ArrayAttr, TensorType, i8, i32, ModuleOp, FunctionType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.utils.exceptions import VerifyException
    HAS_XDSL = True
except Exception:  # noqa: BLE001 - xDSL is an optional prototyping dependency
    HAS_XDSL = False


# The command-buffer ABI's epilogue vocabulary, rendered from the ONE definition
# (merlin.runtime.commandbuffer.EPILOGUE_STAGES) at generation time. A generated dialect is
# standalone, so the value is baked -- but it is baked FROM the single definition, never re-typed.
_KNOWN_EPILOGUE = {epilogue}

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
    _TYPE_CLASSES = [ResidentTensorType, AccumulatorType]
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

_MINIMAL_DIALECT = '''"""Generated xDSL prototype dialect for `{dialect}` (minimal).

The dialect plan declared no concrete ops/types yet (conservative non-toy synthesis), so this
emits a real but empty registered dialect. Fill in ops/types after human review, mirroring the
ToyNPU reference dialect.
"""
from __future__ import annotations

DIALECT_NAME = "{dialect}"
OPS = {ops!r}
TYPES = {types!r}

try:
    from xdsl.ir import Dialect
    HAS_XDSL = True
    {DIALECT_CONST} = Dialect("{dialect}", [], [])

    def get_dialect():
        return {DIALECT_CONST}
except Exception:  # noqa: BLE001
    HAS_XDSL = False

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

_MINIMAL_TEST = '''"""Import/registration test for the generated xDSL dialect `{dialect}`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import {dialect}_dialect as d  # noqa: E402


def test_module_imports():
    assert d.DIALECT_NAME == "{dialect}"
    # get_dialect() never raises whether or not xDSL is installed.
    d.get_dialect()
'''


def generate(dialect_plan: dict[str, Any]) -> list[Artifact]:
    """Return xdsl/ artifacts for the given dialect_plan."""
    dialect = dialect_plan.get("dialect_name", dialect_plan.get("target", "target"))
    ops = [o["name"] for o in dialect_plan.get("ops", []) if isinstance(o, dict) and "name" in o]
    types = [t["name"] for t in dialect_plan.get("types", []) if isinstance(t, dict) and "name" in t]
    const = dialect.upper() + "_DIALECT"

    real = set(ops) >= KNOWN_OPS and set(types) >= KNOWN_TYPES
    dia_tmpl, test_tmpl = (_REAL_DIALECT, _REAL_TEST) if real else (_MINIMAL_DIALECT, _MINIMAL_TEST)
    return [
        Artifact(f"xdsl/{dialect}_dialect.py",
                 dia_tmpl.format(dialect=dialect, ops=ops, types=types, DIALECT_CONST=const,
                                 # a set LITERAL in the canonical tuple order: `repr(set(...))`
                                 # varies with the hash seed and a generated file must not.
                                 epilogue="{" + ", ".join(map(repr, EPILOGUE_STAGES)) + "}")),
        Artifact(f"xdsl/test_{dialect}.py",
                 test_tmpl.format(dialect=dialect, ops=ops, types=types)),
    ]
