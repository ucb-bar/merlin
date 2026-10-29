"""Compiler-valid graph mutations, unsupported scopes, and operational countermodels."""

import json
import os
import subprocess
import sys

import pytest
from test_lean_graph_state_scalar import CASES, CONSTANTS, EXTERNAL, MULTI, OPAQUE, REFERENCE, checked, looping, produce

from merlin.verify import lean_graph_state as stable

PARTIAL_ZERO = looping(upper=0, observe=True).replace(
    "%first = arith.addi", "%partial = arith.divsi %a, %b : i8\n      %first = arith.addi"
)

SOURCES = {
    "basic": CASES["cse-i8"],
    "constants": ("cse", CONSTANTS),
    "opaque": ("cse", OPAQUE),
    "multi": ("cse", MULTI),
    "loop": ("licm", looping()),
    "carries": ("licm", looping(carries=2, observe=True)),
    "partial-zero": ("licm", PARTIAL_ZERO),
    "external": (
        "cse",
        EXTERNAL.replace(
            "func.func private @external(i8) -> i8",
            "func.func private @external(i8) -> i8\nfunc.func private @other(i8) -> i8",
        ),
    ),
}

MUTATIONS = {
    "opcode": "basic",
    "operand": "basic",
    "operand-order": "basic",
    "return": "basic",
    "constant": "constants",
    "type-interface": "basic",
    "poison-flag": "basic",
    "unknown-metadata": "basic",
    "drop-observer-input": "opaque",
    "drop-observer": "opaque",
    "erase-partial": "opaque",
    "effect-order": "opaque",
    "variant-motion": "loop",
    "partial-speculation": "partial-zero",
    "empty-behavior": "partial-zero",
    "carry-init": "carries",
    "yield-order": "carries",
    "unchanged-function": "multi",
    "external-callee": "external",
    "module-metadata": "basic",
}

MUTATOR = """
import json,sys
from xdsl.dialects import arith,builtin,func,scf
from xdsl.rewriter import Rewriter
from merlin.transforms.certificates import parse_ir,ir_text
source,destination,kind=sys.argv[1:]
module=parse_ir(open(source).read());module.verify()
functions=[op for op in module.body.block.ops if isinstance(op,func.FuncOp) and op.body.blocks]
function=functions[0]
operations=list(function.walk())
if kind=='opcode':
 op=next(op for op in operations if type(op) is arith.AddiOp)
 Rewriter.replace_op(op,arith.SubiOp(*op.operands))
elif kind in ('operand','operand-order'):
 op=next(op for op in operations if type(op) is arith.AddiOp)
 op.operands=(function.body.block.args[1],op.operands[1]) if kind=='operand' else tuple(reversed(op.operands))
elif kind=='return':
 op=next(op for op in operations if type(op) is func.ReturnOp);op.operands=(function.body.block.args[0],)
elif kind=='constant':
 op=next(op for op in operations if type(op) is arith.ConstantOp and type(op.result.type) is builtin.IntegerType)
 Rewriter.replace_op(op,arith.ConstantOp.from_int_and_width(op.value.value.data+1,op.result.type))
elif kind=='type-interface':
 module=parse_ir(ir_text(module).replace('i8','i16'))
elif kind in ('poison-flag','unknown-metadata'):
 op=next(op for op in operations if type(op) is arith.AddiOp)
 if kind=='poison-flag':op.properties['overflowFlags']=arith.IntegerOverflowAttr((arith.IntegerOverflowFlag.NSW,))
 else:op.attributes['unmodeled']=builtin.UnitAttr()
elif kind in ('drop-observer-input','drop-observer','effect-order'):
 observers=[op for op in operations if op.name=='builtin.unregistered' and not op.results]
 if kind=='drop-observer-input':observers[0].operands=()
 elif kind=='drop-observer':observers[0].parent_block().erase_op(observers[0])
 else:
  op=observers[-1];op.detach();observers[0].parent_block().insert_op_before(op,observers[0])
elif kind=='erase-partial':
 op=next(op for op in operations if type(op) is arith.DivSIOp);op.parent_block().erase_op(op)
elif kind in ('variant-motion','partial-speculation'):
 loop=next(op for op in operations if type(op) is scf.ForOp)
 op=next(op for op in loop.body.block.ops if type(op) is (arith.SubiOp if kind=='variant-motion' else arith.DivSIOp))
 if kind=='variant-motion':op.operands=(function.body.block.args[0],op.operands[1])
 op.detach();function.body.block.insert_op_before(op,loop)
elif kind=='empty-behavior':
 loop=next(op for op in operations if type(op) is scf.ForOp);op=loop.ub.owner
 Rewriter.replace_op(op,arith.ConstantOp.from_int_and_width(1,op.result.type))
elif kind=='carry-init':
 loop=next(op for op in operations if type(op) is scf.ForOp)
 loop.operands=(*loop.operands[:3],*tuple(reversed(loop.iter_args)))
elif kind=='yield-order':
 op=next(op for op in operations if type(op) is scf.YieldOp);op.operands=tuple(reversed(op.operands))
elif kind=='unchanged-function':
 op=next(op for op in functions[1].walk() if type(op) is arith.SubiOp);op.operands=tuple(reversed(op.operands))
elif kind=='external-callee':
 op=next(op for op in operations if type(op) is func.CallOp);op.properties['callee']=builtin.SymbolRefAttr('other')
elif kind=='module-metadata':module.attributes['unmodeled']=builtin.UnitAttr()
else:raise ValueError(kind)
module.verify()
text=ir_text(module);open(destination,'w').write(text)
print(json.dumps({'compiler_valid':True,'kind':kind,'changed':text!=open(source).read()}))
"""


def mutate(root, after, kind):
    script, destination = root / "mutation.py", root / "mutated.mlir"
    script.write_text(MUTATOR)
    process = subprocess.run(
        [sys.executable, str(script), str(after), str(destination), kind],
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    (root / "mutation.stdout").write_text(process.stdout)
    (root / "mutation.stderr").write_text(process.stderr)
    assert process.returncode == 0, process.stderr
    measured = json.loads(process.stdout)
    assert measured["compiler_valid"] and measured["changed"], measured
    stable._write(
        root / "mutation.json",
        {
            **measured,
            "script": stable.file_ref(script),
            "original": stable.file_ref(after),
            "actual": stable.file_ref(destination),
        },
    )
    return destination


@pytest.mark.parametrize("kind", list(MUTATIONS))
def test_actual_compiler_valid_mutations_refused(kind, tmp_path):
    mode, source = SOURCES[MUTATIONS[kind]]
    before, after, projection = produce(tmp_path, source, mode)
    assert projection["projection_checked"], projection
    changed = mutate(tmp_path, after, kind)
    result = checked(tmp_path, before, changed)
    assert result["status"] in ("unsupported", "unavailable"), result
    if result["status"] == "unavailable":
        kernel = json.loads((tmp_path / "checked/kernel/report.json").read_text())
        assert kernel["returncode"] > 0  # No native signal or setup failure counts as a refusal.
        assert "decide" in (tmp_path / "checked/kernel/kernel.stdout").read_text()


UNSUPPORTED = {
    "unknown-region": """builtin.module {
      func.func @kernel(%a:i8,%b:i8) -> i8 {
        "opaque.region"() ({
          %x = arith.addi %a,%b : i8
          "opaque.observe"(%x) : (i8) -> ()
          "opaque.yield"() : () -> ()
        }) : () -> ()
        func.return %a : i8
      }
    }""",
    "cfg": """builtin.module {
      func.func @kernel(%a:i8,%b:i8) -> i8 {
        func.return %a : i8
      ^unreachable(%u:i8,%v:i8):
        %x = arith.addi %u,%v : i8
        "opaque.observe"(%x) : (i8) -> ()
        func.return %x : i8
      }
    }""",
    "zero-stride": looping(stride=0),
    "internal-call": """builtin.module {
      func.func @kernel(%a:i8,%b:i8) -> i8 {
        %r=func.call @local(%a,%b) : (i8,i8) -> i8
        func.return %r : i8
      }
      func.func @local(%a:i8,%b:i8) -> i8 {
        %x=arith.addi %a,%b : i8
        %y=arith.addi %a,%b : i8
        %r=arith.muli %x,%y : i8
        func.return %r : i8
      }
    }""",
}


@pytest.mark.parametrize("kind", list(UNSUPPORTED))
def test_complete_unsupported_scopes_are_not_empty_successes(kind, tmp_path):
    _, _, projection = produce(tmp_path, UNSUPPORTED[kind], "licm" if kind == "zero-stride" else "cse")
    assert not projection["projection_checked"], projection
    assert projection["public_acceptance_issued"] is False


@pytest.mark.parametrize("kind,arguments", [("drop-observer", [2, 3]), ("partial-speculation", [2, 0])])
def test_actual_typed_countermodels_for_observation_and_empty_speculation(kind, arguments, tmp_path):
    from merlin.verify.lean_graph_state_scalar_kernel import check_query
    from merlin.verify.lean_graph_state_scalar_models import emit_counterexample

    mode, source = SOURCES[MUTATIONS[kind]]
    before, after, _ = produce(tmp_path, source, mode)
    changed = mutate(tmp_path, after, kind)
    refused = checked(tmp_path, before, changed)
    assert refused["status"] == "unavailable"
    projection = json.loads((tmp_path / "checked/projection/projection.json").read_text())
    failure = (
        next(index for index, row in enumerate(projection["registry"]) if row["descriptor"]["name"] == "arith.divsi")
        if kind == "partial-speculation"
        else None
    )
    query, roots = emit_counterexample(projection, arguments, failure=failure)
    model = check_query(query, tmp_path / "countermodel", roots, module="GraphStateScalarCounterexample")
    assert model["status"] == "conditional_graph_checked", model
    stable._write(
        tmp_path / "countermodel-label.json",
        {
            "kind": "operational_counterexample_only",
            "actual_boundary_accepted": False,
            "opaque_interpretation_approved": False,
            "kernel": stable.file_ref(tmp_path / "countermodel/report.json"),
        },
    )
