"""Compiler-valid changes to every relevant pipeline boundary and context."""

import json
import os
import sys
from pathlib import Path

import pytest
from test_lean_graph_state_pipeline import CASES, REFERENCE, capture, checked

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root
from merlin.verify.lean_graph_state_pipeline_receipt import config

MUTATIONS = {
    "step": ("i32", 3),
    "bound": ("i32", 3),
    "offset": ("i32", 3),
    "operand": ("i32", 3),
    "operand-order": ("i32", 3),
    "opcode": ("i32", 3),
    "flags": ("i32", 3),
    "carry-init": ("i64-multicarry", 3),
    "yield-order": ("i64-multicarry", 3),
    "callee": ("i32", 3),
    "drop-observer": ("i32", 4),
    "drop-inner-observer": ("i64-multicarry", 3),
    "observer-arguments": ("i64-multicarry", 4),
    "return-order": ("i64-multicarry", 4),
    "min-to-max": ("i32", 4),
    "min-operands": ("i32", 4),
    "inner-control": ("i32", 4),
    "function-context": ("i32", 4),
    "function-header": ("i32", 4),
    "module-header": ("i32", 4),
}


@pytest.mark.parametrize("kind", list(MUTATIONS))
def test_compiler_valid_boundary_mutations_are_refused(kind, tmp_path):
    name, stage = MUTATIONS[kind]
    case = CASES[name]
    buffers, _ = capture(tmp_path, case)
    script = repo_root() / "src/merlin/verify/lean_graph_state_pipeline_mutate.py"
    target = tmp_path / "mutated.mlir"
    with (tmp_path / "mutation.stdout").open("x") as out, (tmp_path / "mutation.stderr").open("x") as err:
        result = run_owned(
            [sys.executable, str(script), str(buffers[stage]), str(target), kind],
            cwd=REFERENCE,
            env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    assert result.returncode == 0, (tmp_path / "mutation.stderr").read_text()
    measured = json.loads((tmp_path / "mutation.stdout").read_text())
    assert measured["compiler_valid"] and measured["changed"]
    changed = [*buffers]
    changed[stage] = target
    rejected = checked(tmp_path, changed, case)
    assert rejected["status"] in ("unsupported", "refused"), rejected
    if rejected["status"] == "refused":
        kernel = json.loads(Path(rejected["kernel"]["path"]).read_text())
        assert kernel["returncode"] > 0
        assert "error" in Path(kernel["stdout"]["path"]).read_text()
    (tmp_path / "REFUSAL.json").write_text(
        json.dumps(
            {
                "mutation": measured,
                "classification": rejected["status"],
                "complete_boundary": True,
                "native_signal_counted_as_refusal": False,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("field,value", [("factor", True), ("tile", 2.5), ("bits", "32"), ("mode", "approved")])
def test_malformed_configuration_is_not_a_profile(field, value):
    args = {"factor": 2, "tile": 3, "bits": 32, "mode": "graph"}
    args[field] = value
    with pytest.raises(ValueError):
        config(**args)


@pytest.mark.parametrize("bits", [16, 31, 128])
def test_unadmitted_index_width_is_not_claimed(bits, tmp_path):
    case = {**CASES["i32"], "bits": bits}
    buffers, _ = capture(tmp_path, case)
    rejected = checked(tmp_path, buffers, case)
    assert rejected["status"] == "unsupported", rejected


def test_invalid_external_call_signature_is_refused(tmp_path):
    case = CASES["i32"]
    buffers, _ = capture(tmp_path, case)
    script = tmp_path / "malformed-call.py"
    target = tmp_path / "malformed.mlir"
    script.write_text("""import sys
from merlin.transforms.certificates import parse_ir,ir_text
from xdsl.dialects import func
module=parse_ir(open(sys.argv[1]).read());module.verify()
call=next(op for op in module.walk()if type(op)is func.CallOp and op.callee.root_reference.data=='partial')
call.operands=(call.operands[0],)
open(sys.argv[2],'w').write(ir_text(module))
""")
    result = run_owned(
        [sys.executable, str(script), str(buffers[4]), str(target)],
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    changed = [*buffers]
    changed[4] = target
    rejected = checked(tmp_path, changed, case)
    assert rejected["status"] == "unsupported", rejected
    diagnostic = (tmp_path / "checked/projection.stderr").read_text() + (
        tmp_path / "checked/projection.stdout"
    ).read_text()
    assert "signature" in diagnostic or "operand" in diagnostic


@pytest.mark.parametrize("kind", ["operand-type", "result-type", "region-argument"])
def test_malformed_operand_result_and_region_types_are_refused(kind, tmp_path):
    case = CASES["i32"]
    buffers, _ = capture(tmp_path, case)
    script = tmp_path / "malformed-types.py"
    target = tmp_path / "malformed.mlir"
    script.write_text("""import sys
from merlin.transforms.certificates import parse_ir,ir_text
from xdsl.dialects import builtin,func,scf
module=parse_ir(open(sys.argv[1]).read());module.verify()
main=next(f for f in module.body.block.ops if type(f)is func.FuncOp and f.sym_name.data=='main')
call=next(op for op in main.walk()if type(op)is func.CallOp and op.callee.root_reference.data=='partial')
if sys.argv[3]=='operand-type':call.operands=(call.operands[0],main.body.block.args[1])
elif sys.argv[3]=='result-type':call.results[0]._type=builtin.i64
else:next(op for op in main.walk()if type(op)is scf.ForOp).body.block.args[0]._type=builtin.i32
open(sys.argv[2],'w').write(ir_text(module))
""")
    result = run_owned(
        [sys.executable, str(script), str(buffers[4]), str(target), kind],
        cwd=REFERENCE,
        env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    changed = [*buffers]
    changed[4] = target
    rejected = checked(tmp_path, changed, case)
    assert rejected["status"] == "unsupported", rejected
    assert (tmp_path / "checked/projection.stderr").read_text() or (tmp_path / "checked/projection.stdout").read_text()


def test_actual_middle_substitution_is_refused(tmp_path):
    case = CASES["i32"]
    buffers, _ = capture(tmp_path / "original", case)
    altered = {**case, "text": case["text"].replace("arith.addi %bias, %bias", "arith.addi %x, %bias")}
    alternate, _ = capture(tmp_path / "different-source", altered)
    changed = [*buffers]
    changed[2] = alternate[2]
    result = checked(tmp_path, changed, case)
    assert result["status"] in ("unsupported", "refused"), result
    assert result["originals"][2]["sha256"] != __import__("hashlib").sha256(buffers[2].read_bytes()).hexdigest()
