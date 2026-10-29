"""Actual scalar→SCF chain proofs, finite model witnesses and explicit domains."""

import json
import os
import sys
from pathlib import Path

import pytest

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root
from merlin.verify import lean_graph_state as stable
from merlin.verify.lean_graph_state_pipeline_emit import emit
from merlin.verify.lean_graph_state_pipeline_models import countermodels
from merlin.verify.lean_graph_state_pipeline_proof import check_query, guard
from merlin.verify.lean_graph_state_pipeline_receipt import check_boundary, issue

REFERENCE = repo_root().parent / "merlin-graph-proof-intake"


def source(width=32, *, lower=0, upper=24, step=1, multi=False, partial=False):
    t = "i" + str(width)
    if multi:
        declarations = f"""func.func private @partial({t}, {t}, index) -> ({t}, {t})
  func.func private @other({t}, {t}, index) -> ({t}, {t})
  func.func private @watch({t}, index)
  func.func private @final_observe({t}, {t})"""
        loop = f"""%out:2 = scf.for %iv = %lb to %ub step %st iter_args(%acc = %x, %other = %bias) -> ({t}, {t}) {{
      %invariant = arith.muli %bias, %bias : {t}
      %duplicate = arith.muli %bias, %bias : {t}
      %a = arith.addi %duplicate, %acc : {t}
      %a2 = arith.addi %a, %invariant : {t}
      func.call @watch(%a2, %iv) : ({t}, index) -> ()
      %v:2 = func.call @partial(%a2, %other, %iv) : ({t}, {t}, index) -> ({t}, {t})
      %next = arith.subi %v#1, %acc : {t}
      scf.yield %v#0, %next : {t}, {t}
    }}
    func.call @final_observe(%out#0, %out#1) : ({t}, {t}) -> ()
    func.return %out#1, %out#0 : {t}, {t}"""
        result = f"({t}, {t})"
    else:
        declarations = f"""func.func private @partial({t}, index) -> {t}
  func.func private @other({t}, index) -> {t}
  func.func private @final_observe({t})"""
        maybe = (
            f"%div = arith.divsi %a2, %bias : {t}\n      %v = func.call @partial(%div, %iv) : ({t}, index) -> {t}"
            if partial
            else f"%v = func.call @partial(%a2, %iv) : ({t}, index) -> {t}"
        )
        loop = f"""%out = scf.for %iv = %lb to %ub step %st iter_args(%acc = %x) -> ({t}) {{
      %invariant = arith.addi %bias, %bias : {t}
      %duplicate = arith.addi %bias, %bias : {t}
      %a = arith.addi %acc, %duplicate : {t}
      %a2 = arith.addi %a, %invariant : {t}
      {maybe}
      scf.yield %v : {t}
    }}
    func.call @final_observe(%out) : ({t}) -> ()
    func.return %out : {t}"""
        result = t
    return f"""builtin.module {{
  {declarations}
  func.func @main(%x: {t}, %bias: {t}) -> {result} {{
    %lb = arith.constant {lower} : index
    %ub = arith.constant {upper} : index
    %st = arith.constant {step} : index
    %dead = arith.addi %x, %x : {t}
    {loop}
  }}
  func.func @untouched(%x: {t}) -> {t} {{ func.return %x : {t} }}
}}"""


def triple_source():
    return """builtin.module {
  func.func private @partial(i64, i64, i64, index) -> (i64, i64, i64)
  func.func private @other(i64, i64, i64, index) -> (i64, i64, i64)
  func.func private @watch(i64, index)
  func.func private @final_observe(i64, i64, i64)
  func.func @main(%x: i64, %bias: i64) -> (i64, i64, i64) {
    %lb = arith.constant 3 : index
    %ub = arith.constant 27 : index
    %st = arith.constant 2 : index
    %dead = arith.addi %x, %x : i64
    %out:3 = scf.for %iv = %lb to %ub step %st iter_args(%acc = %x, %other = %bias, %third = %x) -> (i64, i64, i64) {
      %invariant = arith.muli %bias, %bias : i64
      %duplicate = arith.muli %bias, %bias : i64
      %a = arith.addi %duplicate, %acc : i64
      %a2 = arith.addi %a, %invariant : i64
      func.call @watch(%a2, %iv) : (i64, index) -> ()
      %v:3 = func.call @partial(%a2, %other, %third, %iv) : (i64, i64, i64, index) -> (i64, i64, i64)
      %next = arith.subi %v#1, %acc : i64
      %last = arith.addi %v#2, %other : i64
      scf.yield %v#0, %next, %last : i64, i64, i64
    }
    func.call @final_observe(%out#0, %out#1, %out#2) : (i64, i64, i64) -> ()
    func.return %out#2, %out#0, %out#1 : i64, i64, i64
  }
  func.func @untouched(%x:i64) -> i64 { func.return %x : i64 }
}"""


CASES = {
    "i32": {"text": source(), "factor": 2, "tile": 3, "bits": 32},
    "i64-multicarry": {"text": source(64, lower=3, upper=27, step=2, multi=True), "factor": 3, "tile": 3, "bits": 64},
    "huge": {"text": source(32, upper=10**12), "factor": 2, "tile": 3, "bits": 64},
    "nonaligned": {"text": source(16, upper=27, step=2), "factor": 2, "tile": 2, "bits": 32},
    "partial-body": {"text": source(16, upper=12, partial=True), "factor": 3, "tile": 2, "bits": 32},
    "triple-carry": {"text": triple_source(), "factor": 3, "tile": 3, "bits": 64},
    "factor-four": {"text": source(8, lower=2, upper=34, step=2), "factor": 4, "tile": 4, "bits": 32},
    "symbolic-provider": {
        "text": source(32, upper=6),
        "factor": 2,
        "tile": 3,
        "bits": 32,
        "mode": "symbolic-provider-counter-v1",
    },
}


def capture(root, case):
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "input.mlir"
    input_path.write_text(case["text"])
    script = repo_root() / "src/merlin/verify/lean_graph_state_pipeline_capture.py"
    command = [
        sys.executable,
        str(script),
        "--reference",
        str(REFERENCE),
        "--source",
        str(input_path),
        "--destination",
        str(root / "actual"),
        "--factor",
        str(case["factor"]),
        "--tile",
        str(case["tile"]),
        "--bits",
        str(case["bits"]),
    ]
    with (root / "producer.stdout").open("x") as out, (root / "producer.stderr").open("x") as err:
        result = run_owned(
            command,
            cwd=REFERENCE,
            env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    assert result.returncode == 0, (root / "producer.stderr").read_text()
    data = json.loads((root / "actual/CAPTURE.json").read_text())
    return [root / "actual" / f"stage{i}.mlir" for i in range(5)], data


def checked(root, buffers, case):
    return check_boundary(
        buffers,
        root / "checked",
        reference=REFERENCE,
        factor=case["factor"],
        tile=case["tile"],
        bits=case["bits"],
        mode=case.get("mode", "graph"),
    )


@pytest.mark.parametrize("name", list(CASES))
def test_actual_complete_pipeline(name, tmp_path):
    case = CASES[name]
    buffers, data = capture(tmp_path, case)
    assert data["each_stage_changed"] == [True] * 4
    assert any(row.get("rule") == "cse" for row in data["diagnostics"][0]["diagnostic_only"])
    result = checked(tmp_path, buffers, case)
    assert result["status"] == "conditional_graph_checked", result
    kernel = json.loads(Path(result["kernel"]["path"]).read_text())
    assert kernel["audit"]["raw"]["violations"] == []
    assert set(kernel["audit"]["raw"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    assert result["claims"]["all_graph_passes_verified"] is False
    if name == "huge":
        assert Path(kernel["source"]["path"]).stat().st_size < 50000
    if name == "symbolic-provider":
        assert result["witness"]["success"] == {"values": [17], "world": 14}
        assert result["witness"]["failure"]["world"] == 16
        assert result["witness"]["source_success_assumed"] is False
    receipt = issue(tmp_path / "checked/report.json", tmp_path / "receipt.json")
    stable._write(
        tmp_path / "result-summary.json",
        {
            "name": name,
            "status": result["status"],
            "receipt": stable.file_ref(tmp_path / "receipt.json"),
            "roots": receipt["roots"],
            "stage_changed": data["each_stage_changed"],
        },
    )


@pytest.mark.parametrize("upper", [0, 1, 2, 7])
def test_inline_empty_and_tail_are_explicit_refusals(upper, tmp_path):
    case = {"text": source(upper=upper), "factor": 2, "tile": 3, "bits": 32}
    buffers, _ = capture(tmp_path, case)
    result = checked(tmp_path, buffers, case)
    assert result["status"] == "unsupported", result
    assert result["claims"]["public_acceptance_issued"] is False


def test_contract_countermodels(tmp_path):
    query, roots = countermodels()
    result = check_query(query, tmp_path / "countermodels", roots, module="GraphStatePipelineCountermodels")
    assert result["status"] == "conditional_graph_checked", result


def test_limited_literal_model_route_stays_unqualified():
    with pytest.raises(RuntimeError, match="retry refused"):
        guard("namespace Witness\nnamespace Execution\n")


def test_checked_literal_and_complete_roster_roots_are_in_emitted_query(tmp_path):
    _, data = capture(tmp_path, CASES["i32"])
    text = emit(data)
    assert "checkLiterals profileBits controls = true" in text
    assert "PreservesEntries (M primitive external) originalEntries finalEntries" in text
