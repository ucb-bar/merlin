"""Complete actual-IR mutations; malformed inputs are a separate denominator."""

from __future__ import annotations

import json
import os
import sys

import pytest
from test_lean_graph_state_integer import REFERENCE, chain_source, nested_source, produce, standalone_source

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root
from merlin.verify.lean_graph_state_integer import check_boundary, configuration
from merlin.verify.lean_graph_state_integer_proof import check_query

VALID = (
    "constant",
    "exposed-site",
    "operand",
    "operand-order",
    "opcode",
    "flags",
    "carry-init",
    "yield-order",
    "capture",
    "step",
    "bound",
    "callee",
    "drop-observer",
    "observer-arguments",
    "return-order",
    "return-value",
    "function-context",
    "signature-width",
    "function-header",
    "module-header",
)


def mutated(tmp_path, kind):
    text = (
        standalone_source()
        if kind in ("constant", "exposed-site")
        else chain_source()
        if kind in ("flags", "opcode")
        else nested_source()
    )
    capture = produce(tmp_path, text)
    buffers = [row["path"] for row in capture["buffers"]]
    target = tmp_path / "mutated.mlir"
    script = repo_root() / "src/merlin/verify/lean_graph_state_integer_mutate.py"
    with (tmp_path / "mutation.stdout").open("x") as out, (tmp_path / "mutation.stderr").open("x") as err:
        run = run_owned(
            [sys.executable, str(script), buffers[1], str(target), kind],
            cwd=REFERENCE,
            env={**os.environ, "PYTHONPATH": str(REFERENCE / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=out,
            stderr=err,
            timeout=60,
            check=False,
        )
    assert run.returncode == 0, (tmp_path / "mutation.stderr").read_text()
    return [buffers[0], target], json.loads((tmp_path / "mutation.stdout").read_text())


@pytest.mark.parametrize("kind", VALID)
def test_valid_adverse_mutations(tmp_path, kind):
    buffers, measurement = mutated(tmp_path, kind)
    assert measurement["compiler_valid"] and measurement["changed"]
    report = check_boundary(buffers, tmp_path / "checked", REFERENCE)
    assert report["status"] in ("unsupported", "refused"), report
    (tmp_path / "REFUSAL.json").write_text(
        json.dumps(
            {
                "mutation": measurement,
                "classification": report["status"],
                "native_failure_counted": False,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("kind", ["malformed-operand", "malformed-result", "malformed-region", "malformed-call"])
def test_malformed_types_regions_and_calls(tmp_path, kind):
    buffers, measurement = mutated(tmp_path, kind)
    assert measurement["malformed"] and not measurement["compiler_valid"]
    report = check_boundary(buffers, tmp_path / "checked", REFERENCE)
    assert report["status"] == "unsupported"


def test_actual_wrong_middle(tmp_path):
    first = produce(tmp_path / "first", chain_source(), chain=True)
    second = produce(tmp_path / "second", chain_source(upper=24), chain=True)
    buffers = [row["path"] for row in first["buffers"]]
    buffers[2] = second["buffers"][2]["path"]
    report = check_boundary(buffers, tmp_path / "checked", REFERENCE)
    assert report["status"] in ("unsupported", "refused")


@pytest.mark.parametrize("field,value", [("factor", True), ("tile", 2.5), ("index_bits", "32"), ("index_bits", 16)])
def test_invalid_configurations(field, value):
    settings = {"factor": 2, "tile": 3, "index_bits": 32}
    settings[field] = value
    with pytest.raises(ValueError):
        configuration(**settings)


def test_wrong_constant_bit_certificate_is_kernel_refused(tmp_path):
    text = """import Merlin.GraphStateIntegerProgram
namespace Merlin.GraphStateIntegerInstance
open Merlin.GraphState.Integer
theorem forged : word 8 0 = word 8 1 := by decide
end Merlin.GraphStateIntegerInstance
"""
    report = check_query(text, tmp_path / "lean", ["Merlin.GraphStateIntegerInstance.forged"])
    assert report["status"] == "refused" and report["returncode"] > 0
    assert report["audit"]["status"] == "not_run"
