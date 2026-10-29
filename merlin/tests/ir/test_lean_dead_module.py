"""Exact total-producer deletion across internal calls, without invented input types."""

from __future__ import annotations

import json
import subprocess

import pytest
from test_lean_w8a8_body import exported
from test_lean_w8a8_byte_model import evaluate

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.passes_xdsl import prune_dead_pure_tensor_ops
from merlin.verify.lean import differential as diff

pytestmark = pytest.mark.usefixtures("lean_oracle")


def source(shape=(2, 3), *, invalid=False):
    tensor = "tensor<" + "".join(f"{size}x" for size in shape) + "f32>"
    dead = (
        "%zero = arith.constant 0 : i64\n%bad = arith.divsi %input, %zero : i64"
        if invalid
        else f"""
      %empty = tensor.empty() : {tensor}
      %zero = arith.constant 0.0 : f32
      %filled = linalg.fill ins(%zero : f32) outs(%empty : {tensor}) -> {tensor}
    """
    )
    return parse_mlir_text(f"""func.func @helper(%input: i64) -> i64 {{
      {dead}
      return %input : i64
    }}
    func.func @forward(%value: i64) -> i64 {{
      %r = func.call @helper(%value) : (i64) -> i64
      return %r : i64
    }}""")


def check(before, after, work, tools, *, index_bits=64, mutate=None):
    documents = []
    for name, program in [("before", before), ("after", after)]:
        directory = work / name
        directory.mkdir()
        documents.append(exported(program, directory, tools, generic=True))
    if mutate is not None:
        left, right = [json.loads(path.read_text()) for path in documents]
        mutate(left, right)
        for path, document in zip(documents, (left, right), strict=True):
            path.write_text(json.dumps(document))
    proc = subprocess.run(
        [
            "lake",
            "env",
            "lean",
            "--run",
            str(repo_root() / "merlin/tests/fixtures/lean_dead_module.lean"),
            *map(str, documents),
            str(index_bits),
        ],
        cwd=repo_root() / "merlin/lean",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["checker"] == "MerlinVerify.Check.DeadModule.check"
    assert report["theorem"] == "MerlinVerify.Check.DeadModule.check_sound"
    assert report["index_bits"] == index_bits
    assert (proc.returncode == 0) == report["exact_module_execution_equality"]
    (work / "dead-module-report.json").write_text(json.dumps(report, indent=2))
    return report["exact_module_execution_equality"], documents


@pytest.mark.parametrize("shape", [(2, 3), (0, 3), ()])
@pytest.mark.parametrize("index_bits", [32, 64])
def test_actual_dead_initializer_pruning_closes_across_calls(tmp_path, lean_tools, shape, index_bits):
    before = source(shape)
    after = before.clone()
    assert prune_dead_pure_tensor_ops(after) >= 1
    accepted, _ = check(before, after, tmp_path, lean_tools, index_bits=index_bits)
    assert accepted


def test_unused_failing_operation_cannot_be_deleted(tmp_path, lean_tools):
    before = source(invalid=True)
    after = before.clone()
    helper = next(op for op in after.walk() if op.name == "func.func")
    divide = next(op for op in helper.walk() if op.name == "arith.divsi")
    helper.body.block.erase_op(divide)
    accepted, _ = check(before, after, tmp_path, lean_tools)
    assert not accepted


@pytest.mark.parametrize(
    "change", ["unmodelled", "shadow", "used", "extra_region", "callee", "signature", "function_attr", "abi_payload"]
)
def test_exported_ast_cannot_forge_totality_freshness_or_call_equivalence(tmp_path, lean_tools, change):
    before = source()
    after = before.clone()
    prune_dead_pure_tensor_ops(after)

    def mutate(left, right):
        helper = left["module"]["ops"][0]
        block = helper["regions"][0]["blocks"][0]
        constant = next(op for op in block["ops"] if op["name"] == "arith.constant")
        if change == "unmodelled":
            constant["attrs"]["unreviewed_mode"] = {"kind": "unit"}
        elif change == "shadow":
            # Rebind the caller's live integer argument with a constant. Deleting it
            # would change the returned value, despite the constant's apparent disuse.
            constant["results"][0] = {"id": block["args"][0]["id"], "type": {"kind": "int", "width": 64}}
            constant["attrs"]["value"] = {"kind": "int", "value": "42", "type": {"kind": "int", "width": 64}}
            block["ops"] = [constant, block["ops"][-1]]
        elif change == "used":
            block["ops"][-1]["operands"] = [constant["results"][0]["id"]]
        elif change == "extra_region":
            helper["regions"].append(helper["regions"][0])
            right["module"]["ops"][0]["regions"].append(right["module"]["ops"][0]["regions"][0])
        elif change == "callee":
            forward = right["module"]["ops"][1]["regions"][0]["blocks"][0]
            call = next(op for op in forward["ops"] if op["name"] == "func.call")
            attrs = call["props"] if "callee" in call["props"] else call["attrs"]
            attrs["callee"]["root"] = "forward"
        elif change == "signature":
            header = helper["attrs"] if "function_type" in helper["attrs"] else helper["props"]
            header["function_type"]["value"]["results"] = [{"kind": "float", "format": "f32"}]
        elif change == "function_attr":
            helper["attrs"]["unreviewed_host_mode"] = {"kind": "unit"}
        else:
            helper["attrs"]["llvm.emit_c_interface"] = {"kind": "bool", "value": True}

    accepted, _ = check(before, after, tmp_path, lean_tools, mutate=mutate)
    assert not accepted


def test_pruned_helper_matches_original_native_and_lean_execution(tmp_path, lean_tools, fidelity_tools):
    before = source()
    after = before.clone()
    prune_dead_pure_tensor_ops(after)
    accepted, documents = check(before, after, tmp_path, lean_tools)
    assert accepted
    samples = [[[str(value)]] for value in (0, 1, 2**63 - 1, 2**63, 2**64 - 1)]
    expected = [[sample[0][0]] for sample in samples]
    for document in documents:
        proc, lean = evaluate(document, samples, document.parent)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        inputs, outputs = diff.signature(json.loads(document.read_text()), "forward")
        native = diff.execute_mlir(
            document.parent / "actual.mlir",
            samples,
            document.parent,
            fidelity_tools,
            inputs=inputs,
            outputs=outputs,
            function="forward",
        )
        diff.assert_equal(lean, expected, outputs, samples)
        assert native["results"] == expected
