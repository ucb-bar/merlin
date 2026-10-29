"""Actual-pass and adversarial validation of the isolated Qinner Lean checker."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
from lean_qinner_cases import arrays, fixture, native_compare

from merlin.common.paths import artifacts_dir
from merlin.verify.lean.export import export_file
from merlin.verify.lean.qinner import actual, check_runtime, prepare, replay, run_checker, write


@pytest.fixture(scope="module")
def compiler():
    path = Path(os.environ.get("MERLIN_M2M_PYTHON", "/home/jeremy/Projects/model2MLIR/.venv/bin/python"))
    if not path.is_file():
        pytest.fail("Qinner qualification requires the MLIR compiler environment")
    prepare()
    return path


def exported(tmp_path, compiler, source=None):
    source = source or fixture()
    generated = actual(source)
    for label, text in (("source", source), ("propagated", generated["propagated"]), ("target", generated["lifted"])):
        (tmp_path / f"{label}.mlir").write_text(text)
        export_file(tmp_path / f"{label}.mlir", tmp_path / f"{label}.json", m2m_python=compiler)
    return generated


@pytest.mark.parametrize("kind", ["f32", "f64", "f16", "i8", "i16", "i32", "i64", "i1"])
@pytest.mark.parametrize("shape", [(4,), (1, 2, 1), (0, 3)])
def test_actual_positive_lifts(tmp_path, compiler, kind, shape):
    result = exported(tmp_path, compiler, fixture(kind, shape))
    assert result["plan"] == result["appended"]
    assert len(result["appended"]) == 2
    report = run_checker(tmp_path / "propagated.json", tmp_path / "target.json")
    assert report["checked"] and report["covered_sites"] == 2 and report["boundary_kind"] == "lifting"
    assert run_checker(tmp_path / "source.json", tmp_path / "propagated.json", mode="propagate")["checked"]
    assert len(check_runtime(result["plan"], arrays(kind, shape))) == 2


def test_nested_and_calls(tmp_path, compiler):
    exported(tmp_path, compiler, fixture(nested=True))
    assert run_checker(tmp_path / "propagated.json", tmp_path / "target.json")["checked"]
    assert run_checker(tmp_path / "source.json", tmp_path / "propagated.json", mode="propagate")["checked"]


def ops(document):
    def walk(operations):
        for op in operations:
            yield op
            for region in op["regions"]:
                for block in region["blocks"]:
                    yield from walk(block["ops"])

    return list(walk(document["module"]["ops"]))


MUTATIONS = [
    "swap_uses",
    "missing_argument",
    "reordered_arguments",
    "wrong_shape",
    "wrong_dtype",
    "altered_return",
    "dropped_output",
    "output_type",
    "original_input_type",
    "extra_operation",
    "module_attribute",
    "helper_body",
    "nested_capture",
    "region_arguments",
    "duplicate_tag",
    "conflicting_tag",
    "wrong_tag",
    "called_entry",
]


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_reject_mutations(tmp_path, compiler, mutation):
    exported(
        tmp_path, compiler, fixture(nested=True) if mutation in {"nested_capture", "region_arguments"} else fixture()
    )
    source = json.loads((tmp_path / "propagated.json").read_text())
    target = json.loads((tmp_path / "target.json").read_text())
    functions = [op for op in ops(target) if op["name"] == "func.func"]
    entry = functions[-1]
    block = entry["regions"][0]["blocks"][0]
    arguments = block["args"]
    all_ops = ops(target)
    returns = [op for op in all_ops if op["name"] == "func.return"]
    if mutation == "swap_uses":
        for op in all_ops:
            op["operands"] = [
                arguments[3]["id"] if v == arguments[2]["id"] else arguments[2]["id"] if v == arguments[3]["id"] else v
                for v in op["operands"]
            ]
    elif mutation == "missing_argument":
        arguments.pop()
    elif mutation == "reordered_arguments":
        arguments[2], arguments[3] = arguments[3], arguments[2]
    elif mutation in {"wrong_shape", "wrong_dtype", "original_input_type"}:
        argument = arguments[0 if mutation == "original_input_type" else 2]
        if mutation == "wrong_dtype":
            argument["type"]["element"] = {"kind": "float", "format": "f64"}
        else:
            argument["type"]["shape"] = [9]
    elif mutation == "altered_return":
        returns[-1]["operands"][0] = arguments[0]["id"]
    elif mutation == "dropped_output":
        returns[-1]["operands"].pop()
    elif mutation == "output_type":
        entry["attrs"]["function_type"]["value"]["results"][0]["shape"] = [9]
    elif mutation == "extra_operation":
        block["ops"].insert(
            0,
            {
                "name": "test.extra",
                "results": [],
                "operands": [],
                "attrs": {},
                "props": {},
                "regions": [],
                "successors": [],
            },
        )
    elif mutation == "module_attribute":
        target["module"]["attrs"]["prov.new"] = {"kind": "string", "value": "changed"}
    elif mutation == "helper_body":
        returns[0]["operands"] = []
    elif mutation == "nested_capture":
        next(op for op in all_ops if op["name"] == "scf.yield")["operands"] = [arguments[3]["id"]]
    elif mutation == "region_arguments":
        next(op for op in all_ops if op["name"] == "scf.if")["regions"][0]["blocks"][0]["args"] = [
            copy.deepcopy(arguments[0])
        ]
    elif mutation in {"duplicate_tag", "conflicting_tag", "wrong_tag"}:
        empties = [op for op in ops(source) if op["name"] == "tensor.empty"]
        empties[1]["attrs"]["prov.quant_inner"] = {
            "kind": "string",
            "value": "left" if mutation == "duplicate_tag" else "wrong",
        }
    elif mutation == "called_entry":
        next(op for op in all_ops if op["name"] == "func.call")["attrs"]["callee"]["root"] = "forward"
    write(tmp_path / "mut-source.json", source)
    write(tmp_path / "mut-target.json", target)
    assert not run_checker(tmp_path / "mut-source.json", tmp_path / "mut-target.json")["checked"]


@pytest.mark.parametrize("mutation", ["swap", "missing", "dtype", "shape", "wrong_tensor", "missing_key", "duplicate"])
def test_runtime_rejects_wrong_binding(mutation):
    from merlin.llvmlower.qinner import QinnerError

    plan = [{"key": key, "shape": [4], "dtype": "f32"} for key in ("left", "right")]
    extra = arrays()
    values = [extra["qinner::left"], extra["qinner::right"]]
    if mutation == "swap":
        values.reverse()
    elif mutation == "missing":
        values.pop()
    elif mutation == "dtype":
        values[0] = values[0].astype(np.float64)
    elif mutation == "shape":
        values[0] = values[0].reshape(2, 2)
    elif mutation == "wrong_tensor":
        values[0] = values[0] + 100
    elif mutation == "missing_key":
        del extra["qinner::left"]
    else:
        plan[1]["key"] = "left"
    with pytest.raises((ValueError, QinnerError)):
        check_runtime(plan, extra, values)


def test_runtime_noncontiguous_normalization():
    plan = [{"key": "left", "shape": [4], "dtype": "f32"}]
    extra = {"qinner::left": np.arange(8, dtype=np.float32)[::2]}
    assert check_runtime(plan, extra)[0]["bytes"] == 16


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind,shape", [("f32", (4,)), ("f64", (1, 2, 1)), ("i8", (4,)), ("f32", (0, 3))])
def test_native_complete_outputs(tmp_path, compiler, kind, shape, nested):
    assert native_compare(compiler, tmp_path, kind=kind, shape=shape, nested=nested)["all_output_bits_match"]


@pytest.mark.slow
def test_retained_receipts_replay(tmp_path, compiler):
    root = Path(
        os.environ.get(
            "MERLIN_QINNER_RECEIPT_ROOT", str(artifacts_dir() / "verification/merlin-passes/frontend/qinner-decoder")
        )
    )
    receipts = sorted((root / "boundaries").glob("*/receipt.json"))
    assert len(receipts) >= 6, "Generate positive and unchanged-model Qinner receipts first"
    for index, path in enumerate(receipts):
        result = replay(path, tmp_path / str(index), compiler_python=compiler)
        assert result["fresh"] and result["checked"] and result["matches_recorded"], (path, result)


@pytest.mark.parametrize(
    "mutation",
    [
        "swap",
        "missing",
        "key",
        "shape",
        "dtype",
        "bytes",
        "duplicate_archive",
        "byte_order",
        "layout",
        "byte_swap",
        "bool_encoding",
    ],
)
def test_lean_runtime_certificate_rejects(tmp_path, compiler, mutation):
    from merlin.verify.lean.qinner import run_binding, runtime_document

    kind = "i1" if mutation == "bool_encoding" else "f32"
    generated = exported(tmp_path, compiler, fixture(kind))
    certificate = runtime_document(generated["plan"], arrays(kind))
    write(tmp_path / "binding.json", certificate)
    assert run_binding(tmp_path / "propagated.json", tmp_path / "target.json", tmp_path / "binding.json")["checked"]
    rows = certificate["delivered"]
    if mutation == "swap":
        rows.reverse()
    elif mutation == "missing":
        rows.pop()
    elif mutation == "key":
        rows[0]["key"] = "wrong"
    elif mutation == "shape":
        rows[0]["type"]["shape"] = [2, 2]
    elif mutation == "dtype":
        rows[0]["type"]["element"] = {"kind": "int", "width": 32}
    elif mutation == "bytes":
        rows[0]["hex"] = "ff" * 16
    elif mutation == "byte_order":
        certificate["byte_order"] = "big"
    elif mutation == "layout":
        certificate["layout"] = "column_major"
    elif mutation == "bool_encoding":
        rows[0]["hex"] = "02" + rows[0]["hex"][2:]
    elif mutation == "byte_swap":
        raw = bytes.fromhex(rows[0]["hex"])
        rows[0]["hex"] = b"".join(raw[i : i + 4][::-1] for i in range(0, len(raw), 4)).hex()
    else:
        certificate["archive"].append(copy.deepcopy(certificate["archive"][0]))
    write(tmp_path / "binding.json", certificate)
    assert not run_binding(tmp_path / "propagated.json", tmp_path / "target.json", tmp_path / "binding.json")["checked"]


@pytest.mark.parametrize("mutation", ["conflict", "duplicate", "wrong_binding", "operand", "unrelated", "out_of_range"])
def test_propagation_rejects(tmp_path, compiler, mutation):
    exported(tmp_path, compiler)
    source = json.loads((tmp_path / "source.json").read_text())
    target = json.loads((tmp_path / "propagated.json").read_text())
    empties = [op for op in ops(target) if op["name"] == "tensor.empty"]
    casts = [op for op in ops(source) if op["name"] == "tensor.cast"]
    if mutation in {"conflict", "duplicate", "wrong_binding"}:
        empties[1]["attrs"]["prov.quant_inner"] = {
            "kind": "string",
            "value": "left" if mutation == "duplicate" else "wrong",
        }
        if mutation == "conflict":
            next(op for op in ops(source) if op["name"] == "tensor.empty")["attrs"]["prov.quant_inner"] = {
                "kind": "string",
                "value": "conflict",
            }
    elif mutation == "operand":
        casts[0]["operands"] = casts[1]["operands"]
    elif mutation == "unrelated":
        next(op for op in ops(target) if op["name"] == "func.return")["operands"] = []
    else:
        casts[0]["attrs"]["prov.quant_inner_99"] = casts[0]["attrs"].pop("prov.quant_inner_0")
    write(tmp_path / "bad-source.json", source)
    write(tmp_path / "bad-propagated.json", target)
    assert not run_checker(tmp_path / "bad-source.json", tmp_path / "bad-propagated.json", mode="propagate")["checked"]


def test_legacy_operand_aliases(tmp_path, compiler):
    generated = exported(tmp_path, compiler, fixture(legacy=True))
    assert [row["key"] for row in generated["appended"]] == ["left", "right"]
    assert run_checker(tmp_path / "source.json", tmp_path / "propagated.json", mode="propagate")["checked"]
    assert run_checker(tmp_path / "propagated.json", tmp_path / "target.json")["checked"]


def test_scalar_archive_rank_change_is_not_hidden():
    from merlin.llvmlower.qinner import QinnerArg, QinnerError, resolve

    with pytest.raises(QinnerError, match="in the IR"):
        resolve({"qinner::scalar": np.array(7, dtype=np.float32)}, [QinnerArg("scalar", (), "f32")])


def test_calls_to_lifted_entry_rejected(tmp_path, compiler):
    exported(tmp_path, compiler)
    for label in ("propagated", "target"):
        doc = json.loads((tmp_path / f"{label}.json").read_text())
        next(op for op in ops(doc) if op["name"] == "func.call")["attrs"]["callee"]["root"] = "forward"
        write(tmp_path / f"{label}.json", doc)
    assert not run_checker(tmp_path / "propagated.json", tmp_path / "target.json")["checked"]


def test_receipt_detects_stale_input(tmp_path, compiler):
    from merlin.verify.lean.qinner import check_boundary

    generated = actual(fixture())
    (tmp_path / "source.mlir").write_text(fixture())
    (tmp_path / "target.mlir").write_text(generated["lifted"])
    np.savez(tmp_path / "extra.npz", **arrays())
    receipt = check_boundary(
        tmp_path / "source.mlir",
        tmp_path / "target.mlir",
        tmp_path / "receipt",
        compiler_python=compiler,
        extra_path=tmp_path / "extra.npz",
    )
    assert receipt["checked"] and receipt["verdict"] == "abstracted" and not receipt["approved"]
    recorded = Path(receipt["inputs"][0]["path"])
    recorded.write_text(recorded.read_text() + "\n")
    result = replay(tmp_path / "receipt/receipt.json", tmp_path / "replay", compiler_python=compiler)
    assert not result["fresh"] and not result["checked"]


@pytest.mark.parametrize("depth", [0, 1, 7, 8])
def test_wrapper_walk_bound(tmp_path, compiler, depth):
    source = fixture()
    lines = []
    previous = "%a"
    for index in range(depth):
        name = f"%view{index}"
        lines.append(f'  {name} = "tensor.cast"({previous}) : (tensor<4xf32>) -> tensor<4xf32>')
        previous = name
    source = source.replace('%ac = "tensor.cast"(%a)', "\n".join(lines) + f'\n  %ac = "tensor.cast"({previous})')
    generated = exported(tmp_path, compiler, source)
    assert len(generated["appended"]) == (2 if depth < 8 else 1)
    assert run_checker(tmp_path / "source.json", tmp_path / "propagated.json", mode="propagate")["checked"] == (
        depth < 8
    )
    assert run_checker(tmp_path / "propagated.json", tmp_path / "target.json")["checked"] == (depth < 8)


def test_arbitrary_operand_zero_walk_is_refused(tmp_path, compiler):
    source = fixture().replace(
        '%ac = "tensor.cast"(%a)',
        '%combined = "test.combine"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>\n'
        '  %ac = "tensor.cast"(%combined)',
    )
    exported(tmp_path, compiler, source)
    assert not run_checker(tmp_path / "source.json", tmp_path / "propagated.json", mode="propagate")["checked"]
    assert not run_checker(tmp_path / "propagated.json", tmp_path / "target.json")["checked"]
