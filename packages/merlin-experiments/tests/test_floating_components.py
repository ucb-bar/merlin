"""Eight public source cases, independent arithmetic, and unchanged integer paths."""

import copy
import json
import math
import os
import subprocess
from fractions import Fraction

import numpy as np
import pytest
import yaml
from merlin_experiments.phase0 import component_float_numerics as reference
from merlin_experiments.phase0 import floating_suite as suite
from merlin_experiments.phase0.writer import _write_capsule_inner

from merlin.common.paths import repo_root
from merlin.frontends.linalg_tensor_graph import read_tensor_graph
from merlin.semantic_compiler.model import KernelRequest
from merlin.targetgen import component_program, corpus_spec, golden_store


def config(**changes):
    return {
        "schema": suite.SCHEMA,
        "operand_dtype": "fp8_e4m3",
        "accumulator_dtype": "bf16",
        "shape": [3, 5, 7],
        "pointwise_compute_dtype": "f32",
        "pointwise_rounding": "rtz",
        "required_oracle_tiers": ["L0", "L1", "L2"],
        "reference_budget": {"max_reference_work": 100000, "max_materialized_elements": 10000},
        **changes,
    }


def binding(configuration):
    return corpus_spec.CorpusBinding(
        target="test_source",
        tile_dim=None,
        operand_dtype=configuration["operand_dtype"],
        accum_dtype=configuration["accumulator_dtype"],
        integer=False,
        tiers=[],
        compare="tolerance_float",
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"shape": [2, 7, 4]},
        {"shape": [1, 4, 1]},
        {"shape": [2, 6, 3]},
        {"operand_dtype": "fp8_e5m2"},
        {"operand_dtype": "fp16", "accumulator_dtype": "f32", "pointwise_rounding": "rne"},
        {"operand_dtype": "fp8_e4m3", "accumulator_dtype": "f32", "pointwise_rounding": "rne"},
        {"operand_dtype": "bf16", "accumulator_dtype": "bf16"},
        {"operand_dtype": "fp16", "accumulator_dtype": "fp16", "pointwise_rounding": "rne"},
        {"operand_dtype": "f32", "accumulator_dtype": "f32", "pointwise_rounding": "rne"},
    ],
)
def test_eight_cases_materialize_complete_goldens_for_distinct_format_contracts(tmp_path, overrides):
    selected = config(**overrides)
    manifest = suite.materialize(selected, tmp_path / "corpus")
    assert manifest["agent_facing_cases"] == 8 and len(manifest["members"]) == 8
    assert manifest["status"] == "source_only"
    assert not manifest["phase1_admitted"] and not manifest["hardware_qualified"]
    for member in manifest["members"]:
        root = tmp_path / "corpus" / member["path"]
        capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
        golden = golden_store.load_golden(root)
        outputs = capsule["component_program"]["outputs"]
        assert set(golden["outputs"]) == {row["name"] for row in outputs}
        assert capsule["numeric_policy"]["atol"] == capsule["numeric_policy"]["rtol"] == 0
        for row in outputs:
            assert np.asarray(golden["outputs"][row["name"]]).shape == tuple(row["shape"])
        # Exercise the same source-only API the backend author sees. Follow actual
        # SSA aliases and all return slots, including multiple slots for one value.
        graph = read_tensor_graph(
            (root / "capsule.interface.mlir").read_text(), entry="forward", target_identity="test"
        )
        assert KernelRequest.from_record(graph.record()) == graph
        program = capsule["component_program"]
        value_ids = {row["name"]: node.id for row, node in zip(program["inputs"], graph.nodes)}
        emitted = iter(graph.nodes[len(value_ids) :])
        for row in program["nodes"]:
            if row["op"] == "alias":
                value_ids[row["name"]] = value_ids[row["actual_inputs"][0]]
                continue
            node = next(emitted)
            matched_kind = {
                "matmul": {"matmul", "widening_matmul"},
                "update": {"add", "mixed_add"},
                "add": {"add", "mixed_add"},
            }.get(row["op"], {row["op"]})
            assert node.op in matched_kind
            assert (node.type.dtype, node.type.shape) == (row["dtype"], tuple(row["shape"]))
            if node.op == "widening_matmul":
                assert dict(node.attrs) == dict(
                    accumulator_first=True, compute_dtype=row["dtype"], reduction_order="source-unspecified"
                )
            assert node.inputs == tuple(value_ids[v] for v in row["actual_inputs"])
            if node.op == "mixed_add":
                assert dict(node.attrs)["compute_dtype"] == row["compute_dtype"]
                assert dict(node.attrs)["narrowing"] == ("bf16_high_bits" if row["rounding"] == "rtz" else "truncf")
            value_ids[row["name"]] = node.id
        assert next(emitted, None) is None
        assert graph.outputs == tuple(value_ids[row["actual_value"]] for row in program["outputs"])
    with pytest.raises(ValueError, match="fresh"):
        suite.materialize(selected, tmp_path / "corpus")


@pytest.mark.parametrize(
    "index,old,new",
    [
        (0, "arith.mulf %lhs_wide, %rhs_wide", "arith.mulf %lhs_wide, %lhs_wide"),
        (0, "arith.addf %old, %product", "arith.addf %product, %product"),
        (0, "arith.addf %old, %product :", "arith.addf %old, %product fastmath<fast> :"),
        (-1, "arith.constant 16 : i32", "arith.constant 15 : i32"),
        (-1, "arith.shrui %bits, %shift", "arith.shrui %shift, %bits"),
        (-1, "linalg.yield %rounded", "linalg.yield %old"),
    ],
)
def test_changed_source_arithmetic_is_not_recognized(tmp_path, index, old, new):
    selected = config()
    root = _write_capsule_inner(suite.entries(selected)[index], binding(selected), tmp_path)
    source = (root / "capsule.interface.mlir").read_text()
    assert old in source
    with pytest.raises(ValueError):
        read_tensor_graph(source.replace(old, new, 1), entry="forward", target_identity="any")


def test_rounding_case_detects_wrong_vector_policy_and_conversion(tmp_path):
    from merlin.targetgen.capsule_golden import compare, golden

    selected = config()
    entry = suite.entries(selected)[-1]
    root = _write_capsule_inner(entry, binding(selected), tmp_path)
    good = golden_store.load_golden(root)["outputs"]
    changed = copy.deepcopy(entry)
    changed["name"] += "_rne"
    changed["program"]["nodes"][0]["rounding"] = "rne"
    bad_root = _write_capsule_inner(changed, binding(selected), tmp_path)
    bad = golden_store.load_golden(bad_root)["outputs"]
    capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
    assert golden(capsule, root) == good
    assert compare(good, bad, capsule["numeric_policy"])["status"] == "fail"
    missing = dict(good)
    missing.pop("out3")
    assert compare(good, missing, capsule["numeric_policy"])["status"] == "fail"
    assert good["out0"] != bad["out0"]
    assert good["out0"] == good["out4"]
    assert np.array_equal(np.asarray(good["out0"]).T, good["out3"])
    assert good["out1"] == good["out2"]  # widening preserves the narrowed value
    assert good["out2"][0][2] == 1.0  # halfway FP8 rounding changes the source 1.0625
    assert "arith.shrui" in (root / "capsule.interface.mlir").read_text()


def test_alias_publication_and_fanout_are_observable():
    selected = config()
    for entry in suite.entries(selected):
        capsule, _ = corpus_spec.build(entry, binding(selected))
        outputs, _ = reference.evaluate(capsule)
        family = entry["source_reference"]
        if family == "alias_update":
            assert outputs["out0"] != outputs["out1"] == outputs["out2"]
            assert {"alias", "epoch_mutation"} <= set(capsule["component_program"]["effects"])
        if family == "residual_fanout":
            assert len(outputs) == 3
            assert {"shared_producer", "escaped_use"} <= set(capsule["component_program"]["effects"])
        if family == "shared_weight":
            assert "immutable_reuse" in capsule["component_program"]["effects"]
        if family == "ordered_contraction":
            assert outputs["out0"][0][0] == 2.0  # one final rounding would produce 3.0


@pytest.mark.parametrize(
    "operand,accumulator",
    [(operand, accumulator) for operand in ("fp8_e4m3", "fp8_e5m2") for accumulator in ("bf16", "fp16", "f32")]
    + [("bf16", "bf16"), ("bf16", "f32"), ("fp16", "fp16"), ("fp16", "f32"), ("f32", "f32")],
)
@pytest.mark.parametrize("shape", [[3, 5, 7], [1, 4, 1]])
def test_ordered_witness_remains_sensitive_to_reassociation(operand, accumulator, shape):
    selected = config(operand_dtype=operand, accumulator_dtype=accumulator, pointwise_rounding="rne", shape=shape)
    entry = suite.entries(selected)[0]
    capsule, _ = corpus_spec.build(entry, binding(selected))
    outputs, inputs = reference.evaluate(capsule)
    lhs, rhs = inputs["A"]["decoded"], inputs["W"]["decoded"]
    _, k, n = selected["shape"]
    exact = sum(Fraction(lhs[p]) * Fraction(rhs[p * n]) for p in range(k))
    dtype = capsule["component_program"]["selected_storage"]["accumulator"]
    assert outputs["out0"][0][0] != reference.rounded(exact, dtype)


def test_rounding_reference_ties_subnormals_signed_zero_and_order():
    assert reference.rounded(Fraction(257, 256), "bf16") == 1.0
    assert reference.rounded(Fraction(259, 256), "bf16") == 1.015625
    assert reference.rounded(Fraction(-259, 256), "bf16", "rtz") == -1.0078125
    assert reference.rounded(Fraction(1, 1 << 133), "bf16") == 2.0**-133
    assert math.copysign(1, reference.rounded(-0.0, "bf16")) == -1
    assert math.copysign(1, reference.add(-0.0, -0.0, "bf16")) == -1
    assert math.copysign(1, reference.multiply(-0.0, 1.0, "bf16")) == -1
    assert reference.add(reference.add(256.0, 1.0, "bf16"), -256.0, "bf16") == 0
    assert reference.add(256.0, reference.add(1.0, -256.0, "bf16"), "bf16") == 1
    for value in [float("nan"), float("inf"), 1e100]:
        with pytest.raises(ValueError, match="finite|overflows"):
            reference.rounded(value, "bf16")


def test_fraction_reference_matches_numpy_half_over_rounding_boundaries():
    # Independent second implementation for every adjacent positive normal FP16 pair.
    codes = np.arange(1, 0x7BFF, dtype=np.uint16)
    lower = codes.view(np.float16).astype(np.float64)
    upper = (codes + 1).view(np.float16).astype(np.float64)
    for index in range(0, len(codes), 13):
        midpoint = (lower[index] + upper[index]) / 2
        for value in [midpoint, -midpoint, np.nextafter(midpoint, float("inf"))]:
            assert reference.rounded(float(value), "f16") == float(np.float16(value))


@pytest.mark.parametrize("mutation", ["missing_policy", "implicit_conversion", "ignored_rounding", "bad_compute"])
def test_incomplete_semantics_refuse(mutation):
    selected = config()
    program = suite.entries(selected)[0]["program"]
    if mutation == "missing_policy":
        program.pop("floating_point")
    elif mutation == "implicit_conversion":
        program["inputs"][0]["dtype"] = "accumulator"
    elif mutation == "ignored_rounding":
        program["nodes"][0]["rounding"] = "rtz"
    else:
        program["nodes"][0]["compute_dtype"] = "f32"
    with pytest.raises(ValueError):
        component_program.render(program, operand_dtype="f8E4M3FN", accumulator_dtype="bf16")


@pytest.mark.parametrize("extension", [
    {"schema": component_program.SCHEMA_V2},
    {"schema": component_program.SCHEMA_V3},
    {"logical_io": component_program.LOGICAL_IO_V1},
])
def test_floating_program_cannot_claim_integer_only_schema_extensions(extension):
    program = suite.entries(config())[0]["program"]
    program.update(extension)
    with pytest.raises(ValueError, match="integer schema or logical IO"):
        component_program.render(program, operand_dtype="f8E4M3FN", accumulator_dtype="bf16")


def test_huge_source_refuses_before_tensor_materialization(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no data allocation before budget admission")

    monkeypatch.setattr(reference, "evaluate", forbidden)
    with pytest.raises(ValueError, match="budget"):
        suite.materialize(config(shape=[10**9, 10**9, 10**9]), tmp_path / "oversized")
    assert not (tmp_path / "oversized").exists()


def test_non_component_float_reference_binding_is_unchanged():
    selected = config(operand_dtype="bf16")
    regime, bound = corpus_spec.entry_binding({"op": "matmul"}, binding(selected))
    assert regime == "simt" and bound.accum_dtype == "f32"
    _, bound = corpus_spec.entry_binding(suite.entries(selected)[0], binding(selected))
    assert bound.accum_dtype == "bf16"


def test_selected_example_has_eight_cases_and_no_hardware_admission():
    selected = yaml.safe_load((repo_root() / "examples/atlas/phase0/floating-tests.yaml").read_bytes())
    assert len(suite.entries(selected)) == 8
    assert "tile" not in json.dumps(selected)


def test_generated_ir_is_accepted_by_selected_upstream_parser(tmp_path):
    parser = os.environ.get("MERLIN_TEST_MLIR_OPT")
    if not parser:
        pytest.skip("select an upstream mlir-opt with MERLIN_TEST_MLIR_OPT")
    manifest = suite.materialize(config(), tmp_path / "sources")
    for row in manifest["members"]:
        source = tmp_path / "sources" / row["path"] / "capsule.interface.mlir"
        result = subprocess.run([parser, str(source)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr


def test_all_sources_execute_through_upstream_cpu_lowering(tmp_path):
    """Execute a second numeric architecture; no candidate/reference implementation is shared."""
    parser = os.environ.get("MERLIN_TEST_MLIR_OPT")
    runner = os.environ.get("MERLIN_TEST_MLIR_RUNNER")
    runtime = os.environ.get("MERLIN_TEST_MLIR_RUNNER_LIB")
    if not all((parser, runner, runtime)):
        pytest.skip("select upstream parser, runner and runtime library for native source execution")
    selected = config(operand_dtype="fp16", accumulator_dtype="f32", pointwise_rounding="rne")
    manifest = suite.materialize(selected, tmp_path / "sources")
    for member in manifest["members"]:
        directory = tmp_path / "sources" / member["path"]
        cap = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        golden = golden_store.load_golden(directory)
        source = (directory / "capsule.interface.mlir").read_text().rstrip()
        # Append an independent constant-input caller to the outer module.
        source = source[: source.rfind("}")]
        source += "\nfunc.func private @printI64(i64)\nfunc.func private @printNewline()\nfunc.func @main() {\n"

        def tensor_type(row):
            return "tensor<" + "x".join(map(str, row["shape"])) + "x" + row["dtype"] + ">"

        for row in cap["inputs"]:
            values = golden["oracle_provenance"]["inputs"][row["name"]]["decoded"]
            values = np.asarray(values).reshape(row["shape"]).tolist()
            source += f"%input_{row['name']} = arith.constant dense<{values}> : {tensor_type(row)}\n"
        outputs = cap["component_program"]["outputs"]
        args = ", ".join("%input_" + row["name"] for row in cap["inputs"])
        types = ", ".join(tensor_type(row) for row in cap["inputs"])
        result_types = ", ".join(tensor_type(row) for row in outputs)
        source += f"%result:{len(outputs)} = func.call @forward({args}) : ({types}) -> ({result_types})\n"
        ordinal = 0
        for output_index, row in enumerate(outputs):
            for i in range(row["shape"][0]):
                for j in range(row["shape"][1]):
                    source += f"%i{ordinal} = arith.constant {i} : index\n%j{ordinal} = arith.constant {j} : index\n"
                    source += (
                        f"%v{ordinal} = tensor.extract %result#{output_index}"
                        f"[%i{ordinal}, %j{ordinal}] : {tensor_type(row)}\n"
                    )
                    value = f"%v{ordinal}"
                    if row["dtype"] != "f32":
                        source += f"%w{ordinal} = arith.extf {value} : {row['dtype']} to f32\n"
                        value = f"%w{ordinal}"
                    source += (
                        f"%bits{ordinal} = arith.bitcast {value} : f32 to i32\n"
                        f"%int{ordinal} = arith.extui %bits{ordinal} : i32 to i64\n"
                        f"func.call @printI64(%int{ordinal}) : (i64) -> ()\n"
                        "func.call @printNewline() : () -> ()\n"
                    )
                    ordinal += 1
        source += "return\n}\n}\n"
        harness, lowered = directory / "native.mlir", directory / "native-lowered.mlir"
        harness.write_text(source)
        pipeline = [
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries",
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--convert-scf-to-cf",
            "--arith-expand",
            "--expand-strided-metadata",
            "--convert-arith-to-llvm",
            "--convert-func-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-cf-to-llvm",
            "--reconcile-unrealized-casts",
        ]
        result = subprocess.run(
            [parser, str(harness), *pipeline, "-o", str(lowered)], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr
        result = subprocess.run(
            [runner, str(lowered), "-e", "main", "-entry-point-result=void", "-shared-libs=" + runtime],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        observed = [int(line) for line in result.stdout.splitlines() if line.strip()]
        expected = (
            np.concatenate([np.asarray(golden["outputs"][row["name"]], dtype=np.float32).ravel() for row in outputs])
            .view(np.uint32)
            .tolist()
        )
        assert observed == expected, member["path"]
