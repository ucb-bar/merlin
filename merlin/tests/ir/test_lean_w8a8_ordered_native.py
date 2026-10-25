"""Compare actual tensor execution in the proved ordered model with native upstream MLIR."""

from __future__ import annotations

import json
import random

import pytest
from lean_w8a8_ordered_cases import source

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff
from merlin.verify.lean.check import _file_ref, lean_root
from merlin.verify.lean.export import export_file


def compare(work, before, samples, tools):
    source_path = work / "source.mlir"
    source_path.write_text(str(before))
    exported = work / "export.json"
    export_file(source_path, exported, backend="mlir", m2m_python=tools.compiler_python)
    inputs, outputs = diff.signature(json.loads(exported.read_text()), "forward")
    samples = diff._samples(samples, inputs, 64)
    request = work / "model-request.json"
    request.write_text(json.dumps({"function": "forward", "samples": samples}))
    fixture = repo_root() / "merlin/tests/fixtures/lean_w8a8_ordered_oracle.lean"
    process = diff._run(
        [str(tools.lake), "env", "lean", "--run", str(fixture), str(exported), str(request)], cwd=lean_root()
    )
    lean = json.loads(process.stdout)
    (work / "model-results.json").write_text(json.dumps(lean, indent=2) + "\n")
    # Only actual defined model results may be compared. Native execution is not a poison oracle.
    assert all(row["status"] == "ok" for row in lean), lean
    assert all(word is not None for row in lean for value in row["values"] for word in value["bits"]), lean
    native = diff.execute_mlir(
        source_path, samples, work / "native", tools, inputs=inputs, outputs=outputs, function="forward"
    )
    differences = []
    for row, (expected, actual) in enumerate(zip(lean, native["results"], strict=True)):
        flat = [int(word) for value in expected["values"] for word in value["bits"]]
        for element, (left, right) in enumerate(zip(flat, actual, strict=True)):
            right = int(right)
            if left != right:
                differences.append(
                    {
                        "sample": row,
                        "element": element,
                        "model_bits": left,
                        "native_bits": right,
                        "both_nan": diff._nan(left, 32) and diff._nan(right, 32),
                    }
                )
    (work / "comparison.json").write_text(
        json.dumps(
            {
                "samples": len(samples),
                "source": _file_ref(source_path),
                "export": _file_ref(exported),
                "oracle_adapter": _file_ref(fixture),
                "logical_model": "W8A8ByteModel.extend (W8A8IEEEPrimitives.model unsupported)",
                "pipeline": native["pipeline"],
                "mlir_revision": native["mlir_revision"],
                "differences": differences,
                "universal_native_fidelity_proved": False,
            },
            indent=2,
        )
        + "\n"
    )
    diff.assert_equal(lean, native["results"], outputs, samples, arithmetic_nans=True)
    return lean


def test_ordered_rounding_and_exceptions_in_actual_matmul(tmp_path, fidelity_tools):
    one = 0x3F800000
    vectors = [
        [0x4B800000, one, 0xCB800000],
        [0x4B800000, 0xCB800000, one],
        [1, 1, 0],
        [0x80000000, 0x80000000, 0x80000000],
        [0x7F7FFFFF, 0x7F7FFFFF, 0],
        [0x7FC00001, one, 0],
        [0xFFC00042, one, 0],
    ]
    result = compare(tmp_path, source(m=1, k=3, n=1), [[values, [one] * 3] for values in vectors], fidelity_tools)
    assert result[0]["values"][0]["bits"] == ["0"]
    assert result[1]["values"][0]["bits"] == [str(one)]
    assert result[2]["values"][0]["bits"] == ["2"]
    assert result[4]["values"][0]["bits"] == [str(0x7F800000)]


@pytest.mark.parametrize("kind", ["matmul", "generic", "batch_matmul"])
def test_multiple_cells_and_batches_use_actual_maps(tmp_path, fidelity_tools, kind):
    generator = random.Random(240127)
    count = 2 if kind == "batch_matmul" else 1

    def finite():
        # Fixed-seed sign/exponent/significand cases, independent of theorem outputs.
        return (generator.randrange(2) << 31) | (generator.randrange(90, 145) << 23) | generator.randrange(1 << 23)

    samples = [[[finite() for _ in range(count * 2 * 3)], [finite() for _ in range(count * 3 * 4)]] for _ in range(5)]
    compare(tmp_path, source(kind=kind), samples, fidelity_tools)


@pytest.mark.parametrize("shape", [(1, 0, 2), (0, 3, 2)])
def test_empty_reductions_and_empty_outputs(tmp_path, fidelity_tools, shape):
    m, k, n = shape
    compare(tmp_path, source(m=m, k=k, n=n), [[[0x3F800000] * (m * k), [0x3F800000] * (k * n)]], fidelity_tools)


def test_negative_zero_initializer_is_preserved_by_actual_execution(tmp_path, fidelity_tools):
    result = compare(tmp_path, source(m=1, k=1, n=1, initial=-0.0), [[[0x80000000], [0x3F800000]]], fidelity_tools)
    assert result[0]["values"][0]["bits"] == [str(0x80000000)]


def test_ordered_bounds_and_material_guard_counterexample_are_kernel_checked(tmp_path, lean_oracle):
    fixture = repo_root() / "merlin/tests/fixtures/lean_w8a8_ordered.lean"
    process = diff._run([str(lean_oracle), "env", "lean", str(fixture)], cwd=lean_root())
    (tmp_path / "kernel.log").write_text(process.stdout + process.stderr)
    for root in (
        "qkv_2048",
        "ordered_rounding_controls",
        "unsafe_limit_overflows",
        "exact_body_controls",
        "existing_context_allows_nan_trig",
    ):
        assert f"'{root}' depends on axioms:" in process.stdout
    assert "sorryAx" not in process.stdout + process.stderr
