"""Native execution of observable nested splat compaction, with separate bit-level references.

Requires the upstream host runner; excluded from binding-only CI qualification.
"""

import hashlib
import json
import math

import pytest
from test_lean_differential import _floats, _tensor_samples
from test_lean_splat_nested import MUTATE, _source, _stage

from merlin.common.paths import repo_root
from merlin.verify.lean import differential as diff

OPTIONS = [
    {},
    {"rank_zero": True},
    {"rank_zero": True, "dirty": True},
    {"rank_zero": True, "captured": True, "escape": True, "multiple": True},
    {"multiple": True, "tensor_capture": True},
    {"outer_shape": ()},
    {"outer_shape": (0,), "rank_zero": True},
    {"element": "i13", "multiple": True},
    {"element": "f32", "captured": True, "escape": True},
    {"element": "f64", "rank_zero": True},
]


def _reference(samples, options):
    element = options.get("element", "i32")
    result = []
    for payloads, outer_init, data, inner_init, captured, alternate in samples:
        values = []
        for payload, old in zip(payloads, outer_init, strict=True):
            value = captured[0] if options.get("captured") else payload
            if element.startswith("i"):
                value += data[0] + old
                if not options.get("dirty"):
                    value += captured[0] if options.get("fresh_init", True) else inner_init[0]
                if options.get("multiple"):
                    value += alternate[0]
                value %= 1 << int(element[1:])
            values.append(value)
        if options.get("escape"):
            values += [captured[0]] * math.prod(options.get("inner_shape", (2, 3)))
        result.append([str(x) for x in values])
    return result


@pytest.mark.parametrize("options", OPTIONS)
def test_nested_compaction_native_and_independent_reference(tmp_path, fidelity_tools, options):
    source = _source(**options)
    stage, count = _stage(tmp_path, source)
    assert count == 1 + int(options.get("multiple", False))
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    element = options.get("element", "i32")
    if element.startswith("f"):
        edges = _floats(int(element[1:])) + [0x7FC01234 if element == "f32" else 0x7FF8000000001234]
        samples.extend([[arg.copy() for arg in samples[0]] for _ in range(max(0, len(edges) - len(samples)))])
        for i, sample in enumerate(samples):
            sample[0] = [edges[(i + j) % len(edges)] for j in range(len(sample[0]))]
            sample[4] = [edges[i % len(edges)]]
    expected = _reference(samples, options)
    for name in ["before", "after"]:
        work = tmp_path / name
        diff.compare((stage / f"{name}.mlir").read_text(), samples, work, fidelity_tools)
        actual = json.loads((work / "native-results.json").read_text())["results"]
        assert actual == expected


def test_nested_wrong_payload_is_observed_natively(tmp_path, fidelity_tools):
    stage, _ = _stage(tmp_path, _source(captured=True, escape=True))
    before, after = stage / "before.mlir", stage / "after.mlir"
    diff._run([str(fidelity_tools.compiler_python), "-c", MUTATE, str(after), "payload"])
    samples = [[[1, 2], [3, 4], [0] * 6, [0] * 6, [7], [13]]]
    for name, program in [("before", before), ("mutated", after)]:
        diff.compare(program.read_text(), samples, tmp_path / name, fidelity_tools)
    original = json.loads((tmp_path / "before/native-results.json").read_text())["results"]
    mutated = json.loads((tmp_path / "mutated/native-results.json").read_text())["results"]
    assert original == [["17", "18"] + ["7"] * 6]
    assert mutated == [["23", "24"] + ["7"] * 6]


def _minimal_captured_destination(fresh=False):
    return f"""func.func @subject(%payloads: tensor<2xi32>, %outer_init: tensor<2xi32>,
        %inner_init: tensor<i32>, %seed: i32) -> tensor<2xi32> {{
      %r = linalg.generic {{ indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>],
          iterator_types = ["parallel"] }} ins(%payloads : tensor<2xi32>) outs(%outer_init : tensor<2xi32>) {{
        ^bb0(%value: i32, %old: i32):
          {"%fresh = tensor.splat %seed : tensor<i32>" if fresh else ""}
          %inner = linalg.generic {{ indexing_maps = [affine_map<()->()>], iterator_types = [] }}
              outs({"%fresh" if fresh else "%inner_init"} : tensor<i32>) {{
            ^bb0(%previous: i32):
              %sum = arith.addi %value, %previous : i32
              linalg.yield %sum : i32
          }} -> tensor<i32>
          %scalar = tensor.extract %inner[] : tensor<i32>
          linalg.yield %scalar : i32
      }} -> tensor<2xi32>
      return %r : tensor<2xi32>
    }}"""


def _evaluate_separately(path, samples, work, tools):
    from merlin.verify.lean.export import export_file

    work.mkdir()
    exported = work / "export.json"
    export_file(path, exported, backend="mlir", m2m_python=tools.compiler_python)
    inputs, outputs = diff.signature(json.loads(exported.read_text()), "subject")
    encoded = [[[str(value) for value in argument] for argument in sample] for sample in samples]
    lean = diff.evaluate_lean(exported, encoded, work, tools)
    native = diff.execute_mlir(path, encoded, work, tools, inputs=inputs, outputs=outputs)
    return lean, native


def test_captured_destination_records_native_lowering_discrepancy(tmp_path, fidelity_tools):
    # S4: no splat operation or compaction is needed to reproduce this native-lowering gap.
    # Tensor semantics read the original immutable destination at each outer iteration.
    # The current bufferization/loop pipeline instead reuses the first result on iteration two.
    stage, count = _stage(tmp_path, _minimal_captured_destination())
    assert count == 0
    samples = [[[1, 2], [0, 0], [1], [1]]]
    for name in ["before", "after"]:
        lean, native = _evaluate_separately(stage / f"{name}.mlir", samples, tmp_path / name, fidelity_tools)
        assert [sample["status"] for sample in lean] == ["ok"]
        assert lean[0]["values"][0]["bits"] == ["2", "3"]
        assert native["results"] == [["2", "4"]]
        (tmp_path / name / "known-discrepancy.json").write_text(
            json.dumps(
                {
                    "id": "S4",
                    "lean_driver_sha256": hashlib.sha256(
                        (repo_root() / "merlin/lean/Differential.lean").read_bytes()
                    ).hexdigest(),
                    "spec_lock_sha256": hashlib.sha256(
                        (repo_root() / "merlin/lean/SPEC_LOCK").read_bytes()
                    ).hexdigest(),
                    "semantics_sha256": {
                        str(path.relative_to(repo_root() / "merlin/lean")): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in sorted((repo_root() / "merlin/lean/MerlinVerify/Sem").rglob("*.lean"))
                    },
                    "independent_reference": [["2", "3"]],
                    "lean": lean,
                    "native": native,
                    "scope": "captured initialized destination reused by nested generic iterations",
                },
                indent=2,
            )
            + "\n"
        )


def test_fresh_nested_destination_restores_native_fidelity(tmp_path, fidelity_tools):
    stage, count = _stage(tmp_path, _minimal_captured_destination(fresh=True))
    assert count == 0
    samples = [[[1, 2], [0, 0], [1], [1]]]
    for name in ["before", "after"]:
        work = tmp_path / name
        diff.compare((stage / f"{name}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == [["2", "3"]]


def test_actual_compaction_retains_captured_destination_discrepancy(tmp_path, fidelity_tools):
    stage, count = _stage(tmp_path, _source(fresh_init=False))
    assert count == 1
    samples = [[[1, 2], [0, 0], [0] * 6, [1] * 6, [7], [13]]]
    for name in ["before", "after"]:
        lean, native = _evaluate_separately(stage / f"{name}.mlir", samples, tmp_path / name, fidelity_tools)
        assert lean[0]["values"][0]["bits"] == ["2", "3"]
        assert native["results"] == [["2", "4"]]


def test_three_nested_generics_match_independent_reference(tmp_path, fidelity_tools):
    from test_lean_splat_nested import _three_level_source

    source = _three_level_source()
    stage, count = _stage(tmp_path, source)
    assert count == 1
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    expected = []
    for payloads, _outer_init, data, _inner_init, captured, _alternate, driver, _driver_init in samples:
        expected.append([str((payloads[0] + data[0] + captured[0] + value) % (1 << 32)) for value in driver])
    for name in ["before", "after"]:
        work = tmp_path / name
        diff.compare((stage / f"{name}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected


def test_four_nested_generics_match_independent_reference(tmp_path, fidelity_tools):
    from test_lean_splat_nested import _four_level_source

    source = _four_level_source()
    stage, count = _stage(tmp_path, source)
    assert count == 1
    samples = _tensor_samples(source, tmp_path, fidelity_tools)
    expected = _reference(samples, {"rank_zero": True})
    for name in ["before", "after"]:
        work = tmp_path / name
        diff.compare((stage / f"{name}.mlir").read_text(), samples, work, fidelity_tools)
        assert json.loads((work / "native-results.json").read_text())["results"] == expected
