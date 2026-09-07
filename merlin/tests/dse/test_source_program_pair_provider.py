"""Real source/oracle validation before a complete-source runtime is admitted."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.perf.source_contraction_witness import extract_source_contraction
from merlin.perf.source_contraction_preparation import _typed_cases
from merlin.perf.source_program_pair import text_digest
from merlin.perf.source_program_pair_provider import _validated_inputs, _output_matches, SourceProgramPairProvider


SOURCE = '''module {func.func @work(%a:tensor<3x9xi8>,%b:tensor<9x4xi8>,%c:tensor<3x4xi8>)->tensor<3x4xi8>{
%r=linalg.matmul ins(%a,%b:tensor<3x9xi8>,tensor<9x4xi8>) outs(%c:tensor<3x4xi8>)->tensor<3x4xi8>
func.return %r:tensor<3x4xi8>}}'''


def fixture(tmp_path):
    bounds = {"max_m": 2, "max_n": 3, "max_k": 5, "max_macs": 100}
    source, extraction = extract_source_contraction(SOURCE, 0, entry="work", **bounds)
    oracle = _typed_cases(source, extraction, lambda: 1)
    (tmp_path/"interface.mlir").write_text(source)
    oracle_file = tmp_path/"oracle.json"
    oracle_file.write_text(json.dumps(oracle))
    artifacts = {arm: {"compiler_sha256": arm*8, "lowered_sha256": "different"+arm,
                       "command_buffer_sha256": "buffer"+arm} for arm in ("before", "after")}
    pair = SimpleNamespace(source=SOURCE, source_sha256=text_digest(SOURCE), graph_sha256="graph",
                           artifacts=artifacts, comparison_binding={"exact": "previous"})
    prepared = {"schema": "source_contraction_preparation_v1", "status": "prepared",
        "source_sha256": pair.source_sha256, "logical_dispatch_digest": "graph",
        "full_model_artifact_binding": copy.deepcopy(artifacts), "comparison_binding": pair.comparison_binding,
        "source_op_index": 0, "entry": "work", "host_bounds": bounds, "extraction": extraction,
        "workdir": str(tmp_path), "independent_oracle": str(oracle_file),
        "independent_oracle_sha256": text_digest(oracle_file.read_text())}
    return prepared, pair, oracle


def test_exact_typed_source_and_three_inputs_are_recomputed(tmp_path):
    prepared, pair, oracle = fixture(tmp_path)
    source, extraction, actual, case, payloads, expected = _validated_inputs(prepared, pair)
    assert source != SOURCE and actual == oracle
    assert len(payloads) == 3 and list(map(len, payloads)) == [10, 15, 6]
    assert case["case"] == oracle["cases"][0]["case"]
    np.testing.assert_array_equal(expected, np.asarray(case["expected"], dtype=np.int8))
    assert extraction["arithmetic"]  # exact source modular semantics, no saturated substitution


@pytest.mark.parametrize("mutation", ["compiler", "source", "interface", "oracle_bytes", "oracle_expected", "shape", "initializer"])
def test_invalid_or_unrelated_source_never_admitted(tmp_path, mutation):
    prepared, pair, oracle = fixture(tmp_path)
    if mutation == "compiler":
        pair.artifacts["after"]["compiler_sha256"] = "changed"
    elif mutation == "source":
        prepared["source_op_index"] = 1
    elif mutation == "interface":
        (tmp_path/"interface.mlir").write_text("unrelated source")
    elif mutation == "oracle_bytes":
        Path(prepared["independent_oracle"]).write_text("{}")
    else:
        if mutation == "oracle_expected":
            oracle["cases"][0]["expected"][0][0] += 1
        elif mutation == "shape":
            oracle["input_shapes"][0] = [1, 10]
        else:
            oracle["cases"][0]["inputs"].pop()
        text = json.dumps(oracle)
        Path(prepared["independent_oracle"]).write_text(text)
        prepared["independent_oracle_sha256"] = text_digest(text)
    with pytest.raises((ValueError, IndexError)):
        _validated_inputs(prepared, pair)


def test_console_shape_is_explicit_not_flattened_arbitrarily():
    expected = np.arange(9, dtype=np.int32).reshape(1, 1, 3, 3)
    assert _output_matches({"Y": [[0,1,2],[3,4,5],[6,7,8]]}, "Y", expected)
    assert not _output_matches({"Y": list(range(9))}, "Y", expected)
    assert not _output_matches({"Y": [[0,1,2],[3,4,5],[6,7,9]]}, "Y", expected)
    assert not _output_matches({"Y": [[0,1,2],[3,4,5],[6,7,8]], "extra": []}, "Y", expected)


@pytest.mark.parametrize("budget", [0, True, 61, float("inf")])
def test_invalid_runtime_budget_precedes_any_adapter_action(tmp_path, budget):
    provider = SourceProgramPairProvider(target="test", adapter=object(), output=tmp_path)
    with pytest.raises(ValueError, match="budget"):
        provider(candidate=tmp_path, experiment=None, prepared={}, timeout_s=budget)
