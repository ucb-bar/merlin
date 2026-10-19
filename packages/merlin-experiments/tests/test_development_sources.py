"""Independent integer references for complete fresh grouped-head source calls."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from merlin_experiments.phase0 import development_sources as D

from merlin.common.cpu_pool import shared_compute_pool


def declaration(**geometry):
    return {
        "schema": D.SCHEMA,
        "scope": "public_development",
        "name": "grouped_mix",
        "pattern": D.PATTERN,
        "geometry": {
            "steps": 2,
            "prefix_tokens": 3,
            "query_tokens": 1,
            "kv_heads": 1,
            "query_heads": 2,
            "head_dim": 5,
            **geometry,
        },
        "arithmetic": {"contract": D.CONTRACT, "readout_shift": 3},
        "max_logical_elements": 100000,
    }


def evaluate(declared):
    pytest.importorskip("torch")
    namespace = {}
    exec(compile(D.render(declared), "fresh-public-development-loader", "exec"), namespace)
    model, inputs = namespace["get_model_and_inputs"]()
    with shared_compute_pool(timeout_s=60):
        outputs = model(*inputs)
    return model, inputs, outputs


def reference(declared, inputs):
    values = [value.numpy().astype(np.int64) for value in inputs]
    cache_k, cache_v = values[:2]
    geometry = declared["geometry"]
    shift = declared["arithmetic"]["readout_shift"]
    outputs = []
    for step in range(geometry["steps"]):
        q, k, v = values[2 + step * 3 : 5 + step * 3]
        cache_k = np.concatenate((cache_k, k), axis=1)
        cache_v = np.concatenate((cache_v, v), axis=1)
        for head in range(geometry["query_heads"]):
            group = head // (geometry["query_heads"] // geometry["kv_heads"])
            scores = q[head] @ cache_k[group].T
            readout = np.clip((scores + 2 ** (shift - 1)) // 2**shift, -128, 127)
            outputs += [scores, readout @ cache_v[group]]
        outputs += [cache_k.copy(), cache_v.copy()]
    return outputs


@pytest.mark.parametrize(
    "geometry",
    [
        {},
        {"query_tokens": 3, "head_dim": 7, "kv_heads": 2, "query_heads": 6},
        pytest.param(
            {"steps": 1, "prefix_tokens": 7, "query_tokens": 2, "head_dim": 1},
            marks=pytest.mark.xfail(strict=True, reason="selected CPU Torch _int_mm returns incorrect K=1 results"),
        ),
    ],
)
def test_complete_integer_outputs_and_carried_state_match_independent_reference(geometry):
    declared = declaration(**geometry)
    model, inputs, actual = evaluate(declared)
    description = D.describe(declared)
    expected = reference(declared, inputs)
    assert len(actual) == len(description["outputs"]) == len(expected)
    assert description["contractions"] == 2 * declared["geometry"]["steps"] * declared["geometry"]["query_heads"]
    for row, value, result in zip(description["outputs"], actual, expected, strict=True):
        np.testing.assert_array_equal(value.numpy(), result)
        assert list(value.shape) == row["shape"]
        assert str(value.dtype) == "torch." + row["dtype"]
    assert any((value.numpy() < 0).any() for value in inputs)
    assert any((value.numpy() > 0).any() for value in inputs)
    assert any((value.numpy() != 0).any() for value in actual)
    # Prefix state affects later results. The complete state is observable,
    # rather than a disconnected collection of isolated contractions.
    changed = list(inputs)
    changed[0] = changed[0].clone()
    changed[0][0, 0, 0] += 1
    assert any(not np.array_equal(a.numpy(), b.numpy()) for a, b in zip(actual, model(*changed), strict=True))


def test_standard_export_retains_all_outputs_and_integer_contractions():
    torch = pytest.importorskip("torch")
    model, inputs, actual = evaluate(declaration())
    exported = torch.export.export(model, inputs)
    targets = [node.target for node in exported.graph.nodes]
    assert targets.count(torch.ops.aten._int_mm.default) == 8
    exported_outputs = exported.module()(*inputs)
    assert len(exported_outputs) == len(actual)
    for expected, value in zip(actual, exported_outputs, strict=True):
        np.testing.assert_array_equal(expected.numpy(), value.numpy())


def test_declared_readout_rounding_and_saturation_on_independent_inputs():
    declared = declaration(steps=1, head_dim=5)
    model, inputs, _ = evaluate(declared)
    # Exercise both saturation endpoints and signed half-up ties. These values
    # are new stimuli, not answers or parameters from an existing model.
    changed = [value.clone() for value in inputs]
    changed[0].fill_(127)
    changed[0][0, 0, 0] = 4
    changed[0][0, 1].fill_(-128)
    changed[0][0, 1, 0] = -4
    changed[2][0].fill_(127)
    changed[2][1].zero_()
    changed[2][1, 0, 0] = 1
    actual = model(*changed)
    expected = reference(declared, changed)
    for value, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value.numpy(), wanted)
    assert (((expected[0] + 4) // 8) > 127).any()
    assert (((expected[0] + 4) // 8) < -128).any()
    assert expected[2][0, :2].tolist() == [4, -4]


def test_source_materializes_through_existing_roster_stager(tmp_path):
    staged = D.stage([declaration()], tmp_path / "development")
    row = staged["members"][0]
    record = staged["staging"]["performance_applications"]["grouped_mix"]
    source = Path(record["workload_root"]) / "loader.py"
    assert source.read_text() == D.render(declaration())
    assert record["loader_sha256"] == row["loader_sha256"]
    assert staged["phase0_admission"] == "not_granted"
    assert "softmax" in row["missing"]


def test_maximum_depth_has_no_constant_integer_input_stream():
    _, inputs, _ = evaluate(declaration(steps=4, head_dim=17))
    assert all(len(np.unique(value.numpy())) == 17 for value in inputs)


@pytest.mark.parametrize("defect", ["private", "heads", "overflow", "budget", "arithmetic", "depth"])
def test_invalid_public_domain_refuses_before_source_creation(tmp_path, defect):
    declared = copy.deepcopy(declaration())
    if defect == "private":
        declared["scope"] = "validation"
    elif defect == "heads":
        declared["geometry"].update(kv_heads=3, query_heads=4)
    elif defect == "overflow":
        declared["geometry"].update(prefix_tokens=65536, query_tokens=65536)
    elif defect == "budget":
        declared["max_logical_elements"] = 1
    elif defect == "arithmetic":
        declared["arithmetic"]["contract"] = "different_rounding"
    else:
        declared["geometry"]["steps"] = 5
    with pytest.raises(ValueError):
        D.stage([declared], tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_ordinary_capture_and_model_writer_preserve_complete_public_source(tmp_path):
    """Optional installed-toolchain integration; no simulator or target compiler."""
    from merlin.targetgen import capsule_source
    from merlin.targetgen.corpus_spec import CorpusBinding

    source = capsule_source.PytorchRefSource(timeout=120)
    if not source.available():
        pytest.skip("selected Model2MLIR capture runtime unavailable")
    declared = declaration()
    staged = D.stage([declared], tmp_path / "sources")
    member = staged["staging"]["performance_applications"][declared["name"]]
    loader = Path(member["workload_root"]) / "loader.py"
    with shared_compute_pool(timeout_s=60):
        captured = source.capture_loader(loader, "int8", workdir=tmp_path / "capture", already_quantized=True)
    assert captured.meta["capture_quantization"] == "already_materialized"
    assert captured.meta["opaque"] == 0
    assert captured.meta["materialized_contractions"] == D.describe(declared)["contractions"]
    _, inputs, _ = evaluate(declared)
    expected = reference(declared, inputs)
    assert len(captured.golden) == len(expected)
    for actual, wanted in zip(captured.golden, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    # The ordinary writer consumes this exact capture, never recaptures it or
    # substitutes a hand-authored interface/reference for a refusal.
    binding = CorpusBinding(
        target="source_only",
        tile_dim=None,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0"],
        compare="exact_int",
    )
    cached = SimpleNamespace(m2m_dir=source.m2m_dir, capture_loader=lambda *args, **kwargs: captured)
    entry = D.capsule_entry(declared, loader)
    historical = {key: value for key, value in entry.items() if key != "output_numeric_policy"}
    with pytest.raises(capsule_source.M2MUnavailable, match="different output dtypes"):
        capsule_source.write_model_capsule(historical, binding, tmp_path / "historical", source=cached)
    with pytest.raises(capsule_source.M2MUnavailable, match="captured_per_output_v1"):
        capsule_source.write_model_capsule(
            {**entry, "output_numeric_policy": "different"}, binding, tmp_path / "invalid", source=cached
        )
    directory = capsule_source.write_model_capsule(entry, binding, tmp_path / "capsules", source=cached)
    import yaml

    capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    golden = yaml.safe_load((directory / "golden.yaml").read_bytes())
    assert capsule["source_reference"] == entry["source_reference"]
    assert set(golden["outputs"]) == set(entry["outs"])
    assert capsule["numeric_policy"]["compare"] == "per_output"
    assert set(capsule["numeric_policy"]["outputs"]) == set(entry["outs"])
    assert all(policy["compare"] == "exact_int" for policy in capsule["numeric_policy"]["outputs"].values())
    assert capsule["operation"]["attributes"]["materialized_contractions"] == 8
