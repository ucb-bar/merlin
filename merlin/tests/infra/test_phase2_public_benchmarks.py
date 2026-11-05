"""Small numerical and real stock-intake checks for the public shape supplement."""

import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from merlin.common.paths import repo_root


@pytest.fixture
def owner():
    path = repo_root() / "examples/gemmini/phase2/benchmarks/prepare.py"
    spec = importlib.util.spec_from_file_location("public_benchmarks_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def profile():
    return yaml.safe_load((repo_root() / "examples/gemmini/phase2/benchmarks/profile.yaml").read_bytes())


def load_source(source):
    namespace = {}
    exec(compile(source, "public_loader.py", "exec"), namespace)
    return namespace["get_model_and_inputs"]()


def test_exact_paper_axes_and_separate_user_variant(owner, profile):
    rows = profile["benchmarks"]
    assert len(rows) == 10
    assert [row["geometry"] for row in rows[:7]] == [
        {"M": m, "N": n, "K": k}
        for m, n, k in [
            (512, 512, 512),
            (12544, 256, 64),
            (12544, 64, 256),
            (3136, 512, 128),
            (3136, 128, 512),
            (784, 1024, 256),
            (12, 512, 512),
        ]
    ]
    assert rows[6]["source_case"] == "user_additional_variant"
    for row, side, channels in zip(rows[7:], [56, 28, 14], [64, 128, 256], strict=True):
        data = owner.describe(row, "native_i8_scale1", seed=profile["seed"])
        assert data["inputs"][0] == {"shape": [4, side + 2, side + 2, channels], "dtype": "i8"}
        assert data["output"] == {"shape": [4, side, side, channels], "dtype": "i8"}
        assert data["inputs"][2] == {"shape": [channels], "dtype": "i32"}
    for row in rows:
        a = owner.describe(row, "native_i8_scale1", seed=19)
        b = owner.describe(row, "raw_i32", seed=19)
        assert a["source_group"] == b["source_group"]
        assert a["arithmetic"]["scale"] == 1
        assert a["arithmetic"]["activation"] == "none"


@pytest.mark.parametrize("readout", ["native_i8_scale1", "raw_i32"])
@pytest.mark.parametrize("kind", ["gemm", "conv2d"])
def test_actual_pytorch_sources_match_independent_complete_integer_reference(owner, profile, kind, readout):
    pytest.importorskip("torch")
    row = copy.deepcopy(next(r for r in profile["benchmarks"] if r["kind"] == kind))
    if kind == "gemm":
        row["geometry"] = {"M": 3, "N": 5, "K": 7}
    else:
        row["geometry"].update(
            batch=2, input_height=5, input_width=6, input_channels=2, output_channels=3, kernel_height=2, kernel_width=3
        )
    module, inputs = load_source(owner.render(row, readout, seed=19))
    arrays = [value.numpy().astype(np.int64) for value in inputs]
    if kind == "gemm":
        expected = arrays[0] @ arrays[1]
    else:
        x, w, bias = arrays
        expected = np.empty((2, 4, 4, 3), dtype=np.int64)
        for b, y, xcoord, co in np.ndindex(expected.shape):
            expected[b, y, xcoord, co] = np.sum(x[b, y : y + 2, xcoord : xcoord + 3, :] * w[:, :, :, co]) + bias[co]
    if readout == "native_i8_scale1":
        expected = np.clip(expected, -128, 127).astype(np.int8)
        assert (expected < 0).any()  # no implicit ReLU
        assert ((expected > -128) & (expected < 127)).any()
        assert np.unique(expected).size > 1
    else:
        expected = expected.astype(np.int32)
    result = module(*inputs).detach().numpy()
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)
    _, again = load_source(owner.render(row, readout, seed=19))
    for first, second in zip(inputs, again, strict=True):
        np.testing.assert_array_equal(first.numpy(), second.numpy())
    if readout == "native_i8_scale1":
        import torch

        extreme = [torch.full_like(t, 127) for t in inputs]
        assert (module(*extreme) == 127).all()
        extreme[0].fill_(-128)
        assert (module(*extreme) == -128).all()


def test_source_geometry_refuses_unknown_or_unsafe_semantics(owner, profile):
    for mutate in [
        lambda r: r["geometry"].update(K=True),
        lambda r: r.update(bias=True),
        lambda r: r["geometry"].update(K=2**20),
        lambda r: r["geometry"].update(transpose=True),
        lambda r: r.update(id=""),
    ]:
        row = copy.deepcopy(profile["benchmarks"][0])
        mutate(row)
        with pytest.raises(ValueError):
            owner.describe(row, "native_i8_scale1", seed=19)
    for mutate in [
        lambda r: r["geometry"].update(padding=[1, 1]),
        lambda r: r["geometry"].update(stride=[True, True]),
        lambda r: r.update(bias=False),
    ]:
        row = copy.deepcopy(profile["benchmarks"][-1])
        mutate(row)
        with pytest.raises(ValueError):
            owner.describe(row, "native_i8_scale1", seed=19)


def test_real_ordinary_roster_stage_preserves_original_objective_and_sources(owner, tmp_path):
    root = repo_root()
    descriptor = root / "examples/gemmini/target/descriptor.yaml"
    roster = root / "examples/gemmini/target/phase0-workloads.yaml"
    original = yaml.safe_load(descriptor.read_bytes())
    output = tmp_path / "prepared"
    report = owner.prepare(
        root / "examples/gemmini/phase2/benchmarks/profile.yaml",
        descriptor=descriptor,
        base_roster=roster,
        software_spec=root / "examples/gemmini/target/software-spec.yaml",
        output=output,
    )
    selected = yaml.safe_load((output / "descriptor.yaml").read_bytes())
    assert selected["phase1_gates"] == original["phase1_gates"]
    assert selected["workload_spec"]["applications"] == original["workload_spec"]["applications"]
    assert {k: v for k, v in selected["workload_spec"].items() if k != "performance_applications"} == {
        k: v for k, v in original["workload_spec"].items() if k != "performance_applications"
    }
    staged = json.loads((output / "staged/staged-workloads.json").read_bytes())
    original_members = yaml.safe_load(roster.read_bytes())["performance_applications"]
    assert set(staged["performance_applications"]) == set(original_members) | {
        row["id"] for row in report["benchmarks"]
    }
    assert len(report["benchmarks"]) == 10
    assert not report["admitted"] and not report["evaluated"] and not report["target_executed"]
    recipe = yaml.safe_load((output / "recipe.yaml").read_bytes())
    assert "public_joint_boundaries" not in recipe
    assert "requant_shift" not in recipe["datapath"]
    assert recipe["performance_source_programs"]["source_mode"] == "native_integer_v1"
    for row in report["benchmarks"]:
        path = Path(row["workload_root"]) / "loader.py"
        assert path.read_bytes() == Path(row["loader"]["path"]).read_bytes()
    # Real ordinary roster staging and its source pins feed the exact stock
    # preselection API; this test never constructs capture/source authority.
    from merlin_experiments.phase0 import capture_selection

    select = Mock(return_value={"status": "selected_not_executed"})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(capture_selection, "select", select)
        options = {
            key: tmp_path / key for key in ("m2m_root", "worker", "venv", "schemas_root", "run_dir", "output_dir")
        }
        selected_sha = owner.pin(output / "preparation.json")["sha256"]
        owner.select_capture(
            output / "preparation.json", report["benchmarks"][0]["id"], expected_sha256=selected_sha, **options
        )
        assert select.call_args.kwargs["dtype"] == "int8"
        assert select.call_args.kwargs["recipe"] is None
        assert select.call_args.kwargs["worker_options"] == {"source_mode": "native_integer_v1"}
        assert "runtime_reuse" not in select.call_args.kwargs
        assert "bwrap_binary" not in select.call_args.kwargs
        reuse = {"schema": "selected_runtime_owned_by_capture_selection"}
        owner.select_capture(
            output / "preparation.json", report["benchmarks"][0]["id"],
            expected_sha256=selected_sha, runtime_reuse=reuse,
            bwrap_binary=tmp_path / "bwrap", **options,
        )
        assert select.call_args.kwargs["runtime_reuse"] is reuse
        assert select.call_args.kwargs["bwrap_binary"] == tmp_path / "bwrap"
        path = Path(report["benchmarks"][0]["workload_root"]) / "loader.py"
        for changed in (path, output / "staged/staged-workloads.json", output / "preparation.json"):
            original_bytes = changed.read_bytes()
            changed.write_bytes(original_bytes + b"\n# changed\n")
            with pytest.raises(ValueError, match="changed"):
                owner.select_capture(
                    output / "preparation.json", report["benchmarks"][0]["id"], expected_sha256=selected_sha, **options
                )
            changed.write_bytes(original_bytes)
        assert select.call_count == 2

        def mutation_during_selection(**kwargs):
            path.write_text(path.read_text() + "# changed\n")
            return {"status": "selected_not_executed"}

        select.side_effect = mutation_during_selection
        with pytest.raises(ValueError, match="changed"):
            owner.select_capture(
                output / "preparation.json", report["benchmarks"][0]["id"], expected_sha256=selected_sha, **options
            )


def test_prospective_affine_copy_source_proof_preserves_public_program_bytes(owner, tmp_path):
    from merlin_experiments.phase0 import public_programs as P

    root = repo_root()
    options = dict(descriptor=root / "examples/gemmini/target/descriptor.yaml",
                   base_roster=root / "examples/gemmini/target/phase0-workloads.yaml",
                   software_spec=root / "examples/gemmini/target/software-spec.yaml")
    original = owner.prepare(root / "examples/gemmini/phase2/benchmarks/profile.yaml",
                             output=tmp_path / "default", **options)
    selected = owner.prepare(root / "examples/gemmini/phase2/benchmarks/profile.yaml",
                             output=tmp_path / "prospective", **options,
                             source_integer_proof="checked_native_integer_affine_copy_v2")
    policy = yaml.safe_load((tmp_path / "prospective/recipe.yaml").read_bytes())["performance_source_programs"]
    assert policy["schema"] == P.AFFINE_INTEGER_POLICY_SCHEMA
    assert policy["source_integer_proof"] == "checked_native_integer_affine_copy_v2"
    for before, after in zip(original["benchmarks"], selected["benchmarks"], strict=True):
        assert Path(before["loader"]["path"]).read_bytes() == Path(after["loader"]["path"]).read_bytes()
    with pytest.raises(ValueError, match="proof selection"):
        owner.prepare(root / "examples/gemmini/phase2/benchmarks/profile.yaml",
                      output=tmp_path / "invalid", **options, source_integer_proof="unknown")
