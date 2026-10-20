"""Original choices precede prepared IR; receipts cannot manufacture authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_source_optimization_build import SOURCE, completed, selected

from merlin.llvmlower.source_optimization_policy import SourceOptimizationPolicy
from merlin.llvmlower.source_optimization_request import GATHER_PASS, PACKING_PASS
from merlin.runtime.source_optimization_build import SourceOptimizationBuild


def policy(**changes):
    values = dict(
        producer_packing=False,
        padding_gather=False,
        maximum_added_operations=4096,
        permitted_passes=(PACKING_PASS, GATHER_PASS),
        original_maximum_added_operations=4096,
    )
    return SourceOptimizationPolicy(**(values | changes))


@pytest.mark.parametrize(
    "change",
    [
        {"producer_packing": 1},
        {"padding_gather": "yes"},
        {"maximum_added_operations": True},
        {"maximum_added_operations": -1},
        {"maximum_added_operations": 4097},
        {"original_maximum_added_operations": 1 << 21},
        {"permitted_passes": [PACKING_PASS]},
        {"permitted_passes": (PACKING_PASS, PACKING_PASS)},
        {"permitted_passes": ("unrelated",)},
        {"producer_packing": True, "permitted_passes": ()},
        {"schema": "unreviewed"},
    ],
)
def test_invalid_policy_or_missing_original_permission_refuses_before_source(change):
    with pytest.raises(ValueError):
        policy(**change)


def test_policy_is_not_restored_from_its_record_or_accepted_as_a_static_request(tmp_path):
    original = policy()
    with pytest.raises(ValueError, match="typed selection"):
        SourceOptimizationBuild(original.record())
    from merlin.llvmlower.lower import lower_model

    with pytest.raises(ValueError, match="typed selection"):
        lower_model(SOURCE, tmp_path / "no-implicit-binding", targets=(), source_optimizations=original)
    assert not (tmp_path / "no-implicit-binding").exists()


@pytest.mark.parametrize("mutation", ["choice", "permission", "budget", "ceiling"])
def test_original_policy_mutation_before_current_ir_refuses(tmp_path, mutation):
    original = policy()
    build = SourceOptimizationBuild(original)
    field, value = {
        "choice": ("producer_packing", True),
        "permission": ("permitted_passes", ()),
        "budget": ("maximum_added_operations", 0),
        "ceiling": ("original_maximum_added_operations", 8192),
    }[mutation]
    object.__setattr__(original, field, value)
    with pytest.raises(ValueError, match="original.*changed"):
        build.bind(tmp_path / "future.mlir")


def test_policy_binds_once_static_request_never_refreshes_and_fresh_invocations_are_independent(tmp_path):
    original = policy()
    first = SourceOptimizationBuild(original)
    path = tmp_path / "prepared.mlir"
    assert first.options()["source_optimizations"] is original
    with pytest.raises(ValueError, match="no actual prepared"):
        first.identity()
    path.write_text(SOURCE)
    first.bind(path)
    bound = first.options()["source_optimizations"]
    assert bound.request.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already bound"):
        first.bind(path)
    path.write_text(SOURCE + "// new preparation epoch\n")
    with pytest.raises(ValueError, match="actual prepared source changed"):
        first.bind(path)
    with pytest.raises(ValueError, match="current source"):
        SourceOptimizationBuild(selected()).bind(path)
    second = SourceOptimizationBuild(original)
    second.bind(path)
    assert second.options()["source_optimizations"].request.source_sha256 != bound.request.source_sha256
    assert original.record() == policy().record()


def test_policy_never_silently_retargets_a_normalized_text_hash(tmp_path):
    original = policy()
    prepared = tmp_path / "prepared.mlir"
    prepared.write_bytes(b"module {}\r\n")
    with pytest.raises(ValueError, match="exact prepared source bytes"):
        SourceOptimizationBuild(original).bind(prepared)


def test_real_completed_policy_receipt_joins_current_bound_selection_and_detaches_data(tmp_path):
    original = policy()
    producer, recipe, elf = completed(tmp_path / "build", original)
    record = producer.record()
    assert record["schema"] == "merlin.source_optimization_build.v2"
    assert record["policy"] == original.record()
    from merlin.llvmlower import source_optimization_policy

    owner = Path(source_optimization_policy.__file__).resolve()
    assert any(
        row["path"] == str(owner) and row["sha256"] == hashlib.sha256(owner.read_bytes()).hexdigest()
        for row in record["library"]
    )
    consumer = SourceOptimizationBuild(original)
    consumer.accept(record, build_dir=recipe.path.parent, executable=elf)
    assert consumer.record() == record
    record["policy"]["original_permitted_passes"].clear()
    assert consumer.record() == producer.record()
    with pytest.raises(ValueError, match="changed original policy"):
        SourceOptimizationBuild(original).accept(record, build_dir=recipe.path.parent, executable=elf)


@pytest.mark.parametrize("change", ["choice", "permissions", "ceiling", "source_sha", "policy", "schema"])
def test_returned_receipt_cannot_change_original_choices_or_authority(tmp_path, change):
    original = policy()
    guard, recipe, elf = completed(tmp_path / "build", original)
    record = guard.record()
    if change == "choice":
        record["selection"]["request"]["producer_packing"] = True
    elif change == "permissions":
        record["selection"]["original_permitted_passes"].clear()
    elif change == "ceiling":
        record["selection"]["original_maximum_added_operations"] = 8192
    elif change == "source_sha":
        record["selection"]["request"]["source_sha256"] = "0" * 64
    elif change == "policy":
        record["policy"]["maximum_added_operations"] = 0
    else:
        record["schema"] = "merlin.source_optimization_build.v1"
    with pytest.raises(ValueError):
        SourceOptimizationBuild(original).accept(record, build_dir=recipe.path.parent, executable=elf)


@pytest.mark.parametrize("change", ["source", "report", "executable", "policy"])
def test_late_build_or_execution_mutation_refuses(tmp_path, change):
    original = policy()
    guard, _recipe, elf = completed(tmp_path / "build", original)
    if change == "policy":
        object.__setattr__(original, "maximum_added_operations", 0)
    else:
        record = guard.record()
        target = {"source": Path(record["input"]["path"]), "report": Path(record["report"]["path"]), "executable": elf}[
            change
        ]
        target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        guard.record()


def test_foreign_actual_recipe_and_output_are_not_the_requested_build(tmp_path):
    original = policy()
    first, recipe, elf = completed(tmp_path / "first", original)
    other, _, other_elf = completed(tmp_path / "other", original)
    with pytest.raises(ValueError, match="foreign consumed report"):
        SourceOptimizationBuild(original).accept(other.record(), build_dir=recipe.path.parent, executable=elf)
    with pytest.raises(ValueError):
        SourceOptimizationBuild(original).accept(first.record(), build_dir=recipe.path.parent, executable=other_elf)


@pytest.mark.parametrize("mutation", [None, "prepare", "returned", "execution"])
def test_highlevel_uses_original_pending_policy_and_real_completed_build(tmp_path, monkeypatch, mutation):
    import runpy

    from merlin.common.paths import merlin_dir
    from merlin.compile import baremetal_model as BM

    helpers = runpy.run_path(str(merlin_dir() / "tests/infra/test_baremetal_saved_model.py"))
    inputs, out = helpers["_fixture"](tmp_path, monkeypatch)
    original = policy()

    def build(bundle, work, **kwargs):
        assert kwargs["source_optimizations"] is original
        info = helpers["_fake_build"](bundle, work, **kwargs)
        info["elf"].unlink()
        work.rmdir()
        guard, _, actual = completed(work, original)
        record = guard.record()
        if mutation == "prepare":
            object.__setattr__(original, "maximum_added_operations", 0)
        elif mutation == "returned":
            record["selection"]["original_maximum_added_operations"] = 8192
        return {**info, "elf": actual, "source_optimizations": record}

    def execute(*args, **kwargs):
        object.__setattr__(original, "maximum_added_operations", 0)
        return {"console": helpers["_console"]()}

    monkeypatch.setattr(BM.spike_model, "build", build)
    monkeypatch.setattr(BM, "_spike_extension_backend", lambda target: None)
    monkeypatch.setattr(BM.spike_model, "run", execute)
    options = {"run": "spike", "reference_file": "golden.npy"} if mutation == "execution" else {"run": "none"}
    if mutation:
        with pytest.raises(ValueError, match="changed"):
            BM.compile_saved_model(**inputs, output=out / "policy", source_optimizations=original, **options)
        assert json.loads((out / "policy/baremetal_model.json").read_text())["status"] == "failed"
    else:
        result = BM.compile_saved_model(**inputs, output=out / "policy", source_optimizations=original, **options)
        assert result["inputs"]["source_optimizations"] == original.record()
        assert result["output"]["source_optimizations"]["schema"] == "merlin.source_optimization_build.v2"
        assert result["output"]["source_optimizations"]["policy"] == original.record()
