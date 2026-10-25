"""Fresh public releases and reviewed authoring reject held-out source metadata."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus import preparation
from merlin_experiments.corpus.publication import require_public_capsules
from merlin_experiments.phase1.corpus_inputs import require_reviewed_bundle
from merlin_experiments.spec import SpecError

_HELD_OUT = "validation_source"


def _capsule(root, fields):
    directory = root / "renamed"
    directory.mkdir(parents=True)
    (directory / "capsule.yaml").write_text(yaml.safe_dump({"name": "renamed", **fields}))
    (directory / "capsule.interface.mlir").write_text("module {}\n")
    return directory


@pytest.mark.parametrize(
    "fields",
    [
        {"name": "SY_model_" + _HELD_OUT},
        {"model": _HELD_OUT},
        {"model_form": {"model": _HELD_OUT}},
        {"loader": "loaders/" + _HELD_OUT + "/loader.py"},
        {"operation": {"attributes": {"model": _HELD_OUT}}},
        {"pytorch_ref": {"model": _HELD_OUT}},
        {"materialized_capture": {"workload_id": _HELD_OUT}},
        {"materialized_capture": {"source_workload_id": _HELD_OUT}},
        {"performance": {"form": {"members": [{"application": _HELD_OUT}]}}},
    ],
)
def test_renaming_does_not_publish_a_held_out_source(tmp_path, fields):
    public = tmp_path / "public"
    _capsule(public, fields)
    target = SimpleNamespace(workload_spec={"models": [_HELD_OUT]})
    with pytest.raises(SpecError, match="held-out validation source") as error:
        require_public_capsules(target, [public])
    assert str(tmp_path) not in str(error.value) and _HELD_OUT not in str(error.value)


def test_public_op_families_are_not_mistaken_for_held_out_programs(tmp_path):
    _capsule(tmp_path, {"model": "iteration_source", "operation": {"op": "attention"}})
    target = SimpleNamespace(workload_spec={"models": [_HELD_OUT]})
    require_public_capsules(target, [tmp_path])


@pytest.mark.parametrize("source", ["baseline", "generated"])
def test_explicit_historical_overlay_rejects_held_out_before_copying(tmp_path, monkeypatch, source):
    baseline = tmp_path / "baseline/isa"
    generated = tmp_path / "generated"
    baseline.mkdir(parents=True)
    generated.mkdir()
    _capsule(baseline if source == "baseline" else generated / "isa", {"model": _HELD_OUT})
    if source == "baseline":  # a release always carries generated members; only the overlay is held out
        _capsule(generated / "isa", {"operation": {"op": "matmul"}})
    target = SimpleNamespace(
        capsule_corpus=baseline,
        workload_spec={"models": [_HELD_OUT]},
        graded_roots=lambda: [baseline],
        hidden_roots=lambda: [],
        perf_roots=lambda: [],
        model_layer_roots=lambda: [],
    )
    monkeypatch.setattr(preparation, "copy_input", lambda *a, **kw: pytest.fail("copied a rejected source"))
    destination = tmp_path / "publication"
    with pytest.raises(SpecError, match="held-out validation source"):
        preparation.assemble(target, generated, destination, generated_only=False)
    assert not destination.exists()


def test_old_reviewed_bundle_cannot_reintroduce_a_public_validation_source(tmp_path):
    experiment = tmp_path / "release/experiment"
    public = experiment / "corpus/isa"
    _capsule(public, {"materialized_capture": {"source_workload_id": _HELD_OUT}})
    target = SimpleNamespace(
        path=experiment / "descriptor.yaml",
        capsule_corpus=public,
        workload_spec={"models": [_HELD_OUT]},
        graded_roots=lambda: [public],
    )
    manifest = experiment / "input_bundles/el4/input_bundle_manifest.yaml"
    with pytest.raises(SpecError, match="held-out validation source"):
        require_reviewed_bundle(target, manifest, {"bundle_id": "el4"})


@pytest.mark.parametrize("operation", ["not a mapping", ["matmul"]])
def test_malformed_source_metadata_refuses(tmp_path, operation):
    _capsule(tmp_path, {"operation": operation})
    with pytest.raises(SpecError, match="operation declaration is malformed"):
        require_public_capsules(SimpleNamespace(workload_spec={}), [Path(tmp_path)])


def _held_out_experiment():
    from merlin.common.paths import repo_root

    return SimpleNamespace(
        workload_spec={"models": [_HELD_OUT]},
        held_out_inputs=("merlin/contract/capsules/_model_layers",),
        source_root=repo_root(),
    )


def _layer_document(name="G_matmul_m3136k64n64_raw"):
    from merlin.common.paths import repo_root

    path = repo_root() / "merlin/contract/capsules/_model_layers" / name / "capsule.yaml"
    return yaml.safe_load(path.read_bytes())


def test_renaming_a_held_out_capsule_does_not_admit_it(tmp_path):
    document = _layer_document()
    for index, mutate in enumerate(
        (
            lambda d: d.update(name="renamed"),
            lambda d: d.update(name="renamed", source_reference="x", label="public"),
        )
    ):
        copy = dict(document)
        mutate(copy)
        directory = tmp_path / str(index) / "c"
        directory.mkdir(parents=True)
        (directory / "capsule.yaml").write_text(yaml.safe_dump(copy))
        with pytest.raises(SpecError, match="excluded historical input"):
            require_public_capsules(_held_out_experiment(), [tmp_path / str(index)])


def test_held_out_shape_is_rejected_under_another_dtype_and_numeric_policy(tmp_path):
    copy = dict(_layer_document(), name="renamed")
    copy["numeric_policy"] = {"compare": "tolerance", "dtype": "f32"}
    copy["inputs"] = [{**item, "dtype": "f32"} for item in copy["inputs"]]
    copy["operation"] = {**copy["operation"], "attributes": {**copy["operation"]["attributes"], "output_dtype": "f32"}}
    directory = tmp_path / "c"
    directory.mkdir()
    (directory / "capsule.yaml").write_text(yaml.safe_dump(copy))
    with pytest.raises(SpecError, match="excluded historical input"):
        require_public_capsules(_held_out_experiment(), [tmp_path])


def test_unidentifiable_model_derived_capsule_fails_closed(tmp_path):
    _capsule(tmp_path, {"source_role": "model_derived"})
    with pytest.raises(SpecError, match="cannot be identified"):
        require_public_capsules(_held_out_experiment(), [tmp_path])


@pytest.mark.parametrize("change", ["omit_epilogue", "reorder_inputs", "rename_operands"])
def test_held_out_shape_identity_uses_semantic_roles(tmp_path, change):
    document = _layer_document()
    document["name"] = "unrelated_public_name"
    if change == "omit_epilogue":
        del document["operation"]["attributes"]["epilogue"]
    elif change == "reorder_inputs":
        document["inputs"].reverse()
    else:
        for index, operand in enumerate(document["inputs"]):
            old = operand["name"]
            operand["name"] = f"renamed_{index}"
            for key, value in document["operation"]["attributes"].items():
                if value == old:
                    document["operation"]["attributes"][key] = operand["name"]
    _capsule(tmp_path, document)
    with pytest.raises(SpecError, match="excluded historical input"):
        require_public_capsules(_held_out_experiment(), [tmp_path])
