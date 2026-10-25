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


@pytest.mark.parametrize(
    "alias",
    [
        "linear",
        "linear_transposed",
        "mm",
        "bmm",
        "batch_matmul",
        "addmm",
        "conv2d",
        "aten.mm.default",
        "aten.bmm.default",
        "aten.linear.default",
        "aten.addmm.default",
    ],
)
@pytest.mark.parametrize("rectangular", [False, True])
def test_held_out_contraction_aliases_cannot_be_published(tmp_path, alias, rectangular):
    import copy

    document = copy.deepcopy(_layer_document())
    experiment = _held_out_experiment()
    if rectangular:
        # A separate exclusion proves orientation without relying on square weights.
        for operand in document["inputs"]:
            operand["shape"] = [7, 5] if operand["role"] == "input" else [5, 11]
        excluded = tmp_path / "excluded"
        _capsule(excluded, document)
        experiment = SimpleNamespace(source_root=tmp_path, held_out_inputs=("excluded",), workload_spec={})
    candidate = copy.deepcopy(document)
    candidate["name"] = "renamed"
    candidate["operation"]["op"] = "linear" if alias == "linear_transposed" else alias
    operands = {item["role"]: item for item in candidate["inputs"]}
    if alias in {"linear_transposed", "aten.linear.default"}:
        operands["weight"]["shape"].reverse()
    elif alias in {"bmm", "batch_matmul", "aten.bmm.default"}:
        for item in candidate["inputs"]:
            item["shape"].insert(0, 1)
    elif alias in {"addmm", "aten.addmm.default"}:
        candidate["operation"]["attributes"].update(alpha=1, beta=0)
    elif alias == "conv2d":
        m, k = operands["input"]["shape"]
        operands["input"]["shape"] = [1, 1, m, k]
        candidate["operation"]["attributes"].update(
            ifm=operands["input"]["name"],
            kh=1,
            kw=1,
            ci=k,
            layout="nhwc",
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
        )
    public = tmp_path / "public"
    _capsule(public, candidate)
    with pytest.raises(SpecError, match="excluded historical input"):
        require_public_capsules(experiment, [public])


def test_distinct_contraction_shapes_remain_public(tmp_path):
    candidate = _layer_document()
    candidate["inputs"][1]["shape"][0] += 1
    candidate["operation"]["op"] = "linear"
    _capsule(tmp_path, candidate)
    require_public_capsules(_held_out_experiment(), [tmp_path])


@pytest.mark.parametrize("role", ["hand-authored", "model_derived"])
def test_unresolved_alias_shapes_fail_closed_independent_of_source_label(tmp_path, role):
    candidate = _layer_document()
    candidate["source_role"] = role
    candidate["operation"]["op"] = "linear"
    candidate["inputs"][0]["shape"] = [61, 67]
    _capsule(tmp_path, candidate)
    with pytest.raises(SpecError, match="cannot be identified"):
        require_public_capsules(_held_out_experiment(), [tmp_path])


@pytest.mark.parametrize("op", ["fused_matmul_bias", "gemm_bias_relu", "custom_contraction"])
def test_unlisted_operation_spellings_cannot_publish_a_held_out_shape(tmp_path, op):
    candidate = _layer_document()
    candidate["name"] = "renamed"
    candidate["operation"]["op"] = op
    _capsule(tmp_path, candidate)
    with pytest.raises(SpecError, match="excluded historical input"):
        require_public_capsules(_held_out_experiment(), [tmp_path])


@pytest.mark.parametrize("change", ["transpose_weight", "batch_one", "rename_roles"])
def test_unlisted_spelling_with_reshaped_operands_cannot_publish_a_held_out_shape(tmp_path, change):
    candidate = _layer_document()
    candidate["name"] = "renamed"
    candidate["operation"]["op"] = "fused_matmul_bias"
    for item in candidate["inputs"]:
        if change == "transpose_weight" and item["role"] == "weight":
            item["shape"].reverse()
        elif change == "batch_one":
            item["shape"].insert(0, 1)
        elif change == "rename_roles":
            item["role"] = {"input": "lhs", "weight": "rhs"}.get(item["role"], item["role"])
    _capsule(tmp_path, candidate)
    with pytest.raises(SpecError, match="excluded historical input"):
        require_public_capsules(_held_out_experiment(), [tmp_path])


def test_extra_operands_cannot_hide_a_held_out_core(tmp_path):
    candidate = _layer_document()
    candidate["name"] = "renamed"
    candidate["operation"]["op"] = "addmm"
    candidate["operation"].setdefault("attributes", {}).update(alpha=1, beta=1)
    m = next(item["shape"][0] for item in candidate["inputs"] if item["role"] == "input")
    n = next(item["shape"][-1] for item in candidate["inputs"] if item["role"] == "weight")
    candidate["inputs"].append({"name": "full_bias", "role": "bias", "shape": [m, n], "dtype": "i32"})
    _capsule(tmp_path, candidate)
    with pytest.raises(SpecError, match="excluded historical input"):
        require_public_capsules(_held_out_experiment(), [tmp_path])
