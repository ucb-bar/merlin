"""Opening a native snapshot binds the installed compiler's actual source bytes."""

import hashlib
import json
import shutil
from dataclasses import replace

import pytest

from merlin.common.paths import module_source_path
from merlin.compiler.semantic import target_binding as snapshots


def test_snapshot_opens_with_current_sources_and_refuses_source_drift(tmp_path, monkeypatch):
    _check_prequantized_bias_preparation()
    with monkeypatch.context() as isolated:
        _check_snapshot_source_drift(tmp_path / "source-drift", isolated)
    _check_nested_program_leaves_preserve_authenticated_parent_metadata(tmp_path / "nested-valid")
    for mutation in ("binary", "source", "extra_directory", "symlink_ancestor", "overlap"):
        _check_nested_program_publication_refuses_changed_or_unaccounted_members(
            tmp_path / ("nested-" + mutation), mutation
        )


def _check_snapshot_source_drift(tmp_path, monkeypatch):
    # Copy only the declared source closure, as an installed package would ship it.
    package = module_source_path("merlin").parent
    sources = snapshots._compiler_sources()
    for name in sources:
        destination = tmp_path / "installed" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, destination)
    monkeypatch.setattr(snapshots, "__file__", str(tmp_path / "installed/compiler/semantic/target_binding.py"))
    profile = snapshots.NativeTargetProfile(
        "fixture",
        (
            snapshots.InstructionDescriptor(
                "add",
                "f32_add",
                ("ram", "ram"),
                "ram",
                "f32",
                "exact",
                (1,),
                input_dtypes=("f32", "f32"),
                input_numerical_policies=("exact", "exact"),
            ),
        ),
        (snapshots.StorageBank("ram", "memory", 1024, "bytes"),),
    )
    root = tmp_path / "snapshot"
    (root / "bin").mkdir(parents=True)
    (root / "profile.json").write_text(json.dumps(profile.record()))
    # Opening validates bridge bytes; execution/building the Rust bridge is a separate boundary.
    (root / "bin/merlin-egg-bridge").write_bytes(b"test bridge identity")
    manifest = {
        "schema": "merlin.native_target_snapshot.v6",
        "status": "selection_only",
        "target_identity": profile.target_identity,
        "profile_sha256": hashlib.sha256((root / "profile.json").read_bytes()).hexdigest(),
        "bridge_sha256": hashlib.sha256((root / "bin/merlin-egg-bridge").read_bytes()).hexdigest(),
        "compiler_sources": sources,
        "runtime_dependencies": snapshots._runtime_dependencies(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    assert snapshots.open_native_snapshot(root).profile == profile

    # Board memory bindings and numerical/input implementations remain bound.
    for name in (
        "compiler/semantic/search/allocation.py",
        "compiler/frontend/input_types.py",
        "compiler/build/memory_budget.py",
        "compiler/build/memory_placement.py",
        "target/soc.py",
        "target/system.py",
        "compiler/quantization/contraction/scaled_fp8.py",
        "compiler/semantic/source_preparation.py",
    ):
        implementation = tmp_path / "installed" / name
        original = implementation.read_bytes()
        implementation.write_bytes(original + b"\n# changed implementation\n")
        with pytest.raises(ValueError, match="sources differ"):
            snapshots.open_native_snapshot(root)
        implementation.unlink()
        with pytest.raises(ValueError, match="sources are unavailable"):
            snapshots.open_native_snapshot(root)
        implementation.write_bytes(original)
    assert snapshots.open_native_snapshot(root).profile == profile

def _nested_publication(root):
    leaf = root / "segments/region_000/segments/body_000"
    leaf.mkdir(parents=True)
    (leaf / "program.bin").write_bytes(b"native program identity")
    detail = {
        "engine": "fixture",
        "request_digest": "leaf-request",
        "target_identity": "fixture-target",
        "artifact_kind": "program",
        "binary_sha256": hashlib.sha256((leaf / "program.bin").read_bytes()).hexdigest(),
    }
    (leaf / "manifest.json").write_text(json.dumps(detail))
    source = root / "segments/region_000/source.mlir"
    source.write_text("module {}")
    manifest = {
        "engine": "fixture",
        "request_digest": "application-request",
        "target_identity": "fixture-target",
        "artifact_kind": "program_set",
        "stage_artifacts": {
            source.relative_to(root).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "segments": [{
            "index": 0,
            "path": leaf.relative_to(root).as_posix(),
            "request_digest": detail["request_digest"],
            "binary_sha256": detail["binary_sha256"],
            "manifest_sha256": hashlib.sha256((leaf / "manifest.json").read_bytes()).hexdigest(),
        }],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _verify_publication(root, manifest):
    snapshots.verify_native_publication(
        root, manifest, engine="fixture", request_digest="application-request", target_identity="fixture-target"
    )


def _check_nested_program_leaves_preserve_authenticated_parent_metadata(tmp_path):
    manifest = _nested_publication(tmp_path)
    _verify_publication(tmp_path, manifest)


def _check_nested_program_publication_refuses_changed_or_unaccounted_members(tmp_path, mutation):
    manifest = _nested_publication(tmp_path)
    region = tmp_path / "segments/region_000"
    if mutation == "binary":
        (region / "segments/body_000/program.bin").write_bytes(b"different native program")
    elif mutation == "source":
        (region / "source.mlir").write_text("module { changed }")
    elif mutation == "extra_directory":
        (region / "unaccounted").mkdir()
    elif mutation == "symlink_ancestor":
        actual = tmp_path / "actual-region"
        region.rename(actual)
        region.symlink_to(actual, target_is_directory=True)
    elif mutation == "overlap":
        manifest["segments"].append({**manifest["segments"][0], "index": 1, "path": "segments/region_000"})
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        _verify_publication(tmp_path, manifest)


def _check_prequantized_bias_preparation():
    from merlin.compiler.semantic.model import (
        IndexMap,
        KernelRequest,
        SemanticNode,
        TensorType,
        pointwise_identity_maps,
    )
    from merlin.compiler.semantic.source_preparation import (
        is_prequantized_bias_add,
        preparation_counts,
        preparation_filename,
        prepare_source,
        split_prequantized_bias,
        verify_prequantized_bias,
        verify_source_preparation,
    )

    output = TensorType((2, 3, 7), "bf16", "raw-bit-pattern")
    types = (
        TensorType((2, 3, 5), "i8", "fp8-code-bits"),
        TensorType((), "i32", "signed-exponent"),
        TensorType((5, 7), "i8", "fp8-code-bits"),
        TensorType((7,), "i32", "signed-exponent"),
        replace(output, shape=(7,)),
    )
    inputs = tuple(SemanticNode(f"arg{i}", "input", (), type_, effect="input") for i, type_ in enumerate(types))
    contraction = SemanticNode(
        "result",
        "fixture_contraction",
        tuple(node.id for node in inputs),
        output,
        (("bias_order", "after_output_cast"), ("semantic_kind", "prequantized_contraction")),
    )
    source = KernelRequest(
        (*inputs, contraction),
        ("result",),
        ("ram",),
        tuple((node.id, "ram") for node in inputs),
        "fixture",
    )
    prepared, receipt = prepare_source(source)
    assert prepared.outputs == source.outputs
    assert prepared.node("result.readout") == replace(contraction, id="result.readout", inputs=contraction.inputs[:4])
    broadcast, addition = prepared.node("result.bias"), prepared.node("result")
    assert broadcast.op == "gather" and broadcast.inputs == ("arg4",)
    assert broadcast.index_maps[0].apply((1, 2, 6)) == (6,)
    assert broadcast.index_maps[1] == pointwise_identity_maps(3, 1)[-1]
    assert addition.op == "bf16_add" and addition.inputs == ("result.readout", "result.bias")
    assert addition.attrs == (("numeric_semantics", "ieee754-bf16-add-rne-v1"),)
    assert is_prequantized_bias_add(prepared, addition)
    assert verify_source_preparation(prepared, json.loads(json.dumps(receipt))) == source
    assert preparation_filename(receipt) == "source-preparation.json"
    assert preparation_counts(receipt)["source_prequantized_bias_splits"] == 1

    # Resealing a changed gather cannot authenticate a different broadcast.
    changed_bias = replace(broadcast, index_maps=(IndexMap(3, ((0, 0, 0),), (0,)), broadcast.index_maps[1]))
    changed = replace(
        prepared, nodes=tuple(changed_bias if node.id == broadcast.id else node for node in prepared.nodes)
    )
    forged = {**receipt["bias"], "after_digest": changed.digest()}
    with pytest.raises(ValueError, match="exact replay"):
        verify_prequantized_bias(changed, forged)
    assert not is_prequantized_bias_add(changed, changed.node("result"))
    forged = json.loads(json.dumps(receipt))
    forged["bias"]["changes"][0]["original"]["attrs"]["bias_order"] = "before_output_cast"
    with pytest.raises(ValueError, match="original changed"):
        verify_source_preparation(prepared, forged)

    wrong_bias = replace(inputs[-1], type=replace(inputs[-1].type, shape=(3,)))
    incompatible = replace(source, nodes=(*inputs[:-1], wrong_bias, contraction))
    with pytest.raises(ValueError, match="column vector after the output cast"):
        prepare_source(incompatible)
    f32 = replace(source, nodes=(*inputs, replace(contraction, type=replace(output, dtype="f32"))))
    unchanged, no_split = split_prequantized_bias(f32)
    assert unchanged == f32 and not no_split["changes"]
