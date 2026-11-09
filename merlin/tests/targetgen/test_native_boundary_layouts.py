"""Physical boundary documents bind source/request identity and actual publication."""

import json
from types import SimpleNamespace

import pytest

from merlin.compiler.build.native import publish_layout
from merlin.compiler.frontend.layouts import bind_physical_layouts, check_layout_artifact
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType


@pytest.fixture
def selected(tmp_path):
    dtype = TensorType((2,), "i32", "exact-i32")
    request = KernelRequest(
        nodes=(SemanticNode("x", "input", (), dtype, effect="input"), SemanticNode("y", "identity", ("x",), dtype)),
        outputs=("y",),
        output_storages=("external",),
        input_storages=(("x", "external"),),
        source_identity="source-fixture",
        target_identity="target-fixture",
    )
    row = {
        "schema": "merlin.physical_boundary_layouts.v2",
        "request_digest": request.digest(),
        "source_identity": request.source_identity,
        "target_identity": request.target_identity,
        "physical_extents": {"x": 2},
        "boundary_layouts": {"x": "logical"},
        "output_layouts": ["logical"],
    }
    path = tmp_path / "layouts.json"
    path.write_text(json.dumps(row))
    return request, row, SimpleNamespace(physical_layouts=path, request=None)


def test_layouts_bind_exact_request_and_tracked_published_bytes(selected, tmp_path):
    request, row, args = selected
    bound, document = bind_physical_layouts(args, request, {"precision": "explicit"})
    assert bound != request and bound.nodes == request.nodes and bound.outputs == request.outputs
    assert bound.source_identity == request.source_identity
    assert bound.physical_extents == (("x", 2),) and bound.boundary_layouts == (("x", "logical"),)
    assert request.digest() == row["request_digest"] and bound.digest() != request.digest()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = publish_layout(artifact, {"stage_artifacts": {}}, document)
    assert json.loads((artifact / "manifest.json").read_text()) == manifest
    check_layout_artifact(artifact, manifest, document)
    with pytest.raises(ValueError, match="unrequested"):
        check_layout_artifact(artifact, manifest, None)
    with pytest.raises(ValueError, match="cannot track"):
        publish_layout(artifact, manifest, document)
    tracked = {"stage_artifacts": {"physical-boundary-layouts.json": "0" * 64}}
    with pytest.raises(ValueError, match="differs"):
        check_layout_artifact(artifact, tracked, document)
    (artifact / "physical-boundary-layouts.json").write_bytes(document + b" ")
    with pytest.raises(ValueError, match="differs"):
        check_layout_artifact(artifact, manifest, document)


def test_unselected_and_legacy_layouts_keep_original_contract(selected, tmp_path):
    request, row, args = selected
    unchanged, document = bind_physical_layouts(SimpleNamespace(), request, None)
    assert unchanged is request and document is None
    check_layout_artifact(tmp_path, {}, None)
    row["schema"] = "merlin.physical_boundary_layouts.v1"
    del row["request_digest"]
    args.physical_layouts.write_text(json.dumps(row))
    bound, _ = bind_physical_layouts(args, request, None)
    assert bound.nodes == request.nodes
    with pytest.raises(ValueError, match="exact-request v2"):
        bind_physical_layouts(args, request, {})
    args.request = "opaque.json"
    with pytest.raises(ValueError, match="parsed Linalg"):
        bind_physical_layouts(args, request, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_digest", "changed"),
        ("source_identity", "changed"),
        ("target_identity", "changed"),
        ("physical_extents", {"x": 0}),
        ("physical_extents", {"x": True}),
        ("boundary_layouts", {"absent": "logical"}),
        ("output_layouts", []),
        ("schema", "unknown"),
        ("unknown", 1),
    ],
)
def test_layout_source_schema_and_complete_boundary_substitutions_refuse(selected, field, value):
    request, row, args = selected
    row[field] = value
    args.physical_layouts.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        bind_physical_layouts(args, request, {})


@pytest.mark.parametrize("command", ["native-compile", "native-verify"])
def test_application_cannot_silently_ignore_selected_layouts(command, tmp_path, capsys):
    from merlin.targetgen.cli import main

    arguments = [
        command,
        "--engine",
        "merlin_native",
        "--support",
        "fixture",
        "--application",
        "--snapshot",
        str(tmp_path / "snapshot"),
        "--compile-inputs",
        str(tmp_path / "inputs"),
        "--physical-layouts",
        str(tmp_path / "layouts.json"),
        "--mode",
        "diagnostic",
        "--out",
        str(tmp_path / "out.json"),
    ]
    if command == "native-compile":
        arguments += ["--abi", str(tmp_path / "abi.json"), "--target-source", str(tmp_path / "target")]
    else:
        arguments += ["--artifact", str(tmp_path / "artifact")]
    assert main(arguments) == 2
    report = (
        json.loads(capsys.readouterr().out)
        if command == "native-compile"
        else json.loads((tmp_path / "out.json").read_text())
    )
    assert report["status"] == "compile_error"
    assert "physical boundary layouts cannot be combined with --application" in report["reason"]
