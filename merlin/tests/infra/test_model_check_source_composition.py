"""Explicit whole-model composition must retain source and build separation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_model_check_full_build import _admission_service, built  # noqa: F401

from merlin.common import mlir_query as mq
from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.device_build import DeviceRouting
from merlin.llvmlower.head_source_route import HeadBatchSourceRoute
from merlin.llvmlower.sibling_source_route import SiblingSourceRoute, _sha, _text
from merlin.model_check import full_build as FB


def _route(stage, *, heads=False):
    identity = _sha(_text(mq.parse((stage / "model.mlir").read_text())))
    effects = BorrowedBufferEffects("test implementation premises only", True, True, True)
    return HeadBatchSourceRoute(identity, effects) if heads else SiblingSourceRoute(identity, effects, partial=True)


@pytest.mark.parametrize("heads", [False, True])
def test_current_source_route_reaches_ordinary_complete_group_routing(tmp_path, monkeypatch, heads):
    from merlin.compile import route_before_build

    (tmp_path / "model.mlir").write_text("module { func.func @forward() { return } }")
    original = DeviceRouting(
        "neutral", str(tmp_path / "candidate"), "int8", "i32", granularity="group", require_complete_groups=True
    )
    monkeypatch.setattr(route_before_build, "plan_before_build", lambda *a, **kw: {"device_routing": original})
    selected = _route(tmp_path, heads=heads)
    result = FB._routing(
        "neutral",
        tmp_path,
        tmp_path / "candidate",
        "forward",
        SimpleNamespace(datapath="int8"),
        None,
        source_route=selected,
    )
    assert result.sibling_source is selected
    assert result.require_complete_groups and original.sibling_source is None
    (tmp_path / "model.mlir").write_text("module { func.func @different() { return } }")
    with pytest.raises(FB.FullBuildError, match="differs from the current captured source"):
        FB._routing(
            "neutral",
            tmp_path,
            tmp_path / "candidate",
            "forward",
            SimpleNamespace(datapath="int8"),
            None,
            source_route=selected,
        )


@pytest.mark.parametrize("route", [object(), {"kind": "independent_heads"}])
def test_untyped_composition_refuses_before_output_creation(tmp_path, route):
    output = tmp_path / "never-created"
    with pytest.raises(FB.FullBuildError, match="explicit typed route"):
        FB.build(
            tmp_path,
            target="neutral",
            label="forward",
            submission=tmp_path,
            submission_sha256="a" * 64,
            settings=None,
            directory=output,
            source_route=route,
        )
    assert not output.exists()


def test_host_only_composition_refuses_before_output_creation(tmp_path):
    (tmp_path / "model.mlir").write_text("module { func.func @forward() { return } }")
    with pytest.raises(FB.FullBuildError, match="device offload"):
        FB.build(
            tmp_path,
            target="neutral",
            label="forward",
            submission=tmp_path,
            submission_sha256="a" * 64,
            settings=None,
            directory=tmp_path / "never-created",
            offload=False,
            source_route=_route(tmp_path),
        )
    assert not (tmp_path / "never-created").exists()


def test_composition_variants_cannot_reuse_baseline_or_each_other(request, tmp_path, monkeypatch):
    _, stage, directory, _, calls = request.getfixturevalue("built")
    settings = FB.BuildSettings("board", tmp_path / "boards", tmp_path / "host.dts", "scalar", 1, tmp_path / "host")
    routed = DeviceRouting("t", str(tmp_path / "submission"), "int8", "i32", granularity="group")
    monkeypatch.setattr(FB, "_routing", lambda *a, source_route=None: replace(routed, sibling_source=source_route))

    def build(route=None):
        return FB.build(
            stage,
            target="t",
            label="model",
            submission=tmp_path / "submission",
            submission_sha256="a" * 64,
            settings=settings,
            directory=directory,
            admission=_admission_service(),
            source_route=route,
        )

    baseline = build()
    assert "source_route" not in baseline["key"] and build()["reused"]
    sibling = _route(stage)
    grouped = build(sibling)
    assert not grouped["reused"] and grouped["key"]["source_route"] == sibling.record()
    assert calls[-1]["device"].sibling_source is sibling and build(sibling)["reused"]
    heads = _route(stage, heads=True)
    assert not build(heads)["reused"]
    assert not build()["reused"] and len(calls) == 4
