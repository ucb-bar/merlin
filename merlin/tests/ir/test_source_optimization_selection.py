"""Bounded source choices reach ordinary compilation without changing legality."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

from merlin.llvmlower.source_optimization_request import GATHER_PASS, PACKING_PASS, SourceOptimizationRequest
from merlin.llvmlower.source_optimization_selection import SourceCompilationOptimization, SourceOptimizationSelection


def selection(source, *, packing=True, gather=True, maximum=4096, permitted=(PACKING_PASS, GATHER_PASS)):
    request = SourceOptimizationRequest(hashlib.sha256(source.encode()).hexdigest(), packing, gather, maximum)
    return SourceOptimizationSelection(request, permitted, 4096)


def test_binding_and_explicit_options_do_not_mutate_environment(monkeypatch):
    monkeypatch.setenv("MERLIN_PASSES", "int-softmax-table,-automatic-host-gathers")
    before = os.environ.copy()
    prepared = SourceCompilationOptimization("module {}", selection("module {}"))
    assert prepared.options(GATHER_PASS) == {"enabled": True, "max_added_operations": 4096}
    assert prepared.options(PACKING_PASS) == {"enabled": True, "max_added_operations": 4096}
    with pytest.raises(ValueError, match="unrelated"):
        prepared.options("int-softmax-table")
    assert dict(os.environ) == before


@pytest.mark.parametrize("mutation", ["permission", "budget", "choice", "library"])
def test_original_permissions_choices_and_implementation_stay_current(monkeypatch, mutation):
    original = selection("module {}")
    prepared = SourceCompilationOptimization("module {}", original)
    if mutation == "permission":
        object.__setattr__(original, "permitted_passes", ())
    elif mutation == "budget":
        object.__setattr__(original, "maximum_added_operations", 0)
    elif mutation == "choice":
        object.__setattr__(original.request, "padding_gather", False)
    else:
        monkeypatch.setattr(prepared, "_library_sources", lambda: {"changed": "0" * 64})
    with pytest.raises(ValueError, match="permission|ceiling|changed"):
        prepared.revalidate()


def test_stale_request_and_permission_failure_refuse_before_lowering(tmp_path):
    from merlin.llvmlower.lower import lower_model

    for source, selected in (
        ("module {}\n", selection("module {}")),
        ("module {}", selection("module {}", permitted=())),
        ("module {}", {}),
    ):
        with pytest.raises(ValueError, match="current source|permission|typed selection"):
            lower_model(source, tmp_path / "must-not-start", targets=(), source_optimizations=selected)
        assert not (tmp_path / "must-not-start").exists()


@pytest.mark.parametrize("when", ["graph", "object"])
def test_graph_and_late_object_callbacks_cannot_change_original_choices(tmp_path, monkeypatch, when):
    from test_padding_gather_connected import source

    from merlin.llvmlower import graph_optimizations, lower

    original = selection(source())
    reached = []
    if when == "graph":

        def graph(text, _policy):
            reached.append("graph")
            object.__setattr__(original.request, "padding_gather", False)
            return text, {}

        monkeypatch.setattr(graph_optimizations, "optimize_host_graph", graph)
        options = {"graph_optimizations": object()}
    else:

        def object_code(_llvm, path):
            reached.append("object")
            object.__setattr__(original, "maximum_added_operations", 4097)
            return path

        monkeypatch.setattr(lower, "build_host_shared", object_code)
        options = {}
    with pytest.raises(ValueError, match="selection.*changed"):
        lower.lower_model(source(), tmp_path / when, source_optimizations=original, **options)
    assert reached == [when]


def test_earlier_source_rewrite_uses_fresh_consumer_ir_not_root_digest(tmp_path, monkeypatch):
    from test_padding_gather_connected import source

    from merlin.llvmlower import graph_optimizations, lower

    original = source()
    derived = original + "\n// original graph stage retained an equivalent source\n"
    monkeypatch.setattr(graph_optimizations, "optimize_host_graph", lambda text, policy: (derived, {}))
    compiled = lower.lower_model(
        original,
        tmp_path / "derived",
        targets=(),
        graph_optimizations=object(),
        source_optimizations=selection(original),
    )
    report = compiled.stats["source_optimizations"]
    assert report["selection"]["request"]["source_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    actual = report["consumers"][GATHER_PASS]
    assert actual["source_file_sha256"] == hashlib.sha256(derived.encode()).hexdigest()
    # The explicit legacy graph route conservatively disables serial source
    # rewrites, independently of the request. It must report no realization.
    assert actual["rewrites"] == 0 and actual["source_file_sha256"] == actual["output_file_sha256"]


def test_emitter_ignoring_selected_ir_fails_independent_conformance():
    from test_im2col_padding import source

    from merlin.llvmlower import im2col_padding
    from merlin.llvmlower.automatic_gathers import prepare_automatic_gathers

    original = source()
    prepared = SourceCompilationOptimization(original, selection(original))
    with pytest.MonkeyPatch.context() as mutation:
        mutation.setattr(im2col_padding, "rewrite_padding_gathers", lambda *args, **kwargs: ({"fake": "plan"},))
        with pytest.raises(ValueError, match="accounting"):
            prepare_automatic_gathers(original, serial=True, **prepared.options(GATHER_PASS))


@pytest.mark.parametrize("maximum,rewrites", [(0, 0), (34, 0), (35, 1)])
def test_actual_gather_receives_validated_construction_budget(maximum, rewrites):
    from test_im2col_padding import source

    from merlin.llvmlower.automatic_gathers import prepare_automatic_gathers

    original = source()
    prepared = SourceCompilationOptimization(original, selection(original, maximum=maximum))
    actual, report = prepare_automatic_gathers(original, serial=True, **prepared.options(GATHER_PASS))
    assert report["rewrites"] == rewrites
    assert (actual != original) is bool(rewrites)


@pytest.mark.parametrize("name", [GATHER_PASS, PACKING_PASS])
def test_explicit_source_choice_wins_over_ambient_switch_without_changing_other_passes(monkeypatch, name):
    from test_im2col_padding import source as gather_source
    from test_packed_producer_connected import source as packing_source

    from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
    from merlin.llvmlower.automatic_gathers import prepare_automatic_gathers
    from merlin.llvmlower.optional_passes import active

    source = gather_source() if name == GATHER_PASS else packing_source()
    for enabled in (False, True):
        monkeypatch.setenv("MERLIN_PASSES", "int-softmax-table," + ("-" if enabled else "") + name)
        prepared = SourceCompilationOptimization(source, selection(source, packing=enabled, gather=enabled))
        kwargs = dict(serial=True, **prepared.options(name))
        if name == GATHER_PASS:
            _, report = prepare_automatic_gathers(source, **kwargs)
        else:
            _, report = prepare_automatic_destinations(source, permuted_outputs=True, **kwargs)
        assert report["rewrites"] == int(enabled)
        assert active().state("int-softmax-table") is True
        assert active().state(name) is not enabled


@pytest.mark.parametrize("value", [1, "yes", {}, ()])
def test_explicit_switches_refuse_nonbooleans(value):
    from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
    from merlin.llvmlower.automatic_gathers import prepare_automatic_gathers

    for prepare in (prepare_automatic_destinations, prepare_automatic_gathers):
        with pytest.raises(ValueError, match="boolean"):
            prepare("module {}", serial=True, enabled=value)


def observe(compiled, fixture, directory):
    """Retain actual native buffers, including inputs and both output guards."""
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy

    directory.mkdir()
    model = HostModel.load(
        str(compiled.host_so), image_policy=PrivateHostImagePolicy(compiled.host_so.parent.resolve())
    )
    state = np.arange(-12, 12, dtype=np.int32).reshape(2, 12)
    records = []
    for epoch in range(3):
        arguments = fixture.inputs(epoch, state)
        originals = [value.copy() for value in arguments]
        expected = fixture.reference(*originals)
        guards = [np.full(value.size + 32, 83, dtype=value.dtype) for value in expected]
        outputs = [raw[16:-16].reshape(value.shape) for raw, value in zip(guards, expected, strict=True)]
        model([(value.ctypes.data, value.shape) for value in (*arguments, *outputs)])
        path = directory / f"call-{epoch}.npz"
        np.savez(
            path,
            **{f"before{i}": v for i, v in enumerate(originals)},
            **{f"after{i}": v for i, v in enumerate(arguments)},
            **{f"output{i}": v for i, v in enumerate(outputs)},
            **{f"guard{i}": v for i, v in enumerate(guards)},
        )
        assert all(a.tobytes() == b.tobytes() for a, b in zip(outputs, expected, strict=True))
        assert all(a.tobytes() == b.tobytes() for a, b in zip(arguments, originals, strict=True))
        assert all(np.all(raw[:16] == 83) and np.all(raw[-16:] == 83) for raw in guards)
        records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        state = outputs[0].copy()
    return records


@pytest.mark.parametrize("case", ["packing", "gather"])
def test_ordinary_native_three_policy_arms_execute_complete_outputs_and_state(tmp_path, monkeypatch, case):
    import test_packed_producer_connected as packing_fixture
    import test_padding_gather_connected as gather_fixture

    from merlin.llvmlower.lower import lower_model_file

    fixture = packing_fixture if case == "packing" else gather_fixture
    source = fixture.source()
    path = tmp_path / "source.mlir"
    path.write_text(source)
    monkeypatch.delenv("MERLIN_PASSES", raising=False)
    monkeypatch.delenv("MERLIN_AUTOMATIC_HOST_GATHERS", raising=False)
    monkeypatch.delenv("MERLIN_AUTOMATIC_HOST_PERMUTATIONS", raising=False)
    arms = {}
    for arm, chosen in (
        ("original_policy", selection(source, packing=False, gather=False)),
        ("shared_default", None),
        ("bounded_choice", selection(source)),
    ):
        compiled = lower_model_file(path, tmp_path / arm, source_optimizations=chosen)
        records = observe(compiled, fixture, tmp_path / (arm + "-observed"))
        actual = compiled.stats["automatic_host_permutations" if case == "packing" else "automatic_host_gathers"]
        expected = int(arm == "bounded_choice" or (case == "packing" and arm == "shared_default"))
        assert actual["rewrites"] == expected
        if chosen is not None:
            report = compiled.stats["source_optimizations"]
            assert report["selection"]["request"] == chosen.request.record()
            assert report["binding_stage"] == "lower_model_input_before_source_optimizations"
            assert json.loads((compiled.workdir / "source-optimization-selection.json").read_text()) == report
        arms[arm] = {
            "calls": records,
            "llvm": {
                "path": str(compiled.ll_path),
                "sha256": hashlib.sha256(compiled.ll_path.read_bytes()).hexdigest(),
            },
            "image": {
                "path": str(compiled.host_so),
                "sha256": hashlib.sha256(compiled.host_so.read_bytes()).hexdigest(),
            },
            "malloc_calls": sum(
                "call " in line and "@malloc(" in line for line in compiled.ll_path.read_text().splitlines()
            ),
            "pass_report": actual,
            "choice_report": compiled.stats.get("source_optimizations"),
        }
    assert arms["original_policy"]["malloc_calls"] > arms["bounded_choice"]["malloc_calls"]
    assert (
        arms["original_policy"]["pass_report"]["library_sources_sha256"]
        == arms["bounded_choice"]["pass_report"]["library_sources_sha256"]
    )
    (tmp_path / "three-arm.json").write_text(
        json.dumps(
            {"arms": arms, "scope": "engineering caller; not formal Phase 2 tuning or timing qualification"}, indent=2
        )
        + "\n"
    )
