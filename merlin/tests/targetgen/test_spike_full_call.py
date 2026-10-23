"""Plain-ISA Spike oracle: routed from the contract, functional tier only, ISA never defaulted."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

capsule_runner = pytest.importorskip("merlin.targetgen.capsule_runner")
spike_full_call = pytest.importorskip("merlin.targetgen.spike_full_call")
_REAL_SPIKE_RUN = pytest.importorskip("merlin.runtime.backends.spike_model").run  # before fixtures patch it


def test_isa_is_derived_not_defaulted(monkeypatch):
    class Manifest:
        contract = {"name": "x"}

    monkeypatch.setattr("merlin.targetgen.target_experiment.load_capability_manifest", lambda target, **_: Manifest())
    with pytest.raises(ValueError, match="isa.march"):
        spike_full_call.sim_adapters("anything")


def test_adapter_exposes_full_call_boundary_only():
    adapter = spike_full_call.full_call_adapter("t", isa="rv64gc")
    assert callable(adapter.run_full_call)
    with pytest.raises(capsule_runner.OracleUnavailable):
        adapter(None, "", None, 1)


def test_explicit_transport_cannot_be_an_arbitrary_candidate_callback():
    with pytest.raises(TypeError, match="selected candidate product transport"):
        spike_full_call.full_call_adapter(
            "t", isa="declared-isa", candidate_transport=SimpleNamespace(compile=lambda: None)
        )


def test_native_binding_requires_every_source_oracle_to_accept_transport():
    def legacy(*args):
        pass

    legacy.compiles_source_bundle = True
    with pytest.raises(NotImplementedError, match="cannot consume explicit"):
        capsule_runner._with_candidate_transport({"L2": legacy}, object())
    with pytest.raises(NotImplementedError, match="no oracle"):
        capsule_runner._with_candidate_transport({}, object())


@pytest.mark.parametrize("build", [{"command": "run-candidate"}, {}, None, []])
def test_source_product_scope_refuses_build_commands_and_other_capsules(build):
    def oracle(*args):
        pass

    oracle._merlin_candidate_transport = object()
    adapters = {"L2": oracle}
    package = SimpleNamespace(manifest={"build": build})
    with pytest.raises(capsule_runner.CertFailure, match="submitted package build commands"):
        capsule_runner.require_candidate_product_scope(package, adapters)
    package.manifest = {}
    with pytest.raises(capsule_runner.CertFailure, match="exported full-call capsules"):
        capsule_runner.require_candidate_product_scope(package, adapters, capsule={"kind": "model"})
    capsule_runner.require_candidate_product_scope(package, adapters)
    package.manifest = {"build": {"command": "legacy-execution"}}
    capsule_runner.require_candidate_product_scope(package, {})


def test_build_refusal_returns_durable_execution_error(tmp_path, monkeypatch):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_provenance

    def refuse(*args, **kwargs):
        raise ValueError("candidate build refused")

    monkeypatch.setattr(spike_model, "build", refuse)
    monkeypatch.setattr(core_aten_provenance, "batch_provenance", lambda *a, **k: {"scope": "test"})
    adapter = spike_full_call.full_call_adapter("t", isa="declared-isa")
    result = adapter.run_full_call(bundle=tmp_path, llvm_mlir="submitted LLVM")
    assert result["output_bytes"] is None
    assert result["execution_error"] == "ValueError: candidate build refused"
    assert result["lane"] == "host" and result["executed_instructions"] == 0


def test_guest_console_never_reaches_the_execution_error(tmp_path, monkeypatch, source_adapter):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_device

    adapter, _ = source_adapter
    guest = tmp_path / "guest-sim"
    guest.write_text("#!/bin/sh\necho 'SECRET-GOLDEN-BYTES /opt/private/answers'\nexit 3\n")
    guest.chmod(0o755)
    monkeypatch.setattr(spike_model, "run", lambda *a, **k: _REAL_SPIKE_RUN(*a, **{**k, "spike_binary": guest}))
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (None, "host"))
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert result["execution_error"].startswith("SpikeModelError: spike exited 3")
    assert "SECRET" not in result["execution_error"] and "/" not in result["execution_error"]


@pytest.fixture
def source_adapter(tmp_path, monkeypatch):
    from merlin.llvmlower import device_offload
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_capsules, core_aten_device, core_aten_provenance

    provider_path = tmp_path / "provider.py"
    provider_path.write_text("# selected support\n")
    provider = SimpleNamespace(prepare_bundle=lambda *a, **k: None, validate_package=lambda *a, **k: None)
    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda *a: provider)
    monkeypatch.setattr(core_aten_device, "selected_facts", lambda *a: nullcontext({}))
    monkeypatch.setattr(core_aten_provenance, "batch_provenance", lambda *a, **k: {"scope": "test"})
    builds = []

    def build(*a, **kwargs):
        builds.append(kwargs.get("device"))
        return {"elf": tmp_path / "model.elf", "mem_bytes": 4096, "build_hash": "bound-build"}

    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(
        spike_model,
        "run",
        lambda *a, **k: {"console": "done", "output_bytes": [b"data"], "metrics": {"build_hash": "bound-build"}},
    )
    monkeypatch.setattr(device_offload, "load_sidecar", lambda p: {"routed": ["source-bound-call"]})
    monkeypatch.setattr(core_aten_device, "executed_device_instructions", lambda *a: {"executed_instructions": 7})
    monkeypatch.setattr(core_aten_capsules, "load_private", lambda c: {"batch_map": {"cases": [{"case_id": "source"}]}})
    monkeypatch.setattr("merlin.targetgen.rtl.facts.ensure_facts", lambda target, explicit=None: tmp_path / "facts")
    monkeypatch.setattr("merlin.targetgen.full_call_build.build_in_sandbox", child_in_process)
    # Simulator jails are exercised for real in test_full_call_build; here the simulators are fakes.
    monkeypatch.setattr("merlin.targetgen.full_call_build.oracle_jail", lambda elf, work, **_: ([], Path(elf)))

    def unjailed(target, engine, elf, work, *, private, timeout):
        from merlin.runtime.backends import base

        return base.get_backend(target).run_elf(elf, simulator=engine, timeout=timeout)

    monkeypatch.setattr("merlin.targetgen.full_call_build.run_elf_jailed", unjailed)
    return spike_full_call.full_call_adapter("t", isa="declared-isa", execution_provider=provider_path), builds


def child_in_process(request, *, bundle, work, package, provider, facts, denied, timeout):
    """The sandboxed child's own code, run in-process so routing semantics stay unit-testable."""
    import json

    from merlin.targetgen import full_call_build

    import shutil

    shutil.rmtree(work, ignore_errors=True)  # a fresh work directory per build, as the sandbox runner makes
    produced = work / "out" / "build"
    produced.parent.mkdir(parents=True)
    request = {**request, "bundle": str(bundle), "build": str(produced), "package": str(package)}
    request |= {"provider": str(provider), "facts": str(facts), "result": str(work / "out" / "result.json")}
    (work / "request.json").write_text(json.dumps(request))
    full_call_build.main([str(work / "request.json")])
    report = json.loads((work / "out" / "result.json").read_text())
    if report.get("build"):
        produced.mkdir(exist_ok=True)
        (produced / "model.elf").write_bytes(b"image")
        report["build"]["elf"] = str(produced / "model.elf")
    return report


def test_selected_transport_validates_inventory_before_host_fallback(tmp_path, monkeypatch, source_adapter):
    from merlin_experiments.phase1.candidate_products import CandidateProductTransport

    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    transport = object.__new__(CandidateProductTransport)
    events = []
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text("module {}")
    names = ("catalog_source.mlir", "device_catalog.json", "kernel.gemmini.mlir")
    tools = {"python": "/selected/bin/python", "llvm_bin": "/selected/llvm/bin"}
    monkeypatch.setattr(CandidateProductTransport, "tool_substitutions", lambda self: tools)

    def compile(self, package, source, **kwargs):
        assert self is transport and source == prepared and package == tmp_path
        assert tuple(p.relative_path for p in kwargs["products"]) == names
        assert sum(p.max_bytes for p in kwargs["products"]) <= 256 * 1024 * 1024
        assert kwargs["private_paths"] == (tmp_path,)
        assert kwargs["substitutions"] == tools
        events.append("isolated-inventory")
        return {"synthetic": "API-control only"}

    monkeypatch.setattr(CandidateProductTransport, "compile", compile)

    def validate(directory, *, compile_products, **kwargs):
        assert directory == tmp_path
        compile_products(prepared, tmp_path / "retained", inventory_only=True)
        events.append("validated")

    provider = SimpleNamespace(
        validate_package_candidate_products=validate,
        validate_package=lambda *a, **k: pytest.fail("legacy candidate Python validation was selected"),
        candidate_catalog_products=lambda *, inventory_only: names if inventory_only else (),
        candidate_catalog_command=lambda **kwargs: ("{python}", "-m", "selected.compiler"),
    )
    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda *a: provider)

    def decline(*args, **kwargs):
        assert callable(kwargs["compile_products"])
        assert events == ["isolated-inventory", "validated"]
        return None, "no eligible contraction"

    monkeypatch.setattr(core_aten_device, "routing_for_bundle", decline)
    bound = adapter._merlin_with_candidate_transport(transport)
    result = bound.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert result["execution_error"] is None
    assert result["lane"] == "host" and builds == [None]
    assert events == ["isolated-inventory", "validated"]


@pytest.mark.parametrize(
    "reason", ["no eligible contraction in captured source", "provider declines captured contraction semantics"]
)
def test_optional_device_decline_runs_real_host_path(tmp_path, monkeypatch, source_adapter, reason):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (None, reason))
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert result["execution_error"] is None and result["output_bytes"] == [b"data"]
    assert builds == [None] and result["lane"] == "host" and result["executed_instructions"] == 0
    assert result["routing"]["reason"] == reason


def test_strict_device_decline_still_refuses(tmp_path, monkeypatch, source_adapter):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (None, "declined"))
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "device"}
    )
    assert "did not route" in result["execution_error"] and not builds


def test_host_guard_never_invokes_candidate_routing(tmp_path, monkeypatch, source_adapter):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: pytest.fail("guard must stay host"))
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": False, "lane_expectation": "host-guard"}
    )
    assert result["execution_error"] is None and builds == [None] and result["lane"] == "host"


@pytest.mark.parametrize("failure", ["catalog source hash mismatch", "invalid catalog", "compiler failure"])
def test_optional_device_errors_do_not_fall_back(tmp_path, monkeypatch, source_adapter, failure):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter

    def refuse(*a, **k):
        raise ValueError(failure)

    if failure == "compiler failure":
        monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (object(), "bound"))
        monkeypatch.setattr(spike_model, "build", refuse)
    else:
        monkeypatch.setattr(core_aten_device, "routing_for_bundle", refuse)
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert failure in result["execution_error"] and result["output_bytes"] is None and not builds


@pytest.mark.parametrize("instructions", [0, 7])
def test_optional_device_requires_bound_calls_and_executed_instructions(
    tmp_path, monkeypatch, source_adapter, instructions
):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    route = object()
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (route, "bound"))
    monkeypatch.setattr(
        core_aten_device, "executed_device_instructions", lambda *a: {"executed_instructions": instructions}
    )
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert builds == [route]
    if instructions:
        assert result["execution_error"] is None and result["routing"]["routed"] is True
        assert result["executed_instructions"] == instructions
    else:
        assert "no target instruction" in result["execution_error"] and result["output_bytes"] is None


@pytest.mark.parametrize("validation", [None, "raises", "invalid-return"])
def test_optional_fallback_requires_independent_validation(tmp_path, monkeypatch, source_adapter, validation):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter

    def validate(*a, **k):
        if validation == "raises":
            raise ValueError("malformed unused catalog")
        return True

    provider = SimpleNamespace(validate_package=validate if validation else None)
    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda *a: provider)
    monkeypatch.setattr(
        core_aten_device, "routing_for_bundle", lambda *a: pytest.fail("validation must precede decline")
    )
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert result["execution_error"] and not builds and result["output_bytes"] is None


@pytest.mark.parametrize("sidecar", ["catalog", "region", "none"])
def test_either_build_sidecar_is_device_call_evidence(tmp_path, monkeypatch, source_adapter, sidecar):
    from merlin.llvmlower import device_build, device_offload
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (object(), "bound"))
    monkeypatch.setattr(
        device_offload, "load_sidecar", lambda p: {"routed": ["catalog"]} if sidecar == "catalog" else {}
    )
    monkeypatch.setattr(
        device_build, "load_region_offload", lambda p: {"routed": ["region"]} if sidecar == "region" else {}
    )
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "device"}
    )
    if sidecar == "none":
        assert "no source-bound device calls" in result["execution_error"]
    else:
        assert result["execution_error"] is None and result["routing"]["routed_operations"] == [sidecar]


def test_admitted_non_contraction_sites_take_the_provider_region_route(tmp_path, monkeypatch):
    from merlin.targetgen import core_aten_device as D

    provider = SimpleNamespace(core_aten_offload=lambda *a, **k: None)
    report = {"cases": [{"status": "bundled", "case": {}}]}
    plan = {"quantized_contractions": 0, "exact_dtype_sites": 1, "sites": []}
    monkeypatch.setattr(D, "routing_for_bundle", lambda *a, **k: pytest.fail("no contraction to catalog"))
    routing, reason = D.select_source_routing(tmp_path, "t", tmp_path, provider, {}, report, plan)
    assert routing.region_offload is not None and "region offload" in reason
    routing, reason = D.select_source_routing(tmp_path, "t", tmp_path, provider, {}, report, plan, compile_products=1)
    assert routing is None and "candidate product transport" in reason
    monkeypatch.setattr(D, "routing_for_bundle", lambda *a, **k: ("catalog", "bound"))
    assert (
        D.select_source_routing(tmp_path, "t", tmp_path, provider, {}, report, {**plan, "quantized_contractions": 2})[0]
        == "catalog"
    )
    assert D.select_source_routing(tmp_path, "t", tmp_path, provider, {}, report, None)[0] == "catalog"


@pytest.mark.parametrize("available", [True, False, "crashes"])
def test_board_linked_oracle_grades_the_rtl_run_of_the_same_image(tmp_path, monkeypatch, source_adapter, available):
    from merlin.runtime.backends import base, spike_model
    from merlin.targetgen import core_aten_device, rtl_simulator_execution

    adapter, _ = source_adapter
    layouts, executed = [], []

    def build(*a, **kwargs):
        a[1].mkdir()  # the board image is built beside, never over, the functional tier's build
        layouts.append({k: kwargs.get(k) for k in ("dram_base", "dram_bytes", "code_reserve", "arena_mb")})
        return {"elf": tmp_path / "board.elf", "mem_bytes": 256 << 20, "build_hash": "bound-build"}

    def run_elf(elf, simulator, timeout):
        executed.append((elf, simulator))
        if available == "crashes":
            raise RuntimeError("simulator exited before a console")
        return "rtl console"

    backend = SimpleNamespace(available=lambda engine: bool(available), run_elf=run_elf)
    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(base, "get_backend", lambda target: backend)
    monkeypatch.setattr(rtl_simulator_execution, "verify_selected", lambda *a: {"execution_identity": "simv"})
    rtl_parse = {
        "console": "rtl console",
        "output_bytes": [b"rtl"],
        "metrics": {"build_hash": "bound-build", "cycles": 9},
    }
    monkeypatch.setattr(spike_model, "parse_console", lambda console: rtl_parse)
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (object(), "bound"))
    board = dict(dram_base=0x80000000, dram_bytes=256 << 20, code_reserve=64 << 20)
    rtl = adapter._merlin_on_board("vcs", board)
    assert not hasattr(rtl, "qualify_full_call") and rtl._merlin_simulator_engine == "vcs"
    result = rtl.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "device"}
    )
    assert layouts == [{**board, "arena_mb": 96}]
    if available == "crashes":  # no RTL console: the tier is not run, never an RTL verdict
        assert result["derived_from_rtl"] is False and "before a console" in result["execution_error"]
    elif available:
        assert executed == [(tmp_path / "vcs-build" / "model.elf", "vcs")]  # the private copy is run
        assert result["output_bytes"] == [b"rtl"] and result["derived_from_rtl"] is True and result["cycles"] == 9
        assert result["lane"] == "device" and result["executed_instructions"] == 7
        assert result["provenance"]["simulator"] == {"execution_identity": "simv"}
    else:
        assert not executed and result["derived_from_rtl"] is False and "not available" in result["execution_error"]
    with pytest.raises(ValueError, match="elaborated-RTL engine"):
        adapter._merlin_on_board("spike", board)


def test_selected_execution_provider_must_be_support_owned(tmp_path, monkeypatch):
    from merlin.targetgen import core_aten_device, plugins
    from merlin.targetgen.rtl import facts

    support = tmp_path / "support"
    support.mkdir()
    (support / "provider.py").write_text("# selected\n")
    (tmp_path / "outside.py").write_text("# foreign\n")
    monkeypatch.setattr(plugins, "resolve_support", lambda target: SimpleNamespace(base=support))
    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda *a: SimpleNamespace(runner_options=dict))
    monkeypatch.setattr(facts, "ensure_facts", lambda target: tmp_path / "facts.json")
    monkeypatch.delenv("MERLIN_CORE_ATEN_EXECUTION_PROVIDER", raising=False)
    assert spike_full_call.selected("t") is None
    monkeypatch.setenv("MERLIN_CORE_ATEN_EXECUTION_PROVIDER", str(tmp_path / "outside.py"))
    with pytest.raises(ValueError, match="inside the selected support"):
        spike_full_call.selected("t")
    monkeypatch.setenv("MERLIN_CORE_ATEN_EXECUTION_PROVIDER", str(support / "provider.py"))
    oracle = spike_full_call.selected("t")
    assert oracle.compiles_source_bundle and callable(oracle.qualify_full_call)


@pytest.mark.parametrize("rtl_config", ["SelectedConfig", None])
def test_rtl_board_comes_from_the_descriptor_host_board(tmp_path, monkeypatch, rtl_config):
    from merlin.runtime import boards
    from merlin.targetgen import corpora, target_experiment

    experiment = SimpleNamespace(host_board="board", selected_board_catalog=lambda: tmp_path / "catalog.yaml")
    board = SimpleNamespace(target="t", rtl_sim_config=rtl_config, dram_base=1, dram_bytes=2, code_reserve=None)
    monkeypatch.setattr(corpora, "descriptor_path", lambda target: tmp_path / "descriptor.yaml")
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: experiment)
    monkeypatch.setattr(boards, "load_boards", lambda catalog: {"board": board})
    calls = []
    adapter = SimpleNamespace(_merlin_on_board=lambda engine, layout: calls.append((engine, layout)) or "rtl")
    selected = spike_full_call.on_rtl_board(adapter, "vcs", "t")
    if rtl_config:
        assert selected == "rtl" and calls == [("vcs", dict(dram_base=1, dram_bytes=2, code_reserve=None))]
    else:
        assert selected is None and not calls
    assert spike_full_call.on_rtl_board(SimpleNamespace(), "vcs", "t") is None


def test_source_product_scope_covers_the_floor_tier(monkeypatch):
    def oracle(*args):
        pass

    oracle._merlin_candidate_transport = object()
    package = SimpleNamespace(manifest={})
    capsule = {"required_oracle_tiers": ["L2"], "scored": True}
    from merlin.targetgen import candidate_scope

    monkeypatch.setattr(candidate_scope, "is_full_call", lambda capsule: True)
    monkeypatch.delenv("MERLIN_CAPSULE_TIER_FLOOR", raising=False)
    candidate_scope.require_candidate_product_scope(package, {"L2": oracle, "L3": object()}, capsule=capsule)
    monkeypatch.setenv("MERLIN_CAPSULE_TIER_FLOOR", "L3")
    with pytest.raises(capsule_runner.CertFailure, match="every required full-call tier"):
        candidate_scope.require_candidate_product_scope(package, {"L2": oracle, "L3": object()}, capsule=capsule)
