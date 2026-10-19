"""Normal source preparation uses explicit neutral adapters, never target guesses."""

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from test_pointwise_source import case, source_text
from test_pointwise_source_provider import Provider, qualify

from merlin.compile.scheduling.fusion_semantics import scalar_graph
from merlin.llvmlower.fresh_tensor_writer import FreshTensorWriterContract
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.shared_pointwise_routing import PointwiseSourceBackend, PointwiseSourceRouting
from merlin.runtime.host_provider import prepare_host_provider


class AutomaticProvider(Provider):
    """Only neutral instruction/glue support; generic routing owns source selection."""

    def __init__(self, work, *, enabled, separate):
        super().__init__(work, enabled=enabled, separate=separate)
        selection = work / "selection"
        selection.mkdir()
        self.contracts = case(selection, separate=separate)["selected_contracts"]
        self.expected_graph = scalar_graph(self.expected_scalar_source, 128, 65536)
        self.fragments, self.qualifications = [], 0
        self.bad_output, self.inject_selection = False, False
        self.loosen_verifier, self.missing_runtime = False, False
        self.disable = False

    def propose(self, source, _operation, directory):
        if self.disable or source.graph != self.expected_graph or len(source.shape) != 2:
            return None
        kwargs = self.case_options(directory, *source.shape)
        supplied = kwargs.pop("selected_contracts")
        assert supplied.hardware_identity == self.contracts.hardware_identity
        assert supplied.numerical_policy == self.contracts.numerical_policy
        if self.inject_selection:
            kwargs["selected_contracts"] = supplied
        if self.missing_runtime:
            kwargs["bindings"] = replace(kwargs["bindings"], external_buffers_disjoint=False)
        budgets = self.schedule_budgets()
        if self.loosen_verifier:
            budgets = replace(budgets, solver_timeout_ms=budgets.solver_timeout_ms + 1)
        return {**kwargs, "budgets": budgets}

    def emit(self, checked, symbol, source, _directory):
        bridge = "_mlir_ciface_borrowed_" + symbol
        image = b"".join(checked.images())
        self.results.append(checked)
        self.records.append(
            {
                "symbol": bridge,
                "shape": list(source.shape),
                "program_section": ".merlin_pointwise." + bridge,
                "program_sha256": sha256(image).hexdigest(),
            }
        )
        self.fragments.append(self._wrapper(bridge, *source.shape, image, self.separate))
        return FreshTensorWriterContract(symbol, 2, (2,), "borrowed_" + symbol, 32)

    def finalize(self, _results, _output):
        self.finish_source(self.fragments)
        if self.bad_output:
            text = self.c_source.read_text()
            for record in self.records:
                _, columns = record["shape"]
                before = f"]=y[i*{columns}+j];"
                assert before in text
                text = text.replace(before, f"]=(uint8_t)(1+y[i*{columns}+j]);")
            self.c_source.write_text(text)

    def verify_selection(self):
        self.contracts.revalidate()


def routing(provider, trace):
    def independent_qualification(results, _output):
        assert tuple(provider.results) == results
        qualify(provider, trace)
        provider.qualifications += 1

    backend = PointwiseSourceBackend(
        provider.contracts,
        provider.propose,
        provider.emit,
        provider.finalize,
        provider.build,
        provider.verify_selection,
    )
    return PointwiseSourceRouting(backend, qualify=independent_qualification)


def inputs(m, n):
    left = (np.arange(m * n, dtype=np.int64) * 73 - 310).reshape(m, n).astype("<i4")
    bias = (np.arange(m * n, dtype=np.int64) * -19 + 74).reshape(m, n).astype("<i4")
    return [(0, left, bias), (0, left + 1, bias - 1), (0, left - 2, bias + 3)]


def prepare(tmp_path, *, enabled=True, shape=(2, 11), change=None):
    provider = AutomaticProvider(tmp_path / "provider", enabled=enabled, separate=True)
    provider.bad_output = change == "wrong_glue"
    provider.inject_selection = change == "selection"
    provider.loosen_verifier = change == "verification"
    provider.missing_runtime = change == "missing_runtime"
    selected = routing(provider, inputs(*shape))
    source = tmp_path / "source.mlir"
    text = source_text(*shape)
    if change == "unsupported":
        text = text.replace("arith.addi", "arith.subi")
    source.write_text(text)
    model = apply_prepared_model_transform(source, tmp_path / "preparation", selected.prepare)
    return selected, provider, source, model


def test_missing_backend_or_original_execution_qualifier_keeps_normal_route_disabled(tmp_path):
    assert PointwiseSourceRouting().build_arguments() == {}
    assert PointwiseSourceRouting().record()["status"] == "disabled"
    provider = AutomaticProvider(tmp_path / "provider", enabled=True, separate=True)
    backend = PointwiseSourceBackend(
        provider.contracts,
        provider.propose,
        provider.emit,
        provider.finalize,
        provider.build,
        provider.verify_selection,
    )
    missing = PointwiseSourceRouting(backend)
    assert missing.build_arguments() == {} and "qualifier" in missing.record()["refusal"]
    assert not provider.results and provider.qualifications == 0


@pytest.mark.parametrize("shape", [(1, 7), (2, 11), (3, 5)])
def test_normal_preparation_matches_source_and_runs_independent_native_full_output_checks(tmp_path, shape):
    route, provider, source, model = prepare(tmp_path, shape=shape)
    assert source.read_text() == source_text(*shape)
    assert provider.qualifications == 1 and len(provider.results) == 1
    report = json.loads((tmp_path / "preparation/source_pointwise_routing.json").read_text())
    assert report["routed_count"] == 1 and report["fallback_count"] == 0
    assert "complete original source restored" in report["inverse"]
    assert "borrowed_merlin_source_pointwise_" in model.read_text()
    assert provider.results[0].conformance["elements"] == np.prod(shape)
    assert provider.results[0].conformance["actual_commands"] == 1
    with pytest.raises(ValueError, match="reuse"):
        route.prepare(source, tmp_path / "preparation")


def test_all_unsupported_source_preserves_bytes_and_skips_only_this_exact_empty_provider(tmp_path):
    route, provider, source, model = prepare(tmp_path, change="unsupported")
    assert model.read_bytes() == source.read_bytes()
    assert provider.qualifications == 0 and provider.results == []
    context = SimpleNamespace(prepared_path=model)
    assert prepare_host_provider(route.build, context, inspector=None) == ((), None)
    report = json.loads((tmp_path / "preparation/source_pointwise_routing.json").read_text())
    assert report["routed_count"] == 0 and report["fallback_count"] == 1
    with pytest.raises(ValueError, match="explicitly callable"):
        prepare_host_provider(object(), context, inspector="not launched")


@pytest.mark.parametrize("change", ["wrong_glue", "selection", "verification"])
def test_incorrect_backend_cannot_publish_or_fall_back_around_original_checks(tmp_path, change):
    with pytest.raises((ValueError, AssertionError), match="(?i)mismatch|contracts|verification"):
        prepare(tmp_path, change=change)
    assert not (tmp_path / "preparation/selected.mlir").exists()
    assert not (tmp_path / "preparation/prepared_model_transform.json").exists()


def test_missing_runtime_ownership_keeps_source_fallback_and_precise_reason(tmp_path):
    route, provider, source, model = prepare(tmp_path, change="missing_runtime")
    assert provider.qualifications == 0 and model.read_bytes() == source.read_bytes()
    assert prepare_host_provider(route.build, SimpleNamespace(prepared_path=model), inspector=None) == ((), None)
    report = json.loads((tmp_path / "preparation/source_pointwise_routing.json").read_text())
    assert report["routed_count"] == 0 and "ownership" in report["members"][0]["reason"]


@pytest.mark.parametrize("change", ["source", "selected", "software"])
def test_provider_reopens_source_selection_before_build_or_empty_route_bypass(tmp_path, change):
    route, provider, source, model = prepare(tmp_path, change="unsupported")
    path = {
        "source": tmp_path / "preparation/input.mlir",
        "selected": model,
        "software": provider.contracts.software_path,
    }[change]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        prepare_host_provider(route.build, SimpleNamespace(prepared_path=model), inspector=None)


def test_normal_recurrent_small_model_routes_without_test_provider_source_matching(tmp_path, record_property):
    from test_pointwise_source_whole_model import test_ordinary_recurrent_model_runs_source_bound_fusion as run_pair

    run_pair(
        tmp_path,
        record_property,
        shape=(2, 5, 11),
        separate=True,
        provider_class=AutomaticProvider,
        routing_factory=routing,
    )


def test_normal_recurrent_build_keeps_complete_host_fallback_with_no_admitted_islands(tmp_path):
    from test_pointwise_source_whole_model import capture

    from merlin.llvmlower import toolchain
    from merlin.model_check import full_build as FB
    from merlin.runtime.backends import spike, spike_model

    if not toolchain.available() or not spike.available():
        pytest.skip("ordinary native lowering and functional simulator unavailable")
    bundle, work = tmp_path / "capture", tmp_path / "model"
    expected, trace = capture(bundle, (2, 5, 11))
    provider = AutomaticProvider(tmp_path / "provider", enabled=True, separate=True)
    provider.disable = True
    route = routing(provider, trace)
    built = spike_model.build(
        bundle,
        work,
        arena_mb=1,
        dram_bytes=64 << 20,
        code_reserve=16 << 20,
        backend="scalar",
        full_readback=True,
        **route.build_arguments(),
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
        host_llvm_bitcode=True,
    )
    console, _wall = FB.run_host(built["elf"], timeout_s=30)
    actual, _metadata = FB.read(console, bundle, built["build_hash"])
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, want)
    assert provider.results == [] and provider.qualifications == 0
    assert not (work / "host_provider/host_provider.json").exists()
    report = json.loads((work / "prepared_model_transform/source_pointwise_routing.json").read_text())
    assert report["routed_count"] == 0 and report["fallback_count"] >= 2


def test_saved_or_foreign_callback_owner_cannot_replace_original_selection(tmp_path):
    provider = AutomaticProvider(tmp_path / "provider", enabled=True, separate=True)
    route = routing(provider, inputs(2, 11))
    with pytest.raises(ValueError, match="completed original preparation"):
        prepare_host_provider(route.build, SimpleNamespace(prepared_path=tmp_path / "absent"), inspector=None)
    route._backend = replace(route._backend, propose=lambda *args: None)
    with pytest.raises(ValueError, match="owner changed"):
        route.build_arguments()
