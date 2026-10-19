"""Observed warmed native calls drive ordinary placement without a device claim.

The observed min/max envelopes describe these samples only. They are not bounds
on future service latency. Compilation, image loading, caller array allocation,
state/counter reset and correctness observation are outside this warm-call scope.
"""

import ctypes
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from test_dense_source_catalog import compilation_memory
from test_dense_source_trace import check_trace
from test_native_offload_model import CASES, SelectedProvider, planned, reference, source
from test_selected_dense_contracts import inputs
from test_source_model_api import reviewed_selection

from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256 as sha
from merlin.llvmlower import abi, toolchain
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.lower import lower_model_file
from merlin.llvmlower.source_dense import offload_context, ordinary_dense_route_identity
from merlin.model_check.full_build import elf_sections
from merlin.perf.component_cost import (
    COMPLETE_STAGES,
    ComponentCostRegion,
    ComponentCostScope,
    ComponentFeatureObservation,
)
from merlin.perf.term import Bounds, PerformanceTerm, Provenance, Validity
from merlin.runtime.host_provider import revalidate_host_provider
from merlin.transforms.certificates import ir_text


def clock_record():
    return vars(time.get_clock_info("perf_counter"))


def query_identity(arguments):
    return sha(
        {
            **{key: asdict(arguments[key]) for key in ("original", "candidate", "scope", "stamp")},
            **{key: arguments[key] for key in ("regime", "original_selection_sha256", "candidate_selection_sha256")},
            **{
                key: {regime: value.to_dict() for regime, value in arguments[key].items()}
                for key in ("original_costs", "candidate_costs")
            },
        }
    )


def sample_effects(native, provider):
    effects = []
    for item in provider.records:
        symbol = item["symbol"]
        count = getattr(native.lib, symbol + "_observe_count")
        count.argtypes, count.restype = [ctypes.c_int], ctypes.c_uint64
        counters = [count(i) for i in range(6)]
        assert counters[5] == 1
        size = getattr(native.lib, symbol + "_trace_size")
        size.restype = ctypes.c_uint64
        query = getattr(native.lib, symbol + "_trace")
        query.argtypes, query.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int64
        rows = [[query(i, j) for j in range(query(i, 1) + 2)] for i in range(size())]
        decoded = check_trace(rows, item["problem"], provider.storage, item["result"].compiled.kernel.args)
        assert counters[:3] == [decoded[key] for key in ("commands", "requested_read_bytes", "requested_write_bytes")]
        effects.append(dict(counters=counters, decoded=decoded, rows=rows))
    return effects


def invoke(arm, args, expected):
    arguments = [array.copy() for array in args]
    outputs = [np.full_like(array, 93) for array in expected]
    for item in arm["provider"].records:
        getattr(arm["native"].lib, item["symbol"] + "_observe_reset")()
    descriptors = [(array.ctypes.data, array.shape) for array in (*arguments, *outputs)]
    started = time.perf_counter_ns()
    arm["native"](descriptors)
    elapsed = time.perf_counter_ns() - started
    for actual, want in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(actual, want)
    for actual, want in zip(arguments[:3], args[:3], strict=True):
        np.testing.assert_array_equal(actual, want)
    return dict(
        ns=elapsed, outputs=[a.tolist() for a in outputs], effects=sample_effects(arm["native"], arm["provider"])
    )


def measured_terms(rows, sample_digest):
    durations = [row["ns"] for row in rows]
    validity = Validity(
        "only these warmed complete native calls with identical reset state, instrumented primitives and ctypes",
        expected_error="observed envelope only; no bound on future invocations or physical devices",
        escalate_when="any executable, runtime, source, state, timer, numerical policy or measurement scope changes",
    )
    provenance = Provenance("measured", (sample_digest,))
    return {
        "cold": PerformanceTerm.unknown("complete_cold_call", "ns", provenance, validity, "cold execution unmeasured"),
        "warm": PerformanceTerm(
            "observed_complete_warm_call",
            sum(durations) / len(durations),
            "ns",
            provenance,
            validity,
            Bounds(min(durations), max(durations)),
        ),
    }


def test_observed_complete_native_cost_is_consumed_by_ordinary_placement(tmp_path, monkeypatch):
    if not toolchain.available() or not toolchain.llvm_nm().is_file():
        pytest.skip("native model compiler and inspector required")
    monkeypatch.setenv("MERLIN_SINK_DEALLOCS", "1")
    shape, tile, width, shared = CASES[0]
    m, k = shape
    module = source(m, k, "i8")
    source_path = tmp_path / "model.mlir"
    source_path.write_text(ir_text(module))
    args, expected, trace = reference(m, k, width)
    input_path = tmp_path / "inputs.npz"
    np.savez(input_path, **{f"in{i}": array for i, array in enumerate(args)})
    facts = tmp_path / "facts"
    facts.mkdir()
    model = inputs(facts, tile=tile, widths=(width, width, 4), shared=shared, shape=(m, k, k))
    arms = {}

    def build(name, indices, comparison=None):
        provider = SelectedProvider(
            tmp_path / (name + "_provider"), model, trace, selected_indices=indices, target_console=False
        )
        selection = reviewed_selection(provider)
        if name != "original":
            selection, plan = planned(selection, module, (1,))
            if comparison is not None:
                selection = replace(selection, offload_comparison=comparison)
        else:
            plan = None
        started = time.perf_counter()
        with compilation_memory() as memory:
            result = lower_model_file(source_path, tmp_path / name, shared_source_lowerings=(selection,))
        compile_seconds = time.perf_counter() - started
        record = result.stats["shared_source_lowering"]
        revalidate_host_provider(record["host_provider"])
        native = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.workdir.resolve()))
        return dict(
            provider=provider,
            selection=selection,
            plan=plan,
            result=result,
            native=native,
            record=record,
            compile_seconds=compile_seconds,
            compilation_memory=memory,
        )

    arms["original"] = build("original", (0, 1))
    arms["candidate"] = build("candidate", (1,))
    for arm in arms.values():
        invoke(arm, args, expected[0])  # Untimed inference history is not reused as a cold measurement.
    samples = {name: [] for name in arms}
    orders = []
    for pair in range(8):
        order = ("original", "candidate") if pair % 2 == 0 else ("candidate", "original")
        orders.append(order)
        for name in order:
            samples[name].append(invoke(arms[name], args, expected[0]))
    sample_path = tmp_path / "warm_samples.json"
    metadata = dict(
        clock=clock_record(),
        python=sys.version,
        platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)),
        orders=orders,
        scope="one warmed synchronous generated native call including ctypes descriptor construction, internal "
        "allocation/packing/host arithmetic/dispatch/instrumented primitives/readout/cleanup; outputs ready on return",
        excluded="compilation, image load, caller array allocation, reset, observer reads and correctness checks",
        cache_policy="one warmup per image; fresh caller arrays, identical private initial state/position and "
        "counters reset per sample; no LLC flush or CPU-frequency control",
        stages="inclusive whole-call accounting, no separately priced stage or presumed zero setup cost",
        cold="UNKNOWN",
        physical_accelerator="unqualified",
        samples=samples,
    )
    sample_path.write_text(json.dumps(metadata, indent=2))
    sample_bytes = sample_path.read_bytes()
    sample_digest = sha256_file(sample_path)
    plan = arms["candidate"]["plan"]
    context = offload_context(arms["candidate"]["selection"])
    scope = ComponentCostScope(
        sha(metadata),
        sha(model[0].numerical_policy),
        sha({"inputs": sha256_file(input_path), "cache": metadata["cache_policy"]}),
    )
    pins = {
        path.resolve(): sha256_file(path)
        for path in (
            source_path,
            input_path,
            sample_path,
            Path(__file__),
            Path(abi.__file__),
            Path(sys.executable).resolve(),
        )
    }
    # Pin the actual already-loaded host runtime objects, not a library name or target assumption.
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = Path(line.split()[-1])
        if path.is_absolute() and path.is_file() and ".so" in path.name and tmp_path not in path.parents:
            pins[path.resolve()] = sha256_file(path)
    terms = {name: measured_terms(rows, sample_digest) for name, rows in samples.items()}
    observations = {}
    target_sha = sha(model[0].record())
    for name, arm in arms.items():
        result = arm["result"]
        region = ComponentCostRegion(
            "whole_call", COMPLETE_STAGES, ("raw_call_ns",), {"samples": sample_digest}, accounting="inclusive"
        )
        cold = ComponentCostRegion(
            "unmeasured_cold_call", COMPLETE_STAGES, ("unmeasured",), {"status": "UNKNOWN"}, accounting="inclusive"
        )
        artifact_files = ((str(result.host_so), sha256_file(result.host_so)),)
        observations[name] = ComponentFeatureObservation(
            sha(arm["record"]["compilation_recipe"]),
            sha256_file(source_path),
            sha256_file(input_path),
            target_sha,
            scope.sha256,
            sha({"arm": name, "metadata": metadata}),
            artifact_files[0][1],
            sha(arm["record"]),
            sha256_file(input_path),
            (sample_digest,),
            (cold,),
            (region,),
            "PASS",
            artifact_files,
            "PASS",
        )
    request = dict(
        original=observations["original"],
        candidate=observations["candidate"],
        original_costs=terms["original"],
        candidate_costs=terms["candidate"],
        scope=scope,
        regime="warm",
        stamp=plan.stamp,
        original_selection_sha256=ordinary_dense_route_identity(plan.stamp),
        candidate_selection_sha256=plan.plan.digest,
    )
    frozen_query = query_identity(request)

    def verify_original(**current):
        if (
            query_identity(current) != frozen_query
            or clock_record() != metadata["clock"]
            or sorted(os.sched_getaffinity(0)) != metadata["affinity"]
        ):
            raise ValueError("original native measurement scope or exact selections changed")
        plan.revalidate(context=offload_context(arms["candidate"]["selection"]))
        assert context == plan.stamp.context
        assert context.hardware == model[0].hardware_identity and target_sha == sha(model[0].record())
        for path, digest in pins.items():
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError("original native timing evidence changed")
        current_samples = json.loads(sample_path.read_text())["samples"]
        independent_args, independent, _ = reference(m, k, width)
        with np.load(input_path) as saved:
            for index, (actual, wanted) in enumerate(zip(args, independent_args, strict=True)):
                assert actual.dtype == wanted.dtype and actual.shape == wanted.shape
                np.testing.assert_array_equal(actual, wanted)
                np.testing.assert_array_equal(saved[f"in{index}"], wanted)
        for name, arm in arms.items():
            revalidate_host_provider(arm["record"]["host_provider"])
            assert (
                arm["native"].image_sha256 == sha256_file(arm["result"].host_so) == observations[name].executable_sha256
            )
            assert {
                key: value.to_dict() for key, value in measured_terms(current_samples[name], sample_digest).items()
            } == {key: value.to_dict() for key, value in current[name + "_costs"].items()}
            for row in current_samples[name]:
                assert row["outputs"] == [a.tolist() for a in independent[0]]
                assert len(row["effects"]) == (2 if name == "original" else 1)

    cost_selection = arms["candidate"]["selection"].with_complete_offload_cost(
        **{
            key: value
            for key, value in request.items()
            if key
            not in (
                "stamp",
                "original_selection_sha256",
                "candidate_selection_sha256",
            )
        },
        verify_original=verify_original,
    )
    decision = cost_selection.offload_comparison
    report = decision.record()
    before, after = terms["original"]["warm"], terms["candidate"]["warm"]
    assert report["choose_candidate"] == (after.bounds.upper < before.bounds.lower)
    assert report["unit"] == "ns" and report["costs"]["original"]["cold"]["value"] == "UNKNOWN"
    damaged = json.loads(sample_bytes)
    damaged["samples"]["candidate"][0]["ns"] += 1
    sample_path.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="timing evidence changed"):
        decision.revalidate()
    sample_path.write_bytes(sample_bytes)
    decision.revalidate()
    indices = (1,) if decision.choose_candidate else (0, 1)
    selected = build("selected", indices, decision)
    observation = invoke(selected, args, expected[0])
    selected_history = [observation]
    receipt_path = Path(selected["record"]["preparation"]["path"]).parent / "source_dense.json"
    (placement,) = json.loads(receipt_path.read_text())["offload_placement"]
    assert placement["actual_route"] == ("candidate" if decision.choose_candidate else "original")
    assert placement["comparison"] == report and len(selected["provider"].records) == len(indices)
    observed_arm = "candidate" if decision.choose_candidate else "original"
    assert observation["effects"] == samples[observed_arm][0]["effects"]
    state = np.asarray(observation["outputs"][2], dtype=args[3].dtype)
    position = np.asarray(observation["outputs"][3], dtype=args[4].dtype)
    for step in (1, 2):
        next_observation = invoke(selected, (*args[:3], state, position), expected[step])
        selected_history.append(next_observation)
        state = np.asarray(next_observation["outputs"][2], dtype=args[3].dtype)
        position = np.asarray(next_observation["outputs"][3], dtype=args[4].dtype)
    product_sha = sha256_file(selected["result"].host_so)
    exact_product = product_sha == observations[observed_arm].executable_sha256
    transfer = dict(
        status="EXACT_OBSERVED_IMAGE" if exact_product else "REFUSAL",
        observed_arm=observed_arm,
        observed_sha256=observations[observed_arm].executable_sha256,
        selected_sha256=product_sha,
        scope="warm observations transfer only to identical executable bytes with the retained runtime/input scope",
    )
    # Recompilation is not permitted to extend the measured envelope to different code.
    if not exact_product:
        transfer["reason"] = "selected executable differs; no timing transfer admitted"
    (tmp_path / "measured_placement.json").write_text(
        json.dumps(
            dict(
                decision=report,
                transfer=transfer,
                selected_observation=observation,
                selected_history=selected_history,
                sources={name: arm["record"] for name, arm in {**arms, "selected": selected}.items()},
                compilation_seconds={
                    name: arm["compile_seconds"] for name, arm in {**arms, "selected": selected}.items()
                },
                compilation_memory={
                    name: arm["compilation_memory"] for name, arm in {**arms, "selected": selected}.items()
                },
                static_artifacts={
                    name: dict(bytes=arm["result"].host_so.stat().st_size, sections=elf_sections(arm["result"].host_so))
                    for name, arm in {**arms, "selected": selected}.items()
                },
                pins={str(path): digest for path, digest in pins.items()},
                scope="observed warm native call only; physical accelerator, cold and service latency unqualified",
            ),
            indent=2,
        )
    )
