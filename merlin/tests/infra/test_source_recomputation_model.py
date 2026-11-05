"""Original host observations for bounded source recomputation choices.

The scalar reference checks complete results; it never selects a rewrite or
prices work. Observed native nanoseconds are not target or future-latency bounds.
"""

import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from test_multi_output_source import CASES, reference, source
from test_recomputation_llvm_payload import descriptor_roster, inspect_buffer_activity

from merlin.common import invocation_record as invocation
from merlin.common.digest import sha256_file
from merlin.common.jsonio import canonical_sha256 as digest
from merlin.common.paths import merlin_dir
from merlin.llvmlower import abi, codegen, lower, toolchain
from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
from merlin.llvmlower.arena_bind import _call_target, _defines, _malloc_size
from merlin.model_check.full_build import elf_sections
from merlin.perf.component_cost import (
    COMPLETE_STAGES,
    ComponentCostRegion,
    ComponentCostScope,
    ComponentFeatureObservation,
)
from merlin.perf.offload_profitability import compare_complete_offload_cost
from merlin.perf.term import Bounds, PerformanceTerm, Provenance, Validity
from merlin.transforms.certificates import ir_digest, ir_text
from merlin.transforms.pipeline_state import PipelineContext

OPTIONS = dict(targets=("host",), vectorize=False, parallel=False)


def tools():
    translator = toolchain.mlir_translate().resolve()
    selected = (
        toolchain.clang(),
        toolchain.compiler_python(),
        translator,
        translator.with_name("llvm-extract"),
        translator.with_name("mlir-opt"),
        toolchain.llvm_install() / "bin/mlir-opt",
        toolchain.host_llc(),
        os.environ.get("CC", "").strip() or toolchain.clang(),
    )
    return tuple(
        dict.fromkeys(
            Path(shutil.which(str(p)) or p).resolve()
            for p in selected
            if p is not None and Path(shutil.which(str(p)) or p).is_file()
        )
    )


def machine_scope():
    return dict(
        machine=platform.machine(),
        platform=platform.platform(),
        python=sys.version,
        clock=vars(time.get_clock_info("perf_counter")),
        affinity=sorted(os.sched_getaffinity(0)),
        environment=invocation.environment_identity(os.environ),
        tools={str(p): sha256_file(p) for p in tools()},
    )


def original_plan(case):
    """Bind actual original tools/runtime and finite options before planning."""
    from merlin.llvmlower import source_multioutput, source_recomputation
    from merlin.perf import (
        address_loop_control,
        component_cost,
        constant_control_cfg,
        host_cfg_activity,
        host_cfg_index,
        offload_profitability,
        term,
    )
    from merlin.targetgen.oot_starterkit import llvm_context
    from merlin.transforms import recomputation
    from merlin.verify import host_call_alignment
    from merlin.xdsl_dialects.lowering import integer_constant_eval

    permissions = recomputation.RecomputationPermissions(
        "original-total-integer-host-source-test-premises-v1",
        repeat_evaluation_no_faults=True,
        repeat_evaluation_unobserved=True,
        suppressed_materialization_no_faults=True,
        suppressed_materialization_unobserved=True,
        inputs_stable_during_region=True,
    )
    budgets = recomputation.RecomputationBudgets(
        max_candidates=4,
        max_source_elements=4096,
        max_repeats=2,
        max_producer_scalar_ops=8,
        max_added_scalar_evaluations=4096,
        max_constructed_operations=2048,
        max_scalar_ir_bytes=65536,
    )
    # These original premises concern this pure tensor source and standard
    # value semantics. They confer no repeatability on unknown native calls.
    paths = tuple(
        dict.fromkeys(
            (
                *tools(),
                Path(sys.executable).resolve(),
                codegen.mlir_runtime_c(),
                *(
                    Path(module.__file__).resolve()
                    for module in (
                        abi,
                        codegen,
                        lower,
                        toolchain,
                        source_multioutput,
                        source_recomputation,
                        component_cost,
                        offload_profitability,
                        term,
                        host_call_alignment,
                        host_cfg_activity,
                        host_cfg_index,
                        constant_control_cfg,
                        address_loop_control,
                        llvm_context,
                        integer_constant_eval,
                    )
                ),
                *(
                    merlin_dir() / "tests/infra" / name
                    for name in (
                        "test_source_recomputation_model.py",
                        "test_recomputation_llvm_payload.py",
                        "test_multi_output_source.py",
                        "test_dense_source_catalog.py",
                    )
                ),
                *recomputation.implementation_paths(),
            )
        )
    )
    pins = {str(path): sha256_file(path) for path in paths}
    context = PipelineContext(
        hardware=digest(machine_scope()),
        numerical_policy=digest(dict(kind="fixed-width-modular-integer", bits=case.width * 8)),
        resources=digest(asdict(budgets)),
        execution=digest(
            dict(
                options=OPTIONS,
                permissions=asdict(permissions),
                machine=machine_scope(),
                runtime=sha256_file(codegen.mlir_runtime_c()),
                abi=sha256_file(abi.__file__),
            )
        ),
        library=digest(pins),
    )
    module = source(case)
    plan = recomputation.plan_source_recomputation(module, context, permissions, budgets)
    assert plan is not None
    return plan, pins


def recheck(pins):
    for name, expected in pins.items():
        path = Path(name)
        assert not path.is_symlink() and path.is_file() and sha256_file(path) == expected


def observe_call(native, arguments, expected, *, timer=False):
    """All caller allocation/reset/validation is outside the optional bracket."""
    owners, views = [], []
    for value in (*arguments, *(np.full_like(v, 83) for v in expected)):
        owner = np.full(value.size + 32, 91, value.dtype)
        view = owner[16:-16].reshape(value.shape)
        view[...] = value
        owners.append(owner)
        views.append(view)
    before = [owner.tobytes() for owner in owners[: len(arguments)]]
    pointers = [(view.ctypes.data, view.shape) for view in views]
    start = time.perf_counter_ns() if timer else None
    native(pointers)
    elapsed = time.perf_counter_ns() - start if timer else None
    outputs = views[len(arguments) :]
    for actual, want in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(actual, want)
    assert [owner.tobytes() for owner in owners[: len(arguments)]] == before
    assert all(np.all(owner[:16] == 91) and np.all(owner[-16:] == 91) for owner in owners)
    return dict(ns=elapsed, outputs=[a.tolist() for a in outputs]), owners, tuple(a.copy() for a in outputs)


def check_saved_owners(path, arguments, expected):
    with np.load(path) as saved:
        assert set(saved.files) == {f"owner{i}" for i in range(len(arguments) + len(expected))}
        for i, value in enumerate((*arguments, *expected)):
            owner = saved[f"owner{i}"]
            assert owner.dtype == value.dtype and owner.shape == (value.size + 32,)
            assert np.all(owner[:16] == 91) and np.all(owner[-16:] == 91)
            np.testing.assert_array_equal(owner[16:-16].reshape(value.shape), value)


def verify_invocations(work):
    receipts = tuple(work.rglob("invocations/*/invocation.json"))
    assert receipts
    for receipt in receipts:
        invocation.verify(receipt)
    return receipts


def join_host_products(result):
    """Join actual LLVM -> model object -> ordinary linked image invocations."""
    receipts = verify_invocations(result.workdir)
    documents = [json.loads(path.read_text()) for path in receipts]
    image, llvm = Path(result.host_so).resolve(), Path(result.ll_path).resolve()
    links = [row for row in documents if any(p["path"] == str(image) for p in row.get("outputs", ()))]
    assert len(links) == 1
    objects = {p["path"]: p["sha256"] for p in links[0]["inputs"]}
    producers = [
        row
        for row in documents
        if any(p["path"] == str(llvm) for p in row.get("inputs", ()))
        and any(p["path"] in objects for p in row.get("outputs", ()))
    ]
    assert len(producers) == 1
    for output in producers[0]["outputs"]:
        if output["path"] in objects:
            assert output["sha256"] == objects[output["path"]] == sha256_file(output["path"])
    return receipts


def allocations(path):
    """Actual consumed model IR, not a selected plan or diagnostic C body."""
    lines = path.read_text().splitlines()
    return [
        dict(function=name, requested_bytes=_malloc_size(line.strip()))
        for name, start, end in _defines(lines)
        for line in lines[start : end + 1]
        if _call_target(line.strip()) == "malloc"
    ]


def terms(rows, evidence):
    durations = [row["ns"] for row in rows]
    validity = Validity(
        "only these observed warmed complete native HostModel calls with identical reset state and ctypes ABI",
        expected_error="observed envelope only; no bound on future invocations, service latency or physical targets",
        escalate_when="source, executable, runtime, settings, input/state, timer, cache or numerical policy changes",
    )
    provenance = Provenance("measured", (evidence,))
    return dict(
        cold=PerformanceTerm.unknown("complete_cold_call", "ns", provenance, validity, "cold execution unmeasured"),
        warm=PerformanceTerm(
            "observed_complete_warm_call",
            sum(durations) / len(durations),
            "ns",
            provenance,
            validity,
            Bounds(min(durations), max(durations)),
        ),
    )


def query_identity(arguments):
    return digest(
        {
            **{key: asdict(arguments[key]) for key in ("original", "candidate", "scope", "stamp")},
            **{key: arguments[key] for key in ("regime", "original_selection_sha256", "candidate_selection_sha256")},
            **{
                key: {regime: value.to_dict() for regime, value in arguments[key].items()}
                for key in ("original_costs", "candidate_costs")
            },
        }
    )


@pytest.mark.parametrize("damage", ("readonly_input", "output_guard", "output_value"))
def test_retained_samples_check_complete_values_inputs_and_guards(tmp_path, damage):
    arguments, expected, _, _ = reference(CASES[0])
    values = (*arguments, *expected[0])
    owners = [np.full(value.size + 32, 91, value.dtype) for value in values]
    for owner, value in zip(owners, values, strict=True):
        owner[16:-16] = value.reshape(-1)
    path = tmp_path / "sample.npz"

    def save():
        np.savez(path, **{f"owner{i}": value for i, value in enumerate(owners)})

    save()
    check_saved_owners(path, arguments, expected[0])
    index, offset = {"readonly_input": (0, 16), "output_guard": (5, 0), "output_value": (5, 16)}[damage]
    owners[index][offset] ^= 1
    save()
    with pytest.raises(AssertionError):
        check_saved_owners(path, arguments, expected[0])


@pytest.mark.parametrize("case", CASES)
def test_source_recomputation_model_plan_has_two_complete_outputs(case):
    if not toolchain.available():
        pytest.skip("original ordinary host compiler required for context binding")
    plan, pins = original_plan(case)
    assert ir_digest(plan.source) != ir_digest(plan.candidate)
    assert plan.baseline_selection_sha256 != plan.candidate_selection_sha256
    before = [operation for operation in plan.source.walk() if operation.name == "linalg.generic"]
    after = [operation for operation in plan.candidate.walk() if operation.name == "linalg.generic"]
    assert len(before) == 4 and len(after) == 3
    assert len(tuple(before[0].results[0].uses)) == 2
    checked = plan.record()["conformance"]
    elements = int(np.prod(case.shape))
    assert checked["before_producer_evaluations"] == elements
    assert checked["after_producer_evaluations"] == 2 * elements
    assert checked["added_scalar_evaluations"] == 2 * elements
    assert checked["logical_producer_bytes_removed"] == elements * case.width
    plan.revalidate()
    recheck(pins)


@pytest.mark.parametrize("case", CASES)
def test_observed_complete_model_cost_controls_ordinary_source_recomputation(tmp_path, case):
    from test_dense_source_catalog import compilation_memory

    from merlin.llvmlower.source_recomputation import SourceRecomputationSelection

    if not toolchain.available():
        pytest.skip("original ordinary host compiler required")
    plan, original_pins = original_plan(case)
    original_machine = machine_scope()
    original_record = plan.record()
    arguments, expected, _, overflow = reference(case)
    assert all(count > 0 for count in overflow)
    source_path = tmp_path / "original.mlir"
    source_path.write_text(ir_text(plan.source))
    candidate_path = tmp_path / "candidate.mlir"
    candidate_path.write_text(ir_text(plan.candidate))
    input_path = tmp_path / "inputs.npz"
    np.savez(input_path, **{f"in{i}": value for i, value in enumerate(arguments)})
    original_pins.update({str(path): sha256_file(path) for path in (source_path, candidate_path, input_path)})
    original_settings = dict(
        targets=("host",),
        textual=False,
        vectorize=False,
        transform_schedule=None,
        hoist_static_allocs=True,
        parallel=False,
        features=None,
        parallel_harts=None,
        parallel_chunks=None,
        static_arena=None,
        data_layout=None,
        index_bits=None,
        op_profile=False,
        borrowed_buffers_selected=False,
        shared_source_lowerings_selected=False,
    )
    boundary_rows, arms = [], {}
    owner_live = True
    expected_comparison = None

    def verify_current(*, stage, source, plan: dict, settings, comparison, result):
        if not owner_live:
            raise ValueError("original complete model owner expired")
        assert stage in ("entry", "return") and (result is None) == (stage == "entry")
        assert ir_digest(source) == original_plan_source_sha
        assert plan == original_record and settings == original_settings
        assert comparison == expected_comparison
        assert machine_scope() == original_machine
        recheck(original_pins)
        if result is not None:
            join_host_products(result)
            report = json.loads((Path(result.workdir) / "source-recomputation.json").read_text())
            assert report == result.stats["source_recomputation"]
            assert report["plan"] == original_record and report["comparison"] == comparison
            recheck({row["path"]: row["sha256"] for row in report["implementation_sources"]})
            selected_path = Path(result.workdir) / "source-recomputation-selected.mlir"
            chosen = comparison is not None and comparison["choose_candidate"]
            assert ir_digest(selected_path.read_text()) == ir_digest(
                candidate_path.read_text() if chosen else source_path.read_text()
            )
            if comparison is not None:
                observed = arms["candidate" if chosen else "original"]
                if sha256_file(result.host_so) != observed["native"].image_sha256:
                    raise ValueError("rebuilt product differs: no observed timing transfer")
        boundary_rows.append(
            dict(
                stage=stage,
                comparison_selected=comparison is not None,
                result=None if result is None else str(result.host_so),
            )
        )

    original_plan_source_sha = plan.stamp.source_sha256

    def build(name, path, selection=None):
        started = time.perf_counter()
        with compilation_memory() as memory:
            result = lower.lower_model_file(path, tmp_path / name, **OPTIONS, source_recomputation=selection)
        elapsed = time.perf_counter() - started
        receipts = join_host_products(result)
        native = HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.workdir.resolve()))
        assert native.image_sha256 == sha256_file(result.host_so)
        candidate = name == "candidate" or name == "selected" and expected_comparison["choose_candidate"]
        inspect_started = time.perf_counter()
        buffer_activity = inspect_buffer_activity(
            result, case, candidate=candidate, diagnostics=tmp_path / (name + "-llvm-observer")
        )
        inspection_seconds = time.perf_counter() - inspect_started
        activity_path = tmp_path / (name + "-buffer-activity.json")
        activity_path.write_text(json.dumps(buffer_activity, indent=2))
        pins = {
            str(p.resolve()): sha256_file(p)
            for p in (
                result.ll_path,
                result.host_so,
                *result.workdir.glob("*.o"),
                *receipts,
                *result.workdir.glob("source-recomputation*"),
                activity_path,
            )
        }
        for kind in ("inputs", "tools", "products"):
            for path, checksum in buffer_activity["import_observation"][kind].items():
                assert path not in pins or pins[path] == checksum
                pins[path] = checksum
        history, state, position = [], arguments[3].copy(), arguments[4].copy()
        for epoch in range(3):
            actual_inputs = (*arguments[:3], state, position)
            row, owners, outputs = observe_call(native, actual_inputs, expected[epoch])
            artifact = tmp_path / f"{name}-epoch{epoch}.npz"
            np.savez(artifact, **{f"owner{i}": value for i, value in enumerate(owners)})
            pins[str(artifact)] = sha256_file(artifact)
            history.append(dict(**row, arrays=str(artifact), sha256=sha256_file(artifact)))
            state, position = outputs[2], outputs[3]
        return dict(
            result=result,
            native=native,
            loaded_image_sha256=native.image_sha256,
            image_custody_scope="PrivateHostImagePolicy freezes and hashes an owned copy for dlopen, then unlinks "
            "that copy; its recorded digest matches this retained ordinary linked image",
            pins=pins,
            history=history,
            compile_seconds=elapsed,
            compilation_memory=memory,
            inspection_seconds=inspection_seconds,
            buffer_activity=dict(path=str(activity_path), sha256=sha256_file(activity_path)),
            allocation_sites=allocations(result.ll_path),
            image_bytes=result.host_so.stat().st_size,
            sections=elf_sections(result.host_so),
        )

    # This is an actual missing-evidence ordinary build, not a forced baseline.
    missing = SourceRecomputationSelection(plan, verify_current=verify_current)
    arms["original"] = build("original", source_path, missing)
    missing_report = json.loads((arms["original"]["result"].workdir / "source-recomputation.json").read_text())
    assert missing_report["selected"] is False and missing_report["status"] == "MISSING_COST"
    # A separately checked source is compiled only to observe a legal alternative.
    arms["candidate"] = build("candidate", candidate_path)
    before, after = (arms[name]["allocation_sites"] for name in ("original", "candidate"))
    assert before == after == []  # The baseline already reuses a final-output buffer for P.
    before_activity, after_activity = (
        json.loads(Path(arms[name]["buffer_activity"]["path"]).read_text())["activity"]
        for name in ("original", "candidate")
    )
    nbytes = int(np.prod(case.shape)) * case.width
    assert before_activity["store_payload_bytes"] - after_activity["store_payload_bytes"] == nbytes
    assert after_activity["load_payload_bytes"] - before_activity["load_payload_bytes"] == nbytes
    assert sum(before_activity[key] for key in ("load_payload_bytes", "store_payload_bytes")) == sum(
        after_activity[key] for key in ("load_payload_bytes", "store_payload_bytes")
    )
    roster, _ = descriptor_roster(case)
    state_root = f"arg:{roster['state']['aligned']}"
    state_before, state_after = (row["buffer_payload"][state_root] for row in (before_activity, after_activity))
    added_input_reads = sum(
        after_activity["buffer_payload"][f"arg:{roster[role]['aligned']}"]["load_payload_bytes"]
        - before_activity["buffer_payload"][f"arg:{roster[role]['aligned']}"]["load_payload_bytes"]
        for role in ("input", "weight0", "prior_state")
    )
    materialization_tradeoff = dict(
        removed_P_state_buffer_store_bytes=state_before["store_payload_bytes"] - state_after["store_payload_bytes"],
        removed_P_state_buffer_load_bytes=state_before["load_payload_bytes"] - state_after["load_payload_bytes"],
        added_original_input_load_bytes=added_input_reads,
        net_direct_scalar_payload_reduction_bytes=0,
        allocation_sites_before=0,
        allocation_sites_after=0,
        scope="one invocation of submitted scalar LLVM under proven constant-loop counts; five declared echo "
        "memcpy intrinsics retained separately; no aggregate traffic or machine memory claim",
    )
    for name, arm in arms.items():
        observe_call(arm["native"], arguments, expected[0])
    samples = {name: [] for name in arms}
    order = []
    for pair in range(8):
        current_order = ("original", "candidate") if pair % 2 == 0 else ("candidate", "original")
        order.append(current_order)
        for name in current_order:
            row, owners, _ = observe_call(arms[name]["native"], arguments, expected[0], timer=True)
            raw_path = tmp_path / f"sample-{pair}-{name}.npz"
            np.savez(raw_path, **{f"owner{i}": value for i, value in enumerate(owners)})
            row.update(arrays=str(raw_path), sha256=sha256_file(raw_path))
            samples[name].append(row)
    metadata = dict(
        machine=original_machine,
        order=order,
        samples=samples,
        scope="warmed complete synchronous native HostModel call including ctypes descriptors, internal allocation, "
        "host arithmetic, output publication, copies and cleanup; outputs ready on return",
        excluded="compilation, image loading, caller allocation/reset, output observation and correctness checking",
        cache_policy="three prior correctness/state-history calls per image, then one explicit fixed-reset warmup; "
        "fresh guarded caller arrays; identical initial state and position for measured calls; "
        "no LLC flush, CPU frequency control or physical target qualification",
        cold="UNKNOWN",
        future_latency_bounds="UNKNOWN",
    )
    sample_path = tmp_path / "warm_samples.json"
    sample_path.write_text(json.dumps(metadata, indent=2))
    sample_bytes, sample_sha = sample_path.read_bytes(), sha256_file(sample_path)
    pins = {
        **original_pins,
        str(sample_path): sample_sha,
        **{p: h for arm in arms.values() for p, h in arm["pins"].items()},
        **{row["arrays"]: row["sha256"] for rows in samples.values() for row in rows},
    }
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = Path(line.split()[-1])
        if path.is_absolute() and path.is_file() and ".so" in path.name and tmp_path not in path.parents:
            pins[str(path.resolve())] = sha256_file(path)
    costs = {name: terms(rows, sample_sha) for name, rows in samples.items()}
    scope = ComponentCostScope(
        digest(metadata),
        plan.stamp.context.numerical_policy,
        digest(dict(inputs=sha256_file(input_path), cache=metadata["cache_policy"])),
    )
    observations = {}
    for name, arm in arms.items():
        artifacts = tuple(arm["pins"].items())
        observations[name] = ComponentFeatureObservation(
            digest(original_machine["tools"]),
            sha256_file(source_path),
            sha256_file(input_path),
            plan.stamp.context.hardware,
            scope.sha256,
            digest(dict(arm=name, metadata=metadata)),
            arm["native"].image_sha256,
            digest(arm["pins"]),
            sha256_file(input_path),
            (sample_sha,),
            (
                ComponentCostRegion(
                    "unmeasured", COMPLETE_STAGES, ("unmeasured",), {"status": "UNKNOWN"}, accounting="inclusive"
                ),
            ),
            (
                ComponentCostRegion(
                    "whole_call",
                    COMPLETE_STAGES,
                    ("observed_call_ns",),
                    {"samples": sample_sha},
                    accounting="inclusive",
                ),
            ),
            "PASS",
            artifacts,
            "PASS",
        )
    request = dict(
        original=observations["original"],
        candidate=observations["candidate"],
        original_costs=costs["original"],
        candidate_costs=costs["candidate"],
        scope=scope,
        regime="warm",
        stamp=plan.stamp,
        original_selection_sha256=plan.baseline_selection_sha256,
        candidate_selection_sha256=plan.candidate_selection_sha256,
    )
    frozen_query = query_identity(request)

    def verify_original(**current):
        if not owner_live:
            raise ValueError("original complete model owner expired")
        if query_identity(current) != frozen_query or machine_scope() != original_machine:
            raise ValueError("original complete model measurement scope changed")
        recheck(pins)
        plan.revalidate()
        assert plan.record() == original_record
        with np.load(input_path) as saved:
            independent_args, independent, _, _ = reference(case)
            for i, value in enumerate(independent_args):
                np.testing.assert_array_equal(saved[f"in{i}"], value)
        reopened = json.loads(sample_path.read_text())["samples"]
        for name in ("original", "candidate"):
            assert arms[name]["native"].image_sha256 == sha256_file(arms[name]["result"].host_so)
            join_host_products(arms[name]["result"])
            assert {k: v.to_dict() for k, v in terms(reopened[name], sample_sha).items()} == {
                k: v.to_dict() for k, v in current[name + "_costs"].items()
            }
            assert all(row["outputs"] == [a.tolist() for a in independent[0]] for row in reopened[name])
            for row in reopened[name]:
                check_saved_owners(row["arrays"], independent_args, independent[0])

    comparison = compare_complete_offload_cost(**request, verify_original=verify_original)
    expected_comparison = comparison.record()
    original_term, candidate_term = costs["original"]["warm"], costs["candidate"]["warm"]
    assert expected_comparison["choose_candidate"] == (candidate_term.bounds.upper < original_term.bounds.lower)
    assert expected_comparison["unit"] == "ns"
    assert expected_comparison["costs"]["original"]["cold"]["value"] == "UNKNOWN"
    assert expected_comparison["costs"]["candidate"]["cold"]["value"] == "UNKNOWN"
    damaged = json.loads(sample_bytes)
    damaged["samples"]["candidate"][0]["ns"] += 1
    sample_path.write_text(json.dumps(damaged))
    with pytest.raises(AssertionError):
        comparison.revalidate()
    sample_path.write_bytes(sample_bytes)
    comparison.revalidate()
    selection = SourceRecomputationSelection(plan, verify_current=verify_current, comparison=comparison)
    transfer = dict(
        status="EXACT_OBSERVED_IMAGE", observed_arm="candidate" if comparison.choose_candidate else "original"
    )
    try:
        selected = build("selected", source_path, selection)
    except ValueError as error:
        if str(error) != "rebuilt product differs: no observed timing transfer":
            raise
        selected = None
        transfer = dict(status="REFUSAL", reason=str(error))
    if selected is not None:
        arms["selected"] = selected
        selected_report = json.loads((selected["result"].workdir / "source-recomputation.json").read_text())
        assert selected_report["selected"] is comparison.choose_candidate
        assert selected_report["comparison"] == expected_comparison
    owner_live = False
    with pytest.raises(ValueError, match="owner expired"):
        comparison.revalidate()
    owner_live = True
    comparison.revalidate()
    # Obtain all owner-derived records before the final callback-free file close.
    final_comparison = comparison.record()
    final_plan = plan.record()
    for arm in arms.values():
        join_host_products(arm["result"])
    recheck(pins)
    for arm in arms.values():
        recheck(arm["pins"])
    (tmp_path / "pair.json").write_text(
        json.dumps(
            dict(
                schema="merlin.source_recomputation_native_pair.v1",
                plan=final_plan,
                comparison=final_comparison,
                transfer=transfer,
                materialization_tradeoff=materialization_tradeoff,
                original_missing_cost=missing_report,
                boundaries=boundary_rows,
                warm_samples=dict(path=str(sample_path), sha256=sample_sha),
                pins=pins,
                arms={
                    name: {key: value for key, value in arm.items() if key not in ("native", "result")}
                    | {
                        "llvm": str(arm["result"].ll_path),
                        "image": str(arm["result"].host_so),
                    }
                    for name, arm in arms.items()
                },
                scope="ordinary neutral native source recomputation and observed warmed ns only; "
                "no target, future-latency "
                "or Phase 2 qualification; allocation census is static final LLVM, not dynamic allocator traffic",
                requested_transfer_bytes="UNKNOWN",
                total_host_traffic_bytes="UNKNOWN",
            ),
            indent=2,
        )
    )
