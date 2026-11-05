"""Opt-in public source recomputation calibration through ordinary V26 builds.

This is an explicit legal-candidate trial, never a positive cost decision or a
Phase 2 authoring release. The functional simulator supplies no qualified cost.
"""

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import recomputation_execution_inputs as R
import recomputation_host_work as W
import yaml
from packing_execution_inputs import (
    PUBLIC_COMPILER_SHA,
    actual_observation,
    committed_custom_words,
    pin,
    read_selection,
    recheck_artifacts,
    recheck_selection,
    verify_event_bytes,
)
from recomputation_execution_inputs import PublicRecomputationCase, materialize, original_case
from test_public_packing_execution import _trace_contract

from merlin.compile.model_execution_inputs import strict_tree_sha256

pytestmark = pytest.mark.target("gemmini")
SELECTION_ENV = "MERLIN_TEST_GEMMINI_RECOMPUTATION_SELECTION"


def _all_pins(value):
    """Retain original producer artifact rows, without interpreting ISA names."""
    result = []
    if type(value) is dict:
        if set(value) >= {"path", "sha256"}:
            result.append({key: value[key] for key in ("path", "sha256")})
        for item in value.values():
            result.extend(_all_pins(item))
    elif type(value) in (tuple, list):
        for item in value:
            result.extend(_all_pins(item))
    return result


def _trace(elf, elf_pin, prepared, trace_path, trace, directory, monkeypatch):
    from merlin.runtime.backends import base
    from merlin.targetgen import target_registry

    before = set((elf.parent / "invocations").glob("*/invocation.json"))
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(trace_path))
    with target_registry.observed_contract("gemmini", trace, source_path=trace_path):
        backend = base.get_backend("gemmini")
        backend.verify_execution_inputs()
        backend.run_elf(elf, simulator="spike", timeout=300, capture_bytes=True)
        backend.verify_execution_inputs()
    path, record = actual_observation(elf.parent, "selected_rocc_process", before=before)
    assert record["inputs"] == [elf_pin]
    accessor, decoder = prepared.check.accessor, prepared.check.selection
    constants = accessor.public_facts()["source_constants"]
    events = committed_custom_words(
        Path(record["stderr"]["path"]).read_bytes(),
        match=constants[decoder.match_constant],
        mask=constants[decoder.mask_constant],
    )
    verify_event_bytes(elf, events)
    decoded = accessor.decode_words(tuple(event["word"] for event in events), directory)
    assert len(decoded["rows"]) == len(events)
    assert all(
        row["word"] == event["word"] and row["length"] == 4 for row, event in zip(decoded["rows"], events, strict=True)
    )
    return {
        "invocation": pin(path),
        "events": events,
        "decoder_receipt": {"path": decoded["receipt_path"], "sha256": decoded["receipt_sha256"]},
        "native_decoder": decoded,
        "executed_custom_commands": len(events),
    }


def _products(result, elf):
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.llvmlower.device_offload import SIDECAR_NAME

    recipe = pin(elf.parent / "compilation_recipe.json")
    verify_completed_recipe(Path(recipe["path"]), executable=elf, expected_recipe_sha256=recipe["sha256"])
    sidecar_path = elf.parent / SIDECAR_NAME
    sidecar = json.loads(sidecar_path.read_bytes())
    assert sidecar["granularity"] == "contraction" and len(sidecar["routed"]) == 2
    assert [(tuple(row["parallel"]), tuple(row["reduction"])) for row in sidecar["routed"]] == [
        ((2, 5), (5,)),
        ((2, 3), (5,)),
    ]
    assert all(tuple(row["dtypes"]) == ("i8", "i8", "i32") for row in sidecar["routed"])
    kernels = result["output"]["device_artifacts"]
    assert len(kernels["kernels"]) == 2
    assert set(kernels["kernels"]) == {row["symbol"] for row in sidecar["routed"]}
    host = result["output"]["host_artifacts"]
    paths = [
        elf.parent / "model.prepared.mlir",
        elf.parent / "device/device_shim.c",
        elf.parent / "device/device_shim.o",
        elf.parent / "lower/lowering_recipe.json",
        sidecar_path,
    ]
    pins = [pin(path) for path in paths] + [recipe, *_all_pins(host), *_all_pins(kernels)]
    recheck_artifacts(pins)
    return {
        "recipe": recipe,
        "artifacts": pins,
        "sidecar": sidecar,
        "device_artifacts": json.loads(json.dumps(kernels)),
        "host_artifacts": host,
    }


def test_original_public_compiler_executes_checked_source_recomputation(tmp_path, monkeypatch):
    selected = os.environ.get(SELECTION_ENV)
    if not selected:
        pytest.skip(f"set {SELECTION_ENV}; absent original selections confer no target qualification")
    document, selected_pin = read_selection(selected)
    if document["trace_contract"] is None:
        pytest.skip("select the original trace runtime; custom execution is unestablished")
    from merlin_experiments.phase1.feedback import private_instruction_coordinator as C
    from merlin_experiments.phase1.feedback import private_instruction_declaration as D

    from merlin.compile.baremetal_model import compile_saved_model, read_full_outputs
    from merlin.compile.model_execution_inputs import selected_execution_firrtl
    from merlin.compile.route_before_build import plan_before_build
    from merlin.llvmlower import compilation_recipe as CR
    from merlin.llvmlower import optional_passes
    from merlin.runtime.input_observation import revalidate_observation
    from merlin.targetgen import target_registry
    from merlin.targetgen.elf_lanes import executable_sections

    case = original_case()  # original plan and implementation pins before issuers
    case_check, case_code = case.revalidate, case.revalidate.__func__.__code__
    work_reader, work_reader_code = W.functions, W.functions.__code__
    observer_pins = tuple(pin(path) for path in W.observer_paths())
    ambient_controls = {
        key: os.environ.get(key)
        for key in (optional_passes.ENV, *(row.switch for row in optional_passes.REGISTRY if row.switch))
    }

    def helpers():
        return (
            materialize,
            _products,
            _trace,
            _all_pins,
            _trace_contract,
            actual_observation,
            committed_custom_words,
            pin,
            recheck_artifacts,
            recheck_selection,
            verify_event_bytes,
            PublicRecomputationCase.revalidate,
            PublicRecomputationCase.source_for,
            R._types,
            CR.verify_completed_recipe,
            *W.functions(),
        )

    helper_reader, helper_code = helpers, helpers.__code__
    frozen_helpers = tuple((function, function.__code__) for function in helpers())
    original_case_fields = (
        case.source,
        case.candidate,
        case.reference,
        case.plan,
        json.dumps(case.source_pins, sort_keys=True),
        case.callbacks,
        case.reference_abi,
    )

    def check():
        assert helpers is helper_reader and helper_reader.__code__ is helper_code
        assert W.functions is work_reader and work_reader.__code__ is work_reader_code
        assert tuple(function for function, _ in frozen_helpers) == helpers()
        assert all(function.__code__ is code for function, code in frozen_helpers)
        assert (
            case.source,
            case.candidate,
            case.reference,
            case.plan,
            json.dumps(case.source_pins, sort_keys=True),
            case.callbacks,
            case.reference_abi,
        ) == original_case_fields
        assert case.revalidate.__func__ is case_check.__func__ and case_check.__func__.__code__ is case_code
        case_check()
        recheck_artifacts(observer_pins)
        assert ambient_controls == {key: os.environ.get(key) for key in ambient_controls}
        recheck_selection(document, selected_pin)

    quiet_path, trace_path = Path(document["quiet_contract"]["path"]), Path(document["trace_contract"]["path"])
    quiet, trace = yaml.safe_load(quiet_path.read_bytes()), yaml.safe_load(trace_path.read_bytes())
    _trace_contract(quiet, trace)
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    monkeypatch.setenv("MERLIN_RTL_FACTS", document["facts"]["path"])
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(quiet_path))
    block = quiet["runner"]["chipyard_rocc"]
    monkeypatch.setenv("MERLIN_SPIKE", block["engines"]["spike"]["binary"]["path"])
    monkeypatch.setenv("MERLIN_RISCV_GCC", block["toolchain"]["compiler"]["path"])
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    prepared = C.prepare(
        D.read(Path(document["instruction_declaration"]["path"]), target="gemmini"),
        target="gemmini",
        output=tmp_path / "live-admission",
    )
    check()
    with target_registry.observed_contract("gemmini", quiet, source_path=quiet_path):
        facts = selected_execution_firrtl(document["facts"]["path"], target="gemmini")
    facts_binding = {"public_effective": facts, "private_input": facts}
    prepared.require_facts(facts_binding)
    pair, bindings, packages = {}, [], []
    for arm in ("baseline", "candidate"):
        package = tmp_path / f"{arm}-compiler"
        shutil.copytree(document["public_compiler"]["path"], package)
        packages.append(package)
        assert strict_tree_sha256(package) == {"sha256": PUBLIC_COMPILER_SHA, "n_files": 16}
        state, calls = case.reference.initial_state(), []
        for epoch in range(3):
            check()
            prepared.require_facts(facts_binding)
            bound, arrays, expected = materialize(
                tmp_path / f"{arm}-inputs-{epoch}", case=case, arm=arm, epoch=epoch, state=state
            )
            bindings.append(bound)
            input_pins = [pin(bound.root / name) for name in ("inputs.npz", "model.mlir", "goldens.npz")]
            monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(quiet_path))
            with target_registry.observed_contract("gemmini", quiet, source_path=quiet_path):
                placement = plan_before_build(
                    "gemmini",
                    case.source_for(arm),
                    datapath="i8",
                    device_package=str(package),
                    capture=bound.root,
                    granularity="contraction",
                    linked_elf_admission=prepared.service,
                    source_kind="typed_static_integer_v1",
                )
                device = placement.get("device_routing")
                assert device is not None, placement.get("device_routing_why")
                assert device.linked_elf_admission is prepared.service and device.granularity == "contraction"
                started = time.perf_counter()
                result = compile_saved_model(
                    capture=bound.root,
                    package=document["host_package"]["path"],
                    board_catalog=document["board_catalog"]["path"],
                    board=document["board"],
                    dts=document["dts"]["path"],
                    output=tmp_path / "out" / f"{arm}-run-{epoch}",
                    target="gemmini",
                    run="spike",
                    arena_mb=16,
                    timeout_s=300,
                    device=device,
                    materialized_inputs=bound,
                    readback="full",
                    tolerance=None,
                    respect_captured_quantization_scope=True,
                    host_llvm_bitcode=True,
                    readonly_input_indices=tuple(range(6)),
                )
                elapsed = time.perf_counter() - started
            check()
            assert result["status"] == "verified_complete_output"
            assert 0 < result["compilation"]["wall_seconds"] <= elapsed
            elf = Path(result["output"]["elf"])
            elf_pin = pin(elf)
            assert elf_pin["sha256"] == result["output"]["elf_sha256"]
            policy_path, policy = actual_observation(elf.parent, "device_final_linked_elf_policy")
            admission = json.loads(Path(policy["stdout"]["path"]).read_bytes())
            assert prepared.service.revalidate(elf=elf, result=admission, target="gemmini") == "accepted"
            quiet_call_path, quiet_call = actual_observation(elf.parent, "selected_rocc_process")
            assert quiet_call["inputs"] == [elf_pin]
            observation = result["output"]["readonly_input_observation"]
            manifest = revalidate_observation(observation["artifacts"])
            assert tuple(row["original_index"] for row in manifest["arguments"]) == tuple(range(6))
            assert tuple(row["original_index"] for row in observation["arguments"]) == tuple(range(6))
            assert [row["sha256"] for row in observation["arguments"]] == [
                hashlib.sha256(value.tobytes()).hexdigest() for value in arrays
            ]
            assert [row["bytes"] for row in observation["arguments"]] == [value.nbytes for value in arrays]
            outputs, _ = read_full_outputs(
                Path(quiet_call["stdout"]["path"]).read_bytes(),
                bound.root,
                result["output"]["build_hash"],
                readonly_input_observation=observation["artifacts"],
            )
            assert len(outputs) == 9
            assert all(
                value.shape == other.shape and value.dtype == other.dtype and value.tobytes() == other.tobytes()
                for value, other in zip(outputs, expected, strict=True)
            )
            actual = tmp_path / f"{arm}-actual-{epoch}.npz"
            np.savez(actual, **dict(zip((row[0] for row in case.reference.OUTPUTS), outputs, strict=True)))
            products = _products(result, elf)
            work = W.inspect_work(
                products["host_artifacts"],
                elf.parent / "model.prepared.mlir",
                bound.root,
                manifest,
                tuple(row["symbol"] for row in products["sidecar"]["routed"]),
                candidate=arm == "candidate",
                diagnostics=tmp_path / f"{arm}-host-work-{epoch}",
            )
            for name, digest in work["import_observation"]["tools"].items():
                assert {"path": name, "sha256": digest} in observer_pins
            work_path = tmp_path / f"{arm}-host-work-{epoch}.json"
            work_path.write_text(json.dumps(work, indent=2))
            products["artifacts"].extend(
                [
                    pin(work_path),
                    *(
                        {"path": path, "sha256": digest}
                        for kind in ("inputs", "tools", "products")
                        for path, digest in work["import_observation"][kind].items()
                    ),
                ]
            )
            traced = _trace(elf, elf_pin, prepared, trace_path, trace, tmp_path / f"{arm}-decode-{epoch}", monkeypatch)
            assert prepared.service.revalidate(elf=elf, result=admission, target="gemmini") == "accepted"
            check()
            calls.append(
                {
                    "epoch": epoch,
                    "input_artifacts": input_pins,
                    "actual_outputs": pin(actual),
                    "outputs_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in outputs],
                    "inputs_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in arrays],
                    "elf": elf_pin,
                    "executable_section_bytes": sum(row[2] for row in executable_sections(elf.read_bytes())),
                    "products": products,
                    "host_work": work,
                    "policy_invocation": pin(policy_path),
                    "quiet_invocation": pin(quiet_call_path),
                    "readonly_input_observation": observation,
                    "trace": traced,
                    "compilation": result["compilation"],
                    "combined_build_and_execution_wall_seconds": elapsed,
                }
            )
            state = outputs[0].copy()  # actual target result, never reference state
        pair[arm] = calls
    for baseline, candidate in zip(pair["baseline"], pair["candidate"], strict=True):
        assert baseline["inputs_sha256"] == candidate["inputs_sha256"]
        assert baseline["outputs_sha256"] == candidate["outputs_sha256"]
        assert [event["word"] for event in baseline["trace"]["events"]] == [
            event["word"] for event in candidate["trace"]["events"]
        ]
    final_plan = json.loads(json.dumps(case.plan.record()))
    prepared.require_facts(facts_binding)
    check()
    for bound in bindings:
        bound.verify(bound.root)
    for package in packages:
        assert strict_tree_sha256(package) == {"sha256": PUBLIC_COMPILER_SHA, "n_files": 16}
    calls = [call for arm in pair.values() for call in arm]
    for call in calls:
        revalidate_observation(call["readonly_input_observation"]["artifacts"])
        recipe = call["products"]["recipe"]
        CR.verify_completed_recipe(
            Path(recipe["path"]), executable=Path(call["elf"]["path"]), expected_recipe_sha256=recipe["sha256"]
        )
    # Recheck after every original callback has completed, including all earlier
    # images and source/library/data products. No old approval is restored.
    recheck_artifacts(
        [
            *case.source_pins,
            *observer_pins,
            *[
                row
                for call in calls
                for row in [
                    call["elf"],
                    call["actual_outputs"],
                    *call["input_artifacts"],
                    *call["products"]["artifacts"],
                ]
            ],
        ],
        invocations=[call[key] for call in calls for key in ("policy_invocation", "quiet_invocation")]
        + [call["trace"]["invocation"] for call in calls],
        decoder_receipts=[call["trace"]["decoder_receipt"] for call in calls],
    )
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "case": "source_recomputation_calibration",
                "arms": pair,
                "selection": selected_pin,
                "source_and_reference": case.source_pins,
                "plan": final_plan,
                "compiler_copies": [
                    dict(path=str(package), sha256=PUBLIC_COMPILER_SHA, n_files=16) for package in packages
                ],
                "optional_controls": ambient_controls,
                "automatic_source_selection": "not_selected; explicit source calibration",
                "cost": "UNKNOWN",
                "accelerator_latency": None,
                "physical_traffic_bytes": None,
                "scope": "six fresh ordinary public compiler model builds; all nine outputs, end-of-call input bytes "
                "and independently reconstructed submitted-LLVM producer work",
                "unknown": [
                    "target_profitability",
                    "formal_phase2_admission",
                    "physical_memory_traffic",
                    "peak_memory",
                ],
            },
            indent=2,
        )
        + "\n"
    )
