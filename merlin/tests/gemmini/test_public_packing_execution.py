"""Opt-in actual public-compiler packing pair; absent inputs grant no qualification.

Run in a fresh pytest process with MERLIN_TEST_GEMMINI_PACKING_SELECTION naming
an original, pinned selection JSON. The live coordinator really reruns its
issuers. No saved approval or protected compiler/reference is restored.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from types import FunctionType

import numpy as np
import pytest
import yaml
from packing_execution_inputs import (
    PUBLIC_COMPILER_SHA,
    actual_observation,
    committed_custom_words,
    initial_state,
    materialize,
    original_case,
    pin,
    public_case,
    read_selection,
    recheck_artifacts,
    recheck_selection,
    verify_event_bytes,
)

from merlin.common.digest import sha256_file
from merlin.compile.model_execution_inputs import strict_tree_sha256

pytestmark = pytest.mark.target("gemmini")
SELECTION_ENV = "MERLIN_TEST_GEMMINI_PACKING_SELECTION"


def _trace_contract(quiet, trace):
    """Permit only the independently selected diagnostic trace resource variant."""
    expected = copy.deepcopy(quiet)
    engine = expected["runner"]["chipyard_rocc"]["engines"]["spike"]
    selected = trace["runner"]["chipyard_rocc"]["engines"]["spike"]
    assert selected["argv"] == ["-l", "--log-commits", *engine["argv"]]
    assert type(selected["max_console_bytes"]) is int and 0 < selected["max_console_bytes"] <= 32 << 20
    engine["argv"] = selected["argv"]
    engine["max_console_bytes"] = selected["max_console_bytes"]
    assert trace == expected, "trace selection must preserve every non-trace contract member"


def test_original_public_compiler_executes_connected_packing_pair(tmp_path, monkeypatch):
    run_public_pair(tmp_path, monkeypatch)


def run_public_pair(tmp_path, monkeypatch, *, case="packing", descriptor_effects=None):
    """Execute a fixed public case through the same original issuer/build gates."""
    specification = public_case(case)
    selected = os.environ.get(SELECTION_ENV)
    if not selected:
        pytest.skip(f"set {SELECTION_ENV}; no compiler, target or numerical qualification established")
    destination_case = case == "destination_forwarding"
    if destination_case and descriptor_effects is None:
        pytest.skip(
            "destination forwarding needs an original live kernel/runtime borrowed-effects owner: "
            "no retained aliases, no releases, completion before return; V26 descriptor identity is insufficient"
        )
    from merlin_experiments.phase1.feedback import private_instruction_coordinator as C
    from merlin_experiments.phase1.feedback import private_instruction_declaration as D

    from merlin.compile.baremetal_model import compile_saved_model, read_full_outputs
    from merlin.compile.model_execution_inputs import selected_execution_firrtl
    from merlin.compile.route_before_build import plan_before_build
    from merlin.llvmlower.optional_passes import Selection, applied
    from merlin.llvmlower.source_optimization_policy import SourceOptimizationPolicy
    from merlin.llvmlower.source_optimization_request import GATHER_PASS, PACKING_PASS
    from merlin.runtime.backends import base
    from merlin.runtime.input_observation import revalidate_observation
    from merlin.targetgen import target_registry
    from merlin.targetgen.elf_lanes import executable_sections

    candidate_policy = (
        None
        if destination_case
        else SourceOptimizationPolicy(
            producer_packing=case == "packing",
            padding_gather=case == "padding_gather",
            maximum_added_operations=4096,
            permitted_passes=(specification.pass_name,),
            original_maximum_added_operations=4096,
        )
    )
    original_policy = None if candidate_policy is None else candidate_policy.record()
    controls = {
        arm: Selection.of(
            enable=(specification.pass_name,) if destination_case and arm == "candidate" else (),
            disable=(PACKING_PASS, GATHER_PASS)
            + ((specification.pass_name,) if destination_case and arm == "baseline" else ()),
        )
        for arm in ("baseline", "candidate")
    }
    original_controls = {arm: value.spell() for arm, value in controls.items()}
    owner_pins, destination_library = [], None
    factory_function = factory_code = None
    if destination_case:
        from merlin.llvmlower import automatic_destinations, automatic_gathers

        # This case exercises the existing optional switch, not the two-choice
        # source policy. Freeze its actual algorithm/checker and parser owners
        # before original preparation callbacks; this is no restored grant.
        destination_library = {path.name: sha256_file(path) for path in automatic_destinations.implementation_paths()}
        owner_pins = [
            pin(path)
            for path in dict.fromkeys(
                (*automatic_destinations.implementation_paths(), *automatic_gathers.implementation_paths())
            )
        ]
        factory_function = getattr(descriptor_effects, "__func__", descriptor_effects)
        assert type(factory_function) is FunctionType, "an original source-bound effects factory is required"
        factory_code = factory_function.__code__
        owner_pins.append(pin(Path(inspect.getsourcefile(factory_function))))
    document, selected_pin = read_selection(selected)
    if document["trace_contract"] is None:
        pytest.skip("select an original trace contract; actual accelerator execution is unestablished")
    quiet_path = Path(document["quiet_contract"]["path"])
    quiet = yaml.safe_load(quiet_path.read_bytes())
    trace_path = Path(document["trace_contract"]["path"]) if document["trace_contract"] is not None else None
    trace = yaml.safe_load(trace_path.read_bytes()) if trace_path is not None else None
    if trace is not None:
        _trace_contract(quiet, trace)
    target = "gemmini"
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")  # Explicitly select no legacy support-provider code.
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(quiet_path))
    monkeypatch.setenv("MERLIN_RTL_FACTS", document["facts"]["path"])
    block = quiet["runner"]["chipyard_rocc"]
    monkeypatch.setenv("MERLIN_SPIKE", block["engines"]["spike"]["binary"]["path"])
    monkeypatch.setenv("MERLIN_RISCV_GCC", block["toolchain"]["compiler"]["path"])
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    source, reference, source_pins = original_case(monkeypatch, case=case)
    prepared = C.prepare(
        D.read(Path(document["instruction_declaration"]["path"]), target=target),
        target=target,
        output=tmp_path / "live-admission",
    )
    with target_registry.observed_contract(target, quiet, source_path=quiet_path):
        actual_facts = selected_execution_firrtl(document["facts"]["path"], target=target)
    # Both original caller roles explicitly select the same public facts for
    # this synthetic case. This creates no private corpus or prior approval.
    facts_binding = {"public_effective": actual_facts, "private_input": actual_facts}
    prepared.require_facts(facts_binding)
    pair = {}
    bindings, packages = [], []
    prepared_seed = None
    for arm in ("baseline", "candidate"):
        package = tmp_path / (arm + "-compiler")
        shutil.copytree(document["public_compiler"]["path"], package)
        packages.append(package)
        assert strict_tree_sha256(package) == {"sha256": PUBLIC_COMPILER_SHA, "n_files": 16}
        state = initial_state(case)
        calls = []
        for epoch in range(3):
            recheck_selection(document, selected_pin)
            prepared.require_facts(facts_binding)
            bound, arrays, expected = materialize(
                tmp_path / f"{arm}-inputs-{epoch}",
                epoch=epoch,
                state=state,
                source=source,
                reference=reference,
                source_pins=source_pins,
                case=case,
            )
            bindings.append(bound)
            input_hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in arrays]
            assert (None if candidate_policy is None else candidate_policy.record()) == original_policy
            assert {name: value.spell() for name, value in controls.items()} == original_controls
            recheck_artifacts(owner_pins)
            assert not destination_case or factory_function.__code__ is factory_code
            source_choices = candidate_policy if arm == "candidate" else None
            monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(quiet_path))
            with target_registry.observed_contract(target, quiet, source_path=quiet_path):
                plan = plan_before_build(
                    target,
                    source,
                    datapath="i8",
                    device_package=str(package),
                    capture=bound.root,
                    granularity="contraction",
                    linked_elf_admission=prepared.service,
                    source_kind="typed_static_integer_v1",
                )
                device = plan.get("device_routing")
                assert device is not None, plan.get("device_routing_why")
                assert device.linked_elf_admission is prepared.service and device.granularity == "contraction"
                if destination_case:
                    # Both arms use the same checked generated-shim identity.
                    device = replace(device, descriptor_results=True, descriptor_effects=descriptor_effects)
                started = time.perf_counter()
                # Packing/gather use the fixed original policy. Destination
                # forwarding uses its existing original optional-pass switch;
                # both mechanisms are engineering controls, not formal grants.
                with applied(controls[arm]):
                    result = compile_saved_model(
                        capture=bound.root,
                        package=document["host_package"]["path"],
                        board_catalog=document["board_catalog"]["path"],
                        board=document["board"],
                        dts=document["dts"]["path"],
                        output=tmp_path / "out" / f"{arm}-run-{epoch}",
                        target=target,
                        run="spike",
                        arena_mb=16,
                        timeout_s=300,
                        device=device,
                        materialized_inputs=bound,
                        readback="full",
                        tolerance=None,
                        respect_captured_quantization_scope=True,
                        host_llvm_bitcode=True,
                        # These fixed public tensor programs preserve every
                        # input during forward; recurrent updates are returned.
                        readonly_input_indices=tuple(range(len(arrays))),
                        **({"source_optimizations": source_choices} if source_choices is not None else {}),
                    )
                wall_seconds = time.perf_counter() - started
            assert result["status"] == "verified_complete_output"
            assert 0 < result["compilation"]["wall_seconds"] <= wall_seconds
            assert result["compilation"]["peak_memory_bytes"] is None
            elf = Path(result["output"]["elf"])
            elf_pin = pin(elf)
            assert elf_pin["sha256"] == result["output"]["elf_sha256"]
            lowering_input = pin(elf.parent / "model.prepared.mlir")
            if prepared_seed is None:
                prepared_seed = lowering_input
            assert lowering_input["sha256"] == prepared_seed["sha256"]
            choice = None
            if source_choices is not None:
                choice_path = elf.parent / "lower" / "source-optimization-selection.json"
                choice = {"artifact": pin(choice_path), "report": json.loads(choice_path.read_bytes())}
                producer = result["output"]["source_optimizations"]
                assert producer["schema"] == "merlin.source_optimization_build.v2"
                assert producer["policy"] == original_policy
                assert choice["report"]["selection"] == producer["selection"]
                assert producer["selection"]["request"]["source_sha256"] == lowering_input["sha256"]
                assert choice["report"]["binding_stage"] == "lower_model_input_before_source_optimizations"
            policy_path, policy_call = actual_observation(elf.parent, "device_final_linked_elf_policy")
            admission = json.loads(Path(policy_call["stdout"]["path"]).read_bytes())
            assert prepared.service.revalidate(elf=elf, result=admission, target=target) == "accepted"
            quiet_call_path, quiet_call = actual_observation(elf.parent, "selected_rocc_process")
            assert quiet_call["inputs"] == [elf_pin]
            observed_inputs = result["output"]["readonly_input_observation"]
            observation = observed_inputs["artifacts"]
            original_roster = tuple(range(len(arrays)))
            manifest = revalidate_observation(observation)
            assert tuple(row["original_index"] for row in manifest["arguments"]) == original_roster
            assert tuple(row["original_index"] for row in observed_inputs["arguments"]) == original_roster
            assert [row["sha256"] for row in observed_inputs["arguments"]] == input_hashes
            assert [row["bytes"] for row in observed_inputs["arguments"]] == [value.nbytes for value in arrays]
            outputs, _ = read_full_outputs(
                Path(quiet_call["stdout"]["path"]).read_bytes(),
                bound.root,
                result["output"]["build_hash"],
                readonly_input_observation=observation,
            )
            assert len(outputs) == len(specification.outputs)
            for value, independent in zip(outputs, expected, strict=True):
                assert value.shape == independent.shape and value.dtype == independent.dtype
                assert value.tobytes() == independent.tobytes()
            if destination_case and epoch == 0:
                # Actual wide output plus the original prior state must cross
                # a signed boundary; a reference-only edge case is insufficient.
                assert any(
                    not -(1 << 31) <= int(wide) + int(prior) < (1 << 31)
                    for wide, prior in zip(outputs[4].flat, arrays[-1].flat, strict=True)
                )
            actual_path = tmp_path / f"{arm}-actual-{epoch}.npz"
            output_names = tuple(name for name, _, _ in specification.outputs)
            np.savez(actual_path, **dict(zip(output_names, outputs, strict=True)))
            trace_record = None
            if trace is not None:
                before = set((elf.parent / "invocations").glob("*/invocation.json"))
                monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(trace_path))
                with target_registry.observed_contract(target, trace, source_path=trace_path):
                    backend = base.get_backend(target)
                    backend.verify_execution_inputs()
                    backend.run_elf(elf, simulator="spike", timeout=300, capture_bytes=True)
                    backend.verify_execution_inputs()
                trace_call_path, trace_call = actual_observation(elf.parent, "selected_rocc_process", before=before)
                assert trace_call["inputs"] == [elf_pin]
                accessor, decoder = prepared.check.accessor, prepared.check.selection
                constants = accessor.public_facts()["source_constants"]
                events = committed_custom_words(
                    Path(trace_call["stderr"]["path"]).read_bytes(),
                    match=constants[decoder.match_constant],
                    mask=constants[decoder.mask_constant],
                )
                verify_event_bytes(elf, events)
                decoded = accessor.decode_words(
                    tuple(event["word"] for event in events), tmp_path / f"{arm}-decode-{epoch}"
                )
                assert len(decoded["rows"]) == len(events)
                assert all(
                    row["word"] == event["word"] and row["length"] == 4
                    for row, event in zip(decoded["rows"], events, strict=True)
                )
                trace_record = {
                    "invocation": pin(trace_call_path),
                    "events": events,
                    "native_decoder": decoded,
                    "decoder_receipt": {"path": decoded["receipt_path"], "sha256": decoded["receipt_sha256"]},
                    "executed_custom_commands": len(events),
                }
            assert prepared.service.revalidate(elf=elf, result=admission, target=target) == "accepted"
            pipeline_path = elf.parent / "lower" / specification.report_name
            pipeline = json.loads(pipeline_path.read_bytes())
            assert (
                pipeline["schema"]
                == {
                    "packing": "merlin.automatic_host_permutations.v1",
                    "padding_gather": "merlin.automatic_host_gathers.v1",
                    "destination_forwarding": "merlin.automatic_host_destinations.v1",
                }[case]
            )
            assert pipeline["rewrites"] == int(arm == "candidate")
            if destination_library is not None:
                assert pipeline["library_sources_sha256"] == destination_library
                recheck_artifacts(owner_pins)
                assert not destination_case or factory_function.__code__ is factory_code
            host_llvm = Path(result["output"]["host_artifacts"]["llvm_ir"]["path"])
            assert sha256_file(host_llvm) == result["output"]["host_artifacts"]["llvm_ir"]["sha256"]
            mallocs = [line for line in host_llvm.read_text().splitlines() if "@malloc(" in line and "call " in line]
            abi = None
            if destination_case:
                from packing_execution_abi import descriptor_abi

                abi = descriptor_abi(elf)
            calls.append(
                {
                    "epoch": epoch,
                    "inputs_sha256": input_hashes,
                    "readonly_input_observation": observed_inputs,
                    "lowering_input": lowering_input,
                    "source_choice": choice,
                    "actual_outputs": pin(actual_path),
                    "outputs_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in outputs],
                    "elf": elf_pin,
                    "executable_section_bytes": sum(row[2] for row in executable_sections(elf.read_bytes())),
                    "host_llvm": pin(host_llvm),
                    "host_llvm_malloc_call_sites": mallocs,
                    "descriptor_abi": abi,
                    "policy_invocation": pin(policy_path),
                    "quiet_invocation": pin(quiet_call_path),
                    "pipeline": pipeline,
                    "pipeline_pin": pin(pipeline_path),
                    "trace": trace_record,
                    "combined_build_and_execution_wall_seconds": wall_seconds,
                    "compilation": result["compilation"],
                    "compilation_peak_memory": None,
                }
            )
            # The actual target result, never the expected array, carries state.
            state = outputs[0].copy()
            prepared.require_facts(facts_binding)
            bound.verify(bound.root)
            recheck_selection(document, selected_pin)
            assert source_pins == [pin(Path(row["path"])) for row in source_pins]
            assert strict_tree_sha256(package) == {"sha256": PUBLIC_COMPILER_SHA, "n_files": 16}
        pair[arm] = calls
    for baseline, candidate in zip(pair["baseline"], pair["candidate"], strict=True):
        assert baseline["inputs_sha256"] == candidate["inputs_sha256"]
        assert baseline["outputs_sha256"] == candidate["outputs_sha256"]
        assert baseline["pipeline"]["library_sources_sha256"] == candidate["pipeline"]["library_sources_sha256"]
        original_allocations = len(baseline["host_llvm_malloc_call_sites"])
        selected_allocations = len(candidate["host_llvm_malloc_call_sites"])
        if case == "packing":
            assert original_allocations == selected_allocations + 1
        else:
            assert original_allocations > selected_allocations
            assert [event["word"] for event in baseline["trace"]["events"]] == [
                event["word"] for event in candidate["trace"]["events"]
            ]
    # All callbacks finish before reopening the complete retained artifact set.
    prepared.require_facts(facts_binding)
    recheck_selection(document, selected_pin)
    assert (None if candidate_policy is None else candidate_policy.record()) == original_policy
    assert {name: value.spell() for name, value in controls.items()} == original_controls
    recheck_artifacts(owner_pins)
    assert not destination_case or factory_function.__code__ is factory_code
    for package in packages:
        assert strict_tree_sha256(package) == {"sha256": PUBLIC_COMPILER_SHA, "n_files": 16}
    for bound in bindings:
        bound.verify(bound.root)
    calls = [call for arm in pair.values() for call in arm]
    for call in calls:
        revalidate_observation(call["readonly_input_observation"]["artifacts"])
        if call["descriptor_abi"] is not None:
            from packing_execution_abi import recheck_descriptor_abi

            recheck_descriptor_abi(call)
    recheck_artifacts(
        [
            *source_pins,
            *[
                call[key]
                for call in calls
                for key in ("elf", "actual_outputs", "host_llvm", "pipeline_pin", "lowering_input")
            ],
            *[call["source_choice"]["artifact"] for call in calls if call["source_choice"] is not None],
            *[
                row
                for call in calls
                if call["descriptor_abi"] is not None
                for row in call["descriptor_abi"]["artifacts"]
            ],
        ],
        invocations=[call[key] for call in calls for key in ("policy_invocation", "quiet_invocation")]
        + [call["trace"]["invocation"] for call in calls],
        decoder_receipts=[call["trace"]["decoder_receipt"] for call in calls],
    )
    (tmp_path / "pair.json").write_text(
        json.dumps(
            {
                "arms": pair,
                "case": case,
                "pass_name": specification.pass_name,
                "original_source_policy": original_policy,
                "original_optional_controls": original_controls,
                "original_descriptor_results": destination_case,
                "original_source_owners": owner_pins,
                "selection": selected_pin,
                "source_and_reference": source_pins,
                "compiler_copies": [
                    {"path": str(package), "sha256": PUBLIC_COMPILER_SHA, "n_files": 16} for package in packages
                ],
                "scope": (
                    "public synthetic source; six fresh ordinary model builds, "
                    "exact complete outputs and external state carry"
                ),
                "unknown": [
                    "accelerator_latency",
                    "physical_correspondence",
                    "requested_transfer_bytes",
                    "formal_phase2_admission",
                    "cold_warm_timing",
                ],
            },
            indent=2,
        )
        + "\n"
    )
