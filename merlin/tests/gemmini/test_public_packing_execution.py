"""Opt-in actual public-compiler packing pair; absent inputs grant no qualification.

Run in a fresh pytest process with MERLIN_TEST_GEMMINI_PACKING_SELECTION naming
an original, pinned selection JSON. The live coordinator really reruns its
issuers. No saved approval or protected compiler/reference is restored.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from packing_execution_inputs import (
    PUBLIC_COMPILER_SHA,
    actual_observation,
    committed_custom_words,
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
from merlin.common.paths import repo_root
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


def run_public_pair(tmp_path, monkeypatch, *, case="packing"):
    """Execute a fixed public case through the same original issuer/build gates."""
    specification = public_case(case)
    selected = os.environ.get(SELECTION_ENV)
    if not selected:
        pytest.skip(f"set {SELECTION_ENV}; no compiler, target or numerical qualification established")
    from merlin_experiments.phase1.feedback import private_instruction_coordinator as C
    from merlin_experiments.phase1.feedback import private_instruction_declaration as D

    from merlin.compile.baremetal_model import compile_saved_model, read_full_outputs
    from merlin.compile.model_execution_inputs import selected_execution_firrtl
    from merlin.compile.route_before_build import plan_before_build
    from merlin.llvmlower.optional_passes import Selection, applied
    from merlin.llvmlower.source_optimization_request import GATHER_PASS, PACKING_PASS, SourceOptimizationRequest
    from merlin.llvmlower.source_optimization_selection import SourceOptimizationSelection
    from merlin.runtime.backends import base
    from merlin.runtime.input_observation import revalidate_observation
    from merlin.targetgen import target_registry
    from merlin.targetgen.elf_lanes import executable_sections

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
        state = np.asarray([-129, -128, -1, 0, 1, 126, 127, 128] * 3, dtype=np.int32).reshape(2, 12)
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
            source_choices = None
            if arm == "candidate":
                assert prepared_seed is not None
                assert pin(Path(prepared_seed["path"])) == prepared_seed
                source_choices = SourceOptimizationSelection(
                    SourceOptimizationRequest(
                        source_sha256=prepared_seed["sha256"],
                        producer_packing=case == "packing",
                        padding_gather=case == "padding_gather",
                        maximum_added_operations=4096,
                    ),
                    permitted_passes=(specification.pass_name,),
                    maximum_added_operations=4096,
                )
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
                started = time.perf_counter()
                # An ambient enable cannot account for a candidate rewrite:
                # only its source-bound request enables the selected pass.
                with applied(Selection.of(disable=(PACKING_PASS, GATHER_PASS))):
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
                assert choice["report"]["selection"] == source_choices.record()
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
            assert len(outputs) == 3
            for value, independent in zip(outputs, expected, strict=True):
                assert value.shape == independent.shape and value.dtype == independent.dtype
                assert value.tobytes() == independent.tobytes()
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
                }[case]
            )
            assert pipeline["rewrites"] == int(arm == "candidate")
            host_llvm = Path(result["output"]["host_artifacts"]["llvm_ir"]["path"])
            assert sha256_file(host_llvm) == result["output"]["host_artifacts"]["llvm_ir"]["sha256"]
            mallocs = [line for line in host_llvm.read_text().splitlines() if "@malloc(" in line and "call " in line]
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
                    "policy_invocation": pin(policy_path),
                    "quiet_invocation": pin(quiet_call_path),
                    "pipeline": pipeline,
                    "pipeline_pin": pin(pipeline_path),
                    "trace": trace_record,
                    "combined_build_and_execution_wall_seconds": wall_seconds,
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
    for package in packages:
        assert strict_tree_sha256(package) == {"sha256": PUBLIC_COMPILER_SHA, "n_files": 16}
    for bound in bindings:
        bound.verify(bound.root)
    calls = [call for arm in pair.values() for call in arm]
    for call in calls:
        revalidate_observation(call["readonly_input_observation"]["artifacts"])
    recheck_artifacts(
        [
            *source_pins,
            *[
                call[key]
                for call in calls
                for key in ("elf", "actual_outputs", "host_llvm", "pipeline_pin", "lowering_input")
            ],
            *[call["source_choice"]["artifact"] for call in calls if call["source_choice"] is not None],
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


def test_public_materialization_validates_exact_source_abi_and_complete_reference(tmp_path, monkeypatch):
    source, reference, pins = original_case(monkeypatch)
    state = np.zeros((2, 12), dtype=np.int32)
    bound, arrays, expected = materialize(
        tmp_path / "materialized", epoch=0, state=state, source=source, reference=reference, source_pins=pins
    )
    assert len(arrays) == 4 and sum(value.size for value in expected) == 72
    assert len(bound.verify(bound.root)) == 8 and not (bound.root / "capture_receipt.json").exists()
    with pytest.raises(AssertionError):
        materialize(
            tmp_path / "wrong-source",
            epoch=0,
            state=state,
            source=source.replace("%sum = arith.addi", "%sum = arith.subi"),
            reference=reference,
            source_pins=pins,
        )
    (bound.root / "inputs.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        bound.verify(bound.root)


def test_commit_word_observation_refuses_disassembly_only_and_malformed_commits():
    with pytest.raises(AssertionError, match="no selected"):
        committed_custom_words(b"core 0: 0x8000 (0x0000000b) custom\n", match=11, mask=127)
    with pytest.raises(AssertionError, match="grammar"):
        committed_custom_words(b"core 0: 3 0x8000 bad\n", match=11, mask=127)
    assert committed_custom_words(b"core 0: 3 0x8000 (0x0000000b)\n", match=11, mask=127) == [
        {"hart": 0, "pc": 32768, "word": 11}
    ]


def test_retired_events_require_actual_executable_bytes(tmp_path, monkeypatch):
    # Reuse the existing synthetic ELF-format fixture; no target build or ISA decoder.
    monkeypatch.syspath_prepend(str(repo_root() / "merlin/tests/targetgen"))
    from test_elf_lane_contract import build_elf

    elf = tmp_path / "program.elf"
    elf.write_bytes(build_elf([(".text", 6, struct.pack("<I", 11)), (".data", 2, struct.pack("<I", 11))]))
    event = {"hart": 0, "pc": 0x80000040, "word": 11}
    verify_event_bytes(elf, [event])
    for changed in ({**event, "word": 12}, {**event, "pc": event["pc"] + 4}, {**event, "pc": event["pc"] + 1}):
        with pytest.raises(AssertionError):
            verify_event_bytes(elf, [changed])


def test_final_reopen_refuses_late_artifact_and_observed_output_changes(tmp_path):
    from merlin.common import invocation_record as I

    first = tmp_path / "earlier.elf"
    first.write_bytes(b"original")
    expected = pin(first)
    observed = tmp_path / "observed"
    I.run([sys.executable, "-c", "print('complete')"], directory=observed, stage="test_custody", capture_output=True)
    invocation = next((observed / "invocations").glob("*/invocation.json"))
    recheck_artifacts([expected], invocations=[pin(invocation)])
    first.write_bytes(b"modified")
    with pytest.raises(AssertionError, match="retained artifact"):
        recheck_artifacts([expected], invocations=[pin(invocation)])
    first.write_bytes(b"original")
    record = json.loads(invocation.read_bytes())
    Path(record["stdout"]["path"]).write_bytes(b"different")
    with pytest.raises(ValueError):
        recheck_artifacts([expected], invocations=[pin(invocation)])


def test_final_reopen_checks_native_decoder_receipt_contents(tmp_path):
    # Synthetic custody metadata exercises no accessor or admission authority.
    output = tmp_path / "decoded.txt"
    output.write_bytes(b"11 4")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"pins": [{"role": "native-invocation", **pin(output)}]}))
    expected = pin(receipt)
    recheck_artifacts([], decoder_receipts=[expected])
    output.write_bytes(b"12 4")
    with pytest.raises(AssertionError, match="native decoder artifact"):
        recheck_artifacts([], decoder_receipts=[expected])
    output.write_bytes(b"11 4")
    receipt.write_text(json.dumps({"pins": []}))
    with pytest.raises(AssertionError, match="native decoder receipt"):
        recheck_artifacts([], decoder_receipts=[expected])
