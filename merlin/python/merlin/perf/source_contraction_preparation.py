"""Prepare two exact short source-contraction programs; do not run either program.

Cached complete-model ownership proofs bind the selected source operation. They
do not establish target execution placement or equivalence of changed lowering.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from time import monotonic
from typing import Any, Mapping

from merlin.frontends.linalg_mlir import parse_mlir_text
from .compiler_plan_evidence import verify_compiler_global_plan
from .source_contraction_witness import extract_source_contraction, evaluate_source_contraction
from .source_program_pair import (bind_source_program_pair, document_digest,
                                  check_artifact_proof, program_plan, source_owners, text_digest)


def task_route_feedback(comparison):
    """Small agent-facing observation; class presence never grants equivalence."""
    status = comparison.get("status", "UNKNOWN")
    return {"status": status,
        "class_presence_changes": comparison.get("class_presence_changes"),
        "known_class_presence_equal_by_arm": comparison.get("known_class_presence_equal_by_arm"),
        "all_static_classes_decoded": comparison.get("all_static_classes_decoded", False),
        "reason": comparison.get("reason"),
        "next_action": ("The reduced probe does not reproduce the observed full-task instruction classes; retain its numerical/timing result only for that reduced program."
            if status == "observed_known_class_presence_mismatch" else
            "Resolve missing source/artifact-bound task observations before attributing this probe to a full-model lowering change."
            if status == "UNKNOWN" else
            "Known class presence is scoped evidence only; descriptor/address equivalence and global cost remain unresolved."),
        "descriptor_semantic_equivalence": "UNKNOWN", "emitted_address_equivalence": "UNKNOWN",
        "timing_calibration_admissible": False, "global_cost_validated": False}


def task_route_presence_for_preparation(*, candidate, experiment, pair, prepared):
    """Join existing full/short observations against independently bound artifacts.

    No model parsing, compilation or simulation. Missing/mismatched observations
    remain visible without revoking an independent reduced numerical result.
    """
    from .task_instruction_evidence import task_instruction_binding, target_instruction_facts
    from .task_route_presence import compare_task_route_presence
    full, short = {}, {}
    extraction = prepared.get("extraction", {})
    try:
        facts = target_instruction_facts(experiment.target)
        policy = experiment.host_policy["sha256"]
        current = experiment.iterations[-1]["analysis"]["diagnostics"]
        before = (current if prepared["comparison_arm"] == "optimization_baseline"
                  else experiment.iterations[-2]["analysis"]["diagnostics"])
        short_source = Path(prepared["workdir"]) / "interface.mlir"
        with short_source.open("rb") as stream:
            raw_source = stream.read(256*1024+1)
        if (len(raw_source) > 256*1024 or text_digest(raw_source.decode()) != extraction["probe_source_sha256"]
                or extraction["source_sha256"] != pair.source_sha256
                or prepared["source_op_index"] != extraction["source_op_index"]):
            raise ValueError("task comparison has stale source/extraction binding")
        for arm, diag, summary_key, proof_key in (
                ("before", before, "baseline" if prepared["comparison_arm"] == "optimization_baseline" else "candidate",
                 "verified_baseline_global_plan_emission" if prepared["comparison_arm"] == "optimization_baseline" else "verified_global_plan_emission"),
                ("after", current, "candidate", "verified_global_plan_emission")):
            artifact, proof = pair.artifacts[arm], diag[proof_key]
            check_artifact_proof(artifact, proof, pair.source_sha256, pair.graph_sha256)
            summary = diag["task_instruction_evidence"][summary_key]
            full[arm] = {"summary": summary, "summary_sha256": document_digest(summary),
                "expected_binding": task_instruction_binding(source_text=pair.source,
                    lowered_text=artifact["lowered_text"], command_buffer_text=artifact["command_buffer_text"],
                    verified_plan=proof, target_facts=facts, host_policy_sha256=policy)}
            item = prepared["arms"][arm]
            texts = {}
            for field, pin in (("lowered_path", "lowered_sha256"),
                               ("command_buffer_path", "command_buffer_canonical_sha256")):
                with Path(item[field]).open("rb") as stream:
                    data = stream.read(2_000_001)
                if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != item[pin]:
                    raise ValueError("short task artifact changed or exceeds the bounded input")
                texts[field] = data.decode()
            short_proof = item["source_task_cfg_proof"]
            cb = json.loads(texts["command_buffer_path"])
            expected = {"source_sha256": extraction["probe_source_sha256"],
                "candidate_sha256": artifact["compiler_sha256"],
                "candidate_lowered_sha256": item["lowered_sha256"],
                "candidate_command_buffer_sha256": item["command_buffer_canonical_sha256"],
                "plan_digest": document_digest(program_plan(cb))}
            if (any(short_proof.get(key) != value for key, value in expected.items())
                    or item["compiler_sha256"] != artifact["compiler_sha256"]):
                raise ValueError("short task proof differs from the selected compiler/artifacts")
            short[arm] = {"summary": item["task_instruction_evidence"],
                "summary_sha256": item["task_instruction_evidence_sha256"],
                "expected_binding": task_instruction_binding(source_text=raw_source.decode(),
                    lowered_text=texts["lowered_path"], command_buffer_text=texts["command_buffer_path"],
                    verified_plan=short_proof, target_facts=facts, host_policy_sha256=policy),
                "short_execution_admission": item.get("short_execution_admission")}
        return compare_task_route_presence(full=full, short=short, extraction=extraction,
            extraction_sha256=document_digest(extraction), short_source_op_index=prepared["probe_source_op_index"])
    except (ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError) as exc:
        result = compare_task_route_presence(full={}, short={}, extraction=extraction,
            extraction_sha256=document_digest(extraction), short_source_op_index=prepared.get("probe_source_op_index"))
        result["reason"] = "cached task comparison unavailable: " + str(exc)
        return result


def _short_source_index(text, entry):
    module = parse_mlir_text(text)
    functions = [op for op in module.body.block.ops if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1:
        raise ValueError("short source entry is not unique")
    selected = [index for index, op in enumerate(op for op in functions[0].body.block.ops if op.name != "func.return")
                if op.name in {"linalg.matmul", "linalg.batch_matmul", "linalg.generic"}]
    if len(selected) != 1:
        raise ValueError("short source must contain exactly one selected contraction")
    return selected[0]


def _typed_cases(probe, extraction, remaining):
    """Shared scalar oracle, deterministic typed boundary values, no target arithmetic."""
    import numpy as np

    cases = []
    for mode in ("signed_coordinate_pattern", "negative_extremes", "modular_overflow"):
        remaining()
        arrays = []
        for index, (shape, dtype) in enumerate(zip(extraction["input_shapes"], extraction["input_dtypes"], strict=True)):
            width = int(dtype[1:])
            if mode == "signed_coordinate_pattern":
                values = [((value*(37+index*12)+17) % 256)-128 for value in range(math.prod(shape))]
            else:
                value = -(1 << (width-1)) if mode == "negative_extremes" and index % 2 == 0 else (1 << (width-1))-1
                values = [value]*math.prod(shape)
            arrays.append(np.asarray(values, dtype="int"+str(width)).reshape(shape))
        expected = evaluate_source_contraction(probe, extraction, arrays)
        remaining()
        cases.append({"case": mode, "inputs": [array.tolist() for array in arrays], "expected": expected.tolist(),
            "input_le_sha256": [hashlib.sha256(array.astype(array.dtype.newbyteorder("<")).tobytes()).hexdigest()
                                for array in arrays],
            "expected_le_sha256": hashlib.sha256(expected.astype(expected.dtype.newbyteorder("<")).tobytes()).hexdigest()})
    return {"schema": "source_contraction_prepared_oracle_v1", "probe_source_sha256": text_digest(probe),
            "input_shapes": extraction["input_shapes"], "input_dtypes": extraction["input_dtypes"],
            "output_shape": extraction["output_shape"], "output_dtype": extraction["output_dtype"],
            "cases": cases, "semantics": extraction["arithmetic"],
            "candidate_numerical_qualification": "UNPROVEN"}


def prepare_source_contraction(*, candidate: Path, experiment, comparison_arm: str,
                               source_op_index: int, entry: str, max_m: int, max_n: int, max_k: int,
                               output: Path, max_macs: int = 100000, timeout_s: float = 60,
                               max_batch_extent: int = 2,
                               expected_route_transition: tuple[str, str] | None = None,
                               expected_short_route_transition: tuple[str | None, str | None] | None = None,
                               portfolio_member: Mapping[str, Any] | None = None,
                               max_source_bytes: int = 2_000_000) -> dict:
    """Prepare normal-entrypoint arms under one caller-charged <=60-second deadline.

    Selection is explicit and host-bound; no predecessor/baseline substitution,
    full-model recompilation, full-model proof rerun, runtime or candidate import.
    """
    if comparison_arm not in {"optimization_baseline", "previous"}:
        raise ValueError("select optimization_baseline or previous explicitly")
    if (not isinstance(entry, str) or not entry or type(source_op_index) is not int or source_op_index < 0
            or any(type(value) is not int or not 1 <= value <= 4096 for value in (max_m, max_n, max_k))
            or type(max_batch_extent) is not int or not 1 <= max_batch_extent <= 16
            or type(max_source_bytes) is not int or not 1 <= max_source_bytes <= 16_000_000
            or type(max_macs) is not int or not 1 <= max_macs <= 100000):
        raise ValueError("source contraction selection and bounds must be explicit and valid")
    if (expected_route_transition is not None
            and (type(expected_route_transition) is not tuple
                 or len(expected_route_transition) != 2
                 or any(route not in {"host", "contraction"}
                        for route in expected_route_transition)
                 or expected_route_transition[0] == expected_route_transition[1])):
        raise ValueError("source contraction route transition must name two distinct supported lanes")
    if (expected_short_route_transition is not None
            and (type(expected_short_route_transition) is not tuple
                 or len(expected_short_route_transition) != 2
                 or any(route not in {None, "host", "contraction"}
                        for route in expected_short_route_transition))):
        raise ValueError("short contraction routes must be supported lanes or explicit wildcards")
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
        raise ValueError("source contraction preparation requires a finite <=60s budget")
    started = monotonic()
    deadline = started+timeout_s
    record = {"schema": "source_contraction_preparation_v1", "status": "UNKNOWN",
        "comparison_arm": comparison_arm, "source_op_index": source_op_index, "entry": entry,
        "host_bounds": {"max_m": max_m, "max_n": max_n, "max_k": max_k,
                        "max_macs": max_macs, "max_batch_extent": max_batch_extent},
        "max_source_bytes": max_source_bytes,
        "expected_route_transition": list(expected_route_transition)
            if expected_route_transition is not None else ["contraction", "contraction"],
        "expected_short_route_transition": list(expected_short_route_transition)
            if expected_short_route_transition is not None else None,
        "arms": {}, "simulator_executed": False, "full_model_executed": False,
        "full_model_recompiled": False, "full_model_proofs_recomputed": False,
        "numerical_qualification": "UNPROVEN", "runtime_admitted": False, "global_speedup_proven": False,
        "emitted_route_correspondence": "UNKNOWN: requires target-bound selected-task lowering evidence",
        "selected_task_emission_change": "UNKNOWN: whole-artifact differences do not establish a selected-task change",
        "scope": "complete reduced source programs and independent oracle prepared, not executed"}
    work = None

    def remaining():
        left = deadline-monotonic()
        if left <= 0:
            raise TimeoutError("paired source contraction preparation exhausted its shared deadline")
        return left

    try:
        pair = bind_source_program_pair(candidate=candidate, experiment=experiment,
            comparison_arm=comparison_arm, portfolio_member=portfolio_member,
            max_source_bytes=max_source_bytes)
        initial_binding = document_digest(pair.comparison_binding)
        initial_controller_binding = experiment.current_probe_binding(candidate)
        remaining()
        full_owners = {arm: pair.owners[arm].get(source_op_index, {}) for arm in ("before", "after")}
        expected_routes = dict(zip(
            ("before", "after"), expected_route_transition or ("contraction", "contraction"), strict=True))
        expected_short_routes = dict(zip(("before", "after"),
            expected_short_route_transition or tuple(expected_routes.values()), strict=True))
        if any(full_owners[arm].get("kind") != route for arm, route in expected_routes.items()):
            raise ValueError("selected source does not have the required verified route in both complete-model arms")
        record.update(source_sha256=pair.source_sha256, logical_dispatch_digest=pair.graph_sha256,
            comparison_binding=pair.comparison_binding, full_model_source_owner=full_owners,
            full_model_artifact_binding={arm: {key: artifact[key] for key in (
                "compiler_sha256", "lowered_sha256", "command_buffer_sha256")}
                for arm, artifact in pair.artifacts.items()})
        probe, extraction = extract_source_contraction(pair.source, source_op_index, entry=entry,
            max_m=max_m, max_n=max_n, max_k=max_k, max_macs=max_macs,
            max_batch_extent=max_batch_extent)
        remaining()
        short_index = _short_source_index(probe, entry)
        record.update(extraction=extraction, probe_source_op_index=short_index)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="source_contraction_", dir=output))
        record["workdir"] = str(work)
        source_path = work/"interface.mlir"
        source_path.write_text(probe)
        for arm, compile_arm in (("before", pair.compile_before), ("after", experiment.compile_probe_candidate)):
            result = compile_arm(candidate, source_path, work/(arm+"_compile"),
                                 timeout_s=remaining(), emit_command_buffer=True)
            lowered, emitted, cb = result["lowered"], result["command_buffer_emission"], result["command_buffer"]
            remaining()
            if source_path.read_text() != probe:
                raise ValueError("hash-bound short source changed during compilation")
            if (lowered.returncode or not lowered.stdout or emitted is None or emitted.returncode
                    or not isinstance(cb, dict) or cb.get("declined") is not None):
                raise ValueError(arm+" short normal-entrypoint compilation failed: "+(lowered.stderr or "")[-1200:])
            raw_cb = json.dumps(cb, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if max(len(lowered.stdout.encode()), len(raw_cb.encode())) > 2_000_000:
                raise ValueError("short emitted artifact exceeds bounded verification size")
            cb_sha = text_digest(raw_cb)
            lowered_path, cb_path = work/(arm+".llvm.mlir"), work/(arm+".command_buffer.json")
            lowered_path.write_text(lowered.stdout)
            cb_path.write_text(raw_cb)
            from merlin.targetgen.rocc import decode as target_decode
            parsed_short = target_decode._parse_module(lowered.stdout)
            proof = verify_compiler_global_plan(source_text=probe, lowered_text=lowered.stdout, command_buffer=cb,
                candidate_sha256=pair.artifacts[arm]["compiler_sha256"], command_buffer_sha256=cb_sha,
                parsed_lowered_module=parsed_short)
            proof = json.loads(json.dumps(proof, allow_nan=False))
            record.setdefault("short_verification", {})[arm] = proof
            remaining()
            plan = program_plan(cb)
            owner = source_owners(plan).get(short_index, {})
            short_admission = None
            if proof.get("status") == "refused":
                from .source_initializer_elision import numerical_probe_admission
                short_admission = numerical_probe_admission(source_text=probe, lowered_text=lowered.stdout,
                    command_buffer=cb, declared_proof=json.loads(json.dumps(proof)),
                    candidate_sha256=pair.artifacts[arm]["compiler_sha256"], source_op_index=short_index)
                record.setdefault("short_execution_admissions", {})[arm] = short_admission
                if document_digest(cb) != cb_sha:
                    raise ValueError("short admission changed the original emitted command buffer")
                if short_admission.get("status") == "source_bound_numerical_probe":
                    expected_admission = {"schema": "short_initializer_execution_admission_v1",
                        "source_sha256": text_digest(probe), "lowered_sha256": text_digest(lowered.stdout),
                        "command_buffer_sha256": cb_sha,
                        "candidate_sha256": pair.artifacts[arm]["compiler_sha256"],
                        "declared_proof_sha256": document_digest(proof)}
                    if any(short_admission.get(key) != value for key, value in expected_admission.items()):
                        raise ValueError("short initializer execution admission has stale source/artifact/proof bindings")
            actual_route = owner.get("kind")
            if ((expected_short_routes[arm] is not None
                    and actual_route != expected_short_routes[arm])
                    or actual_route not in {"host", "contraction"}
                    or (proof.get("status") != "verified"
                    and (short_admission or {}).get("status") != "source_bound_numerical_probe")):
                raise ValueError(arm+" reduced source/task/CFG ownership does not reproduce the selected route")
            entries = plan.get("entry_bindings", [])
            if (len(entries) != len(extraction["inputs"]) or len(set(entries)) != len(entries)
                    or len(plan.get("output_bindings", [])) != 1
                    or (cb.get("kernel_abi") or {}).get("kind") != "whole_program"):
                raise ValueError("short whole-program ABI does not match the actual extracted input/output boundary")
            record["arms"][arm] = {"compiler_sha256": pair.artifacts[arm]["compiler_sha256"],
                "source_sha256": text_digest(probe), "lowered_sha256": text_digest(lowered.stdout),
                "command_buffer_canonical_sha256": cb_sha, "command_buffer_sha256": cb_sha,
                "command_buffer_serialization": "canonical host serialization saved byte-exactly, not original compiler raw formatting",
                "lowered_path": str(lowered_path), "command_buffer_path": str(cb_path),
                "source_task_cfg_proof": proof, "source_owner": owner,
                "declared_route": actual_route,
                "full_model_declared_route": expected_routes[arm],
                "input_tensors": entries, "output_bindings": plan["output_bindings"],
                "actual_target_instruction_semantics": "UNVERIFIED: requires target adapter"}
            if short_admission is not None:
                record["arms"][arm]["short_execution_admission"] = short_admission
            try:
                from .task_instruction_evidence import summarize_task_instructions, target_instruction_facts
                if parsed_short is None:
                    raise ValueError("short emitted LLVM could not be parsed for task instruction observation")
                trace = target_decode.decode_module(parsed_short, target=experiment.target)
                summary = summarize_task_instructions(source_text=probe, lowered_text=lowered.stdout,
                    command_buffer_text=raw_cb, command_buffer=cb, verified_plan=proof,
                    parsed_module=parsed_short, decoded_trace=trace,
                    target_facts=target_instruction_facts(experiment.target),
                    host_policy_sha256=experiment.host_policy["sha256"],
                    decode_module=lambda module: target_decode.decode_module(module, target=experiment.target),
                    short_execution_admission=short_admission)
            except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, OSError) as exc:
                summary = {"status": "UNKNOWN", "route_correspondence": "UNKNOWN",
                    "reason": f"short static task observation unavailable: {exc}",
                    "timing_calibration_admissible": False}
            record["arms"][arm]["task_instruction_evidence"] = summary
            record["arms"][arm]["task_instruction_evidence_sha256"] = document_digest(summary)
        oracle = _typed_cases(probe, extraction, remaining)
        oracle_path = work/"independent_oracle.json"
        oracle_path.write_text(json.dumps(oracle, sort_keys=True, indent=2)+"\n")
        current_pair = bind_source_program_pair(candidate=candidate, experiment=experiment,
            comparison_arm=comparison_arm, portfolio_member=portfolio_member,
            max_source_bytes=max_source_bytes)
        if (experiment.current_probe_binding(candidate) != initial_controller_binding
                or document_digest(current_pair.comparison_binding) != initial_binding
                or current_pair.source_sha256 != pair.source_sha256
                or document_digest(current_pair.comparison_binding) != document_digest(pair.comparison_binding)
                or any(current_pair.artifacts[arm][key] != pair.artifacts[arm][key]
                       for arm in ("before", "after") for key in ("compiler_sha256", "lowered_sha256", "command_buffer_sha256"))):
            raise ValueError("compiler/source/target pair changed during contraction preparation")
        remaining()
        record.update(status="prepared", independent_oracle=str(oracle_path),
            independent_oracle_sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest())
        comparison = task_route_presence_for_preparation(candidate=candidate, experiment=experiment,
            pair=current_pair, prepared=record)
        record.update(task_route_presence=comparison, task_route_presence_sha256=document_digest(comparison),
                      task_route_feedback=task_route_feedback(comparison))
        remaining()
    except (ValueError, KeyError, TypeError, AttributeError, RuntimeError, TimeoutError, OSError,
            subprocess.SubprocessError) as error:
        record.update(status="UNKNOWN", reason=f"{type(error).__name__}: {error}")
    record["elapsed_seconds"] = monotonic()-started
    if work is not None:
        (work/"preparation.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n")
    return record
