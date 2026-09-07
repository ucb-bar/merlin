"""Bounded complete reduced-source A/B observations, not model-cost calibration."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from time import monotonic

from .source_program_pair import bind_source_program_pair, document_digest, text_digest
from .source_contraction_witness import extract_source_contraction, evaluate_source_contraction


def _read(path, expected, limit):
    with Path(path).open("rb") as stream:
        data = stream.read(limit+1)
    if len(data) > limit or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("prepared short-program file changed or exceeds its byte limit")
    return data


def _binding(pair):
    return {arm: {key: artifact[key] for key in (
        "compiler_sha256", "lowered_sha256", "command_buffer_sha256")}
        for arm, artifact in pair.artifacts.items()}


def _record_task_route_presence(result, *, candidate, experiment, pair, prepared):
    """Add scoped relevance without altering raw numerical/runtime observations."""
    from .source_contraction_preparation import task_route_presence_for_preparation, task_route_feedback
    comparison = task_route_presence_for_preparation(candidate=candidate, experiment=experiment,
        pair=pair, prepared=prepared)
    result.update(task_route_presence=comparison,
        task_route_presence_sha256=document_digest(comparison),
        task_route_feedback=task_route_feedback(comparison),
        prepared_task_route_presence_matches=(document_digest(comparison) == prepared.get("task_route_presence_sha256")
            and prepared.get("task_route_presence_sha256") == document_digest(prepared.get("task_route_presence"))))


def _validated_inputs(prepared, pair):
    """Reproduce selected source and oracle before any target build/execution."""
    import numpy as np
    if (prepared.get("schema") != "source_contraction_preparation_v1" or prepared.get("status") != "prepared"
            or prepared.get("source_sha256") != pair.source_sha256
            or prepared.get("logical_dispatch_digest") != pair.graph_sha256
            or prepared.get("full_model_artifact_binding") != _binding(pair)
            or document_digest(prepared.get("comparison_binding")) != document_digest(pair.comparison_binding)):
        raise ValueError("prepared comparison belongs to stale source/compiler/artifacts")
    bounds = prepared["host_bounds"]
    source, extraction = extract_source_contraction(pair.source, prepared["source_op_index"],
        entry=prepared["entry"], **bounds)
    if extraction != prepared["extraction"]:
        raise ValueError("selected source extraction changed")
    path = Path(prepared["workdir"])/"interface.mlir"
    if _read(path, text_digest(source), 256*1024).decode() != source:
        raise ValueError("prepared interface is not the selected source contraction")
    oracle = json.loads(_read(prepared["independent_oracle"], prepared["independent_oracle_sha256"], 256*1024))
    if (oracle.get("schema") != "source_contraction_prepared_oracle_v1"
            or oracle.get("probe_source_sha256") != text_digest(source)
            or any(oracle.get(key) != extraction[key] for key in
                   ("input_shapes", "input_dtypes", "output_shape", "output_dtype"))
            or not isinstance(oracle.get("cases"), list) or not oracle["cases"]):
        raise ValueError("prepared typed oracle has a different source boundary")
    case = oracle["cases"][0]
    arrays, payloads = [], []
    for values, shape, dtype, expected_sha in zip(case["inputs"], oracle["input_shapes"],
            oracle["input_dtypes"], case["input_le_sha256"], strict=True):
        if dtype not in {"i8", "i16", "i32", "i64"} or math.prod(shape)*int(dtype[1:])//8 > 65536:
            raise ValueError("source input dtype/footprint unsupported")
        array = np.asarray(values, dtype="int"+dtype[1:])
        if list(array.shape) != shape:
            raise ValueError("source oracle input shape changed")
        payload = array.astype(array.dtype.newbyteorder("<")).tobytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError("source oracle input bytes changed")
        arrays.append(array)
        payloads.append(payload)
    if sum(map(len, payloads)) > 65536:
        raise ValueError("source input payload exceeds total byte cap")
    expected = evaluate_source_contraction(source, extraction, arrays)
    expected_bytes = expected.astype(expected.dtype.newbyteorder("<")).tobytes()
    if (expected.tolist() != case["expected"]
            or hashlib.sha256(expected_bytes).hexdigest() != case["expected_le_sha256"]):
        raise ValueError("prepared output is not the independently recomputed source result")
    return source, extraction, oracle, case, payloads, expected


def _output_matches(actual, output_name, expected):
    """The common OUT protocol is a matrix view of the exact logical tensor."""
    shape = expected.shape
    rows, cols = (math.prod(shape[:-1]), shape[-1]) if shape else (1, 1)
    matrix = expected.reshape((rows, cols)).tolist()
    return actual == {output_name: matrix}


class SourceProgramPairProvider:
    """One identical source-oracle case, two whole-program warm/measured arms.

    The enclosing controller charges preparation and both executions in finally.
    This provider never imports candidates, recompiles the full graph, uses a
    primitive slice, fits a rate, or infers changed-task relevance from byte diffs.
    """
    def __init__(self, *, target, adapter, output):
        self.target, self.adapter, self.output = target, adapter, Path(output)

    def __call__(self, *, candidate, experiment, prepared, timeout_s=60):
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or not 0 < timeout_s <= 60:
            raise ValueError("paired complete source execution requires a finite <=60s total budget")
        if experiment.target != self.target:
            raise ValueError("source pair target differs from its host adapter")
        started = monotonic()
        deadline = started+timeout_s
        self.output.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="complete_source_pair_", dir=self.output))
        result = {"schema": "complete_source_pair_execution_v1", "status": "UNKNOWN", "arms": {},
            "preparation_sha256": document_digest(prepared), "workdir": str(work),
            "full_model_executed": False, "full_model_recompiled": False, "global_speedup_proven": False,
            "global_cost_calibration": False, "selected_task_route_correspondence": "UNPROVEN",
            "scope": "engine-relative complete reduced-source programs; one identical typed case per arm"}

        def remaining():
            left = deadline-monotonic()
            if left <= 0:
                raise TimeoutError("paired source builds and executions exhausted their shared deadline")
            return left

        try:
            pair = bind_source_program_pair(candidate=candidate, experiment=experiment,
                                            comparison_arm=prepared["comparison_arm"])
            source, extraction, oracle, case, payloads, expected = _validated_inputs(prepared, pair)
            _record_task_route_presence(result, candidate=candidate, experiment=experiment, pair=pair, prepared=prepared)
            initial = experiment.current_probe_binding(candidate)
            result.update(full_model_artifact_binding=_binding(pair), source_sha256=pair.source_sha256,
                probe_source_sha256=text_digest(source), logical_dispatch_digest=pair.graph_sha256,
                selected_case=case["case"], unexecuted_cases=[c["case"] for c in oracle["cases"][1:]],
                expected_le_sha256=case["expected_le_sha256"], input_le_sha256=case["input_le_sha256"])
            from merlin.common.paths import repo_root
            from merlin.targetgen.sandbox.build_dependencies import HostBuildDependencies
            for arm in ("before", "after"):
                remaining()
                entry = prepared["arms"][arm]
                lowered = _read(entry["lowered_path"], entry["lowered_sha256"], 2*1024*1024).decode()
                cb = json.loads(_read(entry["command_buffer_path"], entry["command_buffer_canonical_sha256"], 2*1024*1024))
                if (entry["source_sha256"] != text_digest(source)
                        or entry["compiler_sha256"] != pair.artifacts[arm]["compiler_sha256"]
                        or entry["source_task_cfg_proof"].get("source_sha256") != text_digest(source)
                        or entry["source_task_cfg_proof"].get("candidate_sha256") != entry["compiler_sha256"]
                        or entry["source_task_cfg_proof"].get("plan_digest") != document_digest(cb["params"]["global_program_plan"])
                        or entry["source_task_cfg_proof"].get("candidate_lowered_sha256") != text_digest(lowered)
                        or entry["source_task_cfg_proof"].get("candidate_command_buffer_sha256") != document_digest(cb)):
                    raise ValueError("prepared arm lost exact source/compiler/LLVM/plan binding")
                proof = entry["source_task_cfg_proof"]
                admission = None
                if proof.get("status") != "verified":
                    from .source_initializer_elision import numerical_probe_admission
                    admission = numerical_probe_admission(source_text=source, lowered_text=lowered,
                        command_buffer=cb, declared_proof=proof, candidate_sha256=entry["compiler_sha256"],
                        source_op_index=prepared["probe_source_op_index"])
                    if (admission.get("status") != "source_bound_numerical_probe"
                            or document_digest(admission) != document_digest(entry.get("short_execution_admission"))):
                        raise ValueError("refused declared plan lacks an exact separate short numerical-probe admission")
                inputs = entry["input_tensors"]
                if len(inputs) != len(payloads) or len(set(inputs)) != len(inputs):
                    raise ValueError("source input boundary no longer maps uniquely to ABI arguments")
                plan = cb["params"]["global_program_plan"]
                if plan["entry_bindings"] != inputs or plan["output_bindings"] != entry["output_bindings"]:
                    raise ValueError("source input/output plan bindings changed")
                arm_work = work/arm
                arm_work.mkdir()
                policy = experiment.native_probe_policy(candidate, arm_work)
                environment = self.adapter.short_program_environment(policy)
                build = self.adapter.prepare_short_program_build(source_text=source, lowered_text=lowered,
                    command_buffer=cb, logical_payloads=dict(zip(inputs, payloads, strict=True)),
                    source_evidence={"probe_source_sha256": text_digest(source), "lowered_sha256": text_digest(lowered),
                        "selected_source_op_index": prepared["source_op_index"], "compiler_sha256": entry["compiler_sha256"],
                        "declared_source_plan": proof, "short_execution_admission": admission},
                    workdir=arm_work/"build_worker", python_executable=environment["python_executable"],
                    build_namespace_root=environment["build_namespace_root"],
                    build_path_bindings=environment["build_path_bindings"])
                dependencies = build.dependencies
                cap = HostBuildDependencies(source_root=str(repo_root()),
                    namespace_root=str(environment["build_namespace_root"]), argv=build.argv,
                    file_pins=tuple(json.loads(build.pins_json).items()),
                    source_grants=tuple((d["source"], d["destination"], d["sha256"]) for d in dependencies if d["kind"] == "python"),
                    request_path=build.request_path, worker_path=str(arm_work/"build_worker/worker.py"),
                    format_data=tuple((d["source"], d["destination"], d["sha256"], d["role"])
                                      for d in dependencies if d["kind"] == "format_data"))

                def bounded_build(argv, *, timeout_s):
                    execution = experiment.run_native_probe(candidate, arm_work, argv,
                        timeout_s=min(timeout_s, remaining()), _build_dependencies=cap)
                    (arm_work/"build.stdout").write_text(execution.stdout)
                    (arm_work/"build.stderr").write_text(execution.stderr)
                    return execution

                built = build.run(bounded_build, timeout_s=min(60, remaining()))
                result["arms"][arm] = {"build": built, "status": "built_not_executed",
                    "host_environment": environment, "declared_plan_status": proof["status"],
                    "short_execution_admission": admission}
                if built["warm_invocations_emitted"] != 1 or built["measured_invocations_emitted"] != 1:
                    raise ValueError("complete short caller does not emit warm1/measured1")
                command = self.adapter.prepare_short_program_execution(Path(built["elf_path"]),
                    expected_elf_sha256=built["outputs"]["package_kernel.elf"],
                    expected_engine_provenance=environment["engine_provenance"],
                    python_executable=environment["python_executable"])
                result["arms"][arm]["command"] = command.to_evidence()
                runtime_start = monotonic()
                try:
                    execution = experiment.run_native_probe(candidate, arm_work, command.argv,
                        timeout_s=min(60, remaining()), _execution_dependencies=command.sandbox_dependencies())
                    (arm_work/"runtime.stdout").write_text(execution.stdout)
                    (arm_work/"runtime.stderr").write_text(execution.stderr)
                except Exception as error:
                    for name in ("stdout", "stderr"):
                        raw = getattr(error, name, None)
                        if raw is not None:
                            (arm_work/("runtime."+name)).write_text(raw.decode(errors="replace") if isinstance(raw, bytes) else raw)
                    raise
                finally:
                    result["arms"][arm]["runtime_elapsed_seconds"] = monotonic()-runtime_start
                if execution.returncode:
                    raise ValueError(arm+" complete short executable returned nonzero")
                actual, metrics = self.adapter.parse_output(execution.stdout)
                outputs = cb["kernel_abi"]["outputs"]
                if len(outputs) != 1 or not _output_matches(actual, outputs[0], expected):
                    raise ValueError(arm+" complete short output differs from the independent typed source oracle")
                cycles = metrics.get("cycles")
                if type(cycles) is not int or cycles <= 0:
                    raise ValueError("no positive measured compute cycles in complete source output")
                result["arms"][arm].update(status="passed", correct=True, cycles=cycles, raw_metrics=metrics,
                    output=actual, warm_invocations=1, measured_invocations=1,
                    measurement_scope=built["measurement_scope"],
                    console_sha256=text_digest(execution.stdout))
                if self.adapter.short_program_environment(policy) != environment:
                    raise ValueError("host target environment changed while executing source arm")
                remaining()
            final_pair = bind_source_program_pair(candidate=candidate, experiment=experiment,
                                                  comparison_arm=prepared["comparison_arm"])
            if (_binding(final_pair) != _binding(pair) or experiment.current_probe_binding(candidate) != initial
                    or document_digest(prepared) != result["preparation_sha256"]):
                raise ValueError("complete source pair changed during execution")
            engines = [result["arms"][arm]["command"]["engine_provenance"] for arm in ("before", "after")]
            if engines[0] != engines[1]:
                raise ValueError("source arms executed on different engine/configuration identities")
            result.update(status="passed", correct=True,
                observed_compute_cycle_ratio=result["arms"]["before"]["cycles"]/result["arms"]["after"]["cycles"],
                ratio_scope="same reduced source and typed case on exact engine; not model speedup or route attribution",
                limitations=["other prepared oracle cases not executed", "full-shape numerical equivalence UNPROVEN",
                    "selected changed task/route correspondence UNPROVEN", "no extrapolation to model cost",
                    "format initialization outside ROI does not authorize excluding runtime activation conversion"])
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError, TimeoutError,
                subprocess.SubprocessError) as error:
            result.update(status="UNKNOWN", reason=type(error).__name__+": "+str(error))
        finally:
            result["elapsed_seconds"] = monotonic()-started
            (work/"execution.json").write_text(json.dumps(result, sort_keys=True, indent=2)+"\n")
        return result
