"""Bounded, fixed-work source-order diagnostics from two retained full-model revisions."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
from typing import Any

from .context_probe import extract_queued_movement_context
from .context_program import slice_context_source
from .deps.rocc import INHERITS_DESTINATION
from .fixed_work_context import project_fixed_work_context


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class PairedControlledContextProvider:
    """Keep the old workload fixed while projecting its order through the current compiler edit.

    This provider does not calibrate rates or model latency, reuse old timings, or charge the
    controller budget itself. Both arms execute one warm and one measured reduced workload under
    one deadline. Its current narrow relative proof admits only row-disjoint load/compute moves.
    """

    def __init__(self, *, target: str, adapter: Any, output: Path, max_commands: int = 32):
        if not 3 <= max_commands <= 32:
            raise ValueError("fixed-work command budget must lie in [3,32]")
        self.target, self.adapter, self.output = target, adapter, Path(output)
        self.max_commands = max_commands

    def __call__(self, *, candidate: Path, experiment: Any, timeout_s: float = 60) -> dict[str, Any]:
        from xdsl.printer import Printer
        from merlin.targetgen.rocc import decode

        started = monotonic()
        requested = float(timeout_s)
        if isinstance(timeout_s, bool) or not math.isfinite(requested) or requested <= 0:
            raise ValueError("fixed-work comparison needs a finite positive deadline")
        limit = min(requested, 60.0)

        def remaining():
            left = limit - (monotonic() - started)
            if left <= 0:
                raise TimeoutError("fixed-work preparation and both executions exhausted their deadline")
            return left

        bindings = {"before": experiment.previous_probe_binding(candidate),
                    "after": experiment.current_probe_binding(candidate)}
        if (bindings["before"].graph_digest != bindings["after"].graph_digest
                or bindings["before"].target_digest != bindings["after"].target_digest):
            raise ValueError("fixed-work comparison changed logical graph or target")
        captured = {"before": dict(experiment.previous_artifacts(candidate)),
                    "after": dict(experiment.current_artifacts(candidate))}
        for arm, source in captured.items():
            text = source["lowered_text"]
            if hashlib.sha256(text.encode()).hexdigest() != source["candidate_lowered_sha256"]:
                raise ValueError(f"retained {arm} source artifact changed")
            if len(text.encode()) > 2_000_000:
                raise ValueError("fixed-work source exceeds bounded parsing policy")
            if source.get("parsed_lowered_module") is None:
                source["parsed_lowered_module"] = decode._parse_module(text)
            if source["parsed_lowered_module"] is None:
                raise ValueError("fixed-work emitted source does not parse")
        before = captured["before"]
        contexts = extract_queued_movement_context(
            before["decoded_trace"], target=self.target,
            artifact_sha256=before["candidate_lowered_sha256"], artifact_text=before["lowered_text"],
            command_buffer=before["command_buffer"], parsed_module=before["parsed_lowered_module"],
            max_commands=self.max_commands)
        asm = [op for op in before["parsed_lowered_module"].walk() if op.name == "llvm.inline_asm"]
        anchor, task_compute_count, task_command_count = None, 0, 0
        for row in contexts["motifs"]:
            if row["state_missing"]:
                continue
            owned = [index for index, op in enumerate(asm)
                     if getattr(getattr(op.attributes.get("merlin.global_task"), "value", None), "data", None)
                     == row["task_index"]]
            count = sum(before["decoded_trace"]["instructions"][index]["class"] in INHERITS_DESTINATION
                        for index in owned)
            if count > 1:
                anchor, task_compute_count, task_command_count = row, count, len(owned)
                break
        if anchor is None:
            raise ValueError("no initialized competing-load window strictly reduces a multi-compute source task")
        projected, proof = project_fixed_work_context(before, captured["after"], anchor, target=self.target)
        self.output.mkdir(parents=True, exist_ok=True)
        work = Path(mkdtemp(prefix="fixed_work_", dir=self.output))
        arms, input_contracts = {}, {}
        for arm, context in (("before", anchor), ("after", projected)):
            module, source_slice = slice_context_source(captured[arm]["parsed_lowered_module"], context,
                                                       target=self.target, fixed_work_projection=True)
            source_slice.update({"model_instruction_count": len(captured[arm]["decoded_trace"]["instructions"]),
                                 "source_task_instruction_count": task_command_count,
                                 "source_task_compute_pair_count": task_compute_count,
                                 "executed_compute_pair_count": 1,
                                 "work_contract_sha256": proof["work_contract_sha256"],
                                 "task_reads": context["task_reads"], "task_writes": context["task_writes"]})
            stream = io.StringIO()
            Printer(stream=stream).print_op(module)
            text = stream.getvalue() + "\n"
            if hashlib.sha256(text.encode()).hexdigest() != source_slice["slice_source_sha256"]:
                raise ValueError("fixed-work slice canonical source changed")
            source = work / f"{arm}.target.mlir"
            source.write_text(text)
            budget = int(remaining())
            if budget < 1:
                raise TimeoutError("insufficient fixed-work compilation budget")
            prepared = self.adapter.prepare_primitive_probe(
                source, work / f"{arm}_runtime", timeout_seconds=budget,
                profile_counters=True, include_operand_movement=True, fixed_work_slice=True)
            if (prepared.get("measurement_scope") != "controlled_fixed_work_slice"
                    or prepared["source_artifact_sha256"] != source_slice["slice_source_sha256"]
                    or prepared["timed_instruction_count"] != len(context["instruction_indices"])
                    or prepared["host_input_bytes"] > 65536 or prepared["output_storage_bytes"] > 65536):
                raise ValueError("prepared fixed-work source/scope/footprint changed")
            # Exact wrapper bytes bind initialized values, golden output, arguments and timer ROI.
            input_contracts[arm] = {key: prepared[key] for key in (
                "wrapper_sha256", "computed_tile", "host_input_bytes", "output_storage_bytes",
                "output_row_stride_bytes")}
            input_contracts[arm]["source_argument_of_slice_argument"] = source_slice["source_argument_of_slice_argument"]
            arms[arm] = {"source_slice": source_slice, "prepared": prepared,
                         "work_contract_sha256": proof["work_contract_sha256"]}
        if input_contracts["before"] != input_contracts["after"]:
            raise ValueError("paired work changed initialized inputs, expected output or measured wrapper")
        contract = input_contracts["before"]
        contract_sha = _digest(contract)
        for arm in arms:
            arms[arm]["deterministic_input_contract_sha256"] = contract_sha
            (work / f"{arm}.source_slice.json").write_text(json.dumps(arms[arm]["source_slice"], indent=2) + "\n")
        (work / "projection_proof.json").write_text(json.dumps(proof, indent=2) + "\n")

        def check_binding():
            if (experiment.current_probe_binding(candidate) != bindings["after"]
                    or experiment.previous_probe_binding(candidate) != bindings["before"]):
                raise ValueError("compiler, graph, plan or target changed during fixed-work comparison")

        check_binding()
        executed = set()

        def execute(*, arm: str, timeout_s: float):
            if arm not in arms or arm in executed:
                raise ValueError("fixed-work arm absent or already executed")
            check_binding()
            executed.add(arm)
            actual = self.adapter.execute_prepared_primitive(
                arms[arm]["prepared"], timeout_seconds=min(float(timeout_s), remaining()))
            if actual.get("measurement_scope") != "controlled_fixed_work_slice":
                raise ValueError("runtime returned a different fixed-work scope")
            return actual

        return {"paired_context_inputs": {
            "before_binding": bindings["before"], "after_binding": bindings["after"],
            "before_model_artifact_sha256": captured["before"]["candidate_lowered_sha256"],
            "after_model_artifact_sha256": captured["after"]["candidate_lowered_sha256"],
            "scope": "controlled_fixed_work_slice", "projection_proof": proof,
            "work_contract_sha256": proof["work_contract_sha256"],
            "deterministic_input_contract": contract, "deterministic_input_contract_sha256": contract_sha,
            "arms": arms, "timeout_seconds": remaining()}, "execute": execute}
