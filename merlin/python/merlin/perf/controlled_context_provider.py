"""Host-owned short source-prefix diagnostics, distinct from model-equivalent calibration."""
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


class ControlledSourcePrefixProvider:
    """Prepare one actual current-source prefix; the controller charges its entire action.

    No external probe interface, old executable whitelist, estimated cycle rate or false repetition
    ratio is required. Static footprint bounds and an enforced wall deadline limit this diagnostic.
    It cannot return the equal-signature admission inputs consumed by ``measure_probe``.
    """

    def __init__(self, *, target: str, adapter: Any, output: Path, max_commands: int = 32):
        if not 3 <= max_commands <= 32:
            raise ValueError("controlled source-prefix command budget must lie in [3,32]")
        self.target, self.adapter, self.output = target, adapter, Path(output)
        self.max_commands = max_commands

    def __call__(self, *, candidate: Path, experiment: Any, timeout_s: float = 60) -> dict[str, Any]:
        from xdsl.printer import Printer
        from merlin.targetgen.rocc import decode

        started = monotonic()
        requested = float(timeout_s)
        if isinstance(timeout_s, bool) or not math.isfinite(requested) or requested <= 0:
            raise ValueError("controlled context needs a finite positive deadline no longer than 60 seconds")
        limit = min(60.0, requested)

        def remaining():
            left = limit - (monotonic() - started)
            if left <= 0:
                raise TimeoutError("controlled source-prefix preparation/execution exhausted 60 second budget")
            return left

        binding = experiment.current_probe_binding(candidate)
        captured = experiment.current_artifacts(candidate)
        source = captured["lowered_text"]
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        if source_sha != captured["candidate_lowered_sha256"]:
            raise ValueError("retained full-model source artifact changed")
        if len(source.encode()) > 2_000_000:
            raise ValueError("controlled source extraction exceeds bounded host parsing byte policy")
        module = captured.get("parsed_lowered_module")
        if module is None:
            module = decode._parse_module(source)
        if module is None:
            raise ValueError("current emitted full-model source does not parse")
        trace = captured["decoded_trace"]
        contexts = extract_queued_movement_context(
            trace, target=self.target, artifact_sha256=source_sha, artifact_text=source,
            command_buffer=captured["command_buffer"], parsed_module=module,
            max_commands=self.max_commands)
        asm = [op for op in module.walk() if op.name == "llvm.inline_asm"]
        context = None
        task_compute_count = task_command_count = 0
        for row in contexts["motifs"]:
            if row["state_missing"]:
                continue
            owned = [index for index, op in enumerate(asm)
                     if getattr(getattr(op.attributes.get("merlin.global_task"), "value", None), "data", None)
                     == row["task_index"]]
            count = sum(trace["instructions"][index].get("class") in INHERITS_DESTINATION for index in owned)
            # Never execute a complete source layer's only compute and call it a reduced prefix.
            if count > 1:
                context, task_compute_count, task_command_count = row, count, len(owned)
                break
        if context is None:
            raise ValueError("no bounded initialized competing-load prefix strictly reduces a multi-compute source task")
        short_module, source_slice = slice_context_source(module, context, target=self.target)
        source_slice.update({"model_instruction_count": len(trace["instructions"]),
                             "source_task_instruction_count": task_command_count,
                             "source_task_compute_pair_count": task_compute_count,
                             "executed_compute_pair_count": 1,
                             "task_reads": context["task_reads"], "task_writes": context["task_writes"]})
        self.output.mkdir(parents=True, exist_ok=True)
        work = Path(mkdtemp(prefix="source_prefix_", dir=self.output))
        stream = io.StringIO()
        Printer(stream=stream).print_op(short_module)
        short_text = stream.getvalue() + "\n"
        if hashlib.sha256(short_text.encode()).hexdigest() != source_slice["slice_source_sha256"]:
            raise ValueError("source slice canonical emitted bytes changed")
        short = work / "context.target.mlir"
        short.write_text(short_text)
        (work / "source_slice.json").write_text(json.dumps(source_slice, indent=2) + "\n")
        compile_budget = int(remaining())
        if compile_budget < 1:
            raise TimeoutError("less than one second remains for controlled prefix compilation")
        prepared = self.adapter.prepare_primitive_probe(
            short, work / "runtime", timeout_seconds=compile_budget,
            profile_counters=True, include_operand_movement=True)
        if (prepared.get("measurement_scope") != "controlled_source_prefix"
                or prepared["source_artifact_sha256"] != source_slice["slice_source_sha256"]
                or prepared["timed_instruction_count"] != len(context["instruction_indices"])
                or prepared["host_input_bytes"] > 65536 or prepared["output_storage_bytes"] > 65536):
            raise ValueError("prepared controlled context scope/source/footprint differs from admitted source slice")
        if experiment.current_probe_binding(candidate) != binding:
            raise ValueError("compiler, graph, plan or target changed during controlled prefix preparation")

        # Exact command multiset + ABI/readout footprint supports only a later scoped A/B check;
        # dropping competing loads or narrowing the timer changes this contract.
        work_contract = {
            "source_op_indices": source_slice["source_op_indices"],
            "source_argument_of_slice_argument": source_slice["source_argument_of_slice_argument"],
            "timed_command_multiset": sorted(json.dumps(row, sort_keys=True) for row in context["instruction_semantics"]),
            "initial_configurations": {kind: {key: row.get(key) for key in ("class", "funct", "rs1", "rs2", "decoded")}
                                       for kind, row in context["initial_configurations"].items()},
            "computed_tile": prepared["computed_tile"], "host_input_bytes": prepared["host_input_bytes"],
            "output_storage_bytes": prepared["output_storage_bytes"],
            "output_row_stride_bytes": prepared["output_row_stride_bytes"],
            "competing_load_count": len(source_slice["queued_competing_movement_indices"]),
        }
        source_slice["work_contract"] = work_contract
        source_slice["work_contract_sha256"] = hashlib.sha256(
            json.dumps(work_contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        source_slice["binding"] = binding.to_dict()
        (work / "source_slice.json").write_text(json.dumps(source_slice, indent=2) + "\n")

        def execute(*, timeout_s: float):
            if experiment.current_probe_binding(candidate) != binding:
                raise ValueError("stale compiler or model immediately before controlled context execution")
            actual = self.adapter.execute_prepared_primitive(
                prepared, timeout_seconds=min(float(timeout_s), remaining()))
            if actual.get("measurement_scope") != "controlled_source_prefix":
                raise ValueError("runtime returned a different measurement scope")
            return actual

        return {"controlled_context_inputs": {
            "binding": binding, "model_artifact_sha256": source_sha, "source_slice": source_slice,
            "prepared": prepared, "timeout_seconds": remaining(), "scope": "controlled_source_prefix",
            "source_slice_receipt": str(work / "source_slice.json")}, "execute": execute}
