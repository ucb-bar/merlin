"""Host-owned bridge from current whole-model artifacts to one separate short probe.

This connector calibrates only an isolated primitive reference cost. It does not equate drained
probe timing with the containing layer's latency, and never fits/extrapolates from one sample.
The target adapter supplies executable setup/body/readback instrumentation and target-derived
representation facts. Candidate manifests cannot supply any equivalence or timing evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic
from typing import Any

from merlin.analysis.optimization.probes.context_probe import extract_queued_movement_context
from merlin.analysis.optimization.probes.context_program import slice_context_source
from merlin.analysis.optimization.probes.fixed_work_context import project_fixed_work_context
from merlin.analysis.optimization.probes.instruction_motif import initialized_compute_primitives
from merlin.analysis.optimization.probes.mechanisms import MechanismEvidence, ProbeObservation
from merlin.analysis.perf.dependencies.rocc import INHERITS_DESTINATION
from merlin.analysis.perf.measurement.execution_policy import (
    SimulationBudget,
    WarmComputeReceipt,
    WarmProfileContract,
)
from merlin.common import digest as _mdigest
from merlin.common import digest as _mjson

# ---- isolated_probe_provider -------------------------------------------------------------------

_sha = _mdigest.sha256_bytes


class IsolatedPrimitiveProbeProvider:
    """A real ``provider(candidate, probes, timeout_s)`` for the global action broker.

    ``probes`` retains host-derived artifacts from its latest whole-model analysis, exposes
    ``current_artifacts`` and ``current_probe_binding``, and accounts elapsed preparation through
    ``charge_probe_preparation``. ``adapter`` is selected by the host, never the candidate. The
    first supported extraction family is initialized command-ABI overwrite primitives; missing
    initialization, different instruction state, changed executable, or absent timing evidence
    refuses measurement while leaving structural whole-model optimization available.
    """

    def __init__(
        self,
        *,
        target: str,
        short_interface: Path,
        adapter: Any,
        runtime_receipt: Path,
        output: Path,
        profile_counters: bool = False,
    ):
        self.target = target
        self.short_interface = Path(short_interface).resolve()
        self.short_interface_sha256 = _sha(self.short_interface.read_bytes())
        self.adapter = adapter
        self.runtime_receipt = Path(runtime_receipt).resolve()
        self.runtime_receipt_sha256 = _sha(self.runtime_receipt.read_bytes())
        self.output = Path(output).resolve()
        self.profile_counters = profile_counters

    def __call__(self, *, candidate: Path, probes: Any, timeout_s: float | None = None) -> dict[str, Any]:
        from merlin.target.isa.rocc import decode
        from merlin.target.simulators import gsim_emulator

        started = monotonic()
        limit = min(600.0, float(probes.analysis.timeout_s), 600.0 if timeout_s is None else float(timeout_s))
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("probe provider needs a positive bounded broker deadline")
        deadline = started + limit
        binding = probes.analysis.session.current_probe_binding(candidate)
        captured = probes.analysis.session.current_artifacts(candidate)
        lowered = captured["lowered_text"].encode()
        if _sha(lowered) != captured["candidate_lowered_sha256"]:
            raise ValueError("retained whole-model artifact digest changed")
        if _sha(self.short_interface.read_bytes()) != self.short_interface_sha256:
            raise ValueError("host-selected short probe source changed")
        if _sha(self.runtime_receipt.read_bytes()) != self.runtime_receipt_sha256:
            raise ValueError("host runtime calibration receipt changed")
        # The accessor is host-owned and binds this decoded trace to the same emitted bytes.
        # No second full-model compilation or expensive text parse is performed here.
        model_primitives = initialized_compute_primitives(captured["decoded_trace"], target=self.target)
        self.output.mkdir(parents=True, exist_ok=True)
        from tempfile import mkdtemp

        work = Path(mkdtemp(prefix="primitive_", dir=self.output))

        def remaining() -> float:
            value = deadline - monotonic()
            if value <= 0:
                raise TimeoutError("probe preparation and execution exceeded the broker deadline")
            return value

        try:
            emitted = probes.compile_probe_candidate(
                Path(candidate), self.short_interface, work / "compiler_scratch", timeout_s=remaining()
            )
            (work / "probe_compile.log").write_text(emitted.stderr or "")
            if emitted.returncode or not emitted.stdout:
                raise ValueError("same-candidate short probe compilation failed")
            short = work / "probe.target.mlir"
            short.write_text(emitted.stdout)
            module = decode._parse_module(emitted.stdout)
            if module is None:
                raise ValueError("same-candidate short probe artifact does not parse")
            probe_primitives = initialized_compute_primitives(
                decode.decode_module(module, target=self.target), target=self.target
            )
            if not probe_primitives or probe_primitives[0]["missing"]:
                raise ValueError("short probe has no fully initialized first primitive")
            primitive = probe_primitives[0]
            matched = [
                row
                for row in model_primitives
                if not row["missing"] and row["domain_digest"] == primitive["domain_digest"]
            ]
            if len(matched) <= 1:
                raise ValueError("separate short probe does not reduce a repeated model mechanism")
            profile_options = {"profile_counters": True} if self.profile_counters else {}
            prepared = self.adapter.prepare_primitive_probe(
                short, work / "runtime", timeout_seconds=min(600, int(remaining())), **profile_options
            )
            if prepared["domain_digest"] != primitive["domain_digest"]:
                raise ValueError("prepared executable extracted a different primitive")
            if probes.analysis.session.current_probe_binding(candidate) != binding:
                raise ValueError("compiler, graph, global plan, or target changed during preparation")
            signature = self.adapter.isolated_primitive_signature(primitive["domain"])
            prior = json.loads(self.runtime_receipt.read_text())
            engine = gsim_emulator.citation(self.target)
            for key in ("wrapper_sha256", "primitive_mlir_sha256", "domain_digest"):
                if prior.get(key) != prepared.get(key):
                    raise ValueError(f"prior wall-time calibration has a different {key}")
            prior_elf = self.runtime_receipt.parent / "primitive.elf"
            if _sha(prior_elf.read_bytes()) != prior.get("elf_sha256"):
                raise ValueError("prior measured executable changed")
            runtime_elf_digest = self.adapter.runtime_elf_digest(prior_elf)
            if self.adapter.runtime_elf_digest(Path(prepared["workdir"]) / "primitive.elf") != runtime_elf_digest:
                raise ValueError("prior wall-time calibration has different executable ELF bytes")
            if (
                prior.get("engine_provenance") != engine
                or not prior.get("correct")
                or prior.get("warmup_runs") != 1
                or prior.get("measured_runs") != 1
            ):
                raise ValueError("prior timing does not bind the same engine and warm execution contract")
            # This is measured *effective* ROI cycles per complete diagnostic wall second,
            # including build/setup/boot/readback, not the simulator's hardware cycle throughput.
            # Charging twice the complete historical diagnostic wall time is conservative for
            # this exact ELF/engine. It is only admission evidence, not a statistical guarantee.
            prior_cycles, prior_wall = int(prior["total_compute_cycles"]), float(prior["elapsed_seconds"])
            if prior_cycles <= 0 or not math.isfinite(prior_wall) or not 0 < prior_wall <= 600:
                raise ValueError("prior diagnostic has no finite positive timing observation")
            evidence_path = work / "equivalence.json"
            provenance = str(evidence_path)
            model = MechanismEvidence(binding, signature, _sha(lowered), len(matched), provenance)
            probe = MechanismEvidence(binding, signature, prepared["elf_sha256"], 1, provenance)
            evidence = {
                "schema": "isolated_primitive_provider_v1",
                "binding": binding.to_dict(),
                "model_artifact_sha256": model.artifact_digest,
                "short_source_sha256": self.short_interface_sha256,
                "short_artifact_sha256": _sha(emitted.stdout.encode()),
                "probe_elf_sha256": probe.artifact_digest,
                "runtime_elf_sha256_excluding_debug_metadata": runtime_elf_digest,
                "model_instruction_indices": [row["instruction_indices"] for row in matched],
                "probe_instruction_indices": primitive["instruction_indices"],
                "domain": primitive["domain"],
                "signature": signature.to_dict(),
                "runtime_basis": {
                    "receipt": str(self.runtime_receipt),
                    "sha256": self.runtime_receipt_sha256,
                    "estimate": "twice complete same-ELF same-engine diagnostic wall time",
                },
                "in_context_cycles": None,
                "full_model_cycles": None,
                "full_model_executed": False,
                "scope": "isolated reference cost; model contention and overlap UNKNOWN",
                "counter_uncertainty_scope": "one-cycle counter resolution only; run variance UNKNOWN",
            }
            evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        finally:
            # Preparation is part of this iteration even when extraction or admission fails.
            probes.charge_probe_preparation(candidate, monotonic() - started)

        budget = min(remaining(), float(probes.analysis.timeout_s))
        inputs = {
            "model": model,
            "probe": probe,
            "descriptor": {"kind": "mechanism_probe", "performance": {"measurement_scope": "isolated_primitive"}},
            "budget": SimulationBudget(budget, budget),
            "estimated_cycles": prior_cycles,
            "measured_cycles_per_second": prior_cycles / prior_wall,
            "contract": WarmProfileContract(),
        }

        def execute(*, timeout_s: float) -> ProbeObservation:
            execution_started = monotonic()
            if probes.analysis.session.current_probe_binding(candidate) != binding:
                raise ValueError("stale candidate immediately before primitive execution")
            actual = self.adapter.execute_prepared_primitive(
                prepared, timeout_seconds=min(float(timeout_s), remaining())
            )
            if actual["elf_sha256"] != probe.artifact_digest:
                raise ValueError("executed ELF differs from admitted ELF")
            receipt = WarmComputeReceipt(
                workload=primitive["domain_digest"],
                total_compute_cycles=actual["total_compute_cycles"],
                contract=WarmProfileContract(),
                provenance=str(work / "runtime" / "execution_receipt.json"),
            )
            return ProbeObservation(
                probe,
                receipt,
                monotonic() - execution_started,
                1.0,
                actual["elf_sha256"],
                actual.get("counter_profile"),
            )

        return {
            "admission_inputs": inputs,
            "execute": execute,
            "equivalence_receipt": str(evidence_path),
            "scope": "isolated primitive only; no fitted model or in-context cycle prediction",
        }


# ---- controlled_context_provider ---------------------------------------------------------------
# Host-owned short source-prefix diagnostics, distinct from model-equivalent calibration.


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

    def __call__(self, *, candidate: Path, probes: Any, timeout_s: float = 60) -> dict[str, Any]:
        from xdsl.printer import Printer

        from merlin.target.isa.rocc import decode

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

        binding = probes.analysis.session.current_probe_binding(candidate)
        captured = probes.analysis.session.current_artifacts(candidate)
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
            trace,
            target=self.target,
            artifact_sha256=source_sha,
            artifact_text=source,
            command_buffer=captured["command_buffer"],
            parsed_module=module,
            max_commands=self.max_commands,
        )
        asm = [op for op in module.walk() if op.name == "llvm.inline_asm"]
        context = None
        task_compute_count = task_command_count = 0
        for row in contexts["motifs"]:
            if row["state_missing"]:
                continue
            owned = [
                index
                for index, op in enumerate(asm)
                if getattr(getattr(op.attributes.get("merlin.global_task"), "value", None), "data", None)
                == row["task_index"]
            ]
            count = sum(trace["instructions"][index].get("class") in INHERITS_DESTINATION for index in owned)
            # Never execute a complete source layer's only compute and call it a reduced prefix.
            if count > 1:
                context, task_compute_count, task_command_count = row, count, len(owned)
                break
        if context is None:
            raise ValueError(
                "no bounded initialized competing-load prefix strictly reduces a multi-compute source task"
            )
        short_module, source_slice = slice_context_source(module, context, target=self.target)
        source_slice.update(
            {
                "model_instruction_count": len(trace["instructions"]),
                "source_task_instruction_count": task_command_count,
                "source_task_compute_pair_count": task_compute_count,
                "executed_compute_pair_count": 1,
                "task_reads": context["task_reads"],
                "task_writes": context["task_writes"],
            }
        )
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
            short,
            work / "runtime",
            timeout_seconds=compile_budget,
            profile_counters=True,
            include_operand_movement=True,
        )
        if (
            prepared.get("measurement_scope") != "controlled_source_prefix"
            or prepared["source_artifact_sha256"] != source_slice["slice_source_sha256"]
            or prepared["timed_instruction_count"] != len(context["instruction_indices"])
            or prepared["host_input_bytes"] > 65536
            or prepared["output_storage_bytes"] > 65536
        ):
            raise ValueError("prepared controlled context scope/source/footprint differs from admitted source slice")
        if probes.analysis.session.current_probe_binding(candidate) != binding:
            raise ValueError("compiler, graph, plan or target changed during controlled prefix preparation")

        # Exact command multiset + ABI/readout footprint supports only a later scoped A/B check;
        # dropping competing loads or narrowing the timer changes this contract.
        work_contract = {
            "source_op_indices": source_slice["source_op_indices"],
            "source_argument_of_slice_argument": source_slice["source_argument_of_slice_argument"],
            "timed_command_multiset": sorted(
                json.dumps(row, sort_keys=True) for row in context["instruction_semantics"]
            ),
            "initial_configurations": {
                kind: {key: row.get(key) for key in ("class", "funct", "rs1", "rs2", "decoded")}
                for kind, row in context["initial_configurations"].items()
            },
            "computed_tile": prepared["computed_tile"],
            "host_input_bytes": prepared["host_input_bytes"],
            "output_storage_bytes": prepared["output_storage_bytes"],
            "output_row_stride_bytes": prepared["output_row_stride_bytes"],
            "competing_load_count": len(source_slice["queued_competing_movement_indices"]),
        }
        source_slice["work_contract"] = work_contract
        source_slice["work_contract_sha256"] = hashlib.sha256(
            json.dumps(work_contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        source_slice["binding"] = binding.to_dict()
        (work / "source_slice.json").write_text(json.dumps(source_slice, indent=2) + "\n")

        def execute(*, timeout_s: float):
            if probes.analysis.session.current_probe_binding(candidate) != binding:
                raise ValueError("stale compiler or model immediately before controlled context execution")
            actual = self.adapter.execute_prepared_primitive(
                prepared, timeout_seconds=min(float(timeout_s), remaining())
            )
            if actual.get("measurement_scope") != "controlled_source_prefix":
                raise ValueError("runtime returned a different measurement scope")
            return actual

        return {
            "controlled_context_inputs": {
                "binding": binding,
                "model_artifact_sha256": source_sha,
                "source_slice": source_slice,
                "prepared": prepared,
                "timeout_seconds": remaining(),
                "scope": "controlled_source_prefix",
                "source_slice_receipt": str(work / "source_slice.json"),
            },
            "execute": execute,
        }


# ---- paired_context_provider -------------------------------------------------------------------
# Bounded, fixed-work source-order diagnostics from two retained full-model revisions.


def _digest(value):
    return _mjson.canonical_sha256(value, allow_nan=True)


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

    def __call__(self, *, candidate: Path, probes: Any, timeout_s: float = 60) -> dict[str, Any]:
        from xdsl.printer import Printer

        from merlin.target.isa.rocc import decode

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

        bindings = {
            "before": probes.analysis.session.previous_probe_binding(candidate),
            "after": probes.analysis.session.current_probe_binding(candidate),
        }
        if (
            bindings["before"].graph_digest != bindings["after"].graph_digest
            or bindings["before"].target_digest != bindings["after"].target_digest
        ):
            raise ValueError("fixed-work comparison changed logical graph or target")
        captured = {
            "before": dict(probes.analysis.session.previous_artifacts(candidate)),
            "after": dict(probes.analysis.session.current_artifacts(candidate)),
        }
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
            before["decoded_trace"],
            target=self.target,
            artifact_sha256=before["candidate_lowered_sha256"],
            artifact_text=before["lowered_text"],
            command_buffer=before["command_buffer"],
            parsed_module=before["parsed_lowered_module"],
            max_commands=self.max_commands,
        )
        asm = [op for op in before["parsed_lowered_module"].walk() if op.name == "llvm.inline_asm"]
        anchor, task_compute_count, task_command_count = None, 0, 0
        for row in contexts["motifs"]:
            if row["state_missing"]:
                continue
            owned = [
                index
                for index, op in enumerate(asm)
                if getattr(getattr(op.attributes.get("merlin.global_task"), "value", None), "data", None)
                == row["task_index"]
            ]
            count = sum(
                before["decoded_trace"]["instructions"][index]["class"] in INHERITS_DESTINATION for index in owned
            )
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
            module, source_slice = slice_context_source(
                captured[arm]["parsed_lowered_module"], context, target=self.target, fixed_work_projection=True
            )
            source_slice.update(
                {
                    "model_instruction_count": len(captured[arm]["decoded_trace"]["instructions"]),
                    "source_task_instruction_count": task_command_count,
                    "source_task_compute_pair_count": task_compute_count,
                    "executed_compute_pair_count": 1,
                    "work_contract_sha256": proof["work_contract_sha256"],
                    "task_reads": context["task_reads"],
                    "task_writes": context["task_writes"],
                }
            )
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
                source,
                work / f"{arm}_runtime",
                timeout_seconds=budget,
                profile_counters=True,
                include_operand_movement=True,
                fixed_work_slice=True,
            )
            if (
                prepared.get("measurement_scope") != "controlled_fixed_work_slice"
                or prepared["source_artifact_sha256"] != source_slice["slice_source_sha256"]
                or prepared["timed_instruction_count"] != len(context["instruction_indices"])
                or prepared["host_input_bytes"] > 65536
                or prepared["output_storage_bytes"] > 65536
            ):
                raise ValueError("prepared fixed-work source/scope/footprint changed")
            # Exact wrapper bytes bind initialized values, golden output, arguments and timer ROI.
            input_contracts[arm] = {
                key: prepared[key]
                for key in (
                    "wrapper_sha256",
                    "computed_tile",
                    "host_input_bytes",
                    "output_storage_bytes",
                    "output_row_stride_bytes",
                )
            }
            input_contracts[arm]["source_argument_of_slice_argument"] = source_slice[
                "source_argument_of_slice_argument"
            ]
            arms[arm] = {
                "source_slice": source_slice,
                "prepared": prepared,
                "work_contract_sha256": proof["work_contract_sha256"],
            }
        if input_contracts["before"] != input_contracts["after"]:
            raise ValueError("paired work changed initialized inputs, expected output or measured wrapper")
        contract = input_contracts["before"]
        contract_sha = _digest(contract)
        for arm in arms:
            arms[arm]["deterministic_input_contract_sha256"] = contract_sha
            (work / f"{arm}.source_slice.json").write_text(json.dumps(arms[arm]["source_slice"], indent=2) + "\n")
        (work / "projection_proof.json").write_text(json.dumps(proof, indent=2) + "\n")

        def check_binding():
            if (
                probes.analysis.session.current_probe_binding(candidate) != bindings["after"]
                or probes.analysis.session.previous_probe_binding(candidate) != bindings["before"]
            ):
                raise ValueError("compiler, graph, plan or target changed during fixed-work comparison")

        check_binding()
        executed = set()

        def execute(*, arm: str, timeout_s: float):
            if arm not in arms or arm in executed:
                raise ValueError("fixed-work arm absent or already executed")
            check_binding()
            executed.add(arm)
            actual = self.adapter.execute_prepared_primitive(
                arms[arm]["prepared"], timeout_seconds=min(float(timeout_s), remaining())
            )
            if actual.get("measurement_scope") != "controlled_fixed_work_slice":
                raise ValueError("runtime returned a different fixed-work scope")
            return actual

        return {
            "paired_context_inputs": {
                "before_binding": bindings["before"],
                "after_binding": bindings["after"],
                "before_model_artifact_sha256": captured["before"]["candidate_lowered_sha256"],
                "after_model_artifact_sha256": captured["after"]["candidate_lowered_sha256"],
                "scope": "controlled_fixed_work_slice",
                "projection_proof": proof,
                "work_contract_sha256": proof["work_contract_sha256"],
                "deterministic_input_contract": contract,
                "deterministic_input_contract_sha256": contract_sha,
                "arms": arms,
                "timeout_seconds": remaining(),
            },
            "execute": execute,
        }
