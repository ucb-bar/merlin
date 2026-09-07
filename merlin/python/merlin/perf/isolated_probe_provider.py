"""Host-owned bridge from current whole-model artifacts to one separate short probe.

This connector calibrates only an isolated primitive reference cost. It does not equate drained
probe timing with the containing layer's latency, and never fits/extrapolates from one sample.
The target adapter supplies executable setup/body/readback instrumentation and target-derived
representation facts. Candidate manifests cannot supply any equivalence or timing evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from time import monotonic
from typing import Any

from .execution_policy import SimulationBudget, WarmComputeReceipt, WarmProfileContract
from .instruction_motif import initialized_compute_primitives
from .mechanism_probe import MechanismEvidence, ProbeObservation


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class IsolatedPrimitiveProbeProvider:
    """A real ``provider(candidate, experiment, timeout_s)`` for the global action broker.

    ``experiment`` retains host-derived artifacts from its latest whole-model analysis, exposes
    ``current_artifacts`` and ``current_probe_binding``, and accounts elapsed preparation through
    ``charge_probe_preparation``. ``adapter`` is selected by the host, never the candidate. The
    first supported extraction family is initialized command-ABI overwrite primitives; missing
    initialization, different instruction state, changed executable, or absent timing evidence
    refuses measurement while leaving structural whole-model optimization available.
    """

    def __init__(self, *, target: str, short_interface: Path, adapter: Any,
                 runtime_receipt: Path, output: Path, profile_counters: bool = False):
        self.target = target
        self.short_interface = Path(short_interface).resolve()
        self.short_interface_sha256 = _sha(self.short_interface.read_bytes())
        self.adapter = adapter
        self.runtime_receipt = Path(runtime_receipt).resolve()
        self.runtime_receipt_sha256 = _sha(self.runtime_receipt.read_bytes())
        self.output = Path(output).resolve()
        self.profile_counters = profile_counters

    def __call__(self, *, candidate: Path, experiment: Any,
                 timeout_s: float | None = None) -> dict[str, Any]:
        from merlin.targetgen import gsim_emulator
        from merlin.targetgen.rocc import decode

        started = monotonic()
        limit = min(600.0, float(experiment.timeout_s),
                    600.0 if timeout_s is None else float(timeout_s))
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("probe provider needs a positive bounded broker deadline")
        deadline = started + limit
        binding = experiment.current_probe_binding(candidate)
        captured = experiment.current_artifacts(candidate)
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
            emitted = experiment.compile_probe_candidate(
                Path(candidate), self.short_interface, work / "compiler_scratch",
                timeout_s=remaining())
            (work / "probe_compile.log").write_text(emitted.stderr or "")
            if emitted.returncode or not emitted.stdout:
                raise ValueError("same-candidate short probe compilation failed")
            short = work / "probe.target.mlir"
            short.write_text(emitted.stdout)
            module = decode._parse_module(emitted.stdout)
            if module is None:
                raise ValueError("same-candidate short probe artifact does not parse")
            probe_primitives = initialized_compute_primitives(
                decode.decode_module(module, target=self.target), target=self.target)
            if not probe_primitives or probe_primitives[0]["missing"]:
                raise ValueError("short probe has no fully initialized first primitive")
            primitive = probe_primitives[0]
            matched = [row for row in model_primitives if not row["missing"] and
                       row["domain_digest"] == primitive["domain_digest"]]
            if len(matched) <= 1:
                raise ValueError("separate short probe does not reduce a repeated model mechanism")
            profile_options = {"profile_counters": True} if self.profile_counters else {}
            prepared = self.adapter.prepare_primitive_probe(
                short, work / "runtime", timeout_seconds=min(600, int(remaining())), **profile_options)
            if prepared["domain_digest"] != primitive["domain_digest"]:
                raise ValueError("prepared executable extracted a different primitive")
            if experiment.current_probe_binding(candidate) != binding:
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
            if (prior.get("engine_provenance") != engine or not prior.get("correct")
                    or prior.get("warmup_runs") != 1 or prior.get("measured_runs") != 1):
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
            evidence = {"schema": "isolated_primitive_provider_v1", "binding": binding.to_dict(),
                        "model_artifact_sha256": model.artifact_digest,
                        "short_source_sha256": self.short_interface_sha256,
                        "short_artifact_sha256": _sha(emitted.stdout.encode()),
                        "probe_elf_sha256": probe.artifact_digest,
                        "runtime_elf_sha256_excluding_debug_metadata": runtime_elf_digest,
                        "model_instruction_indices": [row["instruction_indices"] for row in matched],
                        "probe_instruction_indices": primitive["instruction_indices"],
                        "domain": primitive["domain"], "signature": signature.to_dict(),
                        "runtime_basis": {"receipt": str(self.runtime_receipt),
                                          "sha256": self.runtime_receipt_sha256,
                                          "estimate": "twice complete same-ELF same-engine diagnostic wall time"},
                        "in_context_cycles": None, "full_model_cycles": None,
                        "full_model_executed": False,
                        "scope": "isolated reference cost; model contention and overlap UNKNOWN",
                        "counter_uncertainty_scope": "one-cycle counter resolution only; run variance UNKNOWN"}
            evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        finally:
            # Preparation is part of this iteration even when extraction or admission fails.
            experiment.charge_probe_preparation(candidate, monotonic() - started)

        budget = min(remaining(), float(experiment.timeout_s))
        inputs = {"model": model, "probe": probe,
                  "descriptor": {"kind": "mechanism_probe",
                                 "performance": {"measurement_scope": "isolated_primitive"}},
                  "budget": SimulationBudget(budget, budget),
                  "estimated_cycles": prior_cycles,
                  "measured_cycles_per_second": prior_cycles / prior_wall,
                  "contract": WarmProfileContract()}

        def execute(*, timeout_s: float) -> ProbeObservation:
            execution_started = monotonic()
            if experiment.current_probe_binding(candidate) != binding:
                raise ValueError("stale candidate immediately before primitive execution")
            actual = self.adapter.execute_prepared_primitive(
                prepared, timeout_seconds=min(float(timeout_s), remaining()))
            if actual["elf_sha256"] != probe.artifact_digest:
                raise ValueError("executed ELF differs from admitted ELF")
            receipt = WarmComputeReceipt(
                workload=primitive["domain_digest"], total_compute_cycles=actual["total_compute_cycles"],
                contract=WarmProfileContract(),
                provenance=str(work / "runtime" / "execution_receipt.json"))
            return ProbeObservation(probe, receipt, monotonic() - execution_started,
                                    1.0, actual["elf_sha256"], actual.get("counter_profile"))

        return {"admission_inputs": inputs, "execute": execute,
                "equivalence_receipt": str(evidence_path),
                "scope": "isolated primitive only; no fitted model or in-context cycle prediction"}
