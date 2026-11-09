"""Observe actual independent build artifacts without a reference backend import.

The controller supplies its private invocation records. This owner reopens those
records and joins their exact inputs and products. Structural RTL intake may name
derived hardware facts, but does not grant instruction, runtime, timer or semantic
authority. The observation is diagnostic until those separate gates are checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.benchharness import hash_tree
from merlin.common import invocation_record
from merlin.perf.component_cost import COMPLETE_STAGES, ComponentCostScope
from merlin.targetgen.elf_lanes import ElfUnreadable, executable_sections

from .component_baseline import ComponentBaselineAdmission, verify_baseline_admission
from .contracts import StageGateError, document_sha256, sha256_file


def _canonical_file(path, *, owner):
    path = Path(path)
    if (not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_file()
        or not path.is_relative_to(owner)):
        raise StageGateError("component observer evidence escapes the private controller owner")
    return path


@dataclass(frozen=True)
class IndependentComponentArtifactObservation:
    baseline_admission_sha256: str
    compiler_sha256: str
    member_sha256: str
    corpus_sha256: str
    target_sha256: str
    scope_sha256: str
    hardware_intake_sha256: str | None
    invocation_files: tuple[tuple[Path, str], ...]
    artifact_files: tuple[tuple[Path, str], ...]
    stages: tuple[str, ...]
    joins: tuple[tuple[str, str], ...]
    executable_sections: tuple[tuple[str, int], ...]

    def verify(self):
        for path, digest in (*self.invocation_files, *self.artifact_files):
            if path.is_symlink() or path.resolve() != path or sha256_file(path) != digest:
                raise StageGateError("independent component observation changed")
        for path, _ in self.invocation_files:
            invocation_record.verify(path)
        return self.sha256

    @property
    def sha256(self):
        return document_sha256({
            "schema": "merlin.independent_component_artifact_observation.v1",
            "baseline_admission_sha256": self.baseline_admission_sha256,
            "compiler_sha256": self.compiler_sha256, "member_sha256": self.member_sha256,
            "corpus_sha256": self.corpus_sha256, "target_sha256": self.target_sha256,
            "scope_sha256": self.scope_sha256, "hardware_intake_sha256": self.hardware_intake_sha256,
            "invocations": [(str(path), digest) for path, digest in self.invocation_files],
            "artifacts": [(str(path), digest) for path, digest in self.artifact_files],
            "stages": self.stages, "joins": self.joins, "executable_sections": self.executable_sections,
        })

    def public_report(self):
        """Opaque identities and structural counts only; no private program text."""
        self.verify()
        return {
            "schema": "merlin.independent_component_artifact_report.v1",
            "observation_sha256": self.sha256,
            "baseline_admission_sha256": self.baseline_admission_sha256,
            "compiler_sha256": self.compiler_sha256,
            "member_sha256": self.member_sha256,
            "corpus_sha256": self.corpus_sha256,
            "target_sha256": self.target_sha256,
            "scope_sha256": self.scope_sha256,
            "hardware_intake_sha256": self.hardware_intake_sha256,
            "observed_stages": list(self.stages),
            "input_product_joins": len(self.joins),
            "static_executable_bytes": sum(size for _, size in self.executable_sections),
            "executable_artifacts": len({digest for digest, _ in self.executable_sections}),
            "instruction_roles": {"status": "UNKNOWN", "reason": "no independently derived instruction semantics"},
            "source_correspondence": {"status": "UNKNOWN", "reason": "invocation pins do not evaluate semantics"},
            "execution_correctness": {"status": "UNKNOWN", "reason": "original output/effect verifier is separate"},
            "cold": {"status": "UNKNOWN", "missing": list(COMPLETE_STAGES)},
            "warm": {"status": "UNKNOWN", "missing": list(COMPLETE_STAGES)},
            "promotion": "DIAGNOSTIC_ONLY",
        }


def observe_independent_component_artifacts(
    *, baseline_admission, compiler, member, corpus, scope, evidence_root, invocation_paths,
    hardware_intake=None,
):
    """Lift actual parent-owned boundaries, without selecting a compiler adapter.

    This does not execute a target. A separate admitted runtime must produce the
    records under an owner inaccessible to the candidate. Serialized records
    alone cannot issue a baseline, runtime, decoder or performance capability.
    """
    if type(baseline_admission) is not ComponentBaselineAdmission or type(scope) is not ComponentCostScope:
        raise StageGateError("independent component observer requires fresh baseline and exact cost scope")
    verify_baseline_admission(baseline_admission, baseline=baseline_admission.baseline,
                              corpus=corpus, target_descriptor=baseline_admission.target_descriptor)
    if member not in corpus.capsules:
        raise StageGateError("independent observer selected a different generated member")
    compiler, owner = Path(compiler), Path(evidence_root)
    if any(path.is_symlink() or path.resolve() != path for path in (compiler, owner)):
        raise StageGateError("independent observer source/evidence requires canonical paths")
    if (owner.is_relative_to(compiler) or compiler.is_relative_to(owner)
        or owner.is_relative_to(corpus.root) or corpus.root.is_relative_to(owner)):
        raise StageGateError("independent observer private evidence overlaps compiler or source inputs")
    if not isinstance(invocation_paths, tuple) or not invocation_paths:
        raise StageGateError("independent observer requires actual private invocation records")
    hardware_sha = None
    if hardware_intake is not None:
        try:
            from merlin_experiments.phase0.rtl_intake import IndependentHardwareIntake
        except ImportError as error:
            raise StageGateError("independent RTL derivation authority is unavailable") from error
        if type(hardware_intake) is not IndependentHardwareIntake:
            raise StageGateError("component observer hardware facts require independently issued RTL intake")
        hardware_intake.verify()
        selected = baseline_admission.qualification
        from .contracts import mapping_file

        if mapping_file(selected.target_descriptor).get("target") != hardware_intake.target:
            raise StageGateError("independent observer hardware intake targets another descriptor")
        hardware_sha = hardware_intake.sha256
    before = str(hash_tree(compiler)["sha256"])
    records, files, products, joins, stages, sections = [], {}, {}, [], [], []
    seen_executables = set()
    for value in invocation_paths:
        path = _canonical_file(value, owner=owner)
        document = invocation_record.verify(path)
        records.append((path, sha256_file(path)))
        stages.append(document["stage"])
        for pin in (*document["inputs"], *document["outputs"], *document["dependencies"],
                    document["stdout"], document["stderr"], document["executable"]):
            source = Path(pin["path"])
            if source.is_symlink() or source.resolve() != source:
                raise StageGateError("independent observer dependency has path indirection")
            files[source] = pin["sha256"]
        for pin in document["inputs"]:
            prior = products.get((pin["path"], pin["sha256"]))
            if prior is not None:
                producer_stage, finished = prior
                if finished > document["started_ns"]:
                    raise StageGateError("component observer product was not completed before its consuming invocation")
                joins.append((producer_stage, document["stage"]))
        for pin in document["outputs"]:
            output = _canonical_file(pin["path"], owner=owner)
            products[str(output), pin["sha256"]] = document["stage"], document["finished_ns"]
            payload = output.read_bytes()
            if payload.startswith(b"\x7fELF") and pin["sha256"] not in seen_executables:
                try:
                    measured = executable_sections(payload)
                except (ElfUnreadable, ValueError, IndexError) as error:
                    raise StageGateError("independent observer executable section census is incomplete") from error
                if not measured:
                    raise StageGateError("independent observer executable has no inspectable code sections")
                sections.extend((pin["sha256"], size) for _, _, size, _ in measured)
                seen_executables.add(pin["sha256"])
    if str(hash_tree(compiler)["sha256"]) != before:
        raise StageGateError("independent observer compiler changed during artifact observation")
    if hardware_intake is not None:
        hardware_intake.verify()
    observation = IndependentComponentArtifactObservation(
        baseline_admission.sha256, before, member.source_sha256, corpus.capsules_sha256,
        baseline_admission.target_sha256, scope.sha256, hardware_sha, tuple(records),
        tuple(sorted(files.items(), key=lambda row: str(row[0]))), tuple(stages), tuple(joins), tuple(sections),
    )
    observation.verify()
    return observation
