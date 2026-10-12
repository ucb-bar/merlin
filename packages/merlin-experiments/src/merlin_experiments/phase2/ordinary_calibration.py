"""Live public calibration for the normal retained compiler/Spike/RTL route.

Only this fixed producer issues an owner. Imported reports, callback factories,
structural ceilings and engine equivalence cannot qualify an empirical screen.
Pairs compare executable alternatives on the same member and configuration.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.perf.component_screen import ComponentScreenPolicy, qualify_component_screen
from merlin.perf.fast_estimate_validation import Observation, fit_linear_screen
from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME

from . import ordinary_fast_feedback as FAST
from . import ordinary_screen_features as FEATURES
from .contracts import StageGateError, canonical_json, document_sha256, mapping_file, sha256_file, write_json

_ISSUER = object()
PARTITION_SCHEMA = "merlin.ordinary_public_screen_partition.v1"


class CalibrationUnavailable(StageGateError):
    """Original observations cannot qualify a screen; retained costs remain useful."""


def domain(policy):
    binding = policy.binding()
    return document_sha256(
        {
            "target": binding["target"],
            "descriptor": binding["target_descriptor_sha256"],
            "certificate": binding["certificate"],
            "rtl": binding["rtl_identity"],
            "native_runtime_view": binding.get("native_runtime_view"),
            "build_runtime_view": binding.get("build_runtime_view"),
            "contract": binding["contract"],
            "sources": binding["runtime_source_pins"],
            # Temporary output placement is private per job. Every other effective
            # environment selection remains in the empirical execution domain.
            "environment_sha256": document_sha256({key: value for key, value in os.environ.items() if key != "TMPDIR"}),
            "scope": "original selected member timing boundary; external host wall cost UNKNOWN",
            "features": FEATURES.FEATURES,
        }
    )


def public_partition(policy, partition):
    if (
        type(partition) is not dict
        or set(partition) != {"schema", "calibration", "held_validation"}
        or (partition["schema"] != PARTITION_SCHEMA)
    ):
        raise StageGateError("ordinary calibration needs an explicit closed public partition")
    selected, expected = {}, {(row.family, row.capsule) for row in policy.search_corpus.capsules}
    for name in ("calibration", "held_validation"):
        values = partition[name]
        if type(values) is not list or not values:
            raise StageGateError("ordinary calibration partitions must both be nonempty")
        for row in values:
            if type(row) is not list or len(row) != 2 or any(type(value) is not str for value in row):
                raise StageGateError("ordinary calibration requires original public member identities")
            identity = tuple(row)
            if identity not in expected or identity in selected:
                raise StageGateError("ordinary calibration repeats, leaks or substitutes a public member")
            selected[identity] = name
    if set(selected) != expected:
        raise StageGateError("ordinary calibration partition must cover the exact frozen public roster")
    return selected


def _join(timing_root, cell, arm, compiled, package_sha256):
    """Reopen actual normal attempts and original selected ELF/process receipts."""
    matches = []
    for path in timing_root.rglob("*.attempts.json"):
        for attempt in mapping_file(path).get("attempts", []):
            raw = attempt.get("raw") or {}
            execution, measured = raw.get("execution") or {}, raw.get("measurement") or {}
            if (execution.get("family"), execution.get("capsule"), execution.get("arm")) != (
                cell["family"],
                cell["capsule"],
                arm,
            ):
                continue
            observed = (measured.get("gsim_execution") or {}).get("gsim") or {}
            if measured.get("numeric") != "pass" or not (measured.get("gsim_qualification") or {}).get("admitted"):
                continue
            if observed.get("reused_measurement") or not observed.get("elf"):
                continue
            elf = Path(observed["elf"])
            if not elf.is_relative_to(timing_root) or sha256_file(elf) != observed.get("elf_sha256"):
                raise StageGateError("ordinary calibration selected ELF changed or escaped its owner")
            workspace = Path(attempt["workspace"])
            if workspace.resolve() != workspace or not workspace.is_relative_to(timing_root):
                raise StageGateError("ordinary calibration workspace escaped its original timing owner")
            native = compiled.get("artifact_join", {}).get("kind") == "original_native_model_products"
            native_pins = ()
            if native:
                from merlin_experiments.phase1.feedback.candidate_observations import (
                    _reopen_product,
                    reopen_native_products,
                )

                if not isinstance(observed.get("actual_source"), dict):
                    raise CalibrationUnavailable("ordinary native calibration lacks its original model source")
                products, producers = reopen_native_products(
                    workspace, emission=observed.get("actual_candidate"),
                    elf_pin={"path": str(elf), "sha256": observed["elf_sha256"]},
                    source=observed.get("actual_source"),
                )
                identity = compiled["artifact_join"]
                if (sha256_file(products["elf"]) != compiled["elf_sha256"]
                        or sha256_file(products["emitted"]) != compiled["llvm_mlir_sha256"]
                        or sha256_file(products["source"]) != identity["source_sha256"]
                        or sha256_file(products["command_buffer"]) != identity["command_buffer_sha256"]):
                    continue
                console = _reopen_product(observed.get("actual_console"), root=workspace)
                invocation_pin = observed.get("execution_invocations", {}).get("L3")
                actual_invocation = _reopen_product(invocation_pin, root=workspace)
                native_pins = tuple((str(path), sha256_file(path))
                                    for path in (*products.values(), *producers, console))
            else:
                images = {sha256_file(p) for p in workspace.rglob(PACKAGE_ELF_NAME)
                          if p.is_file() and not p.is_symlink()}
                sources = {sha256_file(p) for p in workspace.rglob("lowered.llvm.mlir")
                           if p.is_file() and not p.is_symlink()}
                if compiled["elf_sha256"] not in images or compiled["llvm_mlir_sha256"] not in sources:
                    continue
            process = []
            for invocation in workspace.rglob("invocation.json"):
                if native and invocation != actual_invocation:
                    continue
                row = mapping_file(invocation)
                if (
                    row.get("stage")
                    in {"execution", "selected_rocc_process", "native_oracle_process", "native_runtime_process"}
                    and FAST.successful_record(row)
                    and any(pin.get("sha256") == observed["elf_sha256"] for pin in row.get("inputs", []))
                ):
                    row = I.verify(invocation)
                    if native:
                        engine = observed.get("binary_sha256")
                        selected = [row.get("executable") or {}, *row.get("dependencies", [])]
                        if (not engine or not any(pin.get("sha256") == engine for pin in selected)
                                or row.get("stdout", {}).get("sha256") != observed["actual_console"]["sha256"]
                                or {"path": str(elf), "sha256": observed["elf_sha256"]} not in row.get("inputs", [])):
                            continue
                    process.append((str(invocation), sha256_file(invocation)))
            if not process or mapping_file(path).get("package_sha256") != package_sha256:
                continue
            matches.append((observed["elf_sha256"], (str(path), sha256_file(path)), (*process, *native_pins)))
    if len(matches) != 1:
        raise CalibrationUnavailable(
            "ordinary calibration lacks one fresh joined compile/full-numeric/selected-engine observation"
        )
    return matches[0]


def _cheap_observation(cheap_root, cell, arm, member_index):
    """Reopen deterministic host products; never follow a feedback-supplied path."""
    path = cheap_root / f"m{member_index:03d}_{arm}" / "ordinary_observation.json"
    if path.is_symlink() or path.resolve() != path or sha256_file(path) != cell[arm]["evidence_sha256"]:
        raise StageGateError("ordinary private cheap evidence changed or escaped its original owner")
    private = mapping_file(path)
    original = {key: value for key, value in private.items() if key != "original_grade"}
    visible = {key: value for key, value in cell[arm].items() if key != "evidence_sha256"}
    if canonical_json(original) != canonical_json(visible):
        raise StageGateError("ordinary cheap feedback differs from the actual retained host observation")
    return original, (str(path), sha256_file(path))


@dataclass(frozen=True, eq=False)
class OrdinaryCalibration:
    domain_sha256: str
    partition_json: bytes
    observations_json: bytes
    report_json: bytes
    artifacts: tuple[tuple[str, str], ...]
    _issuer: object

    @property
    def sha256(self):
        return document_sha256(
            [
                self.domain_sha256,
                self.partition_json.decode(),
                self.observations_json.decode(),
                self.report_json.decode(),
                self.artifacts,
            ]
        )

    def verify(self, policy):
        if self._issuer is not _ISSUER or self.domain_sha256 != domain(policy):
            raise StageGateError("ordinary calibration is not live or changed original runtime configuration")
        public_partition(policy, json.loads(self.partition_json))
        for path, digest in self.artifacts:
            try:
                changed = Path(path).is_symlink() or sha256_file(Path(path)) != digest
            except OSError:
                changed = True
            if changed:
                raise StageGateError("ordinary calibration original compiled/measured evidence changed")

    def screen(self, policy, document):
        self.verify(policy)
        report = json.loads(self.report_json)
        if report.get("exposable") is not True:
            return {
                "status": "UNKNOWN",
                "reason": "original held ranking/error/coverage gates did not qualify",
                "order": [],
            }
        rows = [
            Observation(**{**row, "evidence_sha256s": tuple(row["evidence_sha256s"])})
            for row in json.loads(self.observations_json)
        ]
        partition = public_partition(policy, json.loads(self.partition_json))
        training = [row for row in rows if row.group == "calibration"]
        model = _fit(training)
        opportunities = []
        for cell in document["cells"]:
            predictions = []
            for arm in ("baseline", "candidate"):
                value = cell[arm]
                prediction = model.predict(FEATURES.project(value["compiled"]), domain_sha256=self.domain_sha256)
                if value["functional_status"] != "PASS":
                    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

                    prediction = CycleInterval.unknown("ordinary compile/Spike functional observation is incomplete")
                predictions.append(prediction.to_dict())
            baseline, candidate = predictions
            saved = baseline["lo"] - candidate["hi"] if baseline["resolved"] and candidate["resolved"] else None
            opportunities.append(
                {
                    "family": cell["family"],
                    "capsule": cell["capsule"],
                    "baseline": baseline,
                    "candidate": candidate,
                    "conservative_saved_cycles": saved,
                    "partition": partition[(cell["family"], cell["capsule"])],
                }
            )
        return {
            "status": "qualified_public_screening",
            "calibration_sha256": self.sha256,
            "policy": ComponentScreenPolicy().to_dict(),
            "evidence": opportunities,
            "order": [
                [row["family"], row["capsule"]]
                for row in sorted(
                    opportunities,
                    key=lambda row: (
                        row["conservative_saved_cycles"] is None,
                        -(row["conservative_saved_cycles"] or 0),
                    ),
                )
            ],
            "scope": "empirical member-cycle screening; physical service prices and excluded host costs UNKNOWN",
            "feature_pointers": list(model.model.pointers),
            "unpriced_feature_pointers": [key for key in FEATURES.FEATURES if key not in model.model.pointers],
        }


@dataclass(frozen=True)
class _ResidualScreen:
    model: object
    relative_residual: float
    observed_domains: tuple

    def predict(self, features, *, domain_sha256):
        from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

        for key, bounds in self.observed_domains:
            value = features.get(key)
            if bounds is None:
                if value is not None:
                    return CycleInterval.unknown("new feature visibility differs from empirical training domain")
            elif value is None or not bounds[0] <= value <= bounds[1]:
                return CycleInterval.unknown("observed work differs from empirical training domain: " + key)
        prediction = self.model.predict(features, domain_sha256=domain_sha256)
        if not prediction.resolved:
            return prediction

        midpoint = (prediction.lo + prediction.hi) / 2
        radius = midpoint * self.relative_residual
        return CycleInterval(max(0.0, midpoint - radius), midpoint + radius, prediction.provenance, ())


def _fit(rows):
    model = fit_linear_screen(
        rows, pointers=FEATURES.identifiable_pointers(rows), include_fixed=True, maximum_condition=1e8
    )
    residual = max(
        abs(model.predict(row.features, domain_sha256=row.domain).lo - row.cycles) / row.cycles for row in rows
    )
    if not math.isfinite(residual):
        raise ValueError("ordinary training residual is unresolved")
    # Empirical training residual only. The original held coverage/error gate
    # must independently qualify these intervals; this is no confidence claim.
    domains = []
    for key in FEATURES.FEATURES:
        values = [row.features.get(key) for row in rows]
        known = [value for value in values if value is not None]
        domains.append((key, (min(known), max(known)) if known else None))
    return _ResidualScreen(model, residual, tuple(domains))


def _qualify(rows, partition_sha256):
    training, held = [r for r in rows if r.group == "calibration"], [r for r in rows if r.group != "calibration"]
    try:
        fitted, failure = _fit(training), None
    except ValueError as error:
        fitted, failure = None, str(error)

    def training_only(_):
        if fitted is None:
            raise ValueError(failure)
        return fitted

    report = qualify_component_screen(held, training_only, calibration_sha256=partition_sha256)
    # The fixed fitter never reads held labels. The pure validator's generic
    # leave-group-out metadata describes its supplied rows; bind our actual
    # independent calibration inputs explicitly instead.
    for prediction in report["predictions"]:
        prediction["training_programs"] = [row.program for row in training]
    return report


def collect_sample(policy, *, cheap_root, timing_root, cheap, timing, revision, baseline_revision, partition):
    """Private fixed-route consumer; reopen actual products before accepting labels."""
    groups = public_partition(policy, partition)
    labels = {(r["family"], r["capsule"]): r for r in timing["cells"]}
    member_indices = {
        (member.family, member.capsule): index
        for index, member in enumerate(sorted(policy.search_corpus.capsules, key=lambda row: (row.family, row.capsule)))
    }
    rows, artifacts = [], []
    for cell in cheap["cells"]:
        identity = cell["family"], cell["capsule"]
        measured = labels[identity]
        if not measured["measured"] or not measured["comparable"]:
            raise CalibrationUnavailable("ordinary calibration requires complete correct measured RTL roster")
        for arm, source in (("baseline", baseline_revision), ("candidate", revision)):
            value, cheap_pin = _cheap_observation(cheap_root, cell, arm, member_indices[identity])
            compiled = value["compiled"]
            if value["functional_status"] != "PASS" or compiled["status"] != "observed":
                raise CalibrationUnavailable("ordinary calibration lacks original functional/compiled observation")
            program, attempt_pin, process_pins = _join(timing_root, cell, arm, compiled, source.source_sha256)
            pins = (cheap_pin, attempt_pin, *process_pins)
            artifacts.extend(pins)
            rows.append(
                Observation(
                    program,
                    cell["source_sha256"],
                    "calibration" if groups[identity] == "calibration" else document_sha256(identity),
                    domain(policy),
                    FEATURES.project(compiled),
                    measured[arm + "_gsim_cycles"],
                    tuple(digest for _, digest in pins),
                )
            )
    return rows, artifacts


def issue_from_samples(policy, *, partition, samples, artifacts, root):
    """Called only by fixed live producers, never deserializes a qualification owner."""
    observed = {}
    for row in samples:
        prior = observed.get(row.id)
        if prior is not None and (prior.features != row.features or prior.cycles != row.cycles):
            raise StageGateError("ordinary calibration identical executable replicates disagree")
        observed[row.id] = row
    rows = tuple(observed.values())
    selected_domain = domain(policy)
    if any(row.domain != selected_domain for row in rows):
        raise StageGateError("ordinary calibration runtime changed during public observations")
    report = _qualify(rows, document_sha256(partition))
    write_json(root / "qualification.json", report)
    artifacts.append((str(root / "qualification.json"), sha256_file(root / "qualification.json")))
    owner = OrdinaryCalibration(
        selected_domain,
        canonical_json(partition),
        canonical_json([asdict(row) for row in rows]),
        canonical_json(report),
        tuple(sorted(set(artifacts))),
        _ISSUER,
    )
    owner.verify(policy)
    return owner


def produce_calibration(policy, revisions, baseline_revision, *, partition, work_root, timeout_s, workers, allocation):
    """Actually compile/execute public alternatives and qualify held groups on allocated AWS CPU.

    Both original fixed routes execute; complete normal RTL feedback supplies
    labels. No imported observation or caller-selected executable is accepted.
    """
    from .timing_study import _allocated

    policy.validate_search_inputs()
    public_partition(policy, partition)
    if (
        type(revisions) is not tuple
        or not 2 <= len(revisions) <= 64
        or len({r.revision_id for r in revisions}) != len(revisions)
    ):
        raise StageGateError("ordinary calibration needs multiple distinct issued public alternatives")
    _allocated({"resources": allocation})
    root = Path(work_root)
    if root.exists() or root.resolve() != root or not root.is_relative_to(Path(allocation["work_root"])):
        raise StageGateError("ordinary calibration needs a fresh allocated private output owner")
    root.mkdir(parents=True, mode=0o700)
    selected_domain, samples, artifacts = domain(policy), [], []
    for index, revision in enumerate(revisions):
        cheap_root, timing_root = root / f"r{index:03d}_cheap", root / f"r{index:03d}_rtl"
        cheap = _bounded_observe(
            policy,
            revision,
            baseline_revision,
            kind=FAST.KIND,
            work_root=cheap_root,
            call_index=index,
            timeout_s=timeout_s,
            workers=workers,
        )
        timing = _bounded_observe(
            policy,
            revision,
            baseline_revision,
            work_root=timing_root,
            call_index=index,
            timeout_s=timeout_s,
            workers=workers,
        )
        write_json(root / f"r{index:03d}_timing.json", timing)
        artifacts.extend(
            (str(p), sha256_file(p))
            for p in (cheap_root / "ordinary_fast_feedback.json", root / f"r{index:03d}_timing.json")
        )
        values, pins = collect_sample(
            policy,
            cheap_root=cheap_root,
            timing_root=timing_root,
            cheap=cheap,
            timing=timing,
            revision=revision,
            baseline_revision=baseline_revision,
            partition=partition,
        )
        samples.extend(values)
        artifacts.extend(pins)
    if domain(policy) != selected_domain:
        raise StageGateError("ordinary calibration runtime changed during public observations")
    return issue_from_samples(policy, partition=partition, samples=samples, artifacts=artifacts, root=root)


def _bounded_observe(policy, revision, baseline_revision, **kwargs):
    from .supervised_feedback import bounded_feedback

    return bounded_feedback(
        _observe_fixed,
        timeout_s=kwargs["timeout_s"],
        output=Path(kwargs["work_root"]).parent / "workers",
        kwargs={"policy": policy, "revision": revision, "baseline_revision": baseline_revision, "options": kwargs},
    )


def _observe_fixed(*, policy, revision, baseline_revision, options):
    return policy.evaluate_retained(revision, baseline_revision, **options, reuse_artifacts=False)
