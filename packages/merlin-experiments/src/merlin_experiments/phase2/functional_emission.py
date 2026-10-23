"""Compare functional emissions and regrade changed execution dependencies."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes as _sha256
from merlin.targetgen.target_experiment import TargetExperiment
from . import campaign as PC, contracts as CONTRACTS, functional_inputs as FI
from .contracts import StageGateError, sha256_file as _sha256_file

_GUARD_UNCHANGED = "unchanged"
_GUARD_CHANGED = "changed"


def functional_emission_guard(
    baseline: Path,
    candidate: Path,
    target_experiment: TargetExperiment,
    *,
    frozen_functional: FI.FrozenFunctionalInputs,
    active_compiler_library,
    timeout_s: int = 120,
    grade_root: Path | None = None,
    sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs | None = None,
) -> dict[str, Any]:
    """Reuse complete unchanged execution dependencies or regrade full outputs.

    Identical LLVM and command buffers alone cannot establish identical behavior:
    runtime, packing, linking and publication code also contribute. Reuse requires
    the entire package and every selected frozen execution grant to be unchanged.
    Any remaining capsule goes through the ordinary numerical/certification engine.

    Trace findings are advisory diagnostics, not numerical verdicts. Compare
    them against the certified baseline to identify introduced regressions;
    their presence alone does not establish a new failure.
    """
    from merlin.targetgen import oot_runner as OR  # noqa: PLC0415
    from merlin.targetgen import trace_check as TCK  # noqa: PLC0415
    from merlin.targetgen.rocc import decode as RD  # noqa: PLC0415
    from merlin_experiments.phase1 import corpus_inputs

    from . import functional_guard_execution as guard_execution

    if frozen_functional.host_provenance is None:
        raise StageGateError("functional emission guard requires admitted V4 host provenance")
    verified = FI._verify_private_functional_provenance(frozen_functional.host_provenance)
    from merlin_experiments.phase1.brokers.common_transforms import CommonTransformService
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import original_compiler_execution

    profile_service = (
        None
        if verified["bundle"].get("compiler_tool_profile") is None
        else CommonTransformService(
            workspace=verified["workspace"],
            bundle_manifest=Path(frozen_functional.host_provenance["bundle_manifest"]),
            repo=verified["repo"],
        )
    )
    environment_path = Path(frozen_functional.host_provenance["environment"])
    environment = CONTRACTS.mapping_file(environment_path, yaml_file=True)
    # The grading view is a declared host input, not an agent-visible grant. Reuse
    # Phase 1's frozen resolver; never rediscover live descriptor siblings or mix
    # hidden answers and performance-development corpora into its public cohort.
    try:
        view = corpus_inputs.resolve(
            verified["workspace"],
            verified["bundle"],
            environment.get("public_corpus_input"),
            repo=verified["repo"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise StageGateError(f"functional public corpus is unavailable: {exc}") from exc
    if _sha256_file(environment_path) != frozen_functional.host_provenance["environment_sha256"]:
        raise StageGateError("functional host provenance changed during corpus selection")
    capsules = sorted(path.parent for path in view.public.rglob("capsule.yaml"))
    if not capsules:
        raise StageGateError("functional frozen public corpus contains no capsules")

    base_pkg, cand_pkg = OR.load_package(Path(baseline)), OR.load_package(Path(candidate))
    package_identity = (
        CONTRACTS.exact_tree_record(Path(baseline))["sha256"],
        CONTRACTS.exact_tree_record(Path(candidate))["sha256"],
    )
    dependencies_unchanged = package_identity[0] == package_identity[1] and active_compiler_library() is None
    grants = getattr(frozen_functional, "grants", ())
    # An omitted closure is UNKNOWN, including an older synthetic/archive record.
    dependencies_unchanged = dependencies_unchanged and bool(grants)
    for grant in grants:
        source, selected = Path(grant.source), Path(grant.destination)
        identity = CONTRACTS.exact_tree_record if source.is_dir() else _sha256_file
        frozen_identity = identity(source)
        frozen_digest = frozen_identity["sha256"] if isinstance(frozen_identity, dict) else frozen_identity
        if frozen_digest != grant.source_sha256:
            raise StageGateError("functional execution dependency changed after admission")
        if not selected.exists() or selected.is_symlink():
            dependencies_unchanged = False
            continue
        selected_identity = CONTRACTS.exact_tree_record(selected) if selected.is_dir() else _sha256_file(selected)
        selected_digest = selected_identity["sha256"] if isinstance(selected_identity, dict) else selected_identity
        dependencies_unchanged = dependencies_unchanged and selected_digest == frozen_digest
    rows: list[dict[str, Any]] = []
    offenders: list[dict[str, Any]] = []
    for capsule_dir in capsules:
        name = capsule_dir.name
        try:
            descriptor = CONTRACTS.mapping_file(capsule_dir / "capsule.yaml", yaml_file=True)
        except Exception:  # noqa: BLE001 - an unreadable descriptor is reported, never skipped silently
            offenders.append({"capsule": name, "kind": "descriptor_unreadable"})
            continue
        interface = capsule_dir / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
        if not interface.is_file():
            rows.append({"capsule": name, "status": "no_interface"})
            offenders.append({"capsule": name, "kind": "interface_missing"})
            continue
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as raw:
            scratch = Path(raw)
            with original_compiler_execution(
                profile_service,
                target_experiment=target_experiment,
                candidate=baseline,
                work_root=scratch / "profile-baseline",
                sandbox_inputs=sandbox_inputs,
            ):
                base_rc, base_llvm, base_buffer = guard_execution.emit_pair(
                    base_pkg, interface, scratch, "baseline", timeout_s, target_experiment, inputs=sandbox_inputs
                )
            with original_compiler_execution(
                profile_service,
                target_experiment=target_experiment,
                candidate=candidate,
                work_root=scratch / "profile-candidate",
                sandbox_inputs=sandbox_inputs,
            ):
                cand_rc, cand_llvm, cand_buffer = guard_execution.emit_pair(
                    cand_pkg, interface, scratch, "candidate", timeout_s, target_experiment, inputs=sandbox_inputs
                )
        if base_rc != 0:
            offenders.append(
                {"capsule": name, "kind": "baseline_emission_failed", "baseline_rc": base_rc, "candidate_rc": cand_rc}
            )
            continue
        if base_rc == 0 and cand_rc != 0:
            offenders.append(
                {"capsule": name, "kind": "lowering_regressed", "baseline_rc": base_rc, "candidate_rc": cand_rc}
            )
            continue
        emissions_equal = _sha256(base_llvm.encode("utf-8")) == _sha256(cand_llvm.encode("utf-8")) and _sha256(
            base_buffer.encode("utf-8")
        ) == _sha256(cand_buffer.encode("utf-8"))
        if dependencies_unchanged and emissions_equal:
            rows.append({"capsule": name, "status": _GUARD_UNCHANGED})
            continue
        row: dict[str, Any] = {"capsule": name, "status": _GUARD_CHANGED}
        if emissions_equal:
            row["reason"] = "execution dependencies changed or are unavailable"
            rows.append(row)
            continue
        try:
            expected = descriptor.get("expected") or {}
            base_trace = RD.decode_text(base_llvm, source="baseline", target=target_experiment.target)
            cand_trace = RD.decode_text(cand_llvm, source="candidate", target=target_experiment.target)
            base_findings = set(
                TCK.check(
                    base_trace,
                    expected,
                    json.loads(base_buffer) if base_buffer else None,
                    target=target_experiment.target,
                )["violations"]
            )
            cand_findings = set(
                TCK.check(
                    cand_trace,
                    expected,
                    json.loads(cand_buffer) if cand_buffer else None,
                    target=target_experiment.target,
                )["violations"]
            )
            introduced = sorted(cand_findings - base_findings)
            row["introduced_findings"] = introduced
            row["drives_accelerator"] = [
                bool(TCK.drives_accelerator(base_trace)),
                bool(TCK.drives_accelerator(cand_trace)),
            ]
            if row["drives_accelerator"] == [True, False]:
                offenders.append({"capsule": name, "kind": "accelerator_dispatch_regressed"})
            if introduced:
                offenders.append({"capsule": name, "kind": "trace_findings_introduced", "findings": introduced[:8]})
        except Exception as exc:  # noqa: BLE001 - an undecodable candidate trace is absence of proof
            row["decode_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            offenders.append({"capsule": name, "kind": "trace_not_decodable"})
        rows.append(row)

    regrade = None
    if any(row.get("status") == _GUARD_CHANGED for row in rows):
        from merlin.targetgen import capsule_grade as CG  # noqa: PLC0415
        from merlin.targetgen import capsule_runner as CR  # noqa: PLC0415

        # Grade the complete admitted public cohort: a changed shared producer,
        # runtime or host publication routine can affect consumers elsewhere.
        with tempfile.TemporaryDirectory(prefix="functional-regrade-") as scratch:
            output = Path(scratch) if grade_root is None else Path(grade_root)
            if grade_root is not None:
                if output.exists():
                    raise StageGateError("functional regrade evidence destination is not fresh")
                output.mkdir(parents=True)
            try:
                adapters = CR.qa_checkpoint_adapters(target_experiment.target, target_experiment.sim_via)
                import shutil  # noqa: PLC0415

                grading_candidate = output / "compiler"
                shutil.copytree(candidate, grading_candidate)
                with original_compiler_execution(
                    profile_service,
                    target_experiment=target_experiment,
                    candidate=grading_candidate,
                    work_root=output / "profile-package-work",
                    sandbox_inputs=sandbox_inputs,
                ):
                    guard_execution.require_inline_models(capsules)
                    with guard_execution.boxed_guard_package(
                        target_experiment, output, grading_candidate, inputs=sandbox_inputs
                    ):
                        score = CG.grade(
                            grading_candidate,
                            capsules_root=view.public,
                            runs_root=output,
                            model_snapshot_root=output / ".private_model_sources",
                            labels={"public", "dev"},
                            contract=view.contract,
                            oracle_adapters=adapters,
                            timeout=timeout_s,
                            target=target_experiment.target,
                        )
                scored = score.get("per_capsule") or []
                expected_names = {
                    CONTRACTS.mapping_file(path / "capsule.yaml", yaml_file=True).get("name", path.name)
                    for path in capsules
                }
                complete = (
                    len(scored) == len(expected_names)
                    and {row.get("capsule") for row in scored} == expected_names
                    and all(
                        row.get("status") == "pass" and row.get("numeric") == "pass" and not row.get("cert_verdict")
                        for row in scored
                    )
                    and score.get("integrity_status") == "clean"
                )
                regrade = {
                    "status": "pass" if complete else "fail",
                    "score": score,
                    "evidence_root": str(output) if grade_root is not None else None,
                }
                if not complete:
                    offenders.append({"kind": "complete_functional_regrade_failed"})
            except Exception as exc:  # noqa: BLE001 - missing execution remains a refusal
                regrade = {"status": "unavailable", "error_type": type(exc).__name__}
                offenders.append({"kind": "complete_functional_regrade_unavailable"})
    if CONTRACTS.exact_tree_record(Path(candidate))["sha256"] != package_identity[1]:
        offenders.append({"kind": "candidate_changed_during_functional_guard"})
    FI._verify_private_functional_provenance(frozen_functional.host_provenance)

    unchanged = sum(1 for row in rows if row.get("status") == _GUARD_UNCHANGED)
    changed = sum(1 for row in rows if row.get("status") == _GUARD_CHANGED)
    return {
        "status": "clean" if not offenders else "offending",
        "capsules": len(rows),
        "proved_unchanged": unchanged,
        "changed": changed,
        "offenders": offenders,
        "rows": rows,
        "execution_dependencies_unchanged": dependencies_unchanged,
        "full_output_regrade": regrade,
    }
