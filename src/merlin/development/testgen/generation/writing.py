"""Write admitted generated entries and record their evidence and provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from contextlib import closing
from pathlib import Path

import yaml

from merlin.development.testgen.evidence.program_admission import (
    entry_refusal_is_final,
    screen_written,
)
from merlin.development.testgen.generation.capsule_writer import (
    SYNTH_ROLE,
    UnprovableForbid,
    _finish_capsule,
    _write_capsule,
    _write_capsule_source,
)
from merlin.development.testgen.generation.provenance import (
    _capture_failure_reason,
    _scrub_capsule_dir,
    update_provenance_manifest,
)
from merlin.development.testgen.generation.staging import PreparedWrite, prepare_staged, write_staged
from merlin.development.testgen.profiles.screens import diagnostic_entry, screen_entry
from merlin.development.testgen.sealed_capture.generation import verified_capture_failure
from merlin.development.testgen.sealed_capture.selection import (
    _with_selected_model_recipe,
)


def write_generation(
    *,
    PublicationRefusal,
    _form_perf_coverage,
    _is_roster_capsule,
    _performance_errors,
    _prune_superseded_synth,
    _qualifying_group_capsules,
    _runtime_blocked,
    _sweep_skips,
    artifact_root,
    binding,
    capture_selection_failure,
    claim_plan,
    component_coverage_report,
    component_identity,
    component_only,
    declared_families,
    declared_roles,
    deferred_performance,
    entries,
    evidence,
    evidence_manifest,
    evidence_mode,
    facts,
    family_counts,
    generation_scope,
    hardware_intake,
    hardware_target,
    instruction_policy,
    out_root,
    preallocation_denials,
    profile,
    public_source_eligibility,
    requirement_bytes,
    selected_capture,
    selected_requirement,
    semantic_basis,
    software_intake,
    source_components,
    source_preparation,
    target,
    template,
    withhold_unaccelerated,
    writer_options,
    operator_exclusion,
):
    from merlin.development.testgen.generation.entry_planning import (
        _frozen_application_captures,
        _screen_selected_entry,
    )
    from merlin.development.testgen.generation.manifest import _member_path

    written, failures, unbuilt_roster, unprovable_forbids, omitted, admission = [], [], [], [], [], []
    host_placed: list[dict] = []
    refused_generated: list[Path] = []
    from merlin.development.testgen.generation.staging import ordered_jobs, worker_limit

    entry_decisions = {}

    def prepare_entry(e, *, defer_model=False):
        decisions = []
        entry_decisions[e["name"]] = decisions
        if e["name"] in preallocation_denials:
            raise ValueError(preallocation_denials[e["name"]])
        if capture_selection_failure is not None and (e.get("source") == "pytorch" or e.get("pytorch_ref")):
            raise ValueError("independent frontend obligation unavailable: " + capture_selection_failure)
        if evidence is not None and e.get("micro_model"):
            captures = _frozen_application_captures(artifact_root, evidence_manifest)
            declared = set((evidence.application_inventory or {}).get("applications") or {})
            if set(captures) != declared:
                raise ValueError("micro-model capture roster differs from frozen application inventory")
            e = {
                **e,
                "_frozen_application_captures": captures,
                "_frozen_software_spec": copy.deepcopy(evidence.software_spec),
            }
        e = _with_selected_model_recipe(
            evidence_root=artifact_root if evidence is not None else None,
            target=hardware_target,
            binding=binding,
        )
        if evidence is not None and evidence.software_spec:
            decision = _screen_selected_entry(
                evidence.software_spec,
                binding=binding,
                host_capabilities=evidence.host_capabilities,
            )
            decisions.append({"capsule": e.get("name"), **decision})
            # An entry carries no rank/layout/tail/alias/composition observation, so before it is
            # written only a refusal is final; the written program's own screen below decides the
            # rest, and verified mode requires that screen to admit.
            final = entry_refusal_is_final(e, decision)
            if evidence_mode != "diagnostic" and final:
                raise ValueError("SW operation admission: " + decision["reason"])
            if final and not (
                source_preparation is not None and e.get("op") == "component_program" and e.get("cat") == "_perf"
            ):
                e = diagnostic_entry(e, decision)

        def staged_write():
            writer = _write_capsule_source if defer_model else _write_capsule

            def write(root):
                return writer(e, binding, root, facts.get("sha256", ""), **writer_options)

            if defer_model:
                return prepare_staged(
                    write,
                    lambda written, root: _finish_capsule(
                        e, binding, root, written, component_only=writer_options.get("component_only", False)
                    ),
                    out_root,
                    member=_member_path(e),
                    operator_exclusion=operator_exclusion,
                )
            return write_staged(write, out_root, member=_member_path(e), operator_exclusion=operator_exclusion)

        if evidence is None:
            w = staged_write()
        else:
            from merlin.target.registry.identity import observed_contract
            from merlin.target.rtl.facts import observed_facts

            captured_path = artifact_root / "hardware" / "circt" / "facts.json"
            with (
                observed_contract(hardware_target, evidence.contract),
                observed_facts(
                    hardware_target, evidence.refreshed_facts, captured_path if captured_path.is_file() else None
                ),
            ):
                w = staged_write()
        return e, w

    workers = worker_limit() if getattr(selected_capture, "parallel_safe", False) is True else 1
    prepare_model = (lambda e: prepare_entry(e, defer_model=True)) if workers > 1 else None
    print(f"  [capture] independent workers={workers}; dependent model writes remain ordered", flush=True)
    with closing(ordered_jobs(entries, prepare_entry, workers=workers, prepare_model=prepare_model)) as jobs:
        for e, job in jobs:
            family = (e.get("performance") or {}).get("family")
            try:
                try:
                    e, w = job.result()
                    if isinstance(w, PreparedWrite):
                        w = w.finish()
                finally:
                    admission.extend(entry_decisions.pop(e["name"], []))
            except PublicationRefusal:
                # Operator publication checks cannot become per-member author diagnostics.
                raise
            except Exception as exc:  # noqa: BLE001 — reported, never swallowed
                detail = _capture_failure_reason(exc)
                if isinstance(exc, UnprovableForbid) and str(e.get("source_role") or "") == SYNTH_ROLE:
                    # See `UnprovableForbid`. The capsule is removed rather than left on disk: a directory
                    # the corpus does not list is exactly the kind of half-written state the seal cannot
                    # see, and a stale one would be picked up by the next glob as though it had been built.
                    shutil.rmtree(out_root / str(e.get("cat") or "") / str(e.get("name") or ""), ignore_errors=True)
                    unprovable_forbids.append(
                        {"capsule": e.get("name", "?"), "family": e.get("op"), "reason": str(exc)[:400]}
                    )
                    print(f"  [lane] {e.get('name')}: NOT BUILT — its forbid is not provable on this target")
                    continue
                if _is_roster_capsule(e):
                    # A ROSTER MODEL IS A DECLARED INPUT, NOT A COMPILER RESULT. The roster axis emits one
                    # whole-model capsule per model the target's `workload_spec` names, at the format the
                    # target admits -- and whether a given network can be CAPTURED at that format depends on
                    # things outside this repo: whether the loader's dataset is present, whether the m2m venv
                    # has the model's package, whether torchAO's scheme runs on this host, whether
                    # torch.export accepts the module under quantization. Measured on the four declared
                    # models: one captures, one needs an ImageNet npz, one needs a package the venv lacks,
                    # one is refused by torch.export under W8A8 while its weight-only capture succeeds.
                    #
                    # Aborting the corpus for those would make a fact about the ROSTER read as a broken
                    # generator, and would take every other capsule down with it. Recording it keeps the rule
                    # that matters -- a requirement that produced no capsule is never indistinguishable from
                    # one that is met -- because the manifest names the model and the reason, and no capsule
                    # exists to be graded, so nothing can pass in its place.
                    why = _capture_failure_reason(exc)
                    unbuilt_roster.append(
                        {
                            "capsule": e.get("name", "?"),
                            "model": e.get("model"),
                            "operand_dtype": e.get("operand_dtype"),
                            "quant_scheme": e.get("quant_scheme"),
                            "status": "not_built",
                            "reason": why,
                        }
                    )
                    print(f"  [roster] {e.get('name')}: NOT BUILT — {why}")
                    continue
                failures.append((e.get("name", "?"), detail))
                print(f"  [failure] {e.get('name', '?')}: {detail}", flush=True)
                if family:
                    _performance_errors.append(
                        {
                            "family": family,
                            "member": e.get("name", "?"),
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "detail": str(exc)[:500],
                        }
                    )
                continue
            if w:
                if evidence is not None and evidence.software_spec:
                    actual = yaml.safe_load((Path(w) / "capsule.yaml").read_bytes())
                    observed = screen_written(actual, Path(w), target=hardware_target, evidence=evidence)
                    if observed is None:
                        observed = _screen_selected_entry(
                            evidence.software_spec,
                            binding=binding,
                            capsule=actual,
                            host_capabilities=evidence.host_capabilities,
                        )
                    admission.append({"capsule": e.get("name"), "observation": "emitted_capsule", **observed})
                    # PLACEMENT DECIDES PUBLICATION. A written program with no accelerator-placed operation
                    # grades none of the author's work (host placement is Merlin-supplied), so it is removed
                    # and reported, exactly as an unprovable forbid is, never left on disk for a glob.
                    withheld = withhold_unaccelerated(str(e.get("name", "?")), observed, Path(w))
                    if withheld is not None:
                        host_placed.append(withheld)
                        print(f"  [placement] {e.get('name')}: NOT PUBLISHED — {withheld['reason']}")
                        continue
                    # A must-refuse member is refused by the declarations by construction; that refusal is
                    # what it grades, so it neither moves to the diagnostic cohort nor fails generation.
                    must_refuse = e.get("outcome") == "refuse"
                    if (
                        observed["status"] == "unsupported"
                        and not must_refuse
                        and Path(w).parent.name != "_diagnostic"
                        and not (
                            source_preparation is not None
                            and e.get("op") == "component_program"
                            and e.get("cat") == "_perf"
                        )
                        and (e.get("component_coverage") or {}).get("cohort") != "withheld_transfer"
                    ):
                        diagnostic = out_root / "_diagnostic" / Path(w).name
                        diagnostic.parent.mkdir(parents=True, exist_ok=True)
                        if diagnostic.exists():
                            raise ValueError("diagnostic capsule destination already exists")
                        Path(w).rename(diagnostic)
                        w = diagnostic
                    actual["software_screen"] = observed
                    if component_only:
                        from merlin.development.testgen.components.component_generation import require_written

                        try:
                            require_written(
                                actual,
                                **(
                                    {
                                        "source_software": software_intake,
                                        "source_hardware": hardware_intake,
                                        "directory": Path(w),
                                        "source_preparation": source_preparation,
                                    }
                                    if source_components
                                    else {}
                                ),
                            )
                        except ValueError as exc:
                            failures.append((e.get("name", "?"), str(exc)))
                            refused_generated.append(Path(w))
                            _performance_errors.append(
                                {
                                    "family": family,
                                    "member": e.get("name"),
                                    "kind": "component_admission",
                                    "detail": str(exc),
                                }
                            )
                    if observed["status"] == "unsupported" and not must_refuse:
                        actual["source_reference"] = diagnostic_entry(actual, observed)["source_reference"]
                    (Path(w) / "capsule.yaml").write_text(yaml.safe_dump(actual, sort_keys=False))
                    if evidence_mode != "diagnostic" and observed["status"] != "admitted" and not must_refuse:
                        failures.append((e.get("name", "?"), "emitted SW operation admission: " + observed["reason"]))
                        print(
                            f"  [failure] {e.get('name', '?')}: emitted SW operation admission: {observed['reason']}",
                            flush=True,
                        )
                    # Verified evidence admits a generation-time capture only from the sealed runner.
                    capture_refusal = verified_capture_failure(actual)
                    if evidence_mode != "diagnostic" and capture_refusal is not None:
                        failures.append((e.get("name", "?"), "generation-time capture: " + capture_refusal))
                        print(
                            f"  [failure] {e.get('name', '?')}: generation-time capture: {capture_refusal}", flush=True
                        )
                _scrub_capsule_dir(w)
                written.append(w)
                # A qualified model's per-group capsules are corpus members of their own: Phase 1 grades
                # the very programs that qualified it.
                for extra in _qualifying_group_capsules(Path(w), out_root):
                    if extra in written:
                        continue
                    if evidence is not None and evidence.software_spec:
                        member = yaml.safe_load((extra / "capsule.yaml").read_bytes())
                        observed = screen_written(member, extra, target=hardware_target, evidence=evidence)
                        if observed is None:
                            observed = screen_entry(
                                evidence.software_spec,
                                {"name": member.get("name"), "kind": member.get("kind")},
                                defaults={
                                    "operand_dtype": binding.operand_dtype,
                                    "accumulator_dtype": binding.accum_dtype,
                                },
                                capsule=member,
                                host_capabilities=evidence.host_capabilities,
                            )
                        admission.append({"capsule": member.get("name"), "observation": "qualifying_group", **observed})
                        # The same rule every emitted member is held to: an operation the software spec
                        # refuses cannot qualify a model, and verified mode admits only reviewed ones.
                        if observed["status"] == "unsupported" or (
                            evidence_mode != "diagnostic" and observed["status"] != "admitted"
                        ):
                            # The writer already created this group as numerical evidence. Keep its
                            # linked bytes inspectable, but never classify it as hand-authored or put
                            # it in either selected experiment corpus.
                            member["software_screen"] = observed
                            (extra / "capsule.yaml").write_text(yaml.safe_dump(member, sort_keys=False))
                            _scrub_capsule_dir(extra)
                            refused_generated.append(extra)
                            failures.append(
                                (e.get("name", "?"), f"qualifying group {extra.name}: {observed['reason']}")
                            )
                            continue
                        member["software_screen"] = observed
                        (extra / "capsule.yaml").write_text(yaml.safe_dump(member, sort_keys=False))
                    _scrub_capsule_dir(extra)
                    written.append(extra)
                if family and Path(w).parent.name == "_perf":
                    family_counts[family]["written_members"] += 1
            elif family:
                _performance_errors.append(
                    {
                        "family": family,
                        "member": e.get("name", "?"),
                        "status": "error",
                        "error_type": "NoOutput",
                        "detail": "capsule writer returned no output",
                    }
                )
            else:
                omitted.append(
                    {
                        "capsule": e.get("name", "?"),
                        "source": e.get("source"),
                        "status": "not_built",
                        "reason": "required writer source or lowering input unavailable",
                    }
                )
                if evidence_mode == "verified":
                    failures.append((e.get("name", "?"), "required capsule writer produced no artifact"))
    # A performance sweep can fail before it becomes an entry (for example,
    # while resolving a selected oracle). Such failures are recorded in the
    # manifest, but must also fail the run after the other capsules are written;
    # otherwise the controller reports success while a required claim cohort
    # has zero members. Writer failures already appear in both lists.
    failed_names = {name for name, _ in failures}
    for error in _performance_errors:
        name = str(error.get("member") or error.get("family") or "<performance sweep>")
        if name not in failed_names:
            kind = str(error.get("error_type") or "unknown error")
            detail = str(error.get("detail") or "no further detail recorded").replace("\n", " ")[:240]
            failures.append((name, f"performance materialization failed ({kind}): {detail}; inspect MANIFEST.yaml"))
            failed_names.add(name)
    if failures:
        print(f"  [FAIL] {len(failures)} capsule(s) could not be written:")
        for name, why in failures:
            print(f"    - {name}: {why}")
    # Record provenance for what we just emitted. The MANIFEST header has always CLAIMED the generator
    # rewrites it, but no writer existed, so it drifted silently as soon as the corpus grew.
    declared_blocked = [
        {
            "family": row["family"],
            "status": "blocked_unimplemented",
            "reason": row["reason"],
            "emitter": copy.deepcopy(row["performance"]["emitter"]),
            "fit_axes": list(row.get("fit_axes") or []),
            "comparison_roles": list(row.get("comparison_roles") or []),
        }
        for row in (template.get("blocked_unimplemented") or [])
    ]
    generated_members = sum(row["written_members"] for row in family_counts.values())
    performance_record = {
        "shared_template": {"path": template.get("path"), "sha256": template.get("sha256")},
        "facts": {
            "target": hardware_target,
            "sha256": facts["sha256"],
            "digest_kind": "derived_performance_document",
            **({"raw_facts_sha256": evidence.raw_facts_sha256} if evidence is not None else {}),
        },
        "phase": {
            "category": "_perf",
            "label": "dev",
            "included_in_functional_grade": False,
            "exclusion": "TargetExperiment.corpus_siblings excludes underscore-prefixed categories",
        },
        "families": declared_families,
        "counts": {
            "declared_families": len(declared_families),
            "generated_families": sum(1 for row in family_counts.values() if row["written_members"] > 0),
            "generated_members": generated_members,
            "by_family": family_counts,
        },
        "skipped_inapplicable": _sweep_skips,
        "blocked_unimplemented": declared_blocked + _runtime_blocked,
        "errors": _performance_errors,
    }
    if component_identity is not None:
        performance_record["component_generation"] = component_identity
    if unbuilt_roster:
        print(
            f"  [roster] {len(unbuilt_roster)} declared roster model(s) could not be captured at this "
            f"target's derived format; they are recorded as not_built, never as covered"
        )
    if host_placed:
        print(
            f"  [placement] {len(host_placed)} written capsule(s) place no operation on the accelerator; "
            f"they are recorded, never published"
        )
    if unprovable_forbids:
        print(
            f"  [lane] {len(unprovable_forbids)} synthesized host-lane capsule(s) were dropped: their "
            f"program contains regions this target admits, so the forbid they assert is not provable"
        )
    from merlin.development.testgen.generation.hidden_disjointness import check as hidden_disjointness

    disjointness = hidden_disjointness(written)
    if disjointness["status"] != "disjoint":
        failures.append(
            (
                "hidden cohort",
                f"{disjointness['overlapping_hidden_capsules']} hidden capsule(s) repeat a public program",
            )
        )
    source_contract_record = None
    if component_coverage_report is not None:
        from merlin.development.testgen.components.component_coverage import (
            finalize,
            public_summary,
            write_report,
        )

        component_coverage_report = finalize(
            component_coverage_report,
            root=out_root,
            written=written,
            failures=failures,
            generation_identity=component_identity,
            semantic_basis=semantic_basis,
        )
        if source_components:
            from merlin.development.testgen.components.component_source_binding import verify_prepared_sources

            verify_prepared_sources(
                out_root, component_coverage_report, software=software_intake, hardware=hardware_intake
            )
        write_report(out_root, component_coverage_report)
        if source_preparation is not None:
            from merlin.development.testgen.components.source_performance import prepare_source_contracts

            source_contracts = prepare_source_contracts(
                root=out_root,
                coverage=component_coverage_report,
                hardware=hardware_intake,
                software=software_intake,
            )
            source_contract_path = artifact_root / "coverage" / "source-performance-contracts.json"
            source_contract_path.write_text(json.dumps(source_contracts, sort_keys=True, indent=2) + "\n")
            source_contract_record = {
                "path": str(source_contract_path),
                "sha256": hashlib.sha256(source_contract_path.read_bytes()).hexdigest(),
                "scope": "source/reference only; hardware guards and measurement unqualified",
                "source_checked_counts": source_contracts["source_checked_counts"],
            }
            performance_record["source_performance_preparation"] = source_contract_record
        performance_record["component_coverage"] = public_summary(component_coverage_report)
        if component_coverage_report["status"] != "complete":
            failures.append(
                ("component coverage", "mandatory independent obligations unavailable; inspect private coverage report")
            )
    superseded = _prune_superseded_synth(entries, written, target=target)
    if superseded:
        print(
            f"  [prune] {len(superseded)} synthesized capsule(s) whose requirement cell no longer "
            f"exists were removed: {', '.join(superseded)}"
        )
    from merlin.development.testgen.generation.case_redundancy import apply_to_generation

    # Independent derivations reach the same program; one copy is kept and every shape scores once.
    performance_record["case_redundancy"] = apply_to_generation(
        written, out_root, tile=getattr(binding, "tile_dim", None), size_policy=profile.get("correctness_size_classes")
    )
    if performance_record["case_redundancy"]["collapsed"]:
        print(
            f"  [redundancy] {performance_record['case_redundancy']['collapsed']} exact duplicate "
            f"capsule(s) collapsed; record in case-redundancy.json"
        )
    if generation_scope != "full":
        deferred_record = {
            "generation_scope": generation_scope,
            "status": "deferred_not_prepared",
            "phase": "phase2",
            "qualification": "No performance capsule, measurement or Phase 2 readiness is issued",
            "shared_template": performance_record["shared_template"],
            "facts": performance_record["facts"],
            "requirements_sha256": hashlib.sha256(requirement_bytes).hexdigest() if requirement_bytes else None,
            "entries": deferred_performance,
        }
        deferred_raw = (json.dumps(deferred_record, sort_keys=True, indent=2) + "\n").encode()
        deferred_path = out_root / "phase2-deferred.json"
        deferred_path.write_bytes(deferred_raw)
        performance_record["preparation"] = {
            "scope": generation_scope,
            "status": "deferred_not_prepared",
            "deferred_members": len(deferred_performance),
            "ledger": "phase2-deferred.json",
            "sha256": hashlib.sha256(deferred_raw).hexdigest(),
        }
    update_provenance_manifest(
        written,
        cap_root=out_root,
        target=hardware_target,
        refused_generated=refused_generated,
        performance_record=performance_record,
        unbuilt_roster=unbuilt_roster,
        claim_model_evaluation=claim_plan,
        unprovable_forbids=unprovable_forbids,
        host_placement_not_published=host_placed,
        superseded=superseded,
    )
    if evidence is not None:
        from merlin.development.backend.session.inputs.implementation_inventory import fingerprint
        from merlin.development.testgen.evidence.coverage_commitment import observe_cohort
        from merlin.development.testgen.evidence.coverage_obligations_coverage_commitment import (
            selected_inputs,
            write_inputs,
        )

        coverage_root = artifact_root / "coverage"
        coverage_root.mkdir(parents=True, exist_ok=True)
        accounting = json.loads((coverage_root / "operation-accounting.json").read_bytes())
        coverage_inputs = selected_inputs(evidence, accounting=accounting, prohibited_roles=declared_roles)
        coverage_input_record = write_inputs(out_root, coverage_inputs)
        # Absent only when a caller supplies its own inputs; the coverage report then blocks on it.
        drift = coverage_inputs.get("spec_fact_drift") or {"findings": []}
        (coverage_root / "spec-fact-drift.json").write_bytes(
            (json.dumps(drift, sort_keys=True, indent=2) + "\n").encode()
        )
        for row in drift["findings"]:
            if row["classification"] != "ok":
                print(f"  [spec-drift] {row['classification']}: {row['family']}.{row['field']} {row.get('values')}")
        generated_manifest = yaml.safe_load((out_root / "MANIFEST.yaml").read_text()) or {}
        selections = (generated_manifest.get("phase_corpora") or {}).get(hardware_target) or {}
        cohort_reports = {}
        selected_reports = {}
        for phase in ("phase1", "phase2"):
            members = (selections.get(phase) or {}).get("generated_members") or []
            report = observe_cohort(
                coverage_inputs, [out_root / member for member in members], target=hardware_target, phase=phase
            )
            if phase == "phase2":
                from merlin.development.testgen.performance.form_cohort import attach_to_phase2_report

                form_coverage = _form_perf_coverage(profile, selected_requirement, out_root, members)
                attach_to_phase2_report(report, form_coverage)
            selected_reports[phase] = report
            report_path = coverage_root / f"{phase}-capsule-coverage.json"
            raw = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
            report_path.write_bytes(raw)
            cohort_reports[phase] = {
                "path": str(report_path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "status": report["status"],
                "n_capsules": report["cohort"]["n_capsules"],
            }
        if component_only:
            if component_coverage_report is not None:
                from merlin.development.testgen.components.component_coverage import build_guard_link

                guard_link = build_guard_link(component_coverage_report)
            else:
                guard_link = {
                    "schema": "merlin.phase0.component_guard_obligation.v1",
                    "status": "not_established",
                    "guards": [],
                    "reason": (
                        "independent component generation supplies no application conformance or Phase 1 certification"
                    ),
                    "qualification": "generated goldens and program admission do not accept a candidate numerically",
                }
        else:
            from merlin.development.testgen.performance.phase2_guards import build_guard_link

            guard_link = build_guard_link(
                out_root, coverage_inputs, selected_reports["phase1"], selected_reports["phase2"]
            )
        guard_path = coverage_root / "phase2-functional-guards.json"
        guard_raw = (json.dumps(guard_link, sort_keys=True, indent=2) + "\n").encode()
        guard_path.write_bytes(guard_raw)
        guard_record = {
            "path": str(guard_path),
            "sha256": hashlib.sha256(guard_raw).hexdigest(),
            "status": guard_link["status"],
            "n_guards": len(guard_link["guards"]),
        }
        receipt = {
            "schema": "merlin.phase0_generation.v1",
            "target": hardware_target,
            "evidence_status": evidence.status,
            "mode": evidence_mode or "verified",
            "software_spec": profile.get("_software_spec_identity"),
            "synthesis": profile.get("_synth_verification"),
            **({"public_source_eligibility": public_source_eligibility} if public_source_eligibility else {}),
            "operation_accounting": str(artifact_root / "coverage/operation-accounting.json"),
            "quantization_contract": str(artifact_root / "software/quantization-contract.json"),
            "capsules_written": len(written),
            "omitted": omitted,
            "unbuilt_roster": unbuilt_roster,
            "operation_admission": admission,
            "unprovable_forbids": unprovable_forbids,
            "host_placement_not_published": host_placed,
            "failures": [{"capsule": n, "reason": why} for n, why in failures],
            "performance_materialization_errors": len(_performance_errors),
            "qualification": "not_established",
            "corpus_manifest": str(out_root / "MANIFEST.yaml"),
            "coverage_inputs": coverage_input_record,
            "cohort_coverage": cohort_reports,
            "phase2_functional_guards": guard_record,
            "instruction_policy": copy.deepcopy(instruction_policy),
            "hidden_disjointness": disjointness,
            "capsule_commitments": [
                {"member": path.relative_to(out_root).as_posix(), "sha256": fingerprint(path)}
                for path in sorted(written)
            ],
        }
        if component_coverage_report is not None:
            receipt["component_coverage"] = public_summary(component_coverage_report)
        if source_contract_record is not None:
            receipt["source_performance_preparation"] = source_contract_record
        (coverage_root / "generation.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
        manifest_path = out_root / "MANIFEST.yaml"
        manifest = yaml.safe_load(manifest_path.read_text()) or {}
        manifest["coverage_inputs"] = coverage_input_record
        manifest["instruction_policy"] = copy.deepcopy(instruction_policy)
        manifest["phase0_evidence"] = {
            "status": evidence.status,
            "mode": evidence_mode or "verified",
            "raw_facts_sha256": evidence.raw_facts_sha256,
            "software_spec": profile.get("_software_spec_identity"),
            "manifest": str(artifact_root / "evidence-manifest.json"),
            "generation_receipt": str(coverage_root / "generation.json"),
        }
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    if failures:
        raise RuntimeError(
            f"{len(failures)} capsule(s) failed to generate: {', '.join(n for n, _ in failures)}; the "
            f"rest of the corpus was written, so re-running after fixing them is cheap"
        )
    return written
