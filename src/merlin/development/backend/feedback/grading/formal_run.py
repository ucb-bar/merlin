"""Grade a frozen agent run through capsule_bench_v0 (post-agent, full-access phase).

Public/dev phase (repairable in a real run) -> freeze -> hidden/private qualification. Writes
score_capsule.json per phase, an iteration_000 snapshot, run_manifest.yaml, and final_report.md.
Runs OUTSIDE the agent sandbox (needs spike/verilator + the hidden goldens).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

import yaml

from merlin.common.digest import hash_tree
from merlin.development.backend.authoring.library.compiler_library import selected_child
from merlin.development.backend.context import (
    InvocationContext,
    add_context_arguments,
    readback_kwargs,
    resolve_context,
    verify_readback_record,
)
from merlin.development.backend.feedback.certification import hidden_capsule_policy as HCP
from merlin.development.backend.feedback.grading import freeze as freeze_run
from merlin.development.backend.feedback.grading.freeze import main
from merlin.development.backend.qualification.full_models import build as _merlin_build_2
from merlin.development.backend.qualification.full_models import execution as _merlin_execution_1
from merlin.development.backend.qualification.full_models import execution_engine as PFX
from merlin.development.backend.qualification.full_models import private_full_models as PFM
from merlin.development.backend.qualification.full_models.private_facts import selected_input_facts
from merlin.development.backend.qualification.provenance import (
    private_instruction_coordinator as instruction_coordinator,
)
from merlin.development.backend.qualification.provenance import (
    private_instruction_declaration as instruction_declaration,
)
from merlin.development.targetgen.grading import model_checks as CG
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.target.registry import identity as capsule_runner_config
from merlin.target.simulators import oracle_policy as capsule_oracle_policy

# ---- formal ------------------------------------------------------------------------------------


def _main(argv: list[str] | None = None, *, context: InvocationContext | None = None) -> int:
    from merlin.development.backend.feedback.grading.formal import (
        FORMAL_REQUIRED_TIER,
        _cost_phrase,
        _formal_overlap_shares_compute,
        _install_formal_model_simulator,
        _phase_manifest,
        _private_source_freeze_for_formal,
        _score,
        _verify_formal_readback_policy,
    )

    ap = argparse.ArgumentParser()
    add_context_arguments(ap)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--workspace", type=Path, help="Frozen Phase 1 workspace for selected RTL facts")
    ap.add_argument("--rtl-facts", type=Path, help="Selected facts in that workspace's frozen input snapshot")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--qa-timeout", type=int, default=900, help="Selected per-step grading timeout in seconds")
    ap.add_argument(
        "--capsules",
        default=None,
        help="comma-separated capsule roots for the PUBLIC/dev phase. The launcher resolves "
        "these from the target descriptor (primary corpus + its sibling categories); "
        "the default is the shared parent and is only right for a target whose corpus "
        "IS that parent.",
    )
    ap.add_argument(
        "--hidden-capsules",
        default=None,
        help="comma-separated capsule roots for the HIDDEN phase (defaults to --capsules). "
        "The launcher passes the target's own hidden dir so the public-only "
        "materialized subset does not yield an empty hidden grade.",
    )
    ap.add_argument(
        "--contract", default=None, help="optional frozen contract root used to resolve capsule inheritance"
    )
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--validation-scope", choices=("full", "capsules"), default="full")
    ap.add_argument("--skip-hidden", action="store_true")
    ap.add_argument(
        "--hidden-capsule-mode",
        choices=HCP.MODES,
        default="",
        help="operator hidden validation after freeze; functional_spike supplies no RTL certification",
    )
    ap.add_argument(
        "--private-full-model-spec",
        type=Path,
        help="operator-only frozen validation input; never passed to an authoring workspace",
    )
    ap.add_argument(
        "--instruction-selection",
        type=Path,
        help="operator-only reviewed public instruction source/tool/policy declaration; no saved authority",
    )
    a = ap.parse_args(argv)
    if a.instruction_selection is not None and a.private_full_model_spec is None:
        ap.error("--instruction-selection requires --private-full-model-spec")
    if a.qa_timeout < 1:
        ap.error("--qa-timeout must be positive")
    if a.rtl_facts is not None:
        if a.workspace is None:
            ap.error("--rtl-facts requires --workspace")
        from merlin.development.backend.session.inputs.run_record import select

        select(a.workspace, a.rtl_facts)
    legacy_context = context is not None
    context = resolve_context(a, ap, context)
    instruction_inputs = (
        instruction_declaration.read(a.instruction_selection, target=context.target)
        if a.instruction_selection is not None
        else None
    )
    if a.capsules is None:
        if not legacy_context:
            ap.error("installed formal grading requires explicit --capsules")
        a.capsules = str(context.repo / "experiments/corpora")
    if not legacy_context:
        if a.contract is None:
            ap.error("installed formal grading requires explicit --contract")
        if a.hidden_capsules is None and not a.skip_hidden:
            ap.error("installed formal grading requires --hidden-capsules or explicit --skip-hidden")
    run_dir = Path(a.run_dir).absolute() if a.validation_scope == "capsules" else Path(a.run_dir)
    _verify_formal_readback_policy(run_dir, context)
    hidden_policy = HCP.resolve(context.descriptor, a.hidden_capsule_mode)
    functional_hidden = hidden_policy["mode"] == "functional_spike"
    if functional_hidden and (a.no_oracle or a.skip_hidden):
        ap.error("functional hidden corroboration requires the complete measured hidden phase")
    environment_path = run_dir / "environment.yaml"
    environment = yaml.safe_load(environment_path.read_bytes()) if environment_path.is_file() else {}
    from merlin.development.backend.feedback.certification import capsule_certification as CC

    CC.verify_scope(a.validation_scope, environment)
    if a.validation_scope == "capsules" and (a.no_oracle or a.skip_hidden):
        ap.error("capsule certification requires both measured public and selected hidden phases")
    HCP.verify_record(hidden_policy, environment.get("hidden_capsule_validation"))
    pkg = run_dir / "submission"
    from merlin.development.backend.feedback.certification import (
        capsule_qualification_replay as imported_capsules,
    )

    imported = imported_capsules.selected(
        run_dir,
        context=context,
        candidate=pkg,
        capsules=a.capsules,
        hidden_capsules=a.hidden_capsules or a.capsules,
        contract=a.contract,
        facts=a.rtl_facts,
    )
    if imported is not None and (a.no_oracle or a.skip_hidden or a.validation_scope != "full"):
        ap.error("imported capsule observations require complete unpaid external full validation")
    if imported is not None:
        from merlin.development.targetgen.authoring.pass_selection import active_shared_pass_requirement

        if active_shared_pass_requirement() is not None:
            ap.error("imported capsule observations do not qualify the selected shared-pass compiler interfaces")

    from merlin.development.backend.feedback.scalability.resource_gate import require_complete, run_gate

    public_resources = (
        imported["resource_producer_epoch"]["result"]
        if imported is not None
        else run_gate(
            run_dir=run_dir,
            candidate=pkg,
            context=context,
            capsules_root=a.capsules,
            contract=a.contract,
            facts_path=a.rtl_facts,
        )
    )
    require_complete(public_resources)

    # Pin whole-model tile certification to the simulator assigned to the formal RTL tier by the
    # target's own manifest.  An inherited MERLIN_MESH_SIM=spike must not silently turn an L3 claim into
    # a functional-model result; each tile also records and is checked for derived_from_rtl and
    # cycle_accurate evidence by capsule_grade.model_execution_check.
    try:
        _formal_mesh_selection, _inherited_mesh_sim = _install_formal_model_simulator(context.target)
    except Exception as exc:  # noqa: BLE001 — formal grading must fail closed, never demote fidelity
        raise SystemExit(f"cannot resolve formal whole-model simulator: {exc}") from exc
    _formal_mesh_sim = _formal_mesh_selection["engine"]

    # Only answer-free, pinned launch inputs enter public owned-host compilation.
    from merlin.development.backend.feedback.scalability import (
        public_owned_host_inputs,
        public_owned_host_selection,
    )

    public_host_selection = public_owned_host_inputs.install_from_run(
        run_dir, target=context.target, descriptor=context.descriptor
    )
    if public_host_selection is None and a.private_full_model_spec is not None:
        from merlin.runtime.execution.model_inputs import file_sha256

        public_host_selection = public_owned_host_selection.prepare(
            a.private_full_model_spec,
            a.instruction_selection,
            target=context.target,
            output=run_dir.absolute() / "public_owned_host_build_selection.json",
            selected_facts=a.rtl_facts,
            selected_facts_sha256=file_sha256(a.rtl_facts) if a.rtl_facts is not None else None,
            descriptor=context.descriptor,
        )
        os.environ[public_owned_host_selection.ENV] = json.dumps(public_host_selection, sort_keys=True)

    # --- public/dev phase ---
    (run_dir / "grading_public").mkdir(parents=True, exist_ok=True)
    pub = (
        CC._read(imported["numerical_producer_epoch"]["public"]["score"])
        if imported is not None
        else CC.grade(
            run_dir,
            capsules=a.capsules,
            labels={"public", "dev"},
            context=context,
            contract=a.contract,
            facts=a.rtl_facts,
            timeout_s=a.qa_timeout,
        )
        if a.validation_scope == "capsules"
        else _score(
            str(pkg),
            a.capsules,
            str(run_dir / "grading_public"),
            {"public", "dev"},
            a.no_oracle,
            context=context,
            contract=a.contract,
            step_timeout_s=a.qa_timeout,
        )
    )
    if a.validation_scope == "full" and imported is None:
        (run_dir / "grading_public" / "score_capsule.json").write_text(json.dumps(pub, indent=2))

    # The functional public denominator is separate from core RTL and hidden validation.
    from merlin.development.backend.feedback.scalability.public_scalability import (
        attach as attach_scalability,
    )

    attach_scalability(
        pub,
        pkg,
        context=context,
        output=run_dir / "grading_public_scalability",
        timeout=a.qa_timeout,
        contract=a.contract,
    )

    # --- iteration_000 snapshot (dummy = one-shot; a real repairing agent appends more) ---
    it = run_dir / "iterations" / "iteration_000"
    it.mkdir(parents=True, exist_ok=True)
    (it / "capsule_status_after.yaml").write_text(
        yaml.safe_dump({c["capsule"]: c["status"] for c in pub.get("per_capsule", [])}, sort_keys=True)
    )
    (it / "first_failure.yaml").write_text(yaml.safe_dump(pub.get("first_failure_planes", {})))
    (it / "notes.md").write_text(
        "iteration_000: independently reopened original capsule observations; no new capsule execution.\n"
        if imported is not None
        else "iteration_000: initial submission grade (public/dev).\n"
    )

    # --- freeze, then hidden phase (immutability enforced) ---
    frozen = freeze_run.freeze(run_dir, repo=context.repo)
    required_full_models = PFM.requirements_for(context.descriptor)
    required_full_programs = PFM.program_requirements_for(context.descriptor)
    required_loader_env = PFM.loader_env_requirements_for(context.descriptor)
    overlap = False
    # Scheduling preference only: absence retains the historical enabled policy.
    overlap_choice = os.environ.get("MERLIN_FORMAL_OVERLAP_PRIVATE")
    if (
        imported is None
        and a.validation_scope == "full"
        and (overlap_choice is None or overlap_choice == "1")
        and "MERLIN_COMPUTE_CPUS" in os.environ
        and not a.skip_hidden
        and required_full_models
        and a.private_full_model_spec is not None
    ):
        from merlin.common.process.cpu_pool import current_thread_owns_cpu_lease

        overlap = _formal_overlap_shares_compute(os.environ)
        if overlap and (threading.active_count() != 1 or current_thread_owns_cpu_lease()):
            overlap = False  # Never fork a live grader thread or held CPU lease.
        if not overlap:
            print("formal private overlap unavailable; grading serially", file=sys.stderr, flush=True)
    hid = (
        CC._read(imported["numerical_producer_epoch"]["hidden"]["score"])
        if imported is not None
        else {"available": False}
    )
    hidden_pid = None
    if not a.skip_hidden and imported is None:
        # re-hash the submission right before hidden grading; refuse if it changed since freeze
        recheck = hash_tree(pkg)["sha256"]
        mutated = recheck != frozen["submission_sha256"]
        import datetime as _dt

        fj = json.loads((run_dir / "freeze.json").read_text())
        fj["hidden_grading_started_at"] = _dt.datetime.now(_dt.UTC).isoformat()
        fj["submission_sha256_recheck"] = recheck
        fj["workspace_mutable_after_freeze"] = bool(mutated)
        if functional_hidden:
            fj["hidden_capsule_validation"] = hidden_policy
        (run_dir / "freeze.json").write_text(json.dumps(fj, indent=2))
        if mutated:
            raise SystemExit(
                f"FREEZE VIOLATION: submission changed after freeze "
                f"({frozen['submission_sha256'][:12]} -> {recheck[:12]}); refusing hidden grade"
            )
        (run_dir / "grading_hidden").mkdir(parents=True, exist_ok=True)
        if overlap:
            # No hidden source is read before the freeze/recheck. The fork gets its own
            # process-wide facts binding and compiler profile state; neither side changes
            # the CPU environment or affinity, so both lease from the one shared pool.
            # A stale score can never stand in for this child; both join before publication.
            hidden_score_path = run_dir / "grading_hidden" / "score_capsule.json"
            hidden_score_path.unlink(missing_ok=True)
            sys.stdout.flush()
            sys.stderr.flush()
            hidden_pid = os.fork()
            if hidden_pid == 0:
                status = 1
                try:
                    hidden_score = _score(
                        str(pkg),
                        (a.hidden_capsules or a.capsules),
                        str(run_dir / "grading_hidden"),
                        {"hidden"},
                        a.no_oracle,
                        context=context,
                        contract=a.contract,
                        step_timeout_s=a.qa_timeout,
                        **({"hidden_capsule_mode": hidden_policy["mode"]} if functional_hidden else {}),
                    )
                    hidden_score_path.write_text(json.dumps(hidden_score, indent=2))
                    status = 0
                except BaseException as exc:  # noqa: BLE001 -- child cannot certify on any failure
                    print(f"hidden formal child failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                finally:
                    for stream in (sys.stdout, sys.stderr):
                        try:
                            stream.flush()
                        except (OSError, ValueError) as error:
                            os.write(2, f"formal child stream flush failed: {type(error).__name__}\n".encode())
                    os._exit(status)
        else:
            hid = (
                CC.grade(
                    run_dir,
                    capsules=(a.hidden_capsules or a.capsules),
                    labels={"hidden"},
                    context=context,
                    contract=a.contract,
                    facts=a.rtl_facts,
                    timeout_s=a.qa_timeout,
                    hidden_mode=hidden_policy["mode"],
                )
                if a.validation_scope == "capsules"
                else _score(
                    str(pkg),
                    (a.hidden_capsules or a.capsules),
                    str(run_dir / "grading_hidden"),
                    {"hidden"},
                    a.no_oracle,
                    context=context,
                    contract=a.contract,
                    step_timeout_s=a.qa_timeout,
                    **({"hidden_capsule_mode": hidden_policy["mode"]} if functional_hidden else {}),
                )
            )
            if a.validation_scope == "full":
                (run_dir / "grading_hidden" / "score_capsule.json").write_text(json.dumps(hid, indent=2))

    public_phase = _phase_manifest(pub, required_tier=FORMAL_REQUIRED_TIER)
    public_phase.update(
        {
            "numeric_all_exact": pub.get("numeric_all_exact"),
            "trace_all_pass": pub.get("trace_all_pass"),
            "model_execution_all_pass": pub.get("model_execution_all_pass"),
            "structural_evidence_all_pass": pub.get("structural_evidence_all_pass"),
            "structural_evidence_scope": pub.get("structural_evidence_scope"),
            "first_failure_planes": pub.get("first_failure_planes"),
        }
    )
    if a.validation_scope == "capsules":
        try:
            CC.publish(run_dir, descriptor=context.descriptor, resources=public_resources)
        except (ValueError, OSError, KeyError) as exc:
            print(f"capsule certification incomplete: {exc}", file=sys.stderr, flush=True)
            return 1
        return 0
    hidden_status = None
    model_check_background = None
    selected_numerics = None
    try:
        private_models: dict = {
            "schema": PFM.RESULT_SCHEMA,
            "passed": False,
            "required_models": list(required_full_models),
            "models": [],
            "reason": "operator-private full-model validation input was not supplied",
        }
        # The numerical half of the private gate: the static gate's linked programs, executed on the
        # engines the descriptor declares, after the freeze. A malformed declaration is a grading error.
        execution_gate = PFX.gate_for(context.descriptor, required_programs=required_full_programs)
        private_execution: dict | None = (
            None
            if execution_gate is None
            else _merlin_execution_1.not_run(execution_gate, "the static private full-model gate did not pass")
        )
        if required_full_models and a.private_full_model_spec is not None:
            try:
                source_freeze = _private_source_freeze_for_formal(
                    run_dir,
                    a.private_full_model_spec,
                    context.target,
                    workspace=a.workspace,
                    repo=context.repo,
                    descriptor=context.descriptor,
                )
                if source_freeze is None:
                    raise ValueError(
                        "private full-model certification requires a fresh run-owned authored-source freeze"
                    )
                from merlin.development.backend.qualification.provenance import (
                    private_numerical_selection as numerical_selection,
                )

                selected_numerics = numerical_selection.read(a.private_full_model_spec)
                if selected_numerics is not None:
                    model_check_background = numerical_selection.OfficialCheck(
                        pkg,
                        run_dir,
                        required_full_programs,
                        selected_numerics["deadline_unix"],
                    )
                    private_models, private_execution = numerical_selection.consume(
                        run_dir=run_dir,
                        descriptor=context.descriptor,
                        submission=pkg,
                        private_spec_path=a.private_full_model_spec,
                        after_static=model_check_background.start,
                    )
                else:
                    instruction_selection = (
                        instruction_coordinator.prepare(
                            instruction_inputs,
                            target=context.target,
                            output=run_dir.absolute() / "grading_private_instruction_selection",
                        )
                        if instruction_inputs is not None
                        else None
                    )
                    with selected_input_facts(
                        a.private_full_model_spec, target=context.target, required_models=required_full_models
                    ) as facts_binding:
                        from merlin.development.backend.qualification.provenance.private_result_format_selection import (
                            from_run as result_format_for_run,
                        )

                        result_format_selection = result_format_for_run(run_dir, target=context.target)
                        if instruction_selection is not None:
                            instruction_selection.require_facts(facts_binding)
                        private_models = _merlin_build_2.run(
                            pkg,
                            a.private_full_model_spec,
                            target=context.target,
                            required_models=required_full_models,
                            required_programs=required_full_programs,
                            loader_env_requirements=required_loader_env,
                            out=run_dir / "grading_private_full_models",
                            source_freeze=source_freeze,
                            source_freeze_root=(run_dir / "private_full_model_input" / "sources")
                            if source_freeze
                            else None,
                            build_options=PFX.build_options(execution_gate) if execution_gate is not None else None,
                            **(
                                {"native_result_format_selection": result_format_selection}
                                if result_format_selection is not None
                                else {}
                            ),
                            **(
                                {"linked_elf_admission": instruction_selection.service}
                                if instruction_selection is not None
                                else {}
                            ),
                        )
                        if instruction_selection is not None:
                            instruction_selection.require_facts(facts_binding)
                            private_models["instruction_selection"] = instruction_selection.record()
                        if result_format_selection is not None:
                            result_format_selection.verify()
                        if execution_gate is not None and private_models.get("passed") is True:
                            # Inside the facts binding: an RTL engine is cited against the facts the build used.
                            try:
                                private_execution = _merlin_execution_1.run(
                                    private_models,
                                    execution_gate,
                                    target=context.target,
                                    static_out=run_dir / "grading_private_full_models",
                                    max_workers=(
                                        int(os.environ["MERLIN_PRIVATE_EXECUTION_WORKERS"])
                                        if "MERLIN_PRIVATE_EXECUTION_WORKERS" in os.environ
                                        else 1
                                    ),
                                )
                            except Exception as exc:  # noqa: BLE001 -- an execution refusal is an incomplete gate
                                private_execution = _merlin_execution_1.not_run(
                                    execution_gate, f"{type(exc).__name__}: {exc}"
                                )
                    private_models["fact_reader_binding"] = facts_binding
                if (
                    _private_source_freeze_for_formal(
                        run_dir,
                        a.private_full_model_spec,
                        context.target,
                        workspace=a.workspace,
                        repo=context.repo,
                        descriptor=context.descriptor,
                    )
                    != source_freeze
                ):
                    raise ValueError("run-owned private source snapshot changed during linked builds")
            except Exception as exc:  # noqa: BLE001 -- missing private evidence is an incomplete run
                private_models["passed"] = False
                private_models["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if hidden_pid is not None:
            _pid, hidden_status = os.waitpid(hidden_pid, 0)
    if hidden_pid is not None:
        if not os.WIFEXITED(hidden_status) or os.WEXITSTATUS(hidden_status) != 0:
            if model_check_background is not None:
                model_check_background.close()
            raise SystemExit("hidden formal child failed; refusing phase 1 qualification")
        hid = json.loads(hidden_score_path.read_text())
    if hash_tree(pkg)["sha256"] != frozen["submission_sha256"]:
        if model_check_background is not None:
            model_check_background.close()
        raise SystemExit("FREEZE VIOLATION: submission changed during formal grading")
    if (
        imported is not None
        and imported_capsules.verify(imported["receipt"], descriptor=context.descriptor) != imported
    ):
        raise ValueError("imported capsule qualification changed during external validation")
    hidden_phase = _phase_manifest(
        hid,
        required_tier=hidden_policy["required_tier"],
        functional_hidden=functional_hidden,
    )
    if a.skip_hidden:
        hidden_phase["passed"] = "not_run"
        hidden_phase["completion_failures"] = ["hidden_grade_skipped"]
        hidden_phase["formal_complete"] = False
    # A descriptor with no private full-model declaration cannot silently make a new
    # Phase 1 success claim.  Older diagnostic experiments remain runnable, incomplete.
    private_models_complete = bool(required_full_models) and private_models.get("passed") is True
    # A declared, required execution gate holds completion; a declared deferral is recorded on it.
    private_execution_complete = (
        execution_gate is None or execution_gate["required"] is False or (private_execution or {}).get("passed") is True
    )
    if selected_numerics is not None:
        from merlin.runtime.execution.model_inputs import strict_tree_sha256

        private_execution_complete = numerical_selection.complete(
            private_execution,
            static=private_models,
            gate={"programs": required_full_programs},
            candidate_sha256=strict_tree_sha256(pkg)["sha256"],
        )
    model_check_official = {"passed": False, "reason": "private model inputs unavailable"}
    if model_check_background is not None:
        try:
            model_check_official = model_check_background.result()
        except Exception as exc:  # noqa: BLE001 -- unchanged official gate failures remain incomplete
            model_check_official = {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}
    elif required_full_models and private_models_complete and a.private_full_model_spec is not None:
        from merlin.development.backend.feedback.certification import model_checks as MCO

        try:
            model_check_official = MCO.run_gate(pkg, run_dir, required_full_programs, mode="fast", model_workers=3)
        except Exception as exc:  # noqa: BLE001 -- an incomplete host gate never certifies a run
            model_check_official = {"passed": False, "reason": f"{type(exc).__name__}: {exc}"}
    model_check_complete = not required_full_models or model_check_official.get("passed") is True
    public_resources_complete = (
        public_resources.get("required") is False or public_resources.get("resource_complete") is True
    )
    if public_resources.get("required") is True:
        from merlin.development.backend.optimization.contracts import exact_tree_record

        public_resources_complete = bool(
            public_resources_complete and exact_tree_record(pkg)["sha256"] == public_resources.get("candidate_sha256")
        )
    # An undeclared optional scalability check is not requested. A declared one
    # requires its actual passing datum; missing evidence cannot count as pass.
    scalability_complete = "public_scalability" not in pub or pub["public_scalability"].get("passed") is True
    formal_grade_complete = bool(
        public_phase["formal_complete"]
        and hidden_phase["formal_complete"]
        and private_models_complete
        and private_execution_complete
        and model_check_complete
        and public_resources_complete
        and scalability_complete
    )
    completion_failures = [
        *(f"public:{reason}" for reason in public_phase["completion_failures"]),
        *(f"hidden:{reason}" for reason in hidden_phase["completion_failures"]),
        *([] if private_models_complete else ["private_full_models:incomplete"]),
        *([] if private_execution_complete else ["private_full_model_execution:incomplete"]),
        *([] if model_check_complete else ["model_check_official:incomplete"]),
        *([] if public_resources_complete else ["public_codegen_resources:incomplete"]),
        *([] if scalability_complete else ["public_spike_scalability:incomplete"]),
    ]

    # --- process metrics (from launcher), env, run_manifest ---
    environment_path = run_dir / "environment.yaml"
    environment = yaml.safe_load(environment_path.read_bytes()) if environment_path.is_file() else {}
    qualification_only = isinstance(environment, Mapping) and environment.get("qualification_only") is True
    text_files = {
        row.get("name"): row.get("sha256")
        for row in (environment.get("treatment_snapshot") or {}).get("files", [])
        if isinstance(row, Mapping) and row.get("name") in {"served/TASK.md", "archived/TASK.md"}
    }
    errata = environment.get("operator_errata") or {}
    if errata:
        text_files["archived/ERRATA.md"] = errata.get("sha256")
    bundle_path = run_dir / "input_bundle_manifest.yaml"
    bundle_document = yaml.safe_load(bundle_path.read_text(encoding="utf-8")) or {} if bundle_path.is_file() else {}
    exact_grants = list(bundle_document.get("allowed") or [])
    ctt = {}
    ctt_path = run_dir / "cost_time_toolcalls.yaml"
    if ctt_path.is_file():
        ctt = yaml.safe_load(ctt_path.read_text()) or {}

    manifest = {
        "run_id": run_dir.name,
        **({"qualification_only": True, "authoring_converged": False} if qualification_only else {}),
        "arm": a.arm,
        "operator_text_sha256": text_files,
        "assisted_grants": exact_grants if a.arm == "merlin_assisted" else [],
        "model": a.model,
        "grading_budget": {"step_timeout_s": a.qa_timeout},
        "repo_sha": frozen["repo_sha"],
        "submission_sha256": frozen["submission_sha256"],
        "frozen_at": frozen["frozen_at"],
        "integrity_status": pub.get("integrity_status"),
        "integrity_exempt": pub.get("integrity_exempt"),
        "public_dev": public_phase,
        **({"public_scalability": pub["public_scalability"]} if "public_scalability" in pub else {}),
        "hidden": hidden_phase,
        **({"hidden_capsule_validation": hidden_policy} if functional_hidden else {}),
        "private_full_models": private_models,
        **({"model_check_official": model_check_official} if required_full_models else {}),
        "public_codegen_resources": public_resources,
        **(
            {
                "capsule_observation_reuse": {
                    "qualification": imported["receipt"],
                    "numerical_producer_epoch": imported["numerical_producer_epoch"],
                    "resource_producer_epoch": imported["resource_producer_epoch"],
                    "new_capsule_execution": False,
                }
            }
            if imported is not None
            else {}
        ),
        **({} if private_execution is None else {"private_full_model_execution": private_execution}),
        "completion": {
            "formal_grade_complete": formal_grade_complete,
            **({"phase1_authoring_complete": False} if qualification_only else {}),
            "required_tier": FORMAL_REQUIRED_TIER,
            **(
                {"hidden_required_tier": "L2", "hidden_validation_mode": "functional_spike"}
                if functional_hidden
                else {}
            ),
            "required_full_models": list(required_full_models),
            "required_full_programs": {name: list(required_full_programs[name]) for name in required_full_models},
            "failures": completion_failures,
        },
        "cycles_diagnostic": pub.get("cycles_diagnostic", {}),
        "process": {
            "wall_time_seconds": ctt.get("wall_time_seconds"),
            "tokens_total": ctt.get("tokens_total"),
            "tokens_input": ctt.get("tokens_input"),
            "tokens_fresh_input": ctt.get("tokens_fresh_input"),
            "tokens_cache_write": ctt.get("tokens_cache_write"),
            "tokens_cached": ctt.get("tokens_cached"),
            "tokens_output": ctt.get("tokens_output"),
            "tokens_reasoning": ctt.get("tokens_reasoning"),
            "usage_complete": ctt.get("usage_complete"),
            "estimated_cost_usd": ctt.get("estimated_cost_usd"),
            # A subscription-seat run has no per-token spend; its dollars are notional and
            # kept in their own field so nothing sums them into a money budget.
            "billing_mode": ctt.get("billing_mode"),
            "subscription_notional_usd": ctt.get("subscription_notional_usd"),
            "cost_unavailable_reason": ctt.get("cost_unavailable_reason"),
            "tool_calls": ctt.get("tool_calls"),
            "metrics_available": ctt.get("available"),
        },
        "iterations": len(list((run_dir / "iterations").glob("iteration_*"))),
        "oracle_mode": "no_oracle(L0/L1/trace)"
        if a.no_oracle
        else f"contract-routed({'+'.join(sorted(capsule_adapters.oracle_adapters(context.target)))})",
        # HONEST gradeability: a --no-oracle run is a structure-only smoke — the numeric verdict is
        # withheld (capsules read back not_gradeable_no_oracle, never a numeric pass), so the run is NOT
        # gradeable. A graded run carries gradeable=true. Sourced from the grade score (gradeable flag).
        "gradeable": bool(pub.get("gradeable", not a.no_oracle)),
        "gradeable_reason": (
            None if not a.no_oracle else "numeric oracle unavailable — structural (L0/L1/trace) tiers only"
        ),
        "model_mesh_sim": {
            "required_tier": FORMAL_REQUIRED_TIER,
            "simulator": _formal_mesh_sim,
            "inherited_value_overridden": _inherited_mesh_sim,
            "fidelity": _formal_mesh_selection.get("fidelity"),
            "selection": _formal_mesh_selection.get("selection"),
            "reason": _formal_mesh_selection.get("reason"),
            "passed_over": _formal_mesh_selection.get("passed_over", []),
        },
    }
    (run_dir / "run_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    fr = [
        f"# {run_dir.name} — final report ({a.arm})",
        "",
        "| Model | Sealed numerical status |",
        "| --- | --- |",
        *[
            f"| {model} | {(model_check_official.get('models') or {}).get(model, {}).get('status', 'unknown')} |"
            for model in required_full_models
        ],
        f"| Model kernel RTL | {model_check_official.get('rtl_status', 'unknown')} |",
        "",
        f"- whole-model validation: {model_check_official.get('whole_model_status', 'unknown')}",
        f"- integrity: {manifest['integrity_status']} (exempt={manifest['integrity_exempt']})",
        f"- public/dev: functional_pass={manifest['public_dev']['functional_pass']} "
        f"passed={manifest['public_dev']['passed']} highest_tier={manifest['public_dev']['highest_tier']}",
        f"- hidden: {manifest['hidden']['passed']} (after freeze "
        f"{frozen['submission_sha256'][:12] if frozen['submission_sha256'] else 'n/a'})"
        + ("; Spike L2 functional corroboration only, no RTL certification" if functional_hidden else ""),
        f"- process: wall={manifest['process']['wall_time_seconds']}s "
        f"tokens={manifest['process']['tokens_total']} {_cost_phrase(manifest['process'])} "
        f"tool_calls={manifest['process']['tool_calls']} (available={manifest['process']['metrics_available']})",
        f"- oracle_mode: {manifest['oracle_mode']}",
        f"- gradeable: {manifest['gradeable']}"
        + (f" ({manifest['gradeable_reason']})" if manifest.get("gradeable_reason") else ""),
        "",
    ]
    (run_dir / "final_report.md").write_text("\n".join(fr) + "\n")
    print(
        f"graded {run_dir.name}: public functional_pass={pub.get('functional_pass')} "
        f"({manifest['public_dev']['passed']}), hidden={manifest['hidden']['passed']}, "
        f"integrity={manifest['integrity_status']} formal_grade_complete={formal_grade_complete}"
    )
    return 0 if formal_grade_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
