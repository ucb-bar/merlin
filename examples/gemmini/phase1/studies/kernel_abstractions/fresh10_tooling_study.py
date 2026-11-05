"""Prepare the fresh10 capability-removal study without starting an author.

The dispatch envelope is compatible with the separately pinned fleet runner.
Its ``panel`` field aliases ``common_workloads`` solely for that runner's input
integrity check; it is not the earlier study's private 34-case transfer panel.
Only ``run --confirm`` can dispatch. Native preparation and boundary qualification
must finish first. The full-model campaign and continuous promotion stay enabled.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SCHEMA = "merlin.study.xs_ablation.v1"  # The reviewed dispatcher's envelope.
KIND = "fresh10-tooling-v1"
REVIEW_SCHEMA = "merlin.study.fresh10_tooling_launch_review.v1"
BASELINE_REVISION = "f9d7fb5c3d6fa2e5168e982953dd0b1d9b480076"
ARMS = ("infra", "scheduling", "no-reference-kernels")
PROFILES = tuple(KIND + "/" + arm for arm in ARMS)
CPU_GROUPS = (tuple(range(38, 48)) + tuple(range(56, 61)), tuple(range(61, 76)), tuple(range(76, 91)))
RESERVED_EXTERNAL_CPUS = tuple(range(4, 38)) + tuple(range(48, 56))
AUTHOR_S, DRAIN_S = 14400, 900
SNAPSHOTS = (0, 1800, 3600, 7200, 14400)
COUNTS = dict(total_public=152, core=109, scalability=40, connected_programs=3)
REFERENCE_MEMBERS, REFERENCE_ENTRIES = 23, 155
COMMON_FLAGS = (
    "--model",
    "--effort",
    "--driver",
    "--codex-binary",
    "--sandbox",
    "--experiment",
    "--level",
    "--compiler-execution",
    "--compiler-access",
    "--validation-scope",
    "--schedule",
    "--qa-timeout",
    "--model-budget-s",
    "--sim-per-capsule-timeout",
    "--grade-interval",
    "--hidden-capsule-mode",
)
FILE_FLAGS = (
    "--private-full-model-spec",
    "--instruction-selection",
    "--oracle-timing",
    "--instruction-selection-sha256",
    "--native-result-format-selection",
    "--native-result-format-selection-sha256",
)
REVIEW_CHECKS = (
    "common_workloads",
    "common_judge_and_model",
    "empty_backends",
    "fresh_homes_and_state",
    "actual_served_prompt_comparability",
    "profile_source_and_service_boundaries",
    "native_public_data_readability",
    "author_helper_candidate_canaries",
    "no_cca_or_automatic_kernel_learning",
    "reference_content_and_removal",
    "native_preflights",
)
GUIDANCE = """# Compiler construction and measured improvement

Build one reusable compiler from an empty target backend for the complete selected
workload: all public capsule and program suites, the protected checks, and the six
selected full models. Use the supplied numerical, ISA, ordering and execution
contracts. Implement general rules selected by semantics, shape, layout and
resource predicates; preserve correct behavior across the whole selected workload.

Authoring lasts four hours, followed by fifteen minutes of ordinary final draining
and validation. Share fifteen CPUs across all roles and use at most eight helper
agents. Integrate useful changes into the one compiler while ordinary immutable
snapshot checks and continuous promotion run. Start performance comparisons when
a case is correct; use the ordinary self-check, public program, scalability and
full-model workflows advertised by this environment.

Compare variants on the same cases, compiler source and evaluator identity. Keep
numerical evidence separate from provisional model estimates and physical
validation. Missing, refused or unfinished checks remain unknown. Record reusable
rules, emitted effects, measured comparisons, failures and uncertainty with their
source identities. Use only the APIs and files actually disclosed by the selected
author profile. No CCA/reference-analysis service or automatic kernel learning is
enabled in this study. Hardware-loop/sequencer instruction roles are prohibited.
"""


def load_module(path):
    path = Path(path)
    name = "fresh10_operator_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("operator module unavailable: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def campaign_module():
    return load_module(HERE.parents[1] / "launch/fresh_campaign.py")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pin(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("operator inputs must be direct canonical files: " + str(path))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify_pin(selected):
    observed = pin(selected["path"])
    if observed["sha256"] != selected["sha256"]:
        raise ValueError("operator input changed: " + selected["path"])
    return Path(observed["path"])


def reopen(selected):
    return json.loads(verify_pin(selected).read_text())


def option(argv, flag):
    positions = [i for i, value in enumerate(argv) if value == flag]
    if len(positions) > 1 or (positions and positions[0] + 1 == len(argv)):
        raise ValueError("malformed/duplicate campaign option " + flag)
    return argv[positions[0] + 1] if positions else None


def set_option(argv, flag, value):
    option(argv, flag)
    result = list(argv)
    if flag in result:
        result[result.index(flag) + 1] = str(value)
    else:
        result += [flag, str(value)]
    return result


def revision_identity(revision, export_id):
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or export_id != revision[:12]:
        raise ValueError("supply an explicit full source revision and its 12-hex export id")


def cpu_groups(groups):
    groups = [list(group) for group in groups]
    if (
        len(groups) != 3
        or any(len(group) != 15 for group in groups)
        or any(type(cpu) is not int or cpu < 0 for group in groups for cpu in group)
        or len({cpu for group in groups for cpu in group}) != 45
    ):
        raise ValueError("three disjoint pools of exactly fifteen CPUs are required")
    return groups


def policy_identity(policy):
    """Semantic obligations, allowing only the common clock/worker budget edit."""
    result = copy.deepcopy(policy)
    for key in ("duration_s", "drain_s", "final_model_s"):
        result.pop(key, None)
    for key in ("gsim", "gsim_correctness"):
        for field in ("max_jobs", "worker_budget_s"):
            result.get(key, {}).pop(field, None)
    result.setdefault("gsim_correctness", {}).setdefault("continuous_promotion", True)
    return result


def coordinator_pins(path):
    path = Path(path).absolute()
    helper = path.parent.parents[1] / "launch/campaign_capacity.py"
    # A caller supplies an already reviewed frozen coordinator. Both code files
    # are pinned; never silently substitute this checkout's older capacity rule.
    return dict(runner=pin(path), capacity=pin(helper))


def _baseline(audit_path):
    selected = pin(audit_path)
    audit = reopen(selected)
    if (
        audit.get("source_revision") != BASELINE_REVISION
        or audit.get("origin", {}).get("initial_files") != 0
        or any(audit.get("launch_capsules", {}).get(key) != value for key, value in COUNTS.items())
        or audit.get("task_scope", {}).get("held_out_capsules") != 14
    ):
        raise ValueError("audit is not the deployed empty-backend fresh10CCA baseline")
    manifest = reopen(audit["pins"]["reference_manifest"])
    mapping = reopen(audit["pins"]["reference_map"])
    members = sum(len(origin.get("files", [])) for origin in manifest["origins"])
    if (
        members != REFERENCE_MEMBERS
        or len(mapping.get("entries", [])) != REFERENCE_ENTRIES
        or mapping["reference_manifest_sha256"] != audit["pins"]["reference_manifest"]["sha256"]
    ):
        raise ValueError("retain all 23 raw reference members and all 155 map entries")
    policy = reopen(audit["pins"]["joint_policy"])["policy"]
    if policy.get("study_protocol") is not None:
        raise ValueError("fresh10 baseline must retain normal full-model campaign lifecycle")
    return selected, audit, policy


def specs(
    base_spec,
    output,
    *,
    baseline_audit,
    source_revision,
    export_id,
    coordinator,
    guidance=None,
    external_units=(),
    groups=CPU_GROUPS,
):
    """Render three native campaign specs; no builds, CPU claims or authors."""
    revision_identity(source_revision, export_id)
    groups = cpu_groups(groups)
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("select a new canonical study directory")
    campaign = campaign_module()
    context = campaign.Context(base_spec, export_id=export_id)
    base = campaign.load_spec(Path(base_spec).resolve())
    baseline_pin, audit, actual_policy = _baseline(baseline_audit)
    coordinator_selection = coordinator_pins(coordinator)
    template_argv = base["plan"]["argv"]
    if (
        base.get("seed") is not None
        or "--fresh-backend" not in template_argv
        or option(template_argv, "--validation-scope") != "full"
        or option(template_argv, "--schedule") != "continuous"
        or any(option(template_argv, flag) is None for flag in FILE_FLAGS[:2])
    ):
        raise ValueError("base spec must be the full native fresh10 campaign with model and ISA selection")
    base_policy_pin = base["plan"]["policy"]["base"]
    verify_pin(dict(path=str(context.path(base_policy_pin["path"])), sha256=base_policy_pin["sha256"]))
    common_arguments = {
        flag: context.x(option(template_argv, flag)) for flag in COMMON_FLAGS if option(template_argv, flag) is not None
    }
    performance = pin(context.path(base["plan"]["timing_selection"]).resolve())
    if performance["sha256"] != audit["pins"]["performance_selection"]["sha256"]:
        raise ValueError("base spec changed the deployed fresh10 performance model selection")
    file_arguments = {}
    for flag in FILE_FLAGS:
        raw = option(template_argv, flag)
        if raw is not None:
            expanded = context.x(raw)
            file_arguments[flag] = expanded if flag.endswith("-sha256") else pin(Path(expanded).resolve())
    expected_refs = audit["pins"]["reference_manifest"]["sha256"]
    # The plan's manifest is an output of the new native input preparation.
    # Render from the reviewed source bank before that output/export exists.
    references_path = context.path(base["inputs"]["public_references"]) / "reference-manifest.json"
    if pin(references_path.resolve())["sha256"] != expected_refs:
        raise ValueError("base spec does not select the actual fresh10 raw reference manifest")
    original_map = base["plan"].get("reference_map")
    if (
        not original_map
        or context.x(original_map["sha256"]) != audit["pins"]["reference_map"]["sha256"]
        or pin(context.path(original_map["path"]).resolve())["sha256"] != audit["pins"]["reference_map"]["sha256"]
    ):
        raise ValueError("base spec must select the complete pinned fresh10 reference map")
    common_policy = copy.deepcopy(actual_policy)
    common_policy.update(duration_s=AUTHOR_S + DRAIN_S, drain_s=DRAIN_S, final_model_s=DRAIN_S)
    common_policy.setdefault("gsim_correctness", {})["continuous_promotion"] = True
    common_policy["prohibited_instruction_roles"] = sorted(
        set(common_policy.get("prohibited_instruction_roles", [])) | {"loop_descriptor"}
    )
    if policy_identity(common_policy) != policy_identity(actual_policy):
        raise ValueError("common policy edit changed native workload or model obligations")
    output.mkdir(mode=0o700, parents=True)
    policy_path = output / "common-base-policy.json"
    write_json(policy_path, common_policy)
    guide_path = output / "common-guidance.md"
    guide_path.write_text(verify_pin(pin(guidance)).read_text() if guidance else GUIDANCE)
    common = dict(
        schema="merlin.study.fresh10_common_workloads.v1",
        baseline=baseline_pin,
        baseline_source_revision=BASELINE_REVISION,
        capsules=COUNTS,
        protected_capsules=14,
        full_models=6,
        arguments=common_arguments,
        argument_inputs=file_arguments,
        policy_identity=policy_identity(common_policy),
        performance_selection=performance,
        scope="native fresh10 workloads; no additional 34-cell transfer panel",
        results_claimed=False,
    )
    common_path = output / "common_workloads.json"
    write_json(common_path, common)
    rows = []
    for arm, profile, cpus in zip(ARMS, PROFILES, groups, strict=True):
        document = copy.deepcopy(base)
        document.pop("extends", None)
        document.pop("seed", None)
        document["id"] = KIND + "-" + arm
        document["variables"].update(
            STUDY_ROOT=str(output),
            STUDY_ARM=arm,
            INPUTS="${FRAMEWORK}/out/artifacts/fresh10-tooling/${RUN_ID}/inputs",
            ARTIFACTS="${STUDY_ROOT}/arms/${STUDY_ARM}/artifacts",
            RELEASE="${INPUTS}/source-release",
            RUN_STATE="${STUDY_ROOT}/arms/${STUDY_ARM}/state",
            PRIVATE_ROOT="${STUDY_ROOT}/arms/${STUDY_ARM}/private",
            OPERATOR_TMP="${STUDY_ROOT}/arms/${STUDY_ARM}/tmp",
        )
        # Make omitted --export-id incapable of following a subsequently moving HEAD.
        document["variables"]["FRAMEWORK"] = "${REPO}/out/build/run-frameworks/" + export_id
        run_id = output.name + "-" + arm
        unit = "merlinphase1f10tooling" + hashlib.sha256(run_id.encode()).hexdigest()[:16]
        document["run"].update(
            run_id=run_id,
            unit=unit,
            cpus=",".join(map(str, cpus)),
            preparation_cpus="0",
            max_wall_s=AUTHOR_S + DRAIN_S,
            final_model_s=DRAIN_S,
            workers=15,
            sim_max_jobs=15,
            author_helper_effort="medium",
            treatment=arm,
        )
        document["inputs"]["author_profile_common_inputs"] = "fresh10_common"
        plan = document["plan"]
        plan.update(
            author_profile=profile,
            reference_guidance="off",
            reference_experiment_tasks="off",
            reference_experiments=None,
            guidance=str(guide_path),
        )
        plan["policy"]["base"] = pin(policy_path)
        plan["policy"].setdefault("defaults", {}).pop("drain_s", None)
        for flag in ("--codex-max-subagents", "--author-max-subagents"):
            plan["argv"] = set_option(plan["argv"], flag, 8)
        if arm == "no-reference-kernels":
            document["inputs"]["public_references"] = None
            plan["public_reference_manifest"] = None
            plan["reference_map"] = None
        plan.setdefault("record", {}).update(
            study_kind=KIND,
            study_source_revision=source_revision,
            study_baseline_source_revision=BASELINE_REVISION,
            limits=dict(author_helpers=8, shared_compute_cpus=15),
        )
        path = output / (arm + ".yaml")
        path.write_text(yaml.safe_dump(document, sort_keys=False))
        rows.append(dict(arm=arm, profile=profile, cpus=cpus, unit=unit, run_id=run_id, spec=pin(path)))
    record = dict(
        schema=SCHEMA,
        study_kind=KIND,
        preparation_schema="merlin.study.fresh10_tooling.v1",
        state="specifications",
        source_revision=source_revision,
        export_id=export_id,
        operator_task=context.lookup("TASK"),
        baseline=baseline_pin,
        base_spec_chain=[pin(path) for path in campaign.spec_chain(Path(base_spec).resolve())],
        coordinator=coordinator_selection,
        author_s=AUTHOR_S,
        drain_s=DRAIN_S,
        total_s=AUTHOR_S + DRAIN_S,
        final_model_s=DRAIN_S,
        max_ours=3,
        max_machine=5,
        reserved_external_slots=2,
        reserved_external_cpus=list(RESERVED_EXTERNAL_CPUS),
        machine_limit_scope="15-CPU full-size campaigns; reviewed coordinator owns admission",
        external_units=list(external_units),
        common_workloads=pin(common_path),
        panel=pin(common_path),
        dispatch_panel_alias="common_workloads; integrity only, no transfer panel",
        common_guidance=pin(guide_path),
        common_base_policy=pin(policy_path),
        references=dict(
            raw_members=REFERENCE_MEMBERS,
            map_entries=REFERENCE_ENTRIES,
            map_new_elf_subset=68,
            public_abi_files_separate=23,
            manifest_sha256=expected_refs,
            map_sha256=audit["pins"]["reference_map"]["sha256"],
        ),
        measurement=dict(
            snapshot_seconds=list(SNAPSHOTS),
            operator_only=True,
            feedback=False,
            automatic_grade_credit=False,
            full_model_validation="normal; incomplete at deadline remains pending",
        ),
        common_edits=[
            "4h author + 900s drain/final-model budget",
            "common reviewed harness and knob routing fixes",
            "explicit capability profiles; CCA and automatic kernel learning off",
            "one neutral external guidance; actual fresh10 TASK core retained",
            "five operator-only immutable checkpoints, no added transfer cohort",
        ],
        contrasts={
            "A_vs_B": "reference-assisted infrastructure versus identical base plus scheduling semantics",
            "B_vs_C": "changes both supplied library and raw references; not an isolated pass-library ablation",
            "C": "full general tooling without raw reference kernels, CCA or automatic kernel learning",
        },
        arms=rows,
        author_jobs_launched=0,
        model_calls=0,
    )
    write_json(output / "study.json", record)
    return record


def _study(path, states):
    path = Path(path).resolve(strict=True)
    record = json.loads(path.read_text())
    if (
        record.get("schema") != SCHEMA
        or record.get("study_kind") != KIND
        or record.get("state") not in states
        or tuple(row["arm"] for row in record["arms"]) != ARMS
    ):
        raise ValueError("not a fresh10 three-arm study in the required state")
    revision_identity(record["source_revision"], record["export_id"])
    cpu_groups([row["cpus"] for row in record["arms"]])
    if record["panel"] != record["common_workloads"]:
        raise ValueError("dispatch panel must alias the actual common-workload pin")
    for key in ("baseline", "common_workloads", "common_base_policy", "common_guidance"):
        verify_pin(record[key])
    for selected in record["coordinator"].values():
        verify_pin(selected)
    return path, record


def _review(record, path, plan_pins):
    selected = pin(path)
    review = reopen(selected)
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("source_revision") != record["source_revision"]
        or review.get("export_id") != record["export_id"]
        or review.get("common_workloads_sha256") != record["common_workloads"]["sha256"]
        or review.get("baseline_sha256") != record["baseline"]["sha256"]
        or any(review.get(check) is not True for check in REVIEW_CHECKS)
    ):
        raise ValueError("complete native qualification and common-input review are required")
    rows = review.get("arms", [])
    if tuple(row.get("arm") for row in rows) != ARMS:
        raise ValueError("review must cover all three arms in declared order")
    for arm, row, plan_pin in zip(ARMS, rows, plan_pins, strict=True):
        if (
            row.get("plan_sha256") != plan_pin["sha256"]
            or row.get("preflight_returncode") != 0
            or row.get("all_passed") is not True
            or row.get("initial_files") != 0
            or row.get("public_abi_files_read") != 23
        ):
            raise ValueError("native qualification failed or does not bind this plan: " + arm)
        if not row.get("evidence"):
            raise ValueError("native qualification needs pinned evidence: " + arm)
        for evidence in row["evidence"]:
            verify_pin(evidence)
    return selected


def _qualify(record, plans, review_path):
    if len(plans) != 3:
        raise ValueError("all three prepared plans are required")
    selections = [pin(path) for path in plans]
    review_pin = _review(record, review_path, selections)
    common = reopen(record["common_workloads"])
    identities = []
    for row, selected in zip(record["arms"], selections, strict=True):
        verify_pin(row["spec"])
        plan = reopen(selected)
        argv = plan["argv"]
        if (
            plan.get("source_revision") != record["source_revision"]
            or plan.get("execution_started")
            or plan.get("treatment") != row["arm"]
            or plan.get("run_id") != row["run_id"]
            or plan.get("author_profile", {}).get("name") != row["profile"]
            or plan.get("launch_spec", {}).get("sha256") != row["spec"]["sha256"]
            or plan.get("launch_spec", {}).get("unit") != row["unit"]
            or plan.get("cpu_reservation", {}).get("cpus") != row["cpus"]
            or plan["cpu_reservation"].get("currently_acquired") is not False
        ):
            raise ValueError("native plan/source/profile/resource binding differs: " + row["arm"])
        profile = plan["author_profile"]
        expected_capabilities = dict(
            reference_analysis=False,
            automatic_kernel_learning=False,
            raw_kernel_references=row["arm"] != "no-reference-kernels",
            scheduling_semantics=row["arm"] != "infra",
            transform_library=row["arm"] == "no-reference-kernels",
            catalog_advice=row["arm"] == "no-reference-kernels",
            compiler_integration=True,
            performance_diagnostics=True,
        )
        if profile.get("live_cca") is not False or any(
            profile.get("capabilities", {}).get(key) is not expected for key, expected in expected_capabilities.items()
        ):
            raise ValueError("native capability record differs from the declared contrast")
        framework = Path(plan["framework_repository"])
        marker = pin(framework / ".merlin-source-revision.json")
        if framework.name != record["export_id"] or reopen(marker).get("revision") != record["source_revision"]:
            raise ValueError("native framework bytes belong to another source export")
        masks = plan.get("environment", {})
        for key in ("MERLIN_COMPUTE_CPUS", "MERLIN_CODEX_CPUS", "MERLIN_PREPARATION_CPUS"):
            if masks.get(key) != ",".join(map(str, row["cpus"])):
                raise ValueError("native execution CPU mask differs: " + key)
        for key, expected in COUNTS.items():
            if plan.get("capsules", {}).get(key) != expected:
                raise ValueError("native workload inventory differs: " + key)
        for flag, expected in common["arguments"].items():
            if option(argv, flag) != expected:
                raise ValueError("common native option differs: " + flag)
        for flag, expected in common["argument_inputs"].items():
            observed = option(argv, flag)
            if observed is None or (
                observed != expected if isinstance(expected, str) else pin(observed)["sha256"] != expected["sha256"]
            ):
                raise ValueError("full-model/ISA/evaluator selection differs: " + flag)
        if any(
            flag not in argv for flag in ("--fresh-backend", "--author-token-unbounded", "--live-conformance-feedback")
        ):
            raise ValueError("ordinary fresh-backend campaign flags must remain enabled")
        if any(option(argv, flag) != "8" for flag in ("--codex-max-subagents", "--author-max-subagents")):
            raise ValueError("both helper ceilings must equal eight")
        if (
            option(argv, "--author-helper-effort") != "medium"
            or option(argv, "--sim-max-jobs") != "15"
            or option(argv, "--max-wall-s") != str(AUTHOR_S + DRAIN_S)
            or option(argv, "--round-timeout") != str(AUTHOR_S + DRAIN_S)
        ):
            raise ValueError("common helper/worker/clock settings differ")
        if option(argv, "--author-profile") != row["profile"]:
            raise ValueError("actual author profile argument differs")
        if any(
            option(argv, flag) is None
            for flag in (
                "--compiler-library-root",
                "--compiler-library-contract",
                "--compiler-library-contract-sha256",
                "--full-compiler-manifest",
            )
        ):
            raise ValueError("fresh10 compiler/library integration must remain enabled")
        if (
            plan.get("reference_guidance") != "off"
            or plan.get("reference_experiments") is not None
            or any(flag.startswith("--reference-experiments") for flag in argv)
        ):
            raise ValueError("CCA/reference experiments must be disabled in all arms")
        if row["arm"] == "no-reference-kernels":
            if plan.get("reference_manifest") is not None or plan.get("reference_map") is not None:
                raise ValueError("C must remove every raw reference manifest and map")
        else:
            for name, expected in (("reference_manifest", "manifest_sha256"), ("reference_map", "map_sha256")):
                if not plan.get(name) or plan[name]["sha256"] != record["references"][expected]:
                    raise ValueError("A/B must retain the identical full raw reference set and map")
                verify_pin(plan[name])
        policy_pin = pin(option(argv, "--joint-campaign"))
        if record["state"] != "specifications" and row.get("policy") != policy_pin:
            raise ValueError("prepared policy pin changed after qualification")
        policy = reopen(policy_pin)
        if (
            policy.get("study_protocol") is not None
            or policy.get("drain_s") != DRAIN_S
            or policy.get("duration_s") != AUTHOR_S + DRAIN_S
            or policy.get("final_model_s") != DRAIN_S
            or policy.get("gsim_correctness", {}).get("continuous_promotion", True) is not True
            or policy_identity(policy) != common["policy_identity"]
        ):
            raise ValueError("normal full-model policy, calibration and continuous promotion must be preserved")
        guidance = pin(option(argv, "--operator-errata"))
        if guidance["sha256"] != record["common_guidance"]["sha256"]:
            raise ValueError("all arms require identical common external guidance")
        timing = plan["timing_selection"]
        verify_pin(timing)
        if timing["sha256"] != common["performance_selection"]["sha256"]:
            raise ValueError("prepared performance evaluator differs from the actual fresh10 baseline")
        identities.append(
            dict(
                arguments=common["arguments"],
                argument_inputs=common["argument_inputs"],
                policy=policy_identity(policy),
                timing_sha256=timing["sha256"],
                common_guidance_sha256=guidance["sha256"],
            )
        )
        row.update(plan=selected, policy=policy_pin)
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("plans differ in common judge/model/policy identities")
    record.update(state="prepared", prepared_arms=list(ARMS), common_identity=identities[0], launch_review=review_pin)
    return record


def prepare(study_path, plans, review_path):
    """Validate actual native plans and operator evidence, never manufacture it."""
    study_path, record = _study(study_path, {"specifications"})
    record = _qualify(record, plans, review_path)
    write_json(study_path, record)
    return record


def run(study_path, *, confirm=False, poll_s=5):
    if not confirm:
        raise ValueError("run requires explicit --confirm")
    path, record = _study(study_path, {"prepared", "running"})
    plan_pins = [row["plan"] for row in record["arms"]]
    verify_pin(record["launch_review"])
    _qualify(copy.deepcopy(record), [selected["path"] for selected in plan_pins], record["launch_review"]["path"])
    # The runner also reopens plans/policies immediately before every durable
    # CPU/admission claim, including arms that waited for a peer to finish.
    runner = load_module(verify_pin(record["coordinator"]["runner"]))
    return runner.run(path, confirm=True, poll_s=poll_s)


def observe(study_path, runtime_index, *, once=False, interval_s=1, now=None):
    """Operator-only checkpoints; no policy switch, author feedback or grading.

    Runtime rows explicitly bind each native workspace and joint_campaign.json.
    A late observer records missed intervals as unavailable rather than calling
    its current source a historical snapshot. Start this observer before authors.
    """
    _, record = _study(study_path, {"prepared", "running"})
    selected_runtime = pin(runtime_index)
    runtime = reopen(selected_runtime)
    if (
        runtime.get("schema") != "merlin.study.fresh10_runtime_index.v1"
        or runtime.get("source_revision") != record["source_revision"]
        or tuple(row.get("arm") for row in runtime.get("arms", [])) != ARMS
    ):
        raise ValueError("runtime index must bind the three native contexts and source revision")
    from merlin_experiments.phase1 import study_protocol

    owner = HERE.parents[4] / "packages/merlin-experiments/src/merlin_experiments/phase1/study_protocol.py"
    if Path(study_protocol.__file__).resolve() != owner:
        raise ValueError("checkpoint source owner must come from this frozen framework, not ambient PYTHONPATH")
    root = Path(study_path).resolve().parent / "operator-checkpoints"
    root.mkdir(mode=0o700, exist_ok=True)
    index = root / "index.json"
    state = (
        json.loads(index.read_text())
        if index.exists()
        else dict(
            schema="merlin.study.fresh10_checkpoints.v1",
            runtime=selected_runtime,
            source_revision=record["source_revision"],
            author_visible=False,
            automatic_grade_credit=False,
            arms={},
        )
    )
    if state["runtime"] != selected_runtime:
        raise ValueError("runtime contexts changed after observation began")
    while True:
        observed = time.time() if now is None else now
        for row, arm in zip(runtime["arms"], record["arms"], strict=True):
            campaign_path = Path(row["run_dir"]) / "joint_campaign.json"
            if not campaign_path.is_file():
                continue
            campaign = json.loads(campaign_path.read_text())
            started = campaign.get("started_at_epoch")
            if started is None:
                continue
            # Pin the native clock and policy rather than the coordinator start.
            if campaign.get("policy") != reopen(arm["policy"]):
                raise ValueError("runtime campaign policy differs from the prepared arm")
            entry = state["arms"].setdefault(arm["arm"], dict(started_at_epoch=started, snapshots=[]))
            if entry["started_at_epoch"] != started:
                raise ValueError("native author clock changed; checkpoints cannot reset")
            elapsed = observed - started
            done = {item["offset_s"] for item in entry["snapshots"]}
            for offset in SNAPSHOTS:
                if offset in done or elapsed < offset:
                    continue
                late = elapsed - offset
                try:
                    if late > max(5, interval_s * 2):
                        raise ValueError("checkpoint missed; current source cannot replace historical source")
                    item = study_protocol.snapshot(
                        Path(row["workspace"]) / "submission",
                        root / arm["arm"] / str(offset),
                        offset_s=offset,
                        started_at=started,
                        now=observed,
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    item = dict(
                        offset_s=offset,
                        observed_elapsed_s=elapsed,
                        status="unavailable",
                        author_visible=False,
                        reason=str(error),
                    )
                entry["snapshots"].append(item)
        state["observed_at_epoch"] = observed
        write_json(index, state)
        complete = all(len(state["arms"].get(arm, {}).get("snapshots", [])) == len(SNAPSHOTS) for arm in ARMS)
        if once or complete:
            return state
        time.sleep(interval_s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("specs", help="render only; no builds, reservations or authors")
    for flag in ("base-spec", "output", "baseline-audit", "source-revision", "export-id", "coordinator"):
        command.add_argument("--" + flag, required=True)
    command.add_argument("--guidance")
    command.add_argument("--external-unit", action="append", default=[])
    command = commands.add_parser("prepare", help="qualify three native plans and their pinned review")
    command.add_argument("--study", required=True)
    command.add_argument("--plans", nargs=3, required=True)
    command.add_argument("--review", required=True)
    command = commands.add_parser("run", help="dispatch through the separately pinned reviewed fleet runner")
    command.add_argument("--study", required=True)
    command.add_argument("--confirm", action="store_true")
    command = commands.add_parser("observe", help="operator-only immutable snapshots; no author feedback")
    command.add_argument("--study", required=True)
    command.add_argument("--runtime-index", required=True)
    command.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "specs":
        result = specs(
            args.base_spec,
            args.output,
            baseline_audit=args.baseline_audit,
            source_revision=args.source_revision,
            export_id=args.export_id,
            coordinator=args.coordinator,
            guidance=args.guidance,
            external_units=args.external_unit,
        )
    elif args.action == "prepare":
        result = prepare(args.study, args.plans, args.review)
    elif args.action == "observe":
        result = observe(args.study, args.runtime_index, once=args.once)
    else:
        return run(args.study, confirm=args.confirm)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
