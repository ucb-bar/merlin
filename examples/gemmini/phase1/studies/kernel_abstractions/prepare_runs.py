"""Prepare matched offline authors; execution is a separate, explicit command.

Uses an operator-selected existing evaluator configuration. Source-only preparation remains the
default; an explicit continuation preserves the full seed and its prior evidence, with a new clock.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ordinary(root):
    root = Path(root).absolute()
    if root != root.resolve() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError(f"selected input must contain no symlinks: {root}")
    return root


def copy_source(source, destination):
    ordinary(source)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "build", "selfcheck_out", "compiler_library"),
    )


def link_bank(source, destination):
    """Independent directory entries over the same immutable reference bytes."""
    ordinary(source)
    shutil.copytree(source, destination, copy_function=os.link)


def stage_capture(pin, workspace):
    """Stage only the reviewed capture/tool members, separately from common raw references."""
    manifest = Path(pin["path"])
    if digest(manifest) != pin["sha256"]:
        raise ValueError("CCA capture selection pin changed")
    document = json.loads(manifest.read_bytes())
    if document.get("schema") != "merlin.reference_capture_selection.v1" or not document.get("entries"):
        raise ValueError("CCA lifting requires a nonempty captured-reference selection")
    root = ordinary(manifest.parent)
    selected = {}
    for entry in document["entries"]:
        for key in ("capture", "facts", "contract", "observer"):
            member = entry[key]
            relative = Path(member["path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("cca-evidence",):
                raise ValueError("CCA capture member must be inside the reviewed evidence grant")
            relative = relative.relative_to("cca-evidence")
            if digest(root / relative) != member["sha256"]:
                raise ValueError("CCA capture member pin changed")
            selected[relative] = root / relative
    destination = workspace / "cca-evidence"
    for relative, source in selected.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o444)
    target = destination / "manifest.json"
    shutil.copyfile(manifest, target)
    target.chmod(0o444)
    return dict(path=str(target), sha256=pin["sha256"])


def continuation_input(path):
    """Select the existing seed owner without importing historical verdicts as certificates."""
    if path is None:
        return None, None
    path = Path(path).resolve(strict=True)
    selected = json.loads(path.read_bytes())
    if (
        set(selected) != {"from", "run"}
        or set(selected["from"]) != {"path", "sha256"}
        or set(selected["run"]) != {"run_id", "run_dir", "workspace", "framework"}
    ):
        raise ValueError("continuation requires exact seed from pin and original run paths")
    for key in ("run_dir", "workspace", "framework"):
        value = Path(selected["run"][key])
        if not value.is_absolute() or value != value.resolve() or not value.is_dir():
            raise ValueError("continuation requires canonical original run directories")
    owner_path = Path(__file__).resolve().parents[2] / "launch/fresh_seed.py"
    spec = importlib.util.spec_from_file_location("study_seed_owner", owner_path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    owner.verify_source(ordinary(selected["from"]["path"]), selected["from"]["sha256"])
    carry = dict.fromkeys(owner.CARRY, True)
    carry.update(certificates=False, results=False)
    selected.update(notes=owner.NOTES, carry=carry)
    selected["selection"] = {"path": str(path), "sha256": digest(path)}
    selected["owner"] = {"path": str(owner_path), "sha256": digest(owner_path)}
    return selected, owner


def environment(base, *, framework, private, cpu, guidance, reference_map):
    env = {key: value for key, value in base.items() if not key.startswith("MERLIN_REFERENCE")}
    for key in ("MERLIN_CODEX_CPUS", "MERLIN_COMPUTE_CPUS", "MERLIN_PREPARATION_CPUS"):
        env[key] = str(cpu)
    for key in (
        "MERLIN_DEVICE_BUILD_JOBS",
        "MERLIN_GRADE_WORKERS",
        "MERLIN_PRIVATE_EXECUTION_WORKERS",
        "MERLIN_GSIM_MAX_SLOTS",
        "MERLIN_SPIKE_MAX_SLOTS",
        "MERLIN_SELFCHECK_ORDINARY_JOBS",
        "MERLIN_SELFCHECK_VARIANT_JOBS",
        "MERLIN_SELFCHECK_BROKER_JOBS",
    ):
        env[key] = "1"
    env["MERLIN_SELFCHECK_BROKER_JOBS"] = "2"
    for key in ("MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GSIM_SLOT_ROOT", "MERLIN_BUNDLE_CAS"):
        env[key] = str(private / key.lower())
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(private / "tmp"),
        MERLIN_REPO_ROOT=str(framework),
        MERLIN_WORK_DIR=str(framework),
        MERLIN_OUT_ROOT=str(private / "out"),
    )
    if guidance == "on" and reference_map is not None:
        env["MERLIN_REFERENCE_MAP"] = json.dumps(reference_map, sort_keys=True)
    return env


TASK = """Improve the untouched Fresh6 compiler using reusable transformations and scheduling dialect.
This is a fresh {seconds}-second session with {resources}. Do not spawn helpers. Keep working until the
budget ends. No previous author conversation or experiment history has been supplied.

submission/ is editable; baseline/ contains the identical starting compiler. Both contain the
selected shared compiler library. The relevant raw expert sources, required headers and trusted
metric targets are identical across study arms in read-only reference-kernels/. public-cases/
contains the selected training interfaces. Hidden shapes are evaluated after authoring.
Keep precision, fusion, every live output, layout and alias behavior unchanged. Source semantics,
schedule intent, actual emitted effects and performance are distinct evidence. Preserve unknowns.
{tool_instructions}

Use the ORIGINAL brokered evaluator, for example:
  python3 agent_selfcheck.py --suite scalability --sim spike --capsules CASE --workers 1 --wait-budget 5
  python3 agent_selfcheck.py --attach REQUEST_ID --wait-budget 30
Use only the {case_count} selected training sources in public-cases/. Each run has one evaluator.
Optimize the operator family across shapes, rather than selecting a schedule by benchmark name or
one exact shape. Start with the primary anchor, then check promising changes on the other public
development shapes. Derive geometry/storage/capacity predicates and keep a safe fallback. Final
evaluation covers smaller, intermediate and larger shapes without retuning; exact tests stay private.
Selected cases may pass while the complete public suite is still incomplete; inspect case-level
results. Check all public development cases before freezing. Select one final variant using exact
correctness on all public cases and their per-case costs; report regressions as well as improvements.
The original passing seed is the baseline; do not repeat a baseline correctness sweep.
{early_note}
Native/model feedback is provisional, not physical timing. Expert supplied measured targets retain
their original source, machine and ROI scope. Instruction counts are not cycle measurements.

Keep EXPERIMENTS.md with source-pinned hypothesis, actual transformed/emitted change, numeric
result and same-domain model comparison. Keep RULES.md with family/shape/capacity applicability,
neighbors and counterexamples. Retain the best supported incumbent and leave unranked alternatives
in variants/. Finish RESULT.md with evidence and remaining gaps. Scratch belongs under tmp/.
{guidance}
"""

ON = """CCA/reference guidance is enabled. Use the ordinary parser endpoint:
  python3 agent_selfcheck.py --reference-analysis inventory --reference-limit 8
Choose the actual selected expert path from that inventory. Replace REFERENCE_PATH below with
that returned path; the source format is inferred from the selected file:
  python3 agent_selfcheck.py --reference-analysis lift --reference-path \\
    REFERENCE_PATH
  python3 agent_selfcheck.py --reference-analysis candidate --reference-path artifacts/views.json \\
    --reference-format artifact_views
Create views from your own completed source/command-buffer/emission files using
--write-reference-views; use your compiler manifest's ordinary commands to emit them if needed.
Choose a candidate in a supported comparison domain. A captured reference can be compared with
artifact_views. For a static ISA reference, select your own emitted ELF or disassembly using
candidate-format elf or disassembly. Use the ordinary compiler manifest's build commands; this
offline study does not publish a native candidate_build view. A static comparison does not prove
dynamic traffic, source equivalence or timing. Keep unsupported domain pairs as explicit gaps.
Replace CANDIDATE_PATH and CANDIDATE_FORMAT with that actual selection:
  python3 agent_selfcheck.py --reference-analysis compare --reference-path \\
    REFERENCE_PATH \\
    --candidate-path CANDIDATE_PATH --candidate-format CANDIDATE_FORMAT
Fresh analysis is converted into scoped construction/experiment feedback. No old tasks or outcomes
are inherited. Compare both operands, configurations, fences and whole-region cost; residency
alone earns no performance credit. Reanalyze the actual emitted result after a transformation.
Use the report's experiment tasks and named library routes to choose the next bounded attempt.
If a task lacks a goal-bound alternative, implement the general construction and consumed setting
before requesting A/B checks; a declaration alone is insufficient. Associate an ordinary check with
the generated request using --reference-request ID. Keep explicit unknowns and failed attempts.
"""
OFF = """CCA/reference guidance is disabled for this arm. Read the same raw expert references and
trusted metric targets directly. No reference map, CCA endpoint or generated reference tasks is
selected. The ordinary compiler library, numeric evaluator and provisional model remain available.
"""
LIBRARY = """Use .agents/skills/, the transformation catalog and registered library A/B routes. Add a reusable
pass or schedule consumer when needed; never select an optimization by benchmark or model name.
Expose real consumed alternatives in TUNING_KNOBS.json with explicit mechanism/goals and legal values
when binding a generated task. Declarations alone establish no emitted mechanism or performance.
"""
AGENT_ONLY = """This is the agent-only inherited-F6 dependency baseline. The same compiler_library bytes
remain read-only solely as the starting backend's inherited dependency. No library skill, catalog,
recipe, CCA endpoint or generated reference task is selected. Optimize your own backend and new
optimizer code using ordinary source inspection and the same numerical/model feedback. Do not use
new compiler-library pass/catalog/recipe assistance. Preserve the source contract and safe fallback.
"""
PRESETS = {
    "agent_only": (False, False, False),
    "cca_only": (True, False, False),
    "library_only": (False, True, False),
    "schedule_only": (False, False, True),
    "library_schedule": (False, True, True),
    "cca_library_schedule": (True, True, True),
}

STUDIES = {
    "cca": (
        "current",
        "current",
        "Improve the selected operator family using the supplied expert implementation. "
        "Learn a resource/shape rule, preserve numerical semantics, and keep a safe fallback.",
    ),
    "library": (
        "inherited",
        "library",
        "Extend convolution with rolling input reuse, capacity-bounded output groups, "
        "array-weight reuse and wider readout. Derive choices from geometry and resources.",
    ),
    "scheduling": (
        "inherited",
        "scheduling",
        "Reduce redundant K-step completion boundaries. Keep blocking, addresses, arithmetic "
        "and data-command issue order fixed; preserve required block/final completion and fallback.",
    ),
}
STUDY_HELP = {
    "library": "The additional composition APIs are plan_rolling_convolution_output_group, "
    "emit_rolling_convolution_output_group, materialize_grouped_rolling_window and "
    "plan_convolution_rhs_phases. Inspect their contracts; integrate a general consumer and policy. "
    "No oracle settings or completed backend are supplied.",
    "scheduling": "The additional checked consumer is schedule_dense_panel_epochs with DensePanelEpochLaw. "
    "Use the staged public ordering evidence and prove its program obligations. "
    "Compare unchanged completion with merged K streams; defer additional readout changes.",
}


def migrate_library(destination, inherited, selected):
    """Replace only verified pristine dependency files in this independent study copy."""
    inherited.verify()
    selected.verify()
    wanted = {path for path, _, _ in selected.members}
    changed = []
    for relative, expected, _ in inherited.members:
        path = destination / "compiler_library" / relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError("study dependency migration would overwrite an authored seed change")
    for relative, expected, _ in inherited.members:
        path = destination / "compiler_library" / relative
        if relative not in wanted:
            path.unlink()
            changed.append(relative)
    for relative, expected, _ in selected.members:
        path = destination / "compiler_library" / relative
        if path.exists() and digest(path) == expected:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.chmod(0o644)
        shutil.copyfile(selected.root / relative, path)
        path.chmod(0o644)
        changed.append(relative)
    return dict(
        inherited_manifest_sha256=inherited.manifest_sha256,
        selected_manifest_sha256=selected.manifest_sha256,
        changed_members=sorted(changed),
        qualification_issued=False,
    )


def prepare(
    base_config,
    inputs_path,
    output,
    *,
    framework,
    library,
    descriptor=None,
    suite=None,
    reference_map=None,
    library_contract=None,
    seconds=7200,
    replicas=2,
    agent_only_control=False,
    cases=None,
    cca=None,
    library_guidance=None,
    schedule_guidance=None,
    full_compiler=None,
    treatments=None,
    seed_continuation=None,
    study=None,
):
    base = json.loads(Path(base_config).read_bytes())
    inputs = json.loads(Path(inputs_path).read_bytes())
    output = Path(output).absolute()
    if output.exists():
        raise ValueError("preparation destination already exists; use a new study directory")
    framework = Path(framework).absolute()
    if not framework.is_dir() or framework != framework.resolve():
        raise ValueError("framework must be a canonical selected directory")
    library = ordinary(library)
    sys.path[:0] = [str(framework / part) for part in ("src", "packages/merlin-experiments/src")]
    full_record = full_compiler or base.get("full_compiler")
    full_owner = None
    if full_record:
        from merlin_experiments.phase1 import full_compiler_workspace

        full_owner = full_compiler_workspace.select(**full_record)
    continuation, seed_owner = continuation_input(seed_continuation)
    inherited_owner = None
    if study is not None:
        from merlin_experiments.phase1 import full_compiler_workspace

        if study not in STUDIES or continuation is None or treatments is not None or replicas != 1:
            raise ValueError("named study requires a seeded continuation and one matched replica")
        grants = base.get("study_tools", {})
        for key in {"inherited", *STUDIES[study][:2]}:
            if key not in grants:
                raise ValueError(f"missing reviewed study tool selection: {key}")
        inherited_owner = full_compiler_workspace.select(**grants["inherited"])
    seed = ordinary(continuation["from"]["path"] if continuation else base["source_seed"])
    cca_guidance = base.get("cca_guidance")
    if cca_guidance and digest(cca_guidance["path"]) != cca_guidance["sha256"]:
        raise ValueError("CCA guidance pin changed")
    cca_lifting = base.get("cca_lifting")
    if cca_lifting and digest(cca_lifting["path"]) != cca_lifting["sha256"]:
        raise ValueError("CCA capture selection pin changed")
    reference = ordinary(inputs["reference_root"])
    training = [row for row in inputs["cases"] if row["split"] == "train"]
    if cases is not None:
        if len(set(cases)) != len(cases) or not set(cases) <= {row["case"] for row in training}:
            raise ValueError("select unique existing training cases")
        training = [row for row in training if row["case"] in cases]
    if not training or len({row["case"] for row in training}) != len(training):
        raise ValueError("training roster must contain unique cases")
    for row in training:
        if digest(row["source"]["path"]) != row["source"]["sha256"]:
            raise ValueError("training source pin changed")
    if digest(inputs["reference_manifest"]["path"]) != inputs["reference_manifest"]["sha256"]:
        raise ValueError("reference bank pin changed")
    if reference_map is not None and digest(reference_map["path"]) != reference_map["sha256"]:
        raise ValueError("reference map pin changed")
    if seconds <= 0 or not 1 <= replicas <= 2:
        raise ValueError("wall budget must be positive and replicas must be one or two")
    selected = (
        ["control", "treatment"]
        if study
        else list(treatments)
        if treatments is not None
        else (["agent_only"] if agent_only_control else []) + ["library_schedule", "cca_library_schedule"]
    )
    if not selected or len(selected) != len(set(selected)) or (not study and not set(selected) <= set(PRESETS)):
        raise ValueError("select distinct existing treatment presets")
    shared_pool = bool(study) or agent_only_control or treatments is not None
    study_pool = base.get("cpus", [0, 1, 2, 3])
    if shared_pool and (not isinstance(study_pool, list) or len(study_pool) != 4
            or any(type(cpu) is not int or cpu < 0 for cpu in study_pool)
            or study_pool != sorted(set(study_pool))):
        raise ValueError("shared study pool requires four distinct sorted CPU IDs")
    quota = 400 // (len(selected) * replicas) if shared_pool else 100
    contract_pin = None
    if library_contract:
        library_contract = Path(library_contract).absolute()
        document = json.loads(library_contract.read_bytes())
        for member in document["members"]:
            relative = Path(member["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("library contract member escapes root")
            if digest(library / relative) != member["sha256"]:
                raise ValueError("selected library differs from reviewed contract")
        contract_pin = {"path": str(library_contract), "sha256": digest(library_contract)}
    runner = Path(__file__).with_name("run.py").resolve()
    shim_path = framework / "packages/merlin-experiments/src/merlin_experiments/phase1/tools/selfcheck.py"
    spec = importlib.util.spec_from_file_location("study_source_digest", shim_path)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    runs = []
    for replica in range(replicas):
        for tooling in selected:
            mode = (
                tooling
                if treatments is not None or study
                else {"agent_only": "agent_only", "library_schedule": "off", "cca_library_schedule": "on"}[tooling]
            )
            cpu = 2 * replica + (mode == "on")
            cpus = list(study_pool) if shared_pool else [cpu]
            cpu_selection = ",".join(map(str, cpus))
            guidance = "on" if mode == "on" else "off"
            # Independent gates can deactivate assistance without changing a preset's control role.
            preset_cca, preset_library, preset_schedule = (
                (study == "cca" and tooling == "treatment", True, True) if study else PRESETS[tooling]
            )
            enabled_cca = preset_cca and cca is not False
            enabled_library = preset_library and library_guidance is not False
            enabled_schedule = preset_schedule and schedule_guidance is not False
            sdk_readonly = False if study else not (enabled_library or enabled_schedule)
            if study:
                key = STUDIES[study][tooling == "treatment"]
                full_record = base["study_tools"][key]
                full_owner = full_compiler_workspace.select(**full_record)
            guidance = "on" if enabled_cca else "off"
            run = output / f"{mode}-r{replica}"
            private, workspace = run / "private", run / "workspace"
            for child in (
                private / "tmp",
                workspace / "home",
                workspace / "tmp",
                workspace / "qa",
                workspace / "artifacts",
                workspace / "variants",
                workspace / ".qa_channel",
            ):
                child.mkdir(parents=True)
            for name in ("submission", "baseline"):
                if continuation:
                    seed_owner.copy_source(seed, workspace / name, ())
                    if name == "submission":
                        from merlin_experiments.phase1 import continuation_handoff, joint_seed

                        evidence = seed_owner.build_evidence(
                            continuation, seed, workspace / name, private / "submission-seed-evidence"
                        )
                        joint_seed.stage_evidence(evidence["pin"], workspace)
                        private_evidence = joint_seed.stage_run_evidence(evidence["pin"], private)
                        continuation_handoff.write(workspace, continuation["run"], seed)
                        continuation_record = dict(
                            continuation,
                            evidence=evidence,
                            private_evidence=private_evidence,
                            certificate_reuse="none; prior verdicts are historical evidence",
                        )
                    else:
                        seed_owner.strip_stale(
                            workspace / name, private / "baseline-seed-notes", continuation["notes"]
                        )
                else:
                    copy_source(seed, workspace / name)
                if study:
                    migration = migrate_library(workspace / name, inherited_owner, full_owner)
                    write(private / (name + "-dependency-migration.json"), migration)
                destination = workspace / name / "compiler_library"
                if full_owner is not None:
                    staged = full_compiler_workspace.stage(full_owner, workspace / name)
                    write(private / (name + "-full-source-staging.json"), staged)
                for source in library.rglob("*"):
                    if study:
                        break  # All implementations must come from the explicit reviewed grant.
                    relative = source.relative_to(library)
                    if full_owner is not None and source.is_file():
                        module = full_compiler_workspace._module(relative.as_posix())
                        namespace = module or ".".join(relative.with_suffix("").parts)
                        if not full_owner.permits(namespace) or (
                            module is None and source.suffix not in full_compiler_workspace.SUPPORT_SUFFIXES
                        ):
                            continue
                    target = destination / relative
                    if source.is_file() and not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                if full_owner is not None:
                    full_compiler_workspace.candidate_sources(full_owner, workspace / name)
            # Reviewed exports are read-only; only the independent author's copy is editable.
            for path in [workspace / "submission", *(workspace / "submission").rglob("*")]:
                path.chmod(path.stat().st_mode | 0o200)
            link_bank(reference, workspace / "reference-kernels")
            (workspace / "public-cases").mkdir()
            for row in training:
                shutil.copy2(row["source"]["path"], workspace / "public-cases" / (row["case"] + ".mlir"))
            selected_descriptor = Path(descriptor or base["descriptor"]).absolute()
            selected_suite = Path(suite or base["suite"]).absolute()
            selected_contract = Path(base["environment"]["MERLIN_TARGET_CONTRACT"])
            shutil.copy2(selected_contract, workspace / "target_contract.yaml")
            shutil.copytree(Path(base["facts"]).parent, workspace / "rtl_facts")
            shutil.copy2(shim_path, workspace / "agent_selfcheck.py")
            source_sha = shim._submission_digest(workspace / "submission")
            if source_sha != shim._submission_digest(workspace / "baseline"):
                raise ValueError("baseline and initial submission differ")
            discovery = library / "shared-library.json"
            if study and study != "cca":
                discovery = Path(continuation["run"]["workspace"]) / "shared-library.json"
            if not sdk_readonly and discovery.is_file():
                shutil.copy2(discovery, workspace / "shared-library.json")
            skill_framework = Path(continuation["run"]["framework"]) if study and study != "cca" else framework
            skills_module = (
                skill_framework / "packages/merlin-experiments/src/merlin_experiments/phase1/skill_staging.py"
            )
            if not sdk_readonly and skills_module.is_file():
                spec = importlib.util.spec_from_file_location("study_skill_staging", skills_module)
                skills = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(skills)
                collected = skills.collect()
                selected_names = {
                    "feedback-files",
                    "fix-failure-cluster",
                    "inspect-artifacts",
                    "inspect-ir",
                    "inspect-kernel",
                    "inspect-model",
                    "quantization",
                    "read-command-buffer",
                    "cycle-model-and-gsim",
                    "performance-feedback-levels",
                    "roofline-and-utilization",
                }
                if enabled_library:
                    selected_names.update(("pass-library", "ab-library-pass", "compose-pass-pipeline"))
                if enabled_schedule:
                    selected_names.update(("scheduling-api", "explore-schedules", "shape-schedule-selection", "fusion"))
                if enabled_library and enabled_schedule:
                    selected_names.add("heavy-pass-recipes")
                collected = {name: files for name, files in collected.items() if name in selected_names}
                # Reuse the existing validated writer, with an explicit subset of its collected inputs.
                skills.collect = lambda: collected
                skills.stage(
                    workspace,
                    private,
                    {
                        "schema": skills.SCHEMA,
                        "enabled": True,
                        "sha256": skills.digest(collected),
                        "skills": sorted(collected),
                    },
                )
            env = environment(
                base["environment"],
                framework=framework,
                private=private,
                cpu=cpu_selection,
                guidance=guidance,
                reference_map=reference_map,
            )
            env["PYTHONPATH"] = ":".join(
                str(framework / part)
                for part in (
                    "src",
                    "packages/merlin-experiments/src",
                    "packages/merlin-mining/src",
                    "packages/merlin-timing/src",
                )
            )
            dependencies = base.get("read_only_dependencies", [])
            if base.get("parser_dependencies"):
                dependencies = [*dependencies, base["parser_dependencies"]]
            elif base["environment"].get("PYTHONPATH"):
                dependencies = [
                    *dependencies,
                    *[
                        p
                        for p in base["environment"]["PYTHONPATH"].split(":")
                        if p.endswith("reference-analysis-deps/python")
                    ],
                ]
            env["PYTHONPATH"] += "".join(":" + path for path in dependencies)
            env.update(MERLIN_TARGET_EXPERIMENT=str(selected_descriptor), MERLIN_RTL_FACTS=base["facts"])
            for filename, pin in base.get("public_metadata", {}).items():
                if filename not in {"PUBLIC_CAPABILITIES.json", "COMMAND_ORDERING_SELECTION.json"}:
                    raise ValueError("unknown study public metadata")
                if digest(pin["path"]) != pin["sha256"]:
                    raise ValueError("study public metadata pin changed")
                shutil.copyfile(pin["path"], workspace / filename)
                if filename == "COMMAND_ORDERING_SELECTION.json":
                    env["MERLIN_COMPLETION_ORDER_SELECTION"] = json.dumps(
                        dict(path=str(workspace / filename), sha256=pin["sha256"]), sort_keys=True
                    )
            if full_record:
                env["MERLIN_FULL_COMPILER_SELECTION"] = json.dumps(full_record, sort_keys=True)
            captured = stage_capture(cca_lifting, workspace) if enabled_cca and cca_lifting else None
            if captured:
                env["MERLIN_REFERENCE_CAPTURE_SELECTION"] = json.dumps(captured, sort_keys=True)
            write(private / "evaluator-environment.json", env)
            config = {
                **base,
                "source_seed": str(seed),
                "workspace": str(workspace),
                "private": str(private),
                "study": str(output),
                "framework": str(framework),
                "descriptor": str(selected_descriptor),
                "suite": str(selected_suite),
                "reference_guidance": guidance,
                "tooling": tooling,
                "reference_map": reference_map if enabled_cca else None,
                "cca_lifting": captured,
                "library_guidance": enabled_library,
                "schedule_guidance": enabled_schedule,
                "sdk_readonly": sdk_readonly,
                "cpus": cpus,
                "cpu_quota_percent": quota,
                "wall_s": seconds,
                "workers": 1,
                "initial_submission_sha256": source_sha,
                "inputs": str(Path(inputs_path).absolute()),
                "runner": {"path": str(runner), "sha256": digest(runner)},
                "early_slot": inputs.get("early_slot")
                if (inputs.get("early_slot") or {}).get("case") in {row["case"] for row in training}
                else None,
                "early_timeout_s": 180,
                "environment": env,
                "library": str(library),
                "library_contract": contract_pin,
                "full_compiler": full_record,
                "read_only_dependencies": dependencies,
                "training_cases": [row["case"] for row in training],
            }
            if study:
                config.update(
                    study_name=study,
                    study_role=tooling,
                    tool_migration=migration,
                    initial_baseline="arm-specific; automatic dependency gains are not agent gains",
                )
            if continuation:
                config["seed_continuation"] = dict(continuation_record, effective_source_sha256=source_sha)
            write(private / "config.json", config)
            task = TASK.format(
                seconds=seconds,
                case_count=len(training),
                early_note=(
                    "A bounded early check of the selected largest LM source is queued separately."
                    if config["early_slot"]
                    else ""
                ),
                resources=f"{quota}% of one CPU on the shared four-CPU pool" if shared_pool else "one CPU",
                tool_instructions=AGENT_ONLY
                if tooling == "agent_only"
                else (
                    "The inherited compiler_library is frozen read-only. Use the enabled CCA endpoint to guide "
                    "changes to your backend/new optimizer code. Library and scheduling assistance are disabled.\n"
                    if sdk_readonly
                    else LIBRARY
                    if enabled_library and enabled_schedule
                    else "Use only the explicitly staged assistance skills. "
                    f"Library guidance: {enabled_library}; schedule guidance: {enabled_schedule}. "
                    "Disabled assistance supplies no catalog or recipe guidance. "
                    "Inherited SDK dependencies remain available to the starting compiler.\n"
                ),
                guidance=(
                    "CCA and generated tasks are disabled. Raw references and metric targets remain readable; "
                    "the inherited compiler dependency is frozen, without library assistance.\n"
                )
                if tooling == "agent_only"
                else ON
                if enabled_cca
                else OFF,
            )
            if continuation:
                task = task.replace(
                    "No previous author conversation or experiment history has been supplied.",
                    "Prior seed rules, notes, rejected experiments and handoff are supplied below.",
                )
                task = task.replace(
                    "No old tasks or outcomes\nare inherited.",
                    "New CCA tasks start fresh; prior seed outcomes remain in shared evidence.",
                )
                task += "\n" + seed_owner.GUIDANCE.read_text()
                task += "\nCheck the new anchor and changed emissions; inherited results are not new certificates.\n"
            if enabled_cca and cca_guidance:
                shutil.copy2(cca_guidance["path"], workspace / "CCA_REFERENCE.md")
                task += "\nRead CCA_REFERENCE.md for pinned reference analysis; verify candidate effects and cycles.\n"
            if captured:
                task += (
                    "\nThe lift endpoint replays the selected expert's captured command stream from cca-evidence/. "
                    "Use the same raw reference path in lift to recover typed dataflow and CCA evidence, then "
                    "compare it with actual emitted candidate views. This is separate from syntax inspection. "
                    "Hardware completion, overlap and cycles still require their own evidence.\n"
                )
            if study:
                task += "\nStudy task: " + STUDIES[study][2] + "\n"
                if tooling == "treatment" and study in STUDY_HELP:
                    task += STUDY_HELP[study] + "\n"
                task += "Oracle implementations and exact test-shape references are evaluator-only.\n"
            config["task_sha256"] = hashlib.sha256(task.encode()).hexdigest()
            write(private / "config.json", config)
            (workspace / "TASK.md").write_text(task)
            for name in ("EXPERIMENTS.md", "RULES.md", "RESULT.md"):
                (workspace / name).write_text("")
            launch = [
                "taskset",
                "-c",
                cpu_selection,
                base["python"],
                "-B",
                str(runner),
                "launch",
                "--config",
                str(private / "config.json"),
            ]
            runs.append(
                {
                    "replica": replica,
                    "reference_guidance": guidance,
                    "tooling": tooling,
                    "cpu": cpu if not shared_pool else None,
                    "cpus": cpus,
                    "cpu_quota_percent": config["cpu_quota_percent"],
                    "launcher_obligation": f"systemd AllowedCPUs={cpu_selection} CPUQuota={quota}%" if shared_pool else None,
                    "config": str(private / "config.json"),
                    "config_sha256": digest(private / "config.json"),
                    "launch_argv": launch,
                }
            )
    result = {
        "schema": "private.kernel_abstractions.runs.v1",
        "inputs": {"path": str(Path(inputs_path).absolute()), "sha256": digest(inputs_path)},
        "seed": str(seed),
        "runner": {"path": str(runner), "sha256": digest(runner)},
        "reference_manifest": inputs["reference_manifest"],
        "runs": runs,
        "author_launched": False,
    }
    write(output / "runs.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-config", "inputs", "out", "framework", "library-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--library-contract", type=Path)
    parser.add_argument("--full-compiler-root", type=Path)
    parser.add_argument("--full-compiler-manifest", type=Path)
    parser.add_argument("--full-compiler-manifest-sha256")
    parser.add_argument("--reference-map", type=Path, help="existing map JSON; selected only for on")
    parser.add_argument("--wall-s", type=int, default=7200)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--agent-only-control", choices=("off", "on"), default="off")
    parser.add_argument("--treatments", help="comma-separated presets: " + ",".join(PRESETS))
    parser.add_argument("--cases", help="comma-separated training subset; private suite stays unchanged")
    parser.add_argument("--seed-continuation", type=Path, help="JSON with full seed from pin and original run paths")
    parser.add_argument("--study", choices=tuple(STUDIES), help="matched post-Fresh6 capability study")
    for flag in ("cca", "library-guidance", "schedule-guidance"):
        parser.add_argument("--" + flag, choices=("off", "on"))
    args = parser.parse_args()
    full_fields = (args.full_compiler_root, args.full_compiler_manifest, args.full_compiler_manifest_sha256)
    if any(full_fields) and not all(full_fields):
        parser.error("all three full compiler selection fields are required together")
    full = (
        dict(
            root=str(args.full_compiler_root),
            manifest_path=str(args.full_compiler_manifest),
            manifest_sha256=args.full_compiler_manifest_sha256,
        )
        if all(full_fields)
        else None
    )
    selection = (
        {"path": str(args.reference_map.absolute()), "sha256": digest(args.reference_map)}
        if args.reference_map
        else None
    )
    result = prepare(
        args.base_config,
        args.inputs,
        args.out,
        framework=args.framework,
        library=args.library_root,
        descriptor=args.descriptor,
        suite=args.suite,
        reference_map=selection,
        library_contract=args.library_contract,
        seconds=args.wall_s,
        replicas=args.replicas,
        agent_only_control=args.agent_only_control == "on",
        cases=args.cases.split(",") if args.cases else None,
        cca=None if args.cca is None else args.cca == "on",
        library_guidance=None if args.library_guidance is None else args.library_guidance == "on",
        schedule_guidance=None if args.schedule_guidance is None else args.schedule_guidance == "on",
        full_compiler=full,
        treatments=args.treatments.split(",") if args.treatments else None,
        seed_continuation=args.seed_continuation,
        study=args.study,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
