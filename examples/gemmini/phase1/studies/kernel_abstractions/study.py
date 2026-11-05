"""Six-arm xs study preparation and bounded fleet dispatch.

Specs and plans are reviewable before execution. Only ``run --confirm`` starts
authors; it uses the ordinary campaign's prepared reservation command. No
private transfer result enters an author environment.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SCHEMA = "merlin.study.xs_ablation.v1"
AUTHOR_S, DRAIN_S = 14400, 900
MAX_OURS, MAX_MACHINE, EXTERNAL_SLOTS = 3, 5, 2
ARMS = ("fresh6-control", "fresh6-treatment", "el4", "el1", "el2", "el3")
DEFAULT_CPUS = (tuple((*range(38, 48), *range(56, 61))), tuple(range(61, 76)), tuple(range(76, 91)))
RESERVED_CPUS = tuple((*range(4, 38), *range(48, 56)))
PROFILE_PREFIX = "xs-el-v1/"
PROTOCOL = dict(version=1, author_s=AUTHOR_S, evaluation="deferred", scope="selected_capsules",
                snapshot_seconds=[0, 1800, 3600, 7200, 14400])


def _module(name):
    path = HERE.parents[1] / "launch" / (name + ".py")
    spec = importlib.util.spec_from_file_location("xs_study_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pin(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("study inputs must be direct canonical files")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def reopen(value):
    if pin(value["path"]) != value:
        raise ValueError("study input changed: " + value["path"])
    return json.loads(Path(value["path"]).read_text())


def _option(argv, flag):
    positions = [i for i, value in enumerate(argv) if value == flag]
    if len(positions) > 1:
        raise ValueError("duplicate study option " + flag)
    return argv[positions[0] + 1] if positions else None


def _remove(argv, flags):
    result, cursor = [], 0
    while cursor < len(argv):
        if argv[cursor] in flags:
            cursor += 2
        else:
            result.append(argv[cursor])
            cursor += 1
    return result


def _set(argv, flag, value):
    result = _remove(argv, {flag})
    return [*result, flag, str(value)]


def _cpu_groups(groups):
    groups = [list(group) for group in groups]
    if len(groups) != 3 or any(len(g) != 15 or any(type(c) is not int or c < 0 for c in g) for g in groups):
        raise ValueError("study requires three pools of exactly fifteen CPUs")
    flat = [c for group in groups for c in group]
    if len(set(flat)) != 45:
        raise ValueError("study CPU pools must be disjoint")
    if set(flat) & set(RESERVED_CPUS):
        raise ValueError("study CPU pools overlap the operator's reserved CPUs")
    return groups


def specs(base_spec, output, *, panel, seed=None, seed_sha256=None, seed_tools=None,
          seed_library_selection=None, arms=ARMS, external_units=(), cpu_groups=DEFAULT_CPUS):
    """Render the selected ordinary specs; perform no build or author launch."""
    arms = tuple(arms)
    if len(arms) not in {3, 6} or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError("select three or six distinct declared study arms")
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("select a new canonical study directory")
    groups = _cpu_groups(cpu_groups)
    campaign = _module("fresh_campaign")
    base = campaign.load_spec(Path(base_spec).resolve())
    base_context = campaign.Context(base_spec, run_id="xs-study-prepare", cpus=",".join(map(str, groups[0])))
    revision = campaign.git(Path(base_context.lookup("REPO")), "rev-parse", "HEAD")
    panel_pin = pin(panel)
    panel_data = reopen(panel_pin)
    if (panel_data.get("schema") != "merlin.study.transfer_panel.v1"
            or len(panel_data.get("cases", [])) != 34):
        raise ValueError("study requires the complete preregistered transfer panel")
    seeded = any(arm.startswith("fresh6") for arm in arms)
    inherited = treatment = library_pin = None
    if seeded:
        if any(value is None for value in (seed, seed_sha256, seed_tools, seed_library_selection)):
            raise ValueError("seeded arms require the original seed and explicit inherited tool selections")
        tools = json.loads(Path(seed_tools).read_text())
        tools = tools.get("study_tools", tools)
        if not {"inherited", "scheduling"} <= set(tools):
            raise ValueError("seed tools require explicit inherited and scheduling selections")
        from merlin_experiments.phase1 import full_compiler_workspace as full
        inherited, treatment = [full.select(**{k: tools[name][k] for k in
            ("root", "manifest_path", "manifest_sha256")}) for name in ("inherited", "scheduling")]
        original = {p: (sha, mod) for p, sha, mod in inherited.members}
        changed = {p: (sha, mod) for p, sha, mod in treatment.members if original.get(p) != (sha, mod)}
        if set(changed) != {"merlin/compile/scheduling/dense_panel_epochs.py"}:
            raise ValueError("the initial seed contrast may add only the dense completion consumer")
        if set(original) - {p for p, _, _ in treatment.members}:
            raise ValueError("seed treatment removed inherited dependencies")
        seed = Path(seed).resolve(strict=True)
        _module("fresh_seed").verify_source(seed, seed_sha256)
        inherited.verify()
        for relative, digest, _ in inherited.members:
            path = seed / "compiler_library" / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("seed does not retain the pristine inherited library")
        library_pin = pin(seed_library_selection)
    output.mkdir(parents=True, mode=0o700)
    guidance = output / "guidance"
    guidance.mkdir()
    rows = []
    for index, arm in enumerate(arms):
        spec = copy.deepcopy(base)
        spec.pop("extends", None)
        spec.pop("seed", None)
        spec["id"] = "xs-ablation-v1-" + arm
        variables = spec.setdefault("variables", {})
        variables.update(STUDY_ROOT=str(output), STUDY_ARM=arm,
                         INPUTS="${FRAMEWORK}/out/artifacts/xs-study/${RUN_ID}/inputs",
                         ARTIFACTS="${STUDY_ROOT}/arms/${STUDY_ARM}/artifacts",
                         RELEASE="${INPUTS}/source-release",
                         RUN_STATE="${STUDY_ROOT}/arms/${STUDY_ARM}/state",
                         PRIVATE_ROOT="${STUDY_ROOT}/arms/${STUDY_ARM}/private",
                         OPERATOR_TMP="${STUDY_ROOT}/arms/${STUDY_ARM}/tmp")
        cpus = ",".join(map(str, groups[index % 3]))
        run_id = output.name + "-" + arm
        unit = "merlinphase1xs" + hashlib.sha256(run_id.encode()).hexdigest()[:20]
        spec["run"].update(run_id=run_id, unit=unit, cpus=cpus, preparation_cpus=cpus,
                           max_wall_s=AUTHOR_S + DRAIN_S, final_model_s=DRAIN_S,
                           workers=15, sim_max_jobs=15, treatment=arm)
        plan = spec["plan"]
        spec["inputs"]["author_profile_common_inputs"] = "canonical_common"
        spec["inputs"]["environment"]["MERLIN_OUT_ROOT"] = "${FRAMEWORK}/out"
        defaults = plan["policy"].setdefault("defaults", {})
        # Defaults cannot override a base policy's drain; create an exact derived
        # policy instead and pin it below.
        base_policy = Path(base_context.x(plan["policy"]["base"]["path"]))
        if pin(base_policy)["sha256"] != plan["policy"]["base"]["sha256"]:
            raise ValueError("common base policy differs from its reviewed pin")
        policy = json.loads(base_policy.read_text())
        policy.update(duration_s=AUTHOR_S + DRAIN_S, drain_s=DRAIN_S,
                      final_model_s=DRAIN_S, study_protocol=PROTOCOL)
        policy["prohibited_instruction_roles"] = sorted(set(policy.get("prohibited_instruction_roles", []))
                                                       | {"loop_descriptor"})
        policy.get("gsim_correctness", {}).update(continuous_promotion=False)
        policy_path = output / (arm + "-base-policy.json")
        _write(policy_path, policy)
        plan["policy"]["base"] = pin(policy_path)
        defaults.pop("drain_s", None)
        plan["reference_guidance"] = "on" if arm == "el4" else "off"
        plan["reference_experiment_tasks"] = "off"
        if arm != "el4":
            plan["reference_map"] = None
        plan["argv"] = _remove(plan["argv"], {
            "--private-full-model-spec", "--instruction-selection", "--instruction-selection-sha256",
            "--native-result-format-selection", "--native-result-format-selection-sha256",
        })
        removed = {"--private-full-model-spec", "--instruction-selection", "--instruction-selection-sha256",
                   "--native-result-format-selection", "--native-result-format-selection-sha256"}
        for key in ("input_flags", "required_flags"):
            plan[key] = [flag for flag in plan.get(key, []) if flag not in removed]
        for flag in ("--codex-max-subagents", "--author-max-subagents"):
            plan["argv"] = _set(plan["argv"], flag, 8)
        guide = guidance / (arm + ".md")
        text = (
            "Develop one general compiler for the common public xs capsule suite. "
            "Use the supplied environment; preserve every source output, numerical contract and safe fallback. "
            "You have four hours, fifteen shared CPU cores and at most eight helper agents. "
            "Private whole-model campaigns are outside this study. Final transfer and physical evaluation "
            "happens only after all authors freeze. Keep source-pinned EXPERIMENTS.md, RULES.md and RESULT.md.\n"
        )
        if arm.startswith("fresh6"):
            text += (
                "Starting from Fresh6, reduce unnecessary full drains between K steps of its checked "
                "two-window dense stream. Preserve blocking, operand windows, data-command order and "
                "separate output stores. Derive legality from the common public ordering evidence. "
                "Unknown premises retain drains. Focus on the public 16x2048x2048 raw-i32 projection anchor; "
                "retain correctness across every public xs case. An equivalent independently authored rule is allowed.\n"
            )
            if arm == "fresh6-treatment":
                text += ("The additional API is DensePanelEpochLaw/schedule_dense_panel_epochs. "
                         "Inspect its contracts and integrate a reusable consumer; no completed backend is supplied.\n")
            selection = inherited if arm == "fresh6-control" else treatment
            plan["author_profile"] = ("xs-f6-v1/inherited" if arm == "fresh6-control"
                                      else "xs-f6-v1/dense-epochs")
            spec["inputs"]["author_profile_source_selection"] = {
                "root": str(selection.root), "manifest_path": str(selection.manifest_path),
                "manifest_sha256": selection.manifest_sha256,
            }
            plan["library_selection"] = library_pin["path"]
            plan["full_compiler_manifest"] = str(selection.manifest_path)
            for flag, value in (("--full-compiler-root", selection.root),
                                ("--full-compiler-manifest", selection.manifest_path),
                                ("--full-compiler-manifest-sha256", selection.manifest_sha256)):
                plan["argv"] = _set(plan["argv"], flag, value)
            spec["seed"] = dict(
                **{"from": {"path": str(seed), "sha256": seed_sha256}},
                source="${STUDY_ROOT}/arms/${STUDY_ARM}/seed", review_id="xs-ablation-v1-" + arm,
                strip=[], notes=[], carry=dict(certificates=False, results=False, rules=False, handoff=False),
                guidance=str(guide),
            )
        else:
            plan["author_profile"] = PROFILE_PREFIX + arm
            plan["argv"] = _remove(plan["argv"], {
                "--compiler-library-root", "--compiler-library-contract", "--compiler-library-contract-sha256",
                "--required-shared-pass",
            })
            for key in ("input_flags", "required_flags"):
                plan[key] = [flag for flag in plan.get(key, []) if flag not in {
                    "--compiler-library-root", "--compiler-library-contract", "--compiler-library-contract-sha256",
                    "--required-shared-pass"}]
        guide.write_text(text)
        plan["guidance"] = str(guide)
        path = output / (arm + ".yaml")
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        rows.append(dict(arm=arm, wave=index // 3, cpus=groups[index % 3], unit=unit, spec=pin(path)))
    record = dict(schema=SCHEMA, state="specifications", source_revision=revision, export_id=revision[:12],
                  operator_task=base_context.lookup("TASK"),
                  author_s=AUTHOR_S, drain_s=DRAIN_S,
                  max_ours=MAX_OURS, max_machine=MAX_MACHINE, reserved_external_slots=EXTERNAL_SLOTS,
                  reserved_external_cpus=list(RESERVED_CPUS),
                  external_units=list(external_units), panel=panel_pin, arms=rows,
                  seed=dict(path=str(seed), sha256=seed_sha256) if seeded else None,
                  seed_tools=pin(seed_tools) if seeded else None, seed_library_selection=library_pin,
                  raw_reference_review="required before execution; hardware timing targets are operator-only")
    _write(output / "study.json", record)
    return record


def _arguments_identity(plan):
    argv = plan["argv"]
    return {key: _option(argv, key) for key in (
        "--model", "--effort", "--driver", "--codex-binary", "--compiler-execution", "--schedule",
        "--qa-timeout", "--sim-per-capsule-timeout", "--grade-interval",
    )}


def _reference_identity(plan):
    reopen(plan["reference_manifest"])
    return plan["reference_manifest"]["sha256"]


def prepare(study_path, plans, review_path):
    """Pin the selected ready native plans and their qualification receipt."""
    study_path = Path(study_path).resolve()
    record = json.loads(study_path.read_text())
    if record.get("schema") != SCHEMA or record.get("state") != "specifications":
        raise ValueError("study is not awaiting prepared plans")
    if len(plans) not in {3, 6}:
        raise ValueError("select the first three or all six prepared plans in declared arm order")
    review_pin = pin(review_path)
    review = reopen(review_pin)
    required = ("common_xs_sources", "common_raw_references", "reference_content_review",
                "author_sandbox_canaries", "helper_sandbox_canaries", "candidate_sandbox_canaries",
                "initial_backends_qualified", "transfer_contracts_qualified")
    if any(row["arm"].startswith("fresh6") for row in record["arms"][:len(plans)]):
        required += ("seed_time_zero_qualified",)
    if (review.get("schema") != "merlin.study.launch_review.v1"
            or any(review.get(key) is not True for key in required)
            or review.get("panel_sha256") != record["panel"]["sha256"]):
        raise ValueError("study needs complete pinned qualification and visibility evidence")
    identities = []
    for row, path in zip(record["arms"][:len(plans)], plans, strict=True):
        selection = pin(path)
        plan = reopen(selection)
        if plan.get("source_revision") != record["source_revision"]:
            raise ValueError("all plans must use the study's pinned framework export")
        if plan.get("execution_started") or plan.get("treatment") != row["arm"]:
            raise ValueError("selected plan is running or belongs to a different arm")
        if plan["cpu_reservation"]["cpus"] != row["cpus"]:
            raise ValueError("plan CPU reservation differs from its declared study pool")
        argv = plan["argv"]
        policy_path = _option(argv, "--joint-campaign")
        policy = json.loads(Path(policy_path).read_text())
        if (policy.get("study_protocol") != PROTOCOL or policy.get("drain_s") != DRAIN_S
                or policy.get("duration_s") != AUTHOR_S + DRAIN_S):
            raise ValueError("plan does not give four hours of authoring and deferred evaluation")
        if any(_option(argv, key) != "8" for key in ("--codex-max-subagents", "--author-max-subagents")):
            raise ValueError("both helper ceilings must equal eight")
        if _option(argv, "--private-full-model-spec"):
            raise ValueError("private whole-model work is outside this study")
        expected = "on" if row["arm"] == "el4" else "off"
        if plan.get("reference_guidance") != expected:
            raise ValueError("plan reference assistance differs from the declared arm")
        identities.append(dict(source_revision=plan["source_revision"],
                               timing=plan["timing_selection"]["sha256"],
                               author=_arguments_identity(plan), references=_reference_identity(plan)))
        row.update(plan=selection, policy=pin(policy_path))
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("study plans differ in common source, timing, author settings or reference content")
    record.update(state="prepared", prepared_arms=[row["arm"] for row in record["arms"][:len(plans)]],
                  common_identity=identities[0], launch_review=review_pin)
    _write(study_path, record)
    return record


def active_units():
    """All native service identities plus separately counted author campaigns."""
    return _module("campaign_capacity").active_campaigns()


def admissible(active, ours, *, pending=0):
    """Reserve two peer seats; operator evaluation services consume no author seat."""
    counts = getattr(active, "author_counts", {unit: 1 for unit in active})
    own_count = sum(count for unit, count in counts.items() if unit in ours) + pending
    external_count = sum(count for unit, count in counts.items() if unit not in ours)
    return own_count < MAX_OURS and own_count + max(EXTERNAL_SLOTS, external_count) < MAX_MACHINE


def pool_available(row, rows, runs, active):
    for other in rows:
        if other["arm"] == row["arm"]:
            continue
        status = runs.get(other["arm"], {}).get("status")
        occupied = other["unit"] in active or status in {"claimed", "started"}
        if occupied and set(row["cpus"]) & set(other["cpus"]):
            return False
    return True


def _reservation_command(plan):
    command = list(plan["reserved_argv"])
    boundary = command.index("--")
    prefix = command[:boundary]
    if "--host-isolation" in prefix:
        prefix = _set(prefix, "--host-isolation", "preserve")
    else:
        prefix += ["--host-isolation", "preserve"]
    wrapper = Path(prefix[1])
    if wrapper.name != "resource_reservation.py" or "--host-isolation" not in wrapper.read_text():
        raise ValueError("selected frozen wrapper lacks peer-safe host mask preservation")
    return [*prefix, *command[boundary:]]


def run(study_path, *, confirm=False, poll_s=5):
    if not confirm:
        raise ValueError("run requires explicit --confirm")
    study_path = Path(study_path).resolve()
    root = study_path.parent
    study = json.loads(study_path.read_text())
    task = Path(study["operator_task"])
    if task.resolve() != task or not task.is_dir():
        raise ValueError("study operator task is unavailable or indirect")
    admission = task / ".tmp" / "xs-study-dispatch"
    admission.mkdir(parents=True, exist_ok=True, mode=0o700)
    # One coordinator for every study prepared in this task, rather than one
    # independent counter per output directory. Two peer seats remain reserved.
    with ExitStack() as stack:
        host_lock = stack.enter_context((admission / "coordinator.lock").open("a+"))
        fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock = stack.enter_context((root / "dispatch.lock").open("a+"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = json.loads(study_path.read_text())
        if study.get("schema") != SCHEMA or study.get("state") not in {"prepared", "running"}:
            raise ValueError("study needs qualified prepared plans")
        reopen(study["launch_review"])
        reopen(study["panel"])
        units = {row["unit"] for row in study["arms"]}
        selected_rows = [row for row in study["arms"] if row["arm"] in study["prepared_arms"]]
        state_path = root / "dispatch.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(
            schema="merlin.study.dispatch.v1", study=pin(study_path), runs={},
            max_ours=MAX_OURS, max_machine=MAX_MACHINE, reserved_external_slots=EXTERNAL_SLOTS,
        )
        if state.get("study") != pin(study_path):
            raise ValueError("study changed after dispatch")
        children = {}
        while True:
            active = active_units()
            for arm, child in list(children.items()):
                rc = child.poll()
                if rc is not None:
                    state["runs"][arm].update(status="finished" if rc == 0 else "failed",
                                              returncode=rc, finished_at_epoch=time.time())
                    del children[arm]
            pending = sum(1 for arm in children if state["runs"][arm]["unit"] not in active)
            for row in selected_rows:
                previous = state["runs"].get(row["arm"])
                if previous:
                    if previous["status"] in {"claimed", "started"} and row["arm"] not in children and row["unit"] not in active:
                        previous.update(status="interrupted", reason="dispatcher restarted; no automatic retry")
                    continue
                if not admissible(active, units, pending=pending):
                    break
                if not pool_available(row, study["arms"], state["runs"], active):
                    continue
                # Reopen all launch inputs immediately before every start.
                plan = reopen(row["plan"])
                reopen(row["policy"])
                if set(row["cpus"]) & set(study["reserved_external_cpus"]):
                    raise ValueError("launch CPU pool overlaps an external reservation")
                command = _reservation_command(plan)
                log = root / (row["arm"] + "-dispatch.log")
                state["runs"][row["arm"]] = dict(status="claimed", unit=row["unit"],
                    claimed_at_epoch=time.time(), command=command, log=str(log), plan=row["plan"], policy=row["policy"])
                _write(state_path, state)  # Durable intent precedes the process, including crash recovery.
                try:
                    with log.open("ab", buffering=0) as stream:
                        child = subprocess.Popen(command, cwd=plan["cwd"], stdout=stream,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as error:
                    state["runs"][row["arm"]].update(status="failed", reason=str(error))
                    _write(state_path, state)
                    raise
                children[row["arm"]] = child
                state["runs"][row["arm"]].update(status="started", pid=child.pid, started_at_epoch=time.time())
                pending += 1
                _write(state_path, state)
                active = active_units()  # Recheck peers; never rely on the initial wave count.
                pending = sum(1 for arm in children if state["runs"][arm]["unit"] not in active)
            state["active_units"] = sorted(active)
            state["author_campaign_counts"] = getattr(active, "author_counts", {unit: 1 for unit in active})
            state["capacity_observations"] = getattr(active, "observations", {})
            state["observed_at_epoch"] = time.time()
            _write(state_path, state)
            if len(state["runs"]) == len(selected_rows) and not children and not active.intersection(units):
                return 0 if all(row["status"] == "finished" for row in state["runs"].values()) else 1
            time.sleep(poll_s)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("specs")
    for name in ("base-spec", "output", "panel"):
        p.add_argument("--" + name, required=True)
    for name in ("seed", "seed-sha256", "seed-tools", "seed-library-selection"):
        p.add_argument("--" + name)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    p.add_argument("--external-unit", action="append", default=[])
    p = commands.add_parser("prepare")
    p.add_argument("--study", required=True)
    p.add_argument("--plans", nargs="+", required=True)
    p.add_argument("--review", required=True)
    p = commands.add_parser("run")
    p.add_argument("--study", required=True)
    p.add_argument("--confirm", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "specs":
        result = specs(args.base_spec, args.output, panel=args.panel, seed=args.seed,
                       seed_sha256=args.seed_sha256, seed_tools=args.seed_tools,
                       seed_library_selection=args.seed_library_selection, arms=args.arms,
                       external_units=args.external_unit)
    elif args.action == "prepare":
        result = prepare(args.study, args.plans, args.review)
    else:
        return run(args.study, confirm=args.confirm)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
