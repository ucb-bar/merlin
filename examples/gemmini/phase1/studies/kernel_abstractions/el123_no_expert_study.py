"""Render a fresh, reference-free EL123 wave; run only native preparation gates.

The source study is immutable provenance, never a compiler seed or provider
resume. This utility has no author-launch action and reserves no CPUs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import yaml

ARMS = ("el1", "el2", "el3")
PROFILES = {a: "xs-el-no-expert-v1/" + a for a in ARMS}
QUEUED_CPUS = {
    "el1": (*range(41, 48), *range(56, 61), *range(91, 94)),
    "el2": tuple(range(61, 76)),
    "el3": tuple(range(76, 91)),
}
PREPARATION_CPUS = (0, 1, 2, 3)
GATES = ("inputs", "seal", "ordering", "plan", "preflight")
RELATIVE_UTILITY = "examples/gemmini/phase1/studies/kernel_abstractions/el123_no_expert_study.py"
FORBIDDEN_TEXT = (
    "fresh8-reference-kernels-only", "REFERENCE_ANALYSIS_DEPS", "MATCHED_RAW_REFERENCES",
    "FRESH7_REFERENCE", "FRESH8_REFERENCE", "reference-kernels/", "public-references/",
    "REFERENCE_MAP.json", "cca-evidence", "reference-experiments/packet-shards",
)
GUIDANCE = """# Reference-free compiler construction study

Start with an empty target backend. Implement every target lowering, transformation
and scheduling policy yourself. Only the selected public tests, contracts, ISA facts,
ordinary runtime and profile-specific authoring components are supplied.
No expert kernel sources, binaries, disassemblies, example implementations or prior
learned kernel artifacts are supplied. CCA and automatic kernel learning are disabled.
Do not import existing target implementations or resume an earlier authored compiler.

Preserve source semantics, rounding, saturation, all live outputs and the complete
pointer boundary. Hardware-loop/FSM descriptor instructions with the public role
loop_descriptor remain prohibited by the common instruction policy.
Existing complete Spike timing feedback and diagnostics remain available to all
three conditions. Estimates are provisional; unknown checks remain unknown.
Use at most eight helpers sharing the selected fifteen-CPU authoring allowance.
Author for four hours, then freeze before the separate 900-second evaluation drain.
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def checked_pin(value: dict) -> Path:
    path = Path(value["path"])
    if not path.is_file() or sha(path) != value["sha256"]:
        raise ValueError("input pin mismatch: " + str(path))
    return path


def source_pin(revision: str, export_id: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or export_id != revision[:12]:
        raise ValueError("exact source revision and matching export ID are required")


def option(argv: list[str], name: str) -> str:
    at = [i for i, v in enumerate(argv) if v == name]
    if len(at) != 1 or at[0] + 1 == len(argv):
        raise ValueError("expected one argument for " + name)
    return argv[at[0] + 1]


def no_reference_environment(env: dict) -> dict:
    result = {}
    for key, value in env.items():
        if key.startswith("MERLIN_REFERENCE") or key.startswith("MERLIN_CCA"):
            continue
        if key == "PYTHONPATH":
            value = ":".join(p for p in value.split(":") if "REFERENCE" not in p.upper()
                             and "reference-analysis" not in p and "cca-evidence" not in p)
        result[key] = value
    return result


def expanded_path(spec: dict, key: str) -> Path:
    """Expand native declared path variables without a shell or moving-HEAD lookup."""
    variables = spec["variables"]
    def expand(name, active=()):
        if name in active or name not in variables:
            raise ValueError("unresolved or cyclic native path variable: " + name)
        value = variables[name]
        if not isinstance(value, str):
            raise ValueError("native path variable must be text: " + name)
        return re.sub(r"\$\{([A-Z_][A-Z_0-9]*)\}", lambda m: expand(m[1], (*active, name)), value)
    value = expand(key)
    if "${" in value:
        raise ValueError("unexpanded native path: " + key)
    return Path(value)


def clean_spec(original: dict, *, arm: str, output: Path, wave: str, export_id: str) -> dict:
    spec = copy.deepcopy(original)
    run_id = wave + "-" + arm
    spec["id"] = run_id
    run = spec["run"]
    run.update(run_id=run_id, unit="merlinphase1xs" + hashlib.sha256(run_id.encode()).hexdigest()[:20],
               cpus=",".join(map(str, QUEUED_CPUS[arm])),
               preparation_cpus=",".join(map(str, PREPARATION_CPUS)),
               max_wall_s=15300, final_model_s=900, workers=15, sim_max_jobs=15,
               author_helper_effort="medium", treatment=arm)
    variables = spec["variables"]
    for key in list(variables):
        if "REFERENCE" in key.upper() or key.startswith("CCA_"):
            del variables[key]
    variables.update(STUDY_ROOT=str(output), STUDY_ARM=arm,
                     FRAMEWORK="${REPO}/out/build/run-frameworks/" + export_id)
    inputs = spec["inputs"]
    inputs["public_references"] = None
    inputs.pop("forbidden_reference_prefixes", None)
    inputs["environment"] = no_reference_environment(inputs.get("environment", {}))
    plan = spec["plan"]
    plan.update(public_reference_manifest=None, reference_experiments=None,
                reference_guidance="off", reference_map=None, reference_experiment_tasks="off",
                author_profile=PROFILES[arm], guidance=str(output / "guidance.md"))
    plan["environment"] = no_reference_environment(plan["environment"])
    plan["policy"]["base"] = pin(output / "base-policy.json")
    plan["record"]["origin"] = "Fresh empty target backend; no expert kernels or inherited authored state"
    plan["record"]["limits"].update(author_helpers=8, shared_compute_cpus=15)
    plan["record"]["expert_kernel_material"] = "absent"
    return spec


def normalized(spec: dict) -> dict:
    """Only these intentional treatment/resource/identity differences may vary."""
    result = copy.deepcopy(spec)
    result.pop("id")
    for name in ("run_id", "unit", "cpus", "treatment"):
        result["run"].pop(name)
    result["variables"].pop("STUDY_ARM")
    result["plan"].pop("author_profile")
    result["plan"]["record"].pop("compiler_access", None)
    return result


def audit(study: dict) -> dict:
    source_pin(study["source_revision"], study["export_id"])
    if study["state"] != "specifications" or study.get("launch_authorized") is not False:
        raise ValueError("only an unqualified, unlaunched specification wave can be audited")
    checked_pin(study["provenance"]["source_study"])
    source_documents = {}
    for source in study["provenance"]["source_specs"]:
        original = yaml.safe_load(checked_pin(source).read_text())
        source_documents[original["plan"]["author_profile"].rsplit("/", 1)[-1]] = original
    documents, rows = [], []
    for row in study["arms"]:
        arm = row["arm"]
        spec_path = checked_pin(row["spec"])
        spec = yaml.safe_load(spec_path.read_text())
        run, plan, inputs = spec["run"], spec["plan"], spec["inputs"]
        if arm not in ARMS or plan["author_profile"] != PROFILES[arm]:
            raise ValueError("wrong clean author profile")
        if inputs.get("public_references") is not None or plan.get("public_reference_manifest") is not None:
            raise ValueError("expert material was selected")
        if any(plan.get(k) is not None for k in ("reference_map", "reference_experiments")):
            raise ValueError("reference map or prior task selection survives")
        if plan["reference_guidance"] != "off" or plan["reference_experiment_tasks"] != "off":
            raise ValueError("CCA/reference guidance must be off")
        encoded = json.dumps(spec, sort_keys=True)
        if any(s in encoded for s in FORBIDDEN_TEXT):
            raise ValueError("an obsolete reference path or dependency survives")
        checked_pin(plan["policy"]["base"])
        guide = Path(plan["guidance"])
        if guide.read_text() != GUIDANCE:
            raise ValueError("common guidance changed")
        if run["cpus"] != ",".join(map(str, QUEUED_CPUS[arm])) or run["preparation_cpus"] != "0,1,2,3":
            raise ValueError("queued CPU group differs")
        if run["max_wall_s"] != 15300 or run["final_model_s"] != 900 or any(run[k] != 15 for k in ("workers", "sim_max_jobs")):
            raise ValueError("common time/worker budget changed")
        argv = plan["argv"]
        if option(argv, "--model") != study["common_identity"]["model"] or option(argv, "--effort") != study["common_identity"]["effort"]:
            raise ValueError("provider treatment differs")
        if option(argv, "--author-max-subagents") != "8" or option(argv, "--codex-max-subagents") != "8":
            raise ValueError("helper allowance differs")
        if "--fresh-backend" not in argv or any(x in argv for x in ("--seed-submission", "--resume", "--qualify-submission")):
            raise ValueError("clean runs cannot inherit an authored compiler")
        for field in ("MERLIN_SPIKE", "MERLIN_JOINT_PERFORMANCE_SELECTION", "MERLIN_BACKGROUND_PROGRESS"):
            if plan["environment"].get(field) != study["common_identity"]["performance_environment"][field]:
                raise ValueError("existing full Spike feedback changed: " + field)
        if run["run_id"] != study["wave_id"] + "-" + arm:
            raise ValueError("run identity differs")
        repo = expanded_path(spec, "REPO")
        paths = [repo / "out/runs/gemmini/phase1" / run["run_id"],
                 repo / "out/build/agent-workspaces/phase1/gemmini/merlin_assisted" / run["run_id"],
                 Path(study["root"]) / "arms" / arm / "private" / ("codex-" + run["run_id"])]
        if any(p.exists() for p in paths):
            raise ValueError("fresh author paths already exist")
        expected = clean_spec(source_documents[arm], arm=arm, output=Path(study["root"]), wave=study["wave_id"], export_id=study["export_id"])
        if spec != expected:
            raise ValueError("unexplained difference from the pinned source specification")
        documents.append(normalized(spec))
        rows.append({"arm": arm, "run_id": run["run_id"], "profile": plan["author_profile"],
                     "queued_cpus": list(QUEUED_CPUS[arm]), "supplied_module_count": {"el1": 0, "el2": 12, "el3": 34}[arm],
                     "expert_material_selected": False, "fresh_author_paths_absent": True, "spec": pin(spec_path)})
    if [r["arm"] for r in rows] != list(ARMS) or any(d != documents[0] for d in documents[1:]):
        raise ValueError("unexplained difference between clean EL123 specs")
    return {"schema": "operator.el123_no_expert_static_audit.v1", "all_passed": True,
            "source_revision": study["source_revision"], "export_id": study["export_id"],
            "author_jobs_launched": 0, "cpu_reservations_created": 0,
            "native_qualification": "pending", "launch_authorized": False,
            "common_normalized_spec_sha256": hashlib.sha256(json.dumps(documents[0], sort_keys=True).encode()).hexdigest(),
            "common_policy": pin(Path(study["root"]) / "base-policy.json"),
            "common_guidance": pin(Path(study["root"]) / "guidance.md"), "arms": rows}


def render(source_study: Path, output: Path, *, revision: str, export_id: str) -> dict:
    source_pin(revision, export_id)
    source_study, output = source_study.resolve(), output.resolve()
    old = json.loads(source_study.read_text())
    task = Path(old["operator_task"]).resolve()
    if not output.is_relative_to(task / ".tmp") or output.exists() or output == source_study.parent:
        raise ValueError("output must be a fresh directory under the operator task .tmp")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,90}", output.name):
        raise ValueError("wave directory requires a bounded lowercase run ID")
    originals, pins = {}, []
    for row in old["arms"]:
        arm = row["arm"]
        p = checked_pin(row["spec"])
        doc = yaml.safe_load(p.read_text())
        if arm not in ARMS or doc["plan"]["author_profile"] != "xs-el-v1/" + arm:
            raise ValueError("source is not the original EL123 profile study")
        originals[arm] = doc
        pins.append(pin(p))
    if set(originals) != set(ARMS):
        raise ValueError("exactly three source arms required")
    policy_paths = [checked_pin(originals[a]["plan"]["policy"]["base"]) for a in ARMS]
    if len({sha(p) for p in policy_paths}) != 1:
        raise ValueError("source policies differ")
    policy = json.loads(policy_paths[0].read_text())
    if policy["duration_s"] != 15300 or policy["drain_s"] != 900 or policy["study_protocol"]["author_s"] != 14400:
        raise ValueError("source policy is not the fixed four-hour protocol")
    output.mkdir()
    (output / "base-policy.json").write_bytes(policy_paths[0].read_bytes())
    (output / "guidance.md").write_text(GUIDANCE)
    arms = []
    for arm in ARMS:
        spec = clean_spec(originals[arm], arm=arm, output=output, wave=output.name, export_id=export_id)
        path = output / (arm + ".yaml")
        path.write_text(yaml.safe_dump(spec, sort_keys=False))
        arms.append({"arm": arm, "run_id": spec["run"]["run_id"], "unit": spec["run"]["unit"],
                     "cpus": list(QUEUED_CPUS[arm]), "spec": pin(path)})
    base = originals["el1"]["plan"]
    study = {"schema": "operator.el123_no_expert_study.v1", "state": "specifications",
             "wave_id": output.name, "root": str(output), "operator_task": str(task),
             "source_revision": revision, "export_id": export_id,
             "author_s": 14400, "drain_s": 900, "max_ours": 3, "max_machine": 5,
             "cpu_status": "queued future groups; unreserved; recheck live allocations before launch",
             "preparation_cpus": list(PREPARATION_CPUS), "launch_authorized": False,
             "native_qualification": "pending", "seed": None, "seed_tools": None, "seed_library_selection": None,
             "common_identity": {"model": option(base["argv"], "--model"), "effort": option(base["argv"], "--effort"),
                                 "performance_environment": {k: base["environment"][k] for k in
                                     ("MERLIN_SPIKE", "MERLIN_JOINT_PERFORMANCE_SELECTION", "MERLIN_BACKGROUND_PROGRESS")},
                                 "expert_kernel_material": "absent", "live_cca": False},
             "provenance": {"source_study": pin(source_study), "source_specs": pins,
                            "source_campaign_classification": "reference-assisted; not a clean baseline; no authored state reused"},
             "arms": arms}
    (output / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    receipt = audit(study)
    (output / "static-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Queued reference-free EL1/EL2/EL3 wave\n\n"
        "Four hours authoring plus900 seconds drain;15 CPUs and8 helpers per arm. "
        "Same original public tests, contracts, model and complete Spike feedback. "
        "No expert kernels, reference maps, CCA or inherited authored state.\n\n"
        "Status: specifications and static checks only. CPUs are future queued groups, not reservations. "
        "Recheck live allocations before any launch. No author launch is authorized.\n\n"
        "The framework is pinned to " + revision + ". Native gates must use that exact export.\n\n"
        "Run the frozen utility's `gate --study STUDY_DIRECTORY --action ACTION` for each of "
        "`inputs`, `seal`, `ordering`, `plan`, `preflight`, in that order. Each executes sequentially "
        "over the three arms on CPUs0–3. Preflight is render-only: the native script selects "
        "the queued15-CPU author affinity, so execution must wait until those groups are available "
        "and separately reviewed. The utility never accepts --execute or claims native preflight "
        "qualification from a rendered script. These actions never launch an author. "
        "Then require new native author/helper/candidate permission canaries, positive public-input reads, "
        "initial-backend and prompt audits, corrected uniform evaluation checks, and new launch review. "
        "Old readiness, threads, homes, dispatch and authorization receipts are not reusable.\n")
    return receipt


def gate(study_root: Path, action: str) -> dict:
    """A non-author gate only; verify frozen source and every spec before execution."""
    if action not in GATES:
        raise ValueError("only non-author native preparation gates are available")
    study_root = study_root.resolve()
    study = json.loads((study_root / "study.json").read_text())
    audit(study)
    first_spec = yaml.safe_load(checked_pin(study["arms"][0]["spec"]).read_text())
    repo = expanded_path(first_spec, "REPO")
    framework = expanded_path(first_spec, "FRAMEWORK")
    if repo != Path(study["operator_task"]) / "merlin":
        raise ValueError("native repo differs from the operator task")
    marker = json.loads((framework / ".merlin-source-revision.json").read_text())
    if marker.get("revision") != study["source_revision"]:
        raise ValueError("framework export differs from the exact selected source")
    logs = study_root / "preparation"
    logs.mkdir(exist_ok=True)
    if action != GATES[0]:
        previous = json.loads((logs / (GATES[GATES.index(action) - 1] + ".json")).read_text())
        if not previous["all_passed"] or previous["source_revision"] != study["source_revision"] or previous.get("study_sha256") != sha(study_root / "study.json"):
            raise ValueError("preceding native gate has not passed")
    results = []
    for row in study["arms"]:
        spec = yaml.safe_load(checked_pin(row["spec"]).read_text())
        command = ["taskset", "-c", "0,1,2,3", str(expanded_path(spec, "PYTHON")), "-B",
                   str(framework / "examples/gemmini/phase1/launch/fresh_campaign.py"), action,
                   "--spec", row["spec"]["path"], "--export-id", study["export_id"]]
        if action == "seal":
            command += ["--reviewed-by", "Codex operator", "--review-note",
                        "Reference-free EL123 preparation; no expert kernels, inherited state or author launch; "
                        "static input audit: " + str(study_root / "static-audit.json")]
        # Native preflight uses the queued author affinity. Render its script
        # now, but never execute while those CPU groups remain unreserved.
        env = no_reference_environment(dict(os.environ, TMPDIR=str(study_root), PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=str(framework / "src") + ":" + str(framework / "packages/merlin-experiments/src")))
        log = logs / (action + "-" + row["arm"] + "-" + str(time.time_ns()) + ".log")
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        results.append({"arm": row["arm"], "returncode": result.returncode, "log": str(log)})
        if result.returncode:
            break
    receipt = {"schema": "operator.el123_no_expert_preparation_gate.v1", "action": action,
               "source_revision": study["source_revision"], "study_sha256": sha(study_root / "study.json"),
               "all_passed": action != "preflight" and len(results) == 3 and all(r["returncode"] == 0 for r in results),
               "rendered_only": action == "preflight",
               "render_succeeded": action == "preflight" and len(results) == 3 and all(r["returncode"] == 0 for r in results),
               "qualification_status": "pending; queued author CPUs require fresh availability review" if action == "preflight" else "native preparation gate",
               "author_jobs_launched": 0, "results": results}
    (logs / (action + ".json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("render")
    r.add_argument("--source-study", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--source-revision", required=True)
    r.add_argument("--export-id", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--study", type=Path, required=True)
    g = sub.add_parser("gate")
    g.add_argument("--study", type=Path, required=True)
    g.add_argument("--action", choices=GATES, required=True)
    args = p.parse_args()
    if args.command == "render":
        result = render(args.source_study, args.output, revision=args.source_revision, export_id=args.export_id)
    elif args.command == "audit":
        result = audit(json.loads((args.study / "study.json").read_text()))
    else:
        result = gate(args.study, args.action)
    print(json.dumps(result, indent=2))
    if not result["all_passed"] and not result.get("render_succeeded"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
