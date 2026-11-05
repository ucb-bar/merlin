"""Render/audit the versioned raw-RTL EL123 study; non-author preparation only.

Original r2 inputs and judge policy are provenance, never an authored compiler seed.
The existing frontend remains harness-owned. EL1 receives neutral mathematical
programs; EL2/3 receive the shared prelowered interface. No author launch exists.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

_spec = importlib.util.spec_from_file_location(
    "_el123_no_expert_operator_helpers", Path(__file__).with_name("el123_no_expert_study.py")
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

# Reuse only pure rendering/path/environment helpers, without changing their globals.
sha, pin, checked_pin = _base.sha, _base.pin, _base.checked_pin
source_pin, option = _base.source_pin, _base.option
expanded_path, no_reference_environment = _base.expanded_path, _base.no_reference_environment
ARMS, QUEUED_CPUS, PREPARATION_CPUS, GATES = _base.ARMS, _base.QUEUED_CPUS, _base.PREPARATION_CPUS, _base.GATES
FORBIDDEN_TEXT = _base.FORBIDDEN_TEXT
PROFILES = {arm: "xs-el-raw-rtl-v2/" + arm for arm in ARMS}
SOURCE_STUDY_NAME = "xs-el123-20261028-r2"
BASE_SOURCE_REVISION = "adff3ab95caeb3adc738627843ee6671f51d8dc8"
SOURCE_COUNTS = {"el1": 0, "el2": 15, "el3": 34}
DISCLOSURES = {
    arm: dict(
        raw_rtl=True,
        rtl_facts=arm == "el3",
        software_spec=arm == "el3",
        derived_target_contract=arm == "el3",
        derived_test_metadata=arm == "el3",
        input_format="neutral_math_v1" if arm == "el1" else "shared_interface_v1",
    )
    for arm in ARMS
}
RELATIVE_UTILITY = "examples/gemmini/phase1/studies/kernel_abstractions/el123_raw_rtl_study.py"
GUIDANCE = """# Raw-RTL compiler construction study

Start with an empty target backend. Only the selected numerical tests, raw RTL,
ordinary runtime, common input/output protocol and profile-specific inputs/support
are supplied. Author every target lowering, transformation and scheduling policy.
EL1 receives neutral mathematical programs and general runtime/input-output
requirements; it receives no Merlin implementation, interface dialect, extracted
hardware facts or software specification. EL2 receives the existing prelowered
shared interface programs, exact interface definitions/readers and their parser
support; its target backend/dialect must be C++. Its existing upstream frontend is
harness-owned and supplied as prelowered inputs, not an author-callable lowering
service. EL2 receives no extracted facts, software specification, target dialect
builder, scaffold generator, scheduling, pass-library or extraction tooling.
EL3 receives the shared interface and its declared light authoring support plus
selected declarations. Consult the exact profile inventory for supplied APIs.

No expert kernel sources, binaries, disassemblies, example target implementations
or prior learned kernel artifacts are supplied. CCA and automatic kernel learning
are disabled. Do not import an existing target implementation or resume an earlier
compiler. Historical r2 supplies original mathematical tests and judge provenance,
not authored state. EL1's neutral input representation versus EL2/3's shared
interface is an intentional information-set difference and must be disclosed.

Preserve source mathematics, precision, rounding, saturation, all live outputs
and the complete pointer boundary. Hardware-loop/FSM descriptor instructions with
the common policy role loop_descriptor remain prohibited. The same Spike model,
correctness observations and provisional total-cycle feedback remain available.
EL1/EL2 withhold static resource/model summaries and fact-derived optional probes;
the selected mathematical tests and private judge are unchanged. Estimates are
provisional; unknown checks remain unknown. Use at most eight
helpers sharing the selected fifteen CPUs. Author for four hours, then freeze
before the separate 900-second evaluation drain. The model budget, suite, private
judge, numerical policy and evaluation gates are common to all three profiles.
"""


def clean_spec(original: dict, *, arm: str, output: Path, wave: str, export_id: str) -> dict:
    spec = _base.clean_spec(original, arm=arm, output=output, wave=wave, export_id=export_id)
    # The original generic library contains the source-plan validator. Its new
    # typed input adapter must pass dependency review before profile reduction;
    # this grants no extra source to the 0/15/34-file author selections.
    spec["inputs"]["reviewed_modules"].append("merlin.targetgen.oot_starterkit.source_inventory")
    plan = spec["plan"]
    plan["public_owned_host_inputs"] = "${ARTIFACTS}/public-owned-host-inputs.json"
    plan["argv"] += [
        "--public-owned-host-inputs", plan["public_owned_host_inputs"],
        "--public-owned-host-inputs-sha256", "${PUBLIC_OWNED_HOST_INPUTS_SHA256}",
    ]
    # The cohort operator already holds these CPU exclusions across the retired
    # wave and its replacements; an independent peer owns a nested exclusion.
    plan["reservation"]["host_isolation"] = "preserve"
    plan["author_profile"] = PROFILES[arm]
    plan["record"]["origin"] = "Fresh empty target backend; versioned raw-RTL information boundary; no authored seed"
    plan["record"]["author_input_disclosure"] = copy.deepcopy(DISCLOSURES[arm])
    plan["record"]["frontend_provenance"] = (
        "existing harness-owned frontend; original mathematics with profile-specific author input projection"
    )
    return spec


def normalized(spec: dict) -> dict:
    result = _base.normalized(spec)
    result["plan"]["record"].pop("author_input_disclosure")
    return result


def _check_original(study_path: Path, old: dict) -> None:
    if (
        study_path.name != "study.json"
        or study_path.parent.name != SOURCE_STUDY_NAME
        or old.get("source_revision") != BASE_SOURCE_REVISION
        or old.get("export_id") != BASE_SOURCE_REVISION[:12]
    ):
        raise ValueError("source must be the pinned historical r2 original-math study")


def audit(study: dict) -> dict:
    source_pin(study["source_revision"], study["export_id"])
    if study.get("schema") != "operator.el123_raw_rtl_study.v2":
        raise ValueError("wrong raw-RTL study schema")
    if study["state"] != "specifications" or study.get("launch_authorized") is not False:
        raise ValueError("only an unqualified, unlaunched specification wave can be audited")
    if (
        study.get("author_s") != 14400
        or study.get("drain_s") != 900
        or study.get("max_ours") != 3
        or study.get("max_machine") != 5
        or study.get("source_module_counts") != SOURCE_COUNTS
        or any(study.get(key) is not None for key in ("seed", "seed_tools", "seed_library_selection"))
    ):
        raise ValueError("raw-RTL study budget, source inventory or empty-start policy changed")
    source_path = checked_pin(study["provenance"]["source_study"])
    source_study = json.loads(source_path.read_text())
    _check_original(source_path, source_study)
    source_pins = [pin(checked_pin(row["spec"])) for row in source_study["arms"]]
    if study["provenance"]["source_specs"] != source_pins:
        raise ValueError("source specifications differ from the pinned historical r2 study")
    source_documents = {}
    for source in study["provenance"]["source_specs"]:
        original = yaml.safe_load(checked_pin(source).read_text())
        profile = original["plan"]["author_profile"]
        arm = profile.rsplit("/", 1)[-1]
        if arm not in ARMS or profile != "xs-el-v1/" + arm or arm in source_documents:
            raise ValueError("source is not exactly the original EL123 profiles")
        source_documents[arm] = original
    if len(source_pins) != 3 or set(source_documents) != set(ARMS):
        raise ValueError("exactly three historical source specifications required")
    documents, rows = [], []
    for row in study["arms"]:
        arm = row["arm"]
        spec_path = checked_pin(row["spec"])
        spec = yaml.safe_load(spec_path.read_text())
        run, plan, inputs = spec["run"], spec["plan"], spec["inputs"]
        if arm not in ARMS or plan["author_profile"] != PROFILES[arm]:
            raise ValueError("wrong raw-RTL author profile")
        if plan["record"].get("author_input_disclosure") != DISCLOSURES[arm]:
            raise ValueError("author input disclosure differs from the versioned profile")
        if inputs.get("public_references") is not None or plan.get("public_reference_manifest") is not None:
            raise ValueError("expert material was selected")
        if any(plan.get(k) is not None for k in ("reference_map", "reference_experiments")):
            raise ValueError("reference map or prior task selection survives")
        if plan["reference_guidance"] != "off" or plan["reference_experiment_tasks"] != "off":
            raise ValueError("CCA/reference guidance must be off")
        encoded = json.dumps(spec, sort_keys=True)
        if any(s in encoded for s in FORBIDDEN_TEXT):
            raise ValueError("an obsolete reference path or dependency survives")
        policy_path = checked_pin(plan["policy"]["base"])
        if sha(policy_path) != sha(checked_pin(source_documents[arm]["plan"]["policy"]["base"])):
            raise ValueError("original mathematical/evaluation policy changed")
        guide = Path(plan["guidance"])
        if guide.read_text() != GUIDANCE:
            raise ValueError("common guidance changed")
        if run["cpus"] != ",".join(map(str, QUEUED_CPUS[arm])) or run["preparation_cpus"] != "0,1,2,3":
            raise ValueError("queued CPU group differs")
        if (
            run["max_wall_s"] != 15300
            or run["final_model_s"] != 900
            or any(run[k] != 15 for k in ("workers", "sim_max_jobs"))
        ):
            raise ValueError("common time/worker budget changed")
        argv = plan["argv"]
        if (
            option(argv, "--model") != study["common_identity"]["model"]
            or option(argv, "--effort") != study["common_identity"]["effort"]
        ):
            raise ValueError("provider treatment differs")
        if option(argv, "--author-max-subagents") != "8" or option(argv, "--codex-max-subagents") != "8":
            raise ValueError("helper allowance differs")
        if "--fresh-backend" not in argv or any(
            x in argv for x in ("--seed-submission", "--resume", "--qualify-submission")
        ):
            raise ValueError("clean runs cannot inherit an authored compiler")
        for field in ("MERLIN_SPIKE", "MERLIN_JOINT_PERFORMANCE_SELECTION", "MERLIN_BACKGROUND_PROGRESS"):
            if plan["environment"].get(field) != study["common_identity"]["performance_environment"][field]:
                raise ValueError("existing full Spike feedback changed: " + field)
        if run["run_id"] != study["wave_id"] + "-" + arm:
            raise ValueError("run identity differs")
        repo = expanded_path(spec, "REPO")
        paths = [
            repo / "out/runs/gemmini/phase1" / run["run_id"],
            repo / "out/build/agent-workspaces/phase1/gemmini/merlin_assisted" / run["run_id"],
            Path(study["root"]) / "arms" / arm / "private" / ("codex-" + run["run_id"]),
        ]
        if any(p.exists() for p in paths):
            raise ValueError("fresh author paths already exist")
        expected = clean_spec(
            source_documents[arm],
            arm=arm,
            output=Path(study["root"]),
            wave=study["wave_id"],
            export_id=study["export_id"],
        )
        if spec != expected:
            raise ValueError("unexplained difference from the pinned source specification")
        documents.append(normalized(spec))
        rows.append(
            {
                "arm": arm,
                "run_id": run["run_id"],
                "profile": plan["author_profile"],
                "queued_cpus": list(QUEUED_CPUS[arm]),
                "supplied_module_count": SOURCE_COUNTS[arm],
                "author_input_disclosure": copy.deepcopy(DISCLOSURES[arm]),
                "expert_material_selected": False,
                "fresh_author_paths_absent": True,
                "spec": pin(spec_path),
            }
        )
    if [r["arm"] for r in rows] != list(ARMS) or any(d != documents[0] for d in documents[1:]):
        raise ValueError("unexplained difference between clean EL123 specs")
    return {
        "schema": "operator.el123_raw_rtl_static_audit.v2",
        "all_passed": True,
        "source_revision": study["source_revision"],
        "export_id": study["export_id"],
        "author_jobs_launched": 0,
        "cpu_reservations_created": 0,
        "native_qualification": "pending",
        "launch_authorized": False,
        "source_inventory_status": "expected counts only; exact member manifests require native preparation",
        "common_normalized_spec_sha256": hashlib.sha256(json.dumps(documents[0], sort_keys=True).encode()).hexdigest(),
        "common_policy": pin(Path(study["root"]) / "base-policy.json"),
        "common_guidance": pin(Path(study["root"]) / "guidance.md"),
        "arms": rows,
    }


def render(source_study: Path, output: Path, *, revision: str, export_id: str) -> dict:
    source_pin(revision, export_id)
    source_study, output = source_study.resolve(), output.resolve()
    old = json.loads(source_study.read_text())
    _check_original(source_study, old)
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
        if arm not in ARMS or arm in originals or doc["plan"]["author_profile"] != "xs-el-v1/" + arm:
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
        arms.append(
            {
                "arm": arm,
                "run_id": spec["run"]["run_id"],
                "unit": spec["run"]["unit"],
                "cpus": list(QUEUED_CPUS[arm]),
                "spec": pin(path),
            }
        )
    base = originals["el1"]["plan"]
    study = {
        "schema": "operator.el123_raw_rtl_study.v2",
        "state": "specifications",
        "wave_id": output.name,
        "root": str(output),
        "operator_task": str(task),
        "source_revision": revision,
        "export_id": export_id,
        "author_s": 14400,
        "drain_s": 900,
        "max_ours": 3,
        "max_machine": 5,
        "cpu_status": "queued future groups; unreserved; recheck live allocations before launch",
        "preparation_cpus": list(PREPARATION_CPUS),
        "launch_authorized": False,
        "native_qualification": "pending",
        "seed": None,
        "seed_tools": None,
        "seed_library_selection": None,
        "source_module_counts": dict(SOURCE_COUNTS),
        "common_identity": {
            "model": option(base["argv"], "--model"),
            "effort": option(base["argv"], "--effort"),
            "performance_environment": {
                k: base["environment"][k]
                for k in ("MERLIN_SPIKE", "MERLIN_JOINT_PERFORMANCE_SELECTION", "MERLIN_BACKGROUND_PROGRESS")
            },
            "expert_kernel_material": "absent",
            "live_cca": False,
        },
        "provenance": {
            "source_study": pin(source_study),
            "source_specs": pins,
            "source_campaign_classification": "historical reference-assisted r2; no authored state reused",
            "historical_source_revision": BASE_SOURCE_REVISION,
            "math_suite_and_judge": "same original mathematical cases and private evaluation policy",
            "intentional_input_difference": "EL1 neutral mathematics; EL2/3 shared prelowered interface",
        },
        "arms": arms,
    }
    (output / "study.json").write_text(json.dumps(study, indent=2) + "\n")
    receipt = audit(study)
    (output / "static-audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Queued raw-RTL EL1/EL2/EL3 study\n\n"
        "Four hours authoring plus 900 seconds drain, 15 CPUs and at most 8 helpers per arm. "
        "Same original math suite, private judge, numerical policy, model budget and complete Spike feedback. "
        "EL1 receives neutral mathematics without the Merlin interface; EL2/3 retain the shared prelowered interface. "
        "The existing frontend remains harness-owned. Source inventories are 0/15/34 files. "
        "No expert kernels, CCA or inherited authored compiler.\n\n"
        "Status: rendered specifications and static checks only. No CPUs reserved or author launch authorized. "
        "Frozen source: " + revision + ". Use only this exact export.\n\n"
        "Use the frozen utility's gate action for inputs, seal, ordering, plan and preflight in order. "
        "Preflight renders a script and does not execute it. Native author/helper/candidate namespace probes, "
        "positive input reads, source-language and initial/resume prompt audits, corrected uniform evaluation "
        "checks, fresh CPU-availability review and launch review remain necessary. Historical qualification "
        "receipts are not reusable. The utility has no author-launch action.\n"
    )
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
        if (
            not previous["all_passed"]
            or previous["source_revision"] != study["source_revision"]
            or previous.get("study_sha256") != sha(study_root / "study.json")
        ):
            raise ValueError("preceding native gate has not passed")
    results = []
    for row in study["arms"]:
        spec = yaml.safe_load(checked_pin(row["spec"]).read_text())
        command = [
            "taskset",
            "-c",
            "0,1,2,3",
            str(expanded_path(spec, "PYTHON")),
            "-B",
            str(framework / "examples/gemmini/phase1/launch/fresh_campaign.py"),
            action,
            "--spec",
            row["spec"]["path"],
            "--export-id",
            study["export_id"],
        ]
        if action == "seal":
            command += [
                "--reviewed-by",
                "Codex operator",
                "--review-note",
                "Raw-RTL EL123 preparation; no expert kernels, inherited state or author launch; "
                "static input audit: " + str(study_root / "static-audit.json"),
            ]
        # Native preflight uses the queued author affinity. Render its script
        # now, but never execute while those CPU groups remain unreserved.
        env = no_reference_environment(
            dict(
                os.environ,
                TMPDIR=str(study_root),
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=str(framework / "src") + ":" + str(framework / "packages/merlin-experiments/src"),
            )
        )
        log = logs / (action + "-" + row["arm"] + "-" + str(time.time_ns()) + ".log")
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT)
        results.append({"arm": row["arm"], "returncode": result.returncode, "log": str(log)})
        if result.returncode:
            break
    receipt = {
        "schema": "operator.el123_raw_rtl_preparation_gate.v2",
        "action": action,
        "source_revision": study["source_revision"],
        "study_sha256": sha(study_root / "study.json"),
        "all_passed": action != "preflight" and len(results) == 3 and all(r["returncode"] == 0 for r in results),
        "rendered_only": action == "preflight",
        "render_succeeded": action == "preflight" and len(results) == 3 and all(r["returncode"] == 0 for r in results),
        "qualification_status": "pending; queued author CPUs require fresh availability review"
        if action == "preflight"
        else "native preparation gate",
        "author_jobs_launched": 0,
        "results": results,
    }
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
