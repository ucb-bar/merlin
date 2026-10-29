"""Frozen joint-campaign coverage and dispatch through the installed ELF wrapper.

This is an operator post-freeze adapter, not another author runner. Preparation
and dispatch do not claim hardware/source equivalence or numerical success.
Native capsule and complete-chain collectors retain those separate obligations.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from merlin.common.tree_hash import hash_tree
from . import firesim

SCHEMA = "merlin.joint_final_firesim.selection.v1"
DEPLOYED_SCHEMA = "merlin.joint_final_firesim.selection.v2"
DEPLOYED_FIELDS = ("configuration_class", "platform_configuration", "agfi", "target_clock_hz")
HARDWARE_FIELDS = ("configuration_class", "source_firrtl_sha256", "rtl_facts_sha256", "agfi", "target_clock_hz")


def prepare(selection_path):
    """Pin the complete explicit admitted roster; absent jobs remain blockers."""
    selection_path = Path(selection_path).absolute()
    spec = firesim.load(selection_path)
    if spec.get("schema") not in (SCHEMA, DEPLOYED_SCHEMA):
        raise ValueError("select the explicit joint final FireSim campaign schema")
    freeze_pin = spec["freeze_record"]
    summary = firesim.load(firesim.reopen(freeze_pin))
    if summary.get("schema") != "merlin.joint_campaign_completion.v1":
        raise ValueError("final FireSim requires the joint compiler freeze")
    compiler = Path(summary["frozen_compiler"])
    digest = hash_tree(compiler)["sha256"]
    if digest != summary["compiler_sha256"] or summary["completion_reason"] != "freeze_drain":
        raise ValueError("final FireSim requires unchanged final compiler bytes after development")
    hardware = spec["hardware"]
    deployed = spec["schema"] == DEPLOYED_SCHEMA
    fields = DEPLOYED_FIELDS if deployed else HARDWARE_FIELDS
    if set(hardware) != set(fields) or any(not hardware[key] for key in fields):
        raise ValueError("select exact unmodified configuration, source/facts, AGFI and target clock")
    if type(hardware["target_clock_hz"]) is not int or hardware["target_clock_hz"] <= 0:
        raise ValueError("target clock must be explicit")
    for key in (() if deployed else ("source_firrtl_sha256", "rtl_facts_sha256")):
        if not firesim.is_sha256(hardware[key]):
            raise ValueError("hardware needs exact source/facts digests")
    development_profile = spec.get("development_profile")
    if deployed:
        # This pin describes the development gSIM/Spike facts and execution
        # profile. It does not assert that those bytes produced the deployed AGFI.
        profile = firesim.load(firesim.reopen(development_profile))
        if not isinstance(profile, dict) or not profile:
            raise ValueError("development execution profile must be explicit")
    launcher = firesim.reopen(spec["launcher"])
    if launcher.name != "fpga-run-elf" or not os.access(launcher, os.X_OK):
        raise ValueError("use the installed direct-ELF wrapper, not the legacy queue route")
    roster = spec["roster"]
    if set(roster) != {"capsules", "models"} or any(not roster[kind] for kind in roster):
        raise ValueError("final validation must include all admitted capsules and full models")
    expected, chain_capsules = set(), set()
    for kind, rows in roster.items():
        for row in rows:
            identity = (kind, row["id"])
            if not row["id"] or identity in expected:
                raise ValueError("final roster has an absent or repeated identity")
            source = firesim.reopen(row["source"])
            if kind == "capsules":
                metadata = firesim.load(source)
                if metadata.get("name") != row["id"] or not metadata.get("kind"):
                    raise ValueError("capsule roster must bind its actual native capsule metadata")
                if metadata["kind"] == "model":
                    chain_capsules.add(identity)
            expected.add(identity)
    jobs, covered = [], {role: set() for role in ("candidate", "baseline")}
    names = set()
    for row in spec.get("jobs", []):
        if row["role"] not in covered or not firesim._token(row["id"]) or row["id"] in names:
            raise ValueError("job needs a unique safe name and candidate/baseline role")
        names.add(row["id"])
        members = {(kind, name) for kind, ids in row["covers"].items() for name in ids}
        if not members or not members <= expected or members & covered[row["role"]]:
            raise ValueError("job has missing, foreign or duplicated coverage")
        covered[row["role"]].update(members)
        elf = firesim.reopen(row["elf"])
        if elf.read_bytes()[:4] != b"\x7fELF":
            raise ValueError("final job boot asset is not an ELF")
        firesim.reopen(row["build_receipt"])
        if row["role"] == "candidate" and row.get("compiler_sha256") != digest:
            raise ValueError("final job belongs to another compiler")
        # Full-model jobs reuse the original complete-chain admission. A stage
        # image or a hand-written complete=true flag cannot replace it.
        if any(kind == "models" for kind, _ in members) or members & chain_capsules:
            request = firesim.load(firesim.reopen(row["native_model_request"]))
            firesim.verify_request(request)
            if request["status"] != "prepared":
                raise ValueError("native complete-model request remains blocked")
            admitted = {m["id"] for m in request["models"] if any(
                asset["role"] == "boot_elf" and asset["sha256"] == row["elf"]["sha256"] for asset in m["assets"])}
            expected_models = {name for kind, name in members if kind == "models"}
            mapping = row.get("capsule_model_ids", {})
            if set(mapping) != {name for _, name in members & chain_capsules}:
                raise ValueError("model-kind capsule requires its own explicit complete-chain image mapping")
            expected_models.update(mapping.values())
            if expected_models != admitted:
                raise ValueError("model job ELF differs from its native complete-chain request")
            if row["role"] == "candidate" and request["compiler"]["hash_tree_sha256"] != digest:
                raise ValueError("complete-chain request belongs to another compiler")
            if request["deployment"]["agfi"] != hardware["agfi"]:
                raise ValueError("native model request selects another AGFI")
        jobs.append({**row, "launch_argv": [str(launcher), str(elf), row["id"]]})
    blockers = [f"{role}: missing {kind}/{name}" for role, present in covered.items()
                for kind, name in sorted(expected - present)]
    provenance = {"status": "unknown", "reason": "deployed source equivalence has not been established"}
    correspondence = spec.get("hardware_correspondence")
    if correspondence is None:
        blockers.append("deployed FPGA build/source/configuration correspondence has not been bound")
    else:
        evidence = firesim.load(firesim.reopen(correspondence))
        if evidence.get("selected_hardware") != hardware or not evidence.get("build_artifacts"):
            raise ValueError("deployed hardware correspondence does not bind the exact selected configuration")
        if deployed:
            if evidence.get("schema") != "merlin.firesim_deployment_correspondence.v1" or evidence.get("deployment_matches") is not True:
                raise ValueError("explicit reviewed deployment correspondence is required")
            runtime = evidence.get("active_fased_runtime")
            if runtime is None:
                blockers.append("active FASED runtime configuration has not been captured for selected deployment")
            else:
                if runtime.get("selected_hardware") != hardware or not runtime.get("effective_parameters"):
                    raise ValueError("active FASED runtime must bind selected deployment and effective parameters")
                firesim.reopen(runtime["configuration"])
            provenance = evidence.get("development_source_equivalence", {})
            if provenance.get("status") not in ("unknown", "verified") or not provenance.get("reason"):
                raise ValueError("record development/deployed source equivalence and its evidence limits")
            if provenance["status"] == "verified":
                firesim.reopen(provenance["proof"])
        for artifact in evidence["build_artifacts"]:
            firesim.reopen(artifact)
    return {"schema": "merlin.joint_final_firesim.manifest.v1", "selection": firesim.pin(selection_path),
            "freeze_record": freeze_pin, "compiler_sha256": digest, "hardware": hardware,
            "development_profile": development_profile,
            "development_source_equivalence": provenance,
            "clock_scope": "target simulated clock; FPGA implementation clock is separate",
            "development_timing_transfer_qualified": False,
            "roster": roster, "jobs": jobs, "blockers": blockers,
            "status": "blocked" if blockers else "prepared", "paid_jobs_launched": 0,
            "final_validation_complete": False, "scientific_qualification": False,
            "scope": "Explicit frozen coverage and transport selection; native all-output, actual boot image, "
                     "timing-scope and hardware correspondence checks remain required for every result."}


def dispatch(manifest_path, *, job_id, output_dir):
    """Start only one exact wrapper job; never kill its teardown on a local timer.

    The wrapper enforces machine-wide serialization and owns cloud lifecycle.
    Persisted launch identity prevents this API from silently resubmitting a job.
    Recovery uses the wrapper's existing --resume-command route, not a new ELF.
    """
    manifest = firesim.load(Path(manifest_path))
    if prepare(firesim.reopen(manifest["selection"])) != manifest or manifest["status"] != "prepared":
        raise ValueError("final campaign is changed or has unresolved prerequisites")
    rows = [row for row in manifest["jobs"] if row["id"] == job_id]
    if len(rows) != 1:
        raise ValueError("select one prepared final job")
    claims = Path(manifest_path).absolute().with_suffix(".jobs")
    claims.mkdir(exist_ok=True)
    with (claims / (job_id + ".json")).open("x") as claim:
        json.dump({"job_id": job_id, "output_dir": str(Path(output_dir).absolute()),
                   "claimed_at": time.time(), "automatic_resubmission": False}, claim)
    out = Path(output_dir).absolute()
    out.mkdir(parents=True, exist_ok=False)
    # A private launch receipt precedes the process, so partial local failure
    # leaves an explicit recovery question instead of inviting duplicate work.
    receipt = {"job_id": job_id, "argv": rows[0]["launch_argv"], "submitted_at": time.time(),
               "manifest": firesim.pin(Path(manifest_path).absolute()), "status": "launching",
               "numeric_pass": False, "physical_performance_verified": False}
    path = out / "launch.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    with (out / "wrapper.log").open("xb") as log:
        process = subprocess.Popen(receipt["argv"], stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
    receipt.update(pid=process.pid, status="wrapper_started")
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("dispatch")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.selection)
        with args.out.open("x") as stream:
            json.dump(result, stream, indent=2)
    else:
        result = dispatch(args.manifest, job_id=args.job_id, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
