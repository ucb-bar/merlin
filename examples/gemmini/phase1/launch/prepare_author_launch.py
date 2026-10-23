"""Render an explicit future fresh launch and separate no-author preflight.

This does not start either command or acquire CPUs. Historical admission flags
and seed verdicts are not carried into the new selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from merlin_experiments.phase1.joint_control import policy as load_policy
from merlin_experiments.phase2.compiler_library_state import restore_selection


def prepare(*, template, release, library_selection, repo, output, policy, run_id):
    template, release, repo, output = map(lambda p: Path(p).resolve(), (template, release, repo, output))
    old = json.loads(template.read_bytes())
    library = restore_selection(json.loads(Path(library_selection).read_bytes()))
    argv = list(old["argv"])
    previous_run_id = old.get("run_id") or argv[argv.index("--run-id") + 1]
    previous_repo = Path(argv[argv.index("--repo") + 1])
    previous_release = Path(argv[argv.index("--descriptor") + 1]).parents[2]
    task = repo.parent
    output.mkdir(parents=True, exist_ok=True)
    selected_policy = load_policy(policy)
    if "reviewed_compiler_seed" in selected_policy:
        raise ValueError("fresh launch must not carry a reviewed compiler seed")
    policy_copy = output / "policy.json"
    if Path(policy).resolve() != policy_copy:
        policy_copy.write_bytes(Path(policy).read_bytes())
    for flag in ("--seed-submission", "--joint-epoch"):
        if flag in argv:
            i = argv.index(flag)
            del argv[i : i + 2]
    for flag in ("--resume", "--preflight-only", "--codex-canary"):
        if flag in argv:
            argv.remove(flag)

    def set_flag(flag, value):
        if flag in argv:
            argv[argv.index(flag) + 1] = str(value)
        else:
            argv.extend((flag, str(value)))

    contract_path = output / "selected-library-contract.json"
    contract_path.write_text(json.dumps(library.contract.record(), sort_keys=True, indent=2) + "\n")
    values = {
        "--repo": repo,
        "--run-id": run_id,
        "--descriptor": release / "payload/experiment/target_experiment.yaml",
        "--bundle-manifest": release
        / "payload/experiment/input_bundles/merlin_assisted_rtlchecks_public_v0/input_bundle_manifest.yaml",
        "--corpus-seal": release / "private/seal.json",
        "--compiler-library-root": library.root,
        "--compiler-library-contract": contract_path,
        "--compiler-library-contract-sha256": library.contract.sha256,
        "--joint-campaign": policy_copy,
        "--operator-errata": repo / "examples/gemmini/phase1/launch/errata/coevolution-readiness.md",
        "--codex-home-root": task / "private" / ("codex-" + run_id),
        "--author-token-target": 40_000_000,
        "--author-token-reserve": 8_000_000,
    }
    for key, value in values.items():
        set_flag(key, value)
    if "--fresh-backend" not in argv:
        argv.append("--fresh-backend")
    if int(argv[argv.index("--max-wall-s") + 1]) != selected_policy["duration_s"]:
        raise ValueError("native author clock differs from selected joint policy")
    env = old["environment"].copy()
    for key, value in env.items():
        env[key] = (
            value.replace(str(previous_release), str(release))
            .replace(str(previous_repo), str(repo))
            .replace(str(previous_repo.parent / ".tmp"), str(task / ".tmp"))
            .replace(previous_run_id, run_id)
        )
    for key in (
        "MERLIN_JOINT_CAMPAIGN_STATE",
        "MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION",
        "MERLIN_FROZEN_PUBLIC_SCALABILITY",
    ):
        env.pop(key, None)
    cpus = env["MERLIN_COMPUTE_CPUS"]
    if cpus != env["MERLIN_CODEX_CPUS"] or cpus != env["MERLIN_PREPARATION_CPUS"]:
        raise ValueError("all roles must share the same CPU reservation")
    cpu_ids = [int(n) for n in cpus.split(",")]
    if len(cpu_ids) != 20 or len(set(cpu_ids)) != 20:
        raise ValueError("this proposed treatment requires twenty distinct shared CPUs")
    for key in ("MERLIN_GRADE_WORKERS", "MERLIN_DEVICE_BUILD_JOBS", "MERLIN_GSIM_MAX_SLOTS", "MERLIN_SPIKE_MAX_SLOTS"):
        env[key] = "20"
    for key in ("MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GSIM_SLOT_ROOT"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    reserved = reservation_argv(
        argv, env, task=task, output=output, run_id=run_id, cpus=cpus, duration_s=selected_policy["duration_s"]
    )
    record = {
        "schema": "operator.proposed_coevolution_launch.v1",
        "execution_started": False,
        "run_id": run_id,
        "native_preparation": "pending_no_author_preflight",
        "argv": argv,
        "reserved_argv": reserved,
        "reserved_argv_executed": False,
        "cwd": str(repo),
        "environment": env,
        "environment_policy": old["environment_policy"],
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_release": str(release),
        "library_contract_sha256": library.contract.sha256,
        "source_template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
        "cpu_reservation": {"cpus": cpu_ids, "shared_across_roles": True, "currently_acquired": False},
        "origin": "fresh optimized backend; reviewed generic compiler library",
        "token_budget": {
            "target": 40_000_000,
            "reserve": 8_000_000,
            "mode": "soft",
            "scope": "all authors including cached input",
        },
        "model_status": "provisional; no <=4percent qualification claim",
        "capsules": {"total": 250, "original_correctness": 153, "calibration": 20, "scalability_performance": 77},
        "limitations": [
            "Corpus obligations must be checked independently against selected public sources",
            "No author, simulation or CPU reservation is launched by this preparation",
        ],
    }
    (output / "proposed-launch.json").write_text(json.dumps(record, indent=2) + "\n")
    command = reserved
    (output / "proposed-reserved-command.sh").write_text(
        "#!/usr/bin/env bash\n# PROPOSED ONLY: executing this starts the full campaign.\nset -euo pipefail\n"
        + "cd "
        + shlex.quote(str(repo))
        + "\nexec "
        + shlex.join(command)
        + "\n"
    )
    preflight = json.loads(json.dumps(record))
    preflight["argv"][preflight["argv"].index("--run-id") + 1] = run_id + "-preflight"
    preflight["argv"].append("--preflight-only")
    preflight.pop("reserved_argv")
    (output / "preflight-selection.json").write_text(json.dumps(preflight, indent=2) + "\n")
    return record


def reservation_argv(argv, environment, *, task, output, run_id, cpus, duration_s):
    """Restore the selected runtime inside the reservation's environment filter."""
    clean = {key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "LANG") if key in os.environ}
    clean.update(environment)
    inner = ["/usr/bin/env", "-i", *(f"{key}={value}" for key, value in sorted(clean.items())), *argv]
    unit = "merlinphase1" + hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return [
        argv[0],
        str(Path(__file__).with_name("resource_reservation.py")),
        "--cpus",
        cpus,
        "--max-seconds",
        str(duration_s),
        "--unit",
        unit,
        "--state",
        str(task / ".tmp" / run_id / "cpu-reservation.json"),
        "--log",
        str(output / "campaign.log"),
        "--",
        *inner,
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "release", "library-selection", "repo", "output", "policy"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--run-id", required=True)
    record = prepare(**vars(p.parse_args()))
    print(
        json.dumps(
            {
                "prepared": True,
                "execution_started": record["execution_started"],
                "library_contract_sha256": record["library_contract_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
