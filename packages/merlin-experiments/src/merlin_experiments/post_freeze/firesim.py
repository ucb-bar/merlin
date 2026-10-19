"""Prepare registered-workload requests and import remote FireSim observations.

This boundary launches no cloud job and never stages an arbitrary command. It
preserves the distinction between frozen inputs, remote execution observations,
and the existing model checker's numerical/placement qualification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from merlin.common.digest import is_sha256, sha256_file
from merlin.common.tree_hash import hash_tree

SELECTION_SCHEMA = "merlin.post_freeze.firesim.selection.v1"
REQUEST_SCHEMA = "merlin.post_freeze.firesim.request.v1"
RESULT_SCHEMA = "merlin.post_freeze.firesim.observation.v1"
MANAGER_SCHEMA = "merlin.post_freeze.firesim.manager_result.v1"
ASSET_ROLES = frozenset({"boot_elf", "chain_manifest", "input", "reference", "build_receipt"})


def load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping: {path}")
    return value


def pin(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"expected an absolute regular artifact: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def reopen(record: dict[str, Any]) -> Path:
    path = Path(record["path"])
    if not is_sha256(record.get("sha256")) or pin(path)["sha256"] != record["sha256"]:
        raise ValueError(f"artifact bytes changed: {path}")
    return path


def _path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "${" in expanded:
        raise ValueError(f"unbound deployment path: {value}")
    return Path(expanded).absolute()


def _field(value: dict[str, Any], name: str) -> Any:
    for key in name.split("."):
        value = value[key]
    return value


def _bound(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _bound(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_bound(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"unbound selection variable: {value}")
        return expanded
    return value


def _token(value: Any) -> bool:
    return (isinstance(value, str) and 1 <= len(value) <= 64
            and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in value)
            and value not in {".", ".."})


def output_roster(model: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for stage, policy in model["stages"].items():
        for step in range(policy["steps"]):
            for index, output in enumerate(policy["outputs"]):
                rows.append({"stage": stage, "step": step, "index": index,
                             "dtype": output["dtype"], "shape": output["shape"]})
    if len(rows) != model["declared_output_arrays"]:
        raise ValueError("full output roster differs from the declared chain")
    if sum(stage["steps"] for stage in model["stages"].values()) != model["chain_steps"]:
        raise ValueError("program steps differ from the declared complete chain")
    return rows


def prepare(selection: Path) -> dict[str, Any]:
    spec = _bound(load(selection))
    if spec.get("schema") != SELECTION_SCHEMA or spec.get("phase") not in {"phase1", "phase2"}:
        raise ValueError("select a versioned post-freeze Phase 1 or Phase 2 request")
    compiler = _path(spec["compiler"])
    compiler_sha = hash_tree(compiler)["sha256"]
    if not is_sha256(compiler_sha):
        raise ValueError("frozen compiler is absent")
    freeze = pin(_path(spec["freeze_record"]["path"]))
    if _field(load(Path(freeze["path"])), spec["freeze_record"]["compiler_sha256_field"]) != compiler_sha:
        raise ValueError("freeze record and selected compiler bytes disagree")
    contract = pin(_path(spec["workload_contract"]))
    policy = load(Path(contract["path"]))
    if not isinstance(policy.get("models"), dict) or not policy["models"]:
        raise ValueError("workload contract has no complete model roster")
    deployment = spec["deployment"]
    if deployment.get("mode") != "workhorse_locked_ssm_on_demand":
        raise ValueError("select the locked on-demand driver/executor transport")
    launcher = pin(_path(deployment["launcher"]))
    if not os.access(launcher["path"], os.X_OK):
        raise ValueError("selected fpga-run launcher is not executable")
    target_hz = deployment.get("target_clock_hz")
    if type(target_hz) is not int or target_hz <= 0:
        raise ValueError("target clock must be explicit; FPGA host MHz is not target-clock authority")
    for name in ("region", "manager_instance", "ssm_document", "agfi", "results_prefix"):
        if not isinstance(deployment.get(name), str) or not deployment[name].strip():
            raise ValueError(f"deployment needs explicit {name}")
    budget = spec["budget"]
    if type(budget.get("max_jobs")) is not int or budget["max_jobs"] < len(policy["models"]):
        raise ValueError("job ceiling cannot omit any selected complete model")
    if type(budget.get("job_wall_s")) is not int or not 0 < budget["job_wall_s"] <= 21600:
        raise ValueError("select a positive per-job ceiling within the six-hour guardrail")
    models, blockers = [], []
    if not deployment.get("manager_contract"):
        blockers.append("manager build/transport/teardown contract is not yet bound")
    else:
        reopen(deployment["manager_contract"])
    supplied = spec.get("models", {})
    if set(supplied) - set(policy["models"]):
        raise ValueError("selection adds models outside the frozen workload contract")
    for name, model in policy["models"].items():
        selected = supplied.get(name, {})
        assets = []
        for item in selected.get("assets", []):
            if item.get("role") not in ASSET_ROLES:
                raise ValueError(f"unsupported asset role for {name}")
            assets.append({"role": item["role"], **pin(_path(item["path"]))})
        missing = sorted(ASSET_ROLES - {item["role"] for item in assets})
        if missing:
            blockers.append(f"{name}: missing bound assets {','.join(missing)}")
        workload = selected.get("registered_workload")
        registration = selected.get("registration")
        if not _token(workload) or registration is None:
            blockers.append(f"{name}: administrator registration is pending")
        elif not missing:
            registered = load(reopen(registration))
            expected = {"phase": spec["phase"], "model": name, "workload": workload,
                        "compiler_sha256": compiler_sha, "contract_sha256": contract["sha256"],
                        "assets": assets, "agfi": deployment["agfi"]}
            if any(registered.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{name}: registered workload differs from requested frozen inputs")
        steps, arrays = model["selected_chain_steps"], model["declared_arrays_per_chain"]
        if any(type(value) is not int or value < 1 for value in (steps, arrays)):
            raise ValueError("workload contract must declare complete chain and output counts")
        row = {"id": name, "chain_steps": steps, "declared_output_arrays": arrays,
               "chain": model["chain"], "stages": model["stages"], "assets": assets,
               "registered_workload": workload, "registration": registration}
        row["output_roster"] = output_roster(row)
        models.append(row)
    if hash_tree(compiler)["sha256"] != compiler_sha:
        raise ValueError("compiler changed during request preparation")
    return {"schema": REQUEST_SCHEMA, "phase": spec["phase"], "selection": pin(selection),
            "compiler": {"path": str(compiler), "hash_tree_sha256": compiler_sha},
            "freeze_record": freeze, "freeze_compiler_field": spec["freeze_record"]["compiler_sha256_field"],
            "workload_contract": contract, "launcher": launcher, "deployment": deployment,
            "budget": budget, "models": models, "status": "blocked" if blockers else "prepared",
            "blockers": blockers, "queue": "none; submit one registered workload at a time",
            "paid_jobs_launched": 0, "scientific_qualification": False,
            "scope": "Registration/preparation only. Existing model-check engines own all-output numerics, "
                     "source placement and exact accelerator checks. An SSM or UART success is insufficient."}


def verify_request(request: dict[str, Any]) -> None:
    if request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("not a post-freeze FireSim request")
    for key in ("selection", "freeze_record", "workload_contract", "launcher"):
        reopen(request[key])
    compiler = request["compiler"]
    if hash_tree(Path(compiler["path"]))["sha256"] != compiler["hash_tree_sha256"]:
        raise ValueError("frozen compiler changed after request creation")
    if prepare(Path(request["selection"]["path"])) != request:
        raise ValueError("request differs from its independently reopened frozen selection")


def collect(request_path: Path, result_dir: Path, invocation_path: Path | None) -> dict[str, Any]:
    request = load(request_path)
    verify_request(request)
    artifacts, failures = {}, []
    if request["status"] != "prepared":
        failures.append("request still has unresolved payload/registration prerequisites")
    for filename in ("uartlog", "HW_CFG_SUMMARY", "sim-run.sh", "result_manifest.json"):
        path = result_dir / filename
        if path.is_file():
            artifacts[filename] = pin(path)
        else:
            failures.append("missing " + filename)
    invocation = None
    if invocation_path is None:
        failures.append("missing original SSM invocation")
    else:
        invocation = load(invocation_path)
        if (invocation.get("InstanceId") != request["deployment"]["manager_instance"]
                or invocation.get("DocumentName") != request["deployment"]["ssm_document"]
                or invocation.get("Status") != "Success" or invocation.get("ResponseCode") != 0):
            failures.append("SSM invocation did not succeed on the selected manager")
    hardware_matches = False
    if "HW_CFG_SUMMARY" in artifacts:
        values = {}
        for line in Path(artifacts["HW_CFG_SUMMARY"]["path"]).read_text().splitlines():
            name, separator, value = line.partition(":")
            if separator:
                values[name.strip()] = value.strip()
        hardware_matches = values.get("AGFI") == request["deployment"]["agfi"]
        if not hardware_matches:
            failures.append("observed AGFI differs from selection")
    binding = False
    if "result_manifest.json" in artifacts:
        observed = load(Path(artifacts["result_manifest.json"]["path"]))
        model = next((m for m in request["models"] if m["id"] == observed.get("model")), None)
        expected = {"schema": MANAGER_SCHEMA, "phase": request["phase"],
                    "compiler_sha256": request["compiler"]["hash_tree_sha256"],
                    "contract_sha256": request["workload_contract"]["sha256"],
                    "request_sha256": sha256_file(request_path), "agfi": request["deployment"]["agfi"]}
        binding = model is not None and all(observed.get(k) == v for k, v in expected.items())
        if model is not None:
            binding = binding and all(observed.get(k) == model[v] for k, v in (
                ("workload", "registered_workload"), ("assets", "assets"),
                ("completed_chain_steps", "chain_steps"), ("output_array_count", "declared_output_arrays")))
        if invocation is not None:
            binding = binding and observed.get("ssm_command_id") == invocation.get("CommandId")
        outputs = observed.get("outputs")
        if model is None or not isinstance(outputs, list) or len(outputs) != model["declared_output_arrays"]:
            failures.append("missing complete per-step output files")
        else:
            for expected_output, output in zip(model["output_roster"], outputs, strict=True):
                if any(output.get(key) != value for key, value in expected_output.items()):
                    raise ValueError("observed output order/shape/dtype differs from the frozen contract")
                relative = Path(output["file"])
                path = result_dir / relative
                if (relative.is_absolute() or ".." in relative.parts
                        or not path.resolve().is_relative_to(result_dir.resolve())):
                    raise ValueError("output escaped the selected result directory")
                evidence = pin(path)
                if evidence["sha256"] != output.get("sha256"):
                    raise ValueError("raw model output bytes differ from manager manifest")
                artifacts[relative.as_posix()] = evidence
        if not binding:
            failures.append("manager result is not bound to this frozen request and complete chain")
    verify_request(request)
    return {"schema": RESULT_SCHEMA, "request": pin(request_path), "artifacts": artifacts,
            "ssm_invocation": pin(invocation_path) if invocation_path is not None else None,
            "status": "observation_imported" if not failures else "incomplete",
            "failures": failures, "hardware_agfi_matches": hardware_matches,
            "reported_frozen_inputs_match": binding, "model_validation_complete": False,
            "numeric_status": "pending_existing_all_output_checker",
            "placement_status": "pending_existing_source_and_final_elf_audits",
            "timing_status": "raw target-cycle evidence only; no target-latency conversion",
            "scope": "Raw observation intake only; no simulated/FPGA job or numerical gate is executed."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--selection", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--request", type=Path, required=True)
    c.add_argument("--result-dir", type=Path, required=True)
    c.add_argument("--ssm-invocation", type=Path)
    for command in (p, c):
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        value = (prepare(args.selection.absolute()) if args.command == "prepare" else
                 collect(args.request.absolute(), args.result_dir.absolute(),
                         args.ssm_invocation.absolute() if args.ssm_invocation else None))
        source = value if args.command == "prepare" else load(args.request)
        if args.out.resolve().is_relative_to(Path(source["compiler"]["path"]).resolve()):
            raise ValueError("observation output cannot modify the frozen compiler")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        print(json.dumps({"status": value["status"], "output": str(args.out), "paid_jobs_launched": 0}))
        return 0 if value["status"] in {"prepared", "observation_imported"} else 2
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(json.dumps({"status": "refused", "error": str(error), "paid_jobs_launched": 0}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
