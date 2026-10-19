"""Operator-only full-model readiness and exact post-author compiler attribution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

from merlin.common import oot_repo
from merlin.common.digest import sha256_file
from merlin.common.tree_hash import hash_tree
from merlin.model_check.cli import tree_sha256
from merlin.model_check.models import load as load_models


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def source_corresponds(repo: Path, baseline: str, observed: str) -> bool:
    """Permit operator metadata commits, never substitute a different harness tree."""
    if any(
        not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value)
        for value in (baseline, observed)
    ):
        return False
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", baseline, observed], cwd=repo, capture_output=True)
    difference = subprocess.run(
        ["git", "diff", "--quiet", baseline, observed, "--", ".", ":(exclude)examples/gemmini/phase1/launch"],
        cwd=repo,
        capture_output=True,
    )
    return ancestry.returncode == difference.returncode == 0


def selection(launch) -> dict:
    private = launch.path(launch.document["inputs"]["private_full_model_spec"])
    spec = yaml.safe_load(private.read_text())
    bundle = Path(launch.environment["MERLIN_MODEL_CHECK_BUNDLE"])
    names = {row["id"] for row in spec["models"]}
    models = load_models(bundle / "configuration/models.yaml")
    if names != set(models.models) or names != set(read(bundle / "MANIFEST.json")["models"]):
        raise ValueError("full-model roster differs between private spec, models and bundle")
    return {"bundle": bundle, "models": models, "private_spec": private, "names": sorted(names)}


def selected_identity(launch) -> dict:
    selected = selection(launch)
    pins = {}
    for name, pin in launch.document.get("runtime_pins", {}).items():
        actual = sha256_file(launch.path(pin["path"]))
        if actual != pin["sha256"]:
            raise ValueError(f"full-model runtime pin differs: {name}")
        pins[name] = actual
    bundle = selected["bundle"]
    return {
        "source_revision": launch.variables["SOURCE_REVISION"],
        "private_spec_sha256": sha256_file(selected["private_spec"]),
        "bundle_manifest_sha256": sha256_file(bundle / "MANIFEST.json"),
        "bundle_sums_sha256": sha256_file(bundle / "SHA256SUMS"),
        "models_config_sha256": sha256_file(bundle / "configuration/models.yaml"),
        "runtime_pins": pins,
    }


def package_identity(path: Path) -> dict:
    digest = hash_tree(path)
    if not digest.get("present") or not (path / "manifest.yaml").is_file():
        raise ValueError("selected compiler package is absent")
    return {"path": str(path), "hash_tree_sha256": digest["sha256"], "model_check_tree_sha256": tree_sha256(path)}


def report_complete(report: dict, model, policy: dict, package: dict) -> bool:
    """Require real selected gates and every declared step, not a CLI pass string."""
    gates = report.get("gates") or {}
    if report.get("status") != "pass" or report.get("failures") or report.get("model") != model.name:
        return False
    if (
        report.get("command") != "check"
        or report.get("split") != policy["split"]
        or report.get("agent_view") is not False
    ):
        return False
    if report.get("submission", {}).get("tree_sha256") != package["model_check_tree_sha256"]:
        return False
    comparison = report.get("comparison") or {}
    if comparison.get("selected") != "quantized-reference" or comparison.get("float_policy") != {
        "atol": policy["atol"],
        "rtol": policy["rtol"],
        "equal_nan": False,
        "finite_required": True,
    }:
        return False
    for name in ("build", "kernel_exact", "quantized_reference"):
        if gates.get(name, {}).get("status") != "pass" or gates[name].get("partial", False) is not False:
            return False
    expected = sum(program.steps for program in model.programs)
    gate = gates["quantized_reference"]
    if (
        gate.get("inputs_completed"),
        gate.get("inputs_expected"),
        gate.get("steps_completed"),
        gate.get("steps_expected"),
    ) != (1, 1, expected, expected):
        return False
    bindings = [
        {"program": ref["program"], "step": step, "output": int(ref["output"])}
        for ref in model.judged["outputs"]
        for step in range(model.program(ref["program"]).steps)
    ]
    if (
        gate.get("steps_declared") != expected
        or gate.get("outputs_compared") != len(bindings)
        or gate.get("outputs_expected") != len(bindings)
    ):
        return False
    metrics = report.get("model_metrics") or {}
    per_input = metrics.get("per_input") or []
    if metrics.get("source") != "measured" or metrics.get("inputs") != 1 or len(per_input) != 1:
        return False
    row = per_input[0]
    validation = row.get("reference_validation") or {}
    if (
        row.get("id") != policy["input"]
        or row.get("split") != policy["split"]
        or row.get("pass") is not True
        or "reference_consistent" not in row
        or row["reference_consistent"] is not None
    ):
        return False
    if (
        validation.get("source") != "sealed_int8"
        or validation.get("strict_arity") is not True
        or validation.get("unchanged") is not True
    ):
        return False
    compared = row.get("quantized_reference") or []
    outputs = validation.get("outputs") or []
    if len(compared) != len(bindings) or len(outputs) != len(bindings):
        return False
    for binding, measured, reference in zip(bindings, compared, outputs, strict=True):
        if any(measured.get(key) != value or reference.get(key) != value for key, value in binding.items()):
            return False
        if measured.get("pass") is not True or measured.get("mismatched") != 0:
            return False
        if measured.get("dtype") != reference.get("dtype") or measured.get("shape") != reference.get("shape"):
            return False
    if gate.get("float_policy") != comparison["float_policy"]:
        return False
    programs = {row.get("program"): row for row in report.get("programs", [])}
    if len(programs) != len(report.get("programs", [])) or set(programs) != {
        program.name for program in model.programs
    }:
        return False
    for program in model.programs:
        rows = programs[program.name].get("runs", [])
        if len(rows) != program.steps or {row.get("step") for row in rows} != set(range(program.steps)):
            return False
        if any(
            row.get("status") != "completed" or row.get("engine") != "spike" or row.get("input") != policy["input"]
            for row in rows
        ):
            return False
    return True


def validate_readiness(launch) -> dict | None:
    policy = launch.document.get("full_model_verification")
    if policy is None:
        return None
    if (
        policy["comparison"] != "quantized-reference"
        or policy["split"] != "visible"
        or policy["input"] != "visible/v00"
    ):
        raise ValueError("unsupported full-model verification scope")
    receipt_path = launch.path(policy["readiness_receipt"])
    if sha256_file(receipt_path) != policy["readiness_sha256"]:
        raise ValueError("retained infrastructure readiness receipt hash differs")
    receipt = read(receipt_path)
    if (
        set(receipt) != {"schema", "selected_identity", "retained_submission", "checks"}
        or receipt["schema"] != "merlin.phase1.full_model_readiness.v1"
    ):
        raise ValueError("invalid full-model readiness receipt")
    identity = selected_identity(launch)
    if receipt["selected_identity"] != identity:
        raise ValueError("full-model readiness selected source/input/runtime identity differs")
    retained = receipt["retained_submission"]
    if retained != package_identity(Path(retained["path"])):
        raise ValueError("retained readiness compiler bytes differ")
    selected = selection(launch)
    if set(receipt["checks"]) != set(selected["names"]):
        raise ValueError("retained readiness model roster differs")
    repo = launch.path(launch.document["inputs"]["repo"])
    for name, check in receipt["checks"].items():
        if set(check) != {"report", "pin"}:
            raise ValueError("invalid readiness report/pin record")
        values = {}
        for kind in ("report", "pin"):
            record = check[kind]
            if set(record) != {"path", "sha256"} or sha256_file(Path(record["path"])) != record["sha256"]:
                raise ValueError(f"readiness {name} {kind} bytes differ")
            values[kind] = read(Path(record["path"]))
        report, pin = values["report"], values["pin"]
        if not report_complete(report, selected["models"].model(name), policy, retained):
            raise ValueError(f"retained full-chain evidence is incomplete: {name}")
        reference = report["model_metrics"]["per_input"][0]["reference_validation"]
        reference_path = selected["bundle"] / name / policy["input"] / "int8.npz"
        if Path(reference.get("path", "")).resolve() != reference_path.resolve() or sha256_file(
            reference_path
        ) != reference.get("sha256"):
            raise ValueError(f"readiness quantized reference bytes differ: {name}")
        for report_key, identity_key in (
            ("manifest_sha256", "bundle_manifest_sha256"),
            ("sums_sha256", "bundle_sums_sha256"),
            ("models_sha256", "models_config_sha256"),
        ):
            if reference.get(report_key) != identity[identity_key]:
                raise ValueError(f"readiness reference/input identity differs: {name}")
        if not source_corresponds(repo, identity["source_revision"], report.get("tool", {}).get("commit", "")):
            raise ValueError(f"readiness checker source differs: {name}")
        if pin.get("selected_identity") != identity or pin.get("submission") != retained:
            raise ValueError(f"readiness invocation pin differs: {name}")
        command = model_command(
            launch, name, Path(retained["path"]), Path(check["report"]["path"]), Path(pin["work"]), pin["timeout_s"]
        )
        transport = pin.get("transport")
        if transport is not None:
            if set(transport) != {"path", "sha256"} or sha256_file(Path(transport["path"])) != transport["sha256"]:
                raise ValueError(f"readiness worker transport differs: {name}")
            inner = command[4:]
            if pin.get("inner_argv") != inner:
                raise ValueError(f"readiness inner model-check scope differs: {name}")
            command = [command[0], "-u", transport["path"], check["report"]["path"], pin["work"], *inner]
        if pin.get("argv") != command:
            raise ValueError(f"readiness invocation scope differs: {name}")
        binding = report.get("binding") or {}
        bound = selected["bundle"] / str(binding.get("receipt", ""))
        if (
            binding.get("status") != "pass"
            or not bound.is_file()
            or sha256_file(bound) != binding.get("receipt_sha256")
        ):
            raise ValueError(f"readiness hardware binding differs: {name}")
    return receipt


def model_command(launch, name: str, package: Path, report: Path, work: Path, timeout: int) -> list[str]:
    selected = selection(launch)
    policy = launch.document["full_model_verification"]
    return [
        launch.expand(launch.document["launcher"]["python"]),
        "-u",
        "-m",
        "merlin.model_check",
        "check",
        "--submission",
        str(package),
        "--model",
        name,
        "--mode",
        "full",
        "--full-comparison",
        "quantized-reference",
        "--quantized-reference-atol",
        str(policy["atol"]),
        "--quantized-reference-rtol",
        str(policy["rtol"]),
        "--split",
        policy["split"],
        "--input",
        policy["input"],
        "--engine",
        "spike",
        "--jobs",
        "1",
        "--bundle",
        str(selected["bundle"]),
        "--models",
        str(selected["models"].path),
        "--timeout-s",
        str(timeout),
        "--out",
        str(report),
        "--work",
        str(work),
    ]


def terminal_author(run: Path, records: dict[int, tuple[int, int]]) -> bool:
    for pid in records:
        try:
            if (Path("/proc") / str(pid) / "comm").read_text().strip() == "codex":
                return False
        except OSError:
            pass
    transcripts = sorted((run / "rounds").glob("round_*.transcript.jsonl"))
    if not transcripts:
        return False
    try:
        events = [json.loads(line) for line in transcripts[-1].read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return False
    return any(row.get("type") == "result" for row in events)


def native_snapshot(run: Path, author: Path) -> dict | None:
    """Prove the actual native commit now; ledger publication may still be grading."""
    try:
        live = hash_tree(author)
        if not live.get("present") or not (author / "manifest.yaml").is_file():
            return None
        history = run / "oot"
        commit = oot_repo.resolve(history, "HEAD")
        if oot_repo.tree_digest(history, commit) != live["sha256"]:
            return None
        for candidate in sorted((run / "_qa_work").glob("cand_*/submission")):
            if hash_tree(candidate).get("sha256") == live["sha256"]:
                return {
                    "sha256": live["sha256"],
                    "candidate": str(candidate),
                    "author": str(author),
                    "native_oot_commit": commit,
                }
    except (OSError, ValueError, oot_repo.OotRepoError):
        return None
    return None


def copy_snapshot(run: Path, snapshot: dict, destination: Path) -> dict:
    oot_repo.export(run / "oot", snapshot["native_oot_commit"], destination)
    identity = package_identity(destination)
    if any(
        hash_tree(Path(snapshot[key])).get("sha256") != identity["hash_tree_sha256"] for key in ("author", "candidate")
    ):
        raise ValueError("author/native snapshot changed while freezing model compiler")
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    return identity


def final_correspondence(run: Path, snapshot: dict, package: dict) -> dict:
    published = hash_tree(run / "submission")
    ledger = run / "oot_commits.jsonl"
    try:
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    except (OSError, ValueError):
        rows = []
    matched = any(
        row.get("oot", {}).get("package_digest") == snapshot["sha256"]
        and row.get("oot", {}).get("commit") == snapshot["native_oot_commit"]
        for row in rows
    )
    return {
        "native_oot_commit_verified": oot_repo.tree_digest(run / "oot", snapshot["native_oot_commit"])
        == snapshot["sha256"],
        "native_ledger": "exact" if matched else "pending",
        "native_final_publication": "exact"
        if published.get("sha256") == snapshot["sha256"]
        else "different_compiler"
        if published.get("present")
        else "pending",
        "frozen_copy_unchanged": package_identity(Path(package["path"])) == package,
    }
