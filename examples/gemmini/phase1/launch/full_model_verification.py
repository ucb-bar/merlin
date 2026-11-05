"""Operator-only full-model readiness and exact post-author compiler attribution."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

from merlin.common import oot_repo
from merlin.common.digest import sha256_file
from merlin.common.tree_hash import hash_tree
from merlin.model_check.cli import tree_sha256
from merlin.model_check.full_chain_validation import report_complete
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



def validate_readiness(launch) -> dict | None:
    policy = launch.document.get("full_model_verification")
    if policy is None:
        return None
    if policy.get("output_scope", "judged_only") != "judged_only" or "all_output_plan" in policy:
        output_verification().validate(launch, identity=selected_identity(launch), models=selection(launch)["models"])
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
    if receipt.get("schema") == "merlin.phase1.infrastructure_readiness.v2":
        validate_infrastructure_readiness(launch, receipt)
        return receipt
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


def _pinned(record: dict, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} has no exact artifact identity")
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise ValueError(f"{label} bytes differ")
    return path


def _witness(launch, record: dict, schema: str) -> dict:
    value = read(_pinned(record, schema))
    if value.get("schema") != schema:
        raise ValueError("infrastructure witness schema differs")
    repo = launch.path(launch.document["inputs"]["repo"])
    if not source_corresponds(repo, launch.variables["SOURCE_REVISION"], value.get("source_revision", "")):
        raise ValueError("infrastructure witness source differs")
    producer = value.get("producer")
    selected = launch.document.get("runtime_pins", {}).get("operator_infrastructure_witness")
    if selected is None or producer != {"path": str(launch.path(selected["path"])), "sha256": selected["sha256"]}:
        raise ValueError("infrastructure collector is not the selected pinned operator")
    _pinned(producer, "infrastructure collector")
    return value


def _invocation(record: dict, stage: str) -> dict:
    value = read(_pinned(record, stage))
    if value.get("schema") != "merlin.invocation_record.v1" or value.get("stage") != stage:
        raise ValueError("infrastructure process observation differs")
    for group in ("inputs", "dependencies", "outputs"):
        for pin in value.get(group, []):
            _pinned(pin, f"{stage} {group}")
    _pinned(value.get("executable"), f"{stage} executable")
    return value


def _target_witness(launch, record: dict) -> None:
    import numpy as np

    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.model_check import full_build as FB
    from merlin.model_check import full_quantized as FQ
    from merlin.model_check import full_reference_cache as RC
    from merlin.model_check import replay as RP
    from merlin.runtime.backends import spike_model

    value = _witness(launch, record, "merlin.phase1.infrastructure_target_witness.v1")
    if set(value) != {"schema", "source_revision", "producer", "target", "stage", "build_record",
                      "dram_base", "dram_bytes", "runs"}:
        raise ValueError("infrastructure target witness fields differ")
    if value["target"] != selection(launch)["models"].target:
        raise ValueError("infrastructure witness target differs")
    stage = value["stage"]
    if set(stage) != {"path", "sha256"} or strict_tree_sha256(Path(stage["path"]))["sha256"] != stage["sha256"]:
        raise ValueError("infrastructure fixture stage bytes differ")
    stage = Path(stage["path"])
    build = read(_pinned(value["build_record"], "infrastructure full build"))
    if build.get("status") != "linked" or sha256_file(Path(build["elf"])) != build["elf_sha256"]:
        raise ValueError("infrastructure original full image differs")
    original = verify_completed_recipe(Path(build["compilation_recipe"]["path"]), executable=Path(build["elf"]),
                                       expected_recipe_sha256=build["compilation_recipe"]["sha256"])
    if original["producer"]["sha256"] != sha256_file(Path(spike_model.__file__)):
        raise ValueError("infrastructure full build used another producer")
    if not isinstance(value["runs"], list) or len(value["runs"]) != 2:
        raise ValueError("infrastructure input rebinding needs two actual runs")
    input_digests, output_digests = set(), set()
    policy = launch.document["full_model_verification"]
    for run in value["runs"]:
        if set(run) != {"inputs", "expected", "elf", "compilation_recipe", "execution"}:
            raise ValueError("infrastructure run fields differ")
        inputs = _pinned(run["inputs"], "infrastructure bound inputs")
        expected = _pinned(run["expected"], "infrastructure independent expected outputs")
        elf = _pinned(run["elf"], "infrastructure rebound image")
        recipe = _pinned(run["compilation_recipe"], "infrastructure relink recipe")
        compiled = verify_completed_recipe(
            recipe, executable=elf, expected_recipe_sha256=run["compilation_recipe"]["sha256"]
        )
        if compiled["producer"]["sha256"] != sha256_file(Path(FB.__file__)):
            raise ValueError("infrastructure input rebinding used another producer")
        execution = _invocation(run["execution"], "phase1.infrastructure.target")
        argv, _engine = RP.spike_argv(
            value["target"], elf, dram_base=value["dram_base"], dram_bytes=value["dram_bytes"]
        )
        if (execution.get("argv") != argv or execution.get("kind") != "subprocess"
                or Path(execution["executable"]["path"]).resolve() != Path(argv[0]).resolve()
                or execution.get("status") != "completed" or execution.get("returncode") != 0
                or any(execution.get(key) is not True for key in
                       ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged"))
                or any(pin not in execution.get("inputs", []) for pin in (run["inputs"], run["elf"]))):
            raise ValueError("infrastructure target execution is incomplete or uses other inputs")
        stdout = _pinned(execution.get("stdout"), "infrastructure target console")
        _pinned(execution.get("stderr"), "infrastructure target stderr")
        actual, _info = FB.read(stdout.read_bytes(), stage, build["build_hash"])
        with np.load(expected, allow_pickle=False) as data:
            names = [f"out{i}" for i in range(len(data.files))]
            if set(data.files) != set(names):
                raise ValueError("infrastructure expected output arity/order differs")
            wanted = [np.asarray(data[name]) for name in names]
        if not {np.dtype("int32"), np.dtype("float32")} <= {array.dtype for array in wanted}:
            raise ValueError("infrastructure witness needs mixed int32 and float32 output readback")
        compared = FQ.compare_outputs(actual, wanted, [{}] * len(wanted), policy)
        if not compared or any(not row["pass"] for row in compared):
            raise ValueError("infrastructure target outputs differ from independent expected outputs")
        with np.load(inputs, allow_pickle=False) as data:
            input_digests.add(RC.arrays_digest({name: np.asarray(data[name]) for name in data.files}))
        output_digests.add(RC.arrays_digest({f"out{i}": array for i, array in enumerate(actual)}))
    if len(input_digests) != 2 or len(output_digests) != 2:
        raise ValueError("infrastructure input rebinding did not exercise changed arrays and outputs")


def _cleanup_witness(launch, record: dict) -> None:
    value = _witness(launch, record, "merlin.phase1.infrastructure_cleanup_witness.v1")
    if set(value) != {"schema", "source_revision", "producer", "invocation", "stdout", "children",
                      "owned_process_cleanup"}:
        raise ValueError("infrastructure cleanup witness fields differ")
    invocation = _invocation(value["invocation"], "phase1.infrastructure.timeout")
    if (invocation.get("kind") != "subprocess" or invocation.get("status") != "interrupted"
            or invocation.get("error") != "TimeoutExpired"):
        raise ValueError("infrastructure cleanup did not observe an actual timeout")
    stdout = _pinned(value["stdout"], "infrastructure timeout child observations")
    if invocation.get("stdout") != value["stdout"]:
        raise ValueError("infrastructure timeout observations differ from captured process output")
    children = [json.loads(line) for line in stdout.read_text().splitlines() if line.strip()]
    if (children != value["children"] or len(children) < 2
            or len({(item.get("pid"), item.get("start_ticks")) for item in children}) != len(children)):
        raise ValueError("infrastructure cleanup has no nested-child observation")
    cleanup = value["owned_process_cleanup"]
    if (cleanup.get("complete") is not True or cleanup.get("guardian_reaped") is not True
            or cleanup.get("error") is not None or cleanup.get("reaped_children", 0) < len(children)):
        raise ValueError("infrastructure process owner did not confirm complete cleanup")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for child in children:
        if (set(child) != {"pid", "start_ticks", "boot_id"} or type(child["pid"]) is not int or child["pid"] <= 0
                or child["boot_id"] != boot or type(child["start_ticks"]) is not int or child["start_ticks"] < 1):
            raise ValueError("infrastructure child process identity is incomplete")
        try:
            stat = (Path("/proc") / str(child["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
        except FileNotFoundError:
            continue
        if child["boot_id"] == boot and int(stat[19]) == int(child["start_ticks"]):
            raise ValueError("infrastructure timeout left an observed owned child alive")


def _capture_correspondence(selected: dict) -> None:
    """Cross-join selected capture/attestation source graphs; native preflight owns full tree checks."""
    rows = {row["id"]: row for row in yaml.safe_load(selected["private_spec"].read_text())["models"]}
    roots = read(selected["bundle"] / "MANIFEST.json").get("selected_capture_roots") or {}
    for name in selected["names"]:
        row, model = rows[name], selected["models"].model(name)
        capture = Path(row["capture"]).resolve()
        if name not in roots or capture != (Path(roots[name]) / "run/capture").resolve():
            raise ValueError(f"private spec and numerics bundle select different captures: {name}")
        selection_pin = {"path": row["capture_selection"], "sha256": row["capture_selection_sha256"]}
        _pinned(selection_pin, "selected capture preselection")
        attestation = read(_pinned({"path": row["capture_execution_attestation"],
                                   "sha256": row["capture_execution_attestation_sha256"]}, "capture attestation"))
        attested = attestation.get("capture") or {}
        if (attestation.get("selection") != selection_pin
                or Path(attested.get("capture_path", "")).resolve() != capture
                or not attested.get("capture_tree_sha256")):
            raise ValueError(f"capture attestation does not bind the selected source: {name}")
        programs = {item["name"]: item for item in attested.get("programs", [])}
        for program in model.programs:
            stage = capture / "stages" / program.stage if program.stage else capture
            catalog = read(selected["bundle"] / name / "programs" / program.name / "catalog.json")
            expected_sha = (programs.get(program.stage or program.name) or attested).get("model_sha256")
            if Path(catalog["stage"]).resolve() != stage or sha256_file(stage / "model.mlir") != expected_sha:
                raise ValueError(f"catalog differs from the selected attested source: {name}/{program.name}")


def validate_infrastructure_readiness(launch, receipt: dict) -> None:
    """Admit authoring from positive mechanisms; never qualify the retained or fresh compiler."""
    from merlin.model_check import bundle as B
    from merlin.model_check import full_quantized as FQ

    if (set(receipt) != {"schema", "purpose", "selected_identity", "retained_submission",
                         "retained_diagnostics", "target_witness", "cleanup_witness"}
            or receipt["schema"] != "merlin.phase1.infrastructure_readiness.v2"
            or receipt["purpose"] != "fresh_author_preflight"):
        raise ValueError("invalid infrastructure readiness receipt")
    if receipt["selected_identity"] != selected_identity(launch):
        raise ValueError("infrastructure readiness selected source/input/runtime identity differs")
    retained = receipt["retained_submission"]
    if retained != package_identity(Path(retained["path"])):
        raise ValueError("historical retained compiler bytes differ")
    selected = selection(launch)
    _capture_correspondence(selected)
    if set(receipt["retained_diagnostics"]) != set(selected["names"]):
        raise ValueError("historical diagnostic model roster differs")
    for name, records in receipt["retained_diagnostics"].items():
        if set(records) != {"report", "pin"}:
            raise ValueError("historical diagnostic fields differ")
        diagnostic = read(_pinned(records["report"], "historical retained diagnostic"))
        pin = read(_pinned(records["pin"], "historical retained invocation"))
        if diagnostic.get("schema") == "merlin.readiness.partial.v1":
            if diagnostic.get("complete") is not False or not diagnostic.get("reason"):
                raise ValueError("historical partial diagnostic was relabelled complete")
            diagnostic = diagnostic["model_check_report_so_far"]
        if (diagnostic.get("command") != "check" or diagnostic.get("model") != name
                or diagnostic.get("submission", {}).get("tree_sha256") != retained["model_check_tree_sha256"]
                or diagnostic.get("status") not in {"pass", "fail", "error"}
                or pin.get("submission") != retained or not pin.get("selected_identity")):
            raise ValueError("historical diagnostic compiler/model attribution differs")
        # This separately reopens CURRENT inputs; historical failures never certify their bytes or policy.
        FQ.sealed_reference(B.Bundle(selected["bundle"]), selected["models"].model(name),
                            selected["models"].path, launch.document["full_model_verification"]["input"])
    _target_witness(launch, receipt["target_witness"])
    _cleanup_witness(launch, receipt["cleanup_witness"])


def build_infrastructure_receipt(launch, retained: Path, diagnostics: dict, target_witness: Path,
                                 cleanup_witness: Path) -> dict:
    """Build and validate a v2 receipt from actual artifact paths; caller publishes/pins its bytes."""
    def pin(path):
        return {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))}

    receipt = {"schema": "merlin.phase1.infrastructure_readiness.v2", "purpose": "fresh_author_preflight",
               "selected_identity": selected_identity(launch), "retained_submission": package_identity(retained),
               "retained_diagnostics": {name: {kind: pin(path) for kind, path in records.items()}
                                        for name, records in diagnostics.items()},
               "target_witness": pin(target_witness), "cleanup_witness": pin(cleanup_witness)}
    validate_infrastructure_readiness(launch, receipt)
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


def output_verification():
    """Load the exact sibling operator even with safe-path Python startup."""
    source = Path(__file__).with_name("full_model_outputs.py")
    spec = importlib.util.spec_from_file_location("phase1_full_model_outputs", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
