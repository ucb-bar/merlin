"""Compare prepared launch inputs with their reviewed configuration.

Read-only checks cover argv, environments, pins, numerical lineage and
exclusions. An optional process reader observes /proc and affinity; these
observations do not start a run or establish compiler qualification.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from merlin.common.digest import sha256_file

if TYPE_CHECKING:
    from merlin.development.backend.session.launch_config import Launch, Prepend, Report


def _load_json(path: Path, report: Report, subject: str) -> dict | None:

    if not path.is_file():
        report.drift("missing_file", subject, str(path), None)
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report.drift("unreadable", subject, "JSON", f"{type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        report.drift("unreadable", subject, "a JSON object", type(value).__name__)
        return None
    return value


def _load_yaml(path: Path, report: Report, subject: str) -> dict | None:

    if not path.is_file():
        report.drift("missing_file", subject, str(path), None)
        return None
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        report.drift("unreadable", subject, "YAML", f"{type(exc).__name__}: {exc}")
        return None
    if not isinstance(value, dict):
        report.drift("unreadable", subject, "a YAML mapping", type(value).__name__)
        return None
    return value


def _dig(value: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _same_path(a: object, b: object) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return a == b
    return os.path.normpath(a) == os.path.normpath(b) or os.path.realpath(a) == os.path.realpath(b)


def _within(path: str | Path, root: str | Path) -> bool:
    return Path(os.path.normpath(path)).is_relative_to(Path(os.path.normpath(root)))


def split_argv(argv: Iterable[str]) -> tuple[list[str], dict[str, list[str | None]], list[str]]:
    """``(interpreter, -m, module)`` head, the ``--flag value`` occurrences, and stray words."""
    argv = list(argv)
    flags: dict[str, list[str | None]] = {}
    strays: list[str] = []
    index = 3
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            strays.append(token)
            index += 1
            continue
        name, equals, inline = token.partition("=")
        if equals:
            value, index = inline, index + 1
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            value, index = argv[index + 1], index + 2
        else:
            value, index = None, index + 1
        flags.setdefault(name, []).append(value)
    return argv[:3], flags, strays


def _compare_argv(launch: Launch, actual: list[str], subject: str, report: Report, *, template: bool) -> None:

    head, expected, _ = split_argv(launch.argv)
    found_head, found, strays = split_argv(actual)
    report.expect(f"{subject} interpreter/module", " ".join(head), " ".join(found_head), "argv_head")
    for word in strays:
        report.drift("extra_argument", subject, None, word)
    for flag, values in found.items():
        if len(values) > 1:
            report.drift("duplicate_flag", f"{subject} {flag}", 1, len(values))
    for flag, values in expected.items():
        if flag not in found:
            report.drift("missing_flag", f"{subject} {flag}", values[0], None)
        elif template and flag == "--run-id" and found[flag][-1] == launch.placeholder:
            report.ok(f"{subject} {flag} (template)")
        else:
            report.expect(f"{subject} {flag}", values[0], found[flag][-1], "flag_value")
    for flag in found:
        if flag not in expected:
            report.drift("extra_flag", f"{subject} {flag}", None, found[flag][-1])


def _compare_environment(
    expected: Mapping[str, str | Prepend],
    actual: Mapping[str, str | Prepend],
    subject: str,
    report: Report,
    policy: Mapping,
    *,
    live: bool,
) -> None:
    """A file must say exactly the config; a live process may also inherit unowned variables.

    In a live environment a search path matches when it starts with the configured prefix;
    in a file it must keep the inherited value (``{prepend: ...}``), not replace it.
    """
    from merlin.development.backend.session.launch_config import Prepend

    forbidden = set(policy.get("forbidden") or ())
    owned = tuple(policy.get("owned_prefixes") or ())
    for name, want in expected.items():
        if name not in actual:
            report.drift("missing_env", f"{subject} {name}", want, None)
            continue
        found = actual[name]
        matches = want.accepts(found) if live and isinstance(want, Prepend) else want == found
        if matches:
            report.ok(f"{subject} {name}")
        else:
            report.drift("env_value", f"{subject} {name}", want, found)
    for name, found in actual.items():
        if name in expected:
            continue
        if name in forbidden:
            report.drift("forbidden_env", f"{subject} {name}", "unset", found)
        elif not live or (owned and name.startswith(owned)):
            report.drift("extra_env", f"{subject} {name}", None, found)


def _check_pins(launch: Launch, report: Report) -> None:
    from merlin.development.backend.session.launch_config import Report

    for label, (path, digest) in launch.pins.items():
        target = Path(path)
        if not target.is_file():
            report.drift("missing_file", label, path, None)
        else:
            report.expect(f"{label} sha256", digest, sha256_file(target), "hash")
    selection = launch.document.get("instruction_selection")
    if selection:
        approval = _load_json(launch.path(selection["approval"]["path"]), Report(), "approval")
        if approval is not None:
            report.expect(
                "instruction_selection.approval names sha256",
                selection["sha256"],
                approval.get("selection_sha256"),
                "lineage",
            )
            if _same_path(launch.expand(selection["path"]), approval.get("selection")):
                report.ok("instruction_selection.approval names path")
            else:
                report.drift(
                    "lineage",
                    "instruction_selection.approval names path",
                    launch.expand(selection["path"]),
                    approval.get("selection"),
                )


def _check_environment_file(launch: Launch, path: Path, report: Report) -> None:
    from merlin.development.backend.session.launch_config import parse_environment_script

    if not path.is_file():
        report.drift("missing_file", "environment file", str(path), None)
        return
    actual, problems = parse_environment_script(path.read_text(encoding="utf-8"))
    for problem in problems:
        report.drift("env_syntax", f"environment file {path.name}", "export NAME=VALUE", problem)
    policy = launch.document.get("environment_policy") or {}
    _compare_environment(launch.environment, actual, "environment file", report, policy, live=False)


def _check_plan(launch: Launch, path: Path, report: Report) -> dict | None:
    from merlin.development.backend.session.launch_config import launch_record

    plan = _load_json(path, report, "launch plan")
    if plan is None:
        return None
    _compare_argv(launch, list(plan.get("argv") or []), "plan argv", report, template=True)
    document = launch.document
    origin = document["origin"]
    seed = None if origin["seed_submission"] is None else launch.expand(origin["seed_submission"])
    for key, want in (
        ("environment_file", launch.expand(document["launcher"]["environment_file"])),
        ("manager_affinity", list(launch.manager_cpus)),
        ("seed_submission", seed),
        ("resume", origin["resume"]),
        ("initial_submission", origin["initial_submission"]),
        ("proposed_authoring_budget_s", document["budgets"]["authoring_s"]),
    ):
        report.expect(f"plan {key}", want, plan.get(key, "<absent>"), "plan_field")
    if "selected_runtime_hashes" in plan:
        report.expect(
            "plan selected_runtime_hashes",
            launch_record(launch)["runtime_pins"],
            plan["selected_runtime_hashes"],
            "hash",
        )
    return plan


def _check_numerics(launch: Launch, plan: dict | None, report: Report) -> None:

    numerics = launch.document.get("numerics")
    if not numerics:
        return
    approved = numerics["approved"]
    bundle = launch.path(approved["bundle"])
    manifest = bundle / "MANIFEST.json"
    if manifest.is_file():
        report.expect(
            f"numerics {approved['id']} manifest sha256", approved["manifest_sha256"], sha256_file(manifest), "hash"
        )
    else:
        report.drift("missing_file", f"numerics {approved['id']} manifest", str(manifest), None)
    signoff = approved["signoff"]
    source = launch.path(signoff["source"])
    if not source.is_file():
        report.drift("missing_file", "numerics sign-off source", str(source), None)
    else:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        needles = (signoff["marker"], approved["manifest_sha256"], signoff["verdict"])
        if any(all(needle in line for needle in needles) for line in lines):
            report.ok("numerics sign-off")
        else:
            report.drift("lineage", "numerics sign-off", " + ".join(needles), "no line with all three")
    launch_bundle = launch.expand(numerics["launch_bundle"])
    launch_manifest = Path(launch_bundle) / "MANIFEST.json"
    launch_sha = sha256_file(launch_manifest) if launch_manifest.is_file() else None
    if launch_sha is None:
        report.drift("missing_file", "numerics launch bundle manifest", str(launch_manifest), None)
    elif plan is not None:
        report.expect("plan manifest_sha256", launch_sha, plan.get("manifest_sha256"), "lineage")
    if numerics.get("reuse_proof"):
        proof = _load_json(launch.path(numerics["reuse_proof"]), report, "numerics reuse proof")
        if proof is not None:
            if _same_path(str(bundle), proof.get("parent_bundle")):
                report.ok("reuse proof parent bundle")
            else:
                report.drift("lineage", "reuse proof parent bundle", str(bundle), proof.get("parent_bundle"))
            report.expect(
                "reuse proof parent manifest sha256",
                approved["manifest_sha256"],
                proof.get("parent_manifest_sha256"),
                "lineage",
            )
            report.expect("reuse proof changed original files", [], proof.get("changed_original_files"), "lineage")
    spec = launch.document["inputs"].get("private_full_model_spec")
    models = _load_yaml(launch.path(spec), report, "private full-model spec") if spec else None
    if models is not None:
        named = _dig(models, "numerics_bundle", "path")
        _expect_path(report, "private full-model spec numerics_bundle.path", launch_bundle, named)
    if numerics.get("finalization_report"):
        status = _load_json(launch.path(numerics["finalization_report"]), report, "finalization report")
        if status is not None:
            _expect_path(report, "finalization report bundle", launch_bundle, status.get("bundle"))
            if launch.config.release:
                release = launch.expand(launch.config.release["prepare"]["output"])
                _expect_path(report, "finalization report release", release, status.get("release"))
            if launch_sha:
                report.expect(
                    "finalization report manifest_sha256", launch_sha, status.get("manifest_sha256"), "lineage"
                )
    if numerics.get("verify_report"):
        verified = _load_json(launch.path(numerics["verify_report"]), report, "bundle verify report")
        if verified is not None:
            report.expect("bundle verify status", "pass", verified.get("status"), "lineage")
            if launch_sha:
                report.expect(
                    "bundle verify manifest_sha256",
                    launch_sha,
                    _dig(verified, "bundle", "verify", "manifest_sha256"),
                    "lineage",
                )


def _expect_path(report: Report, subject: str, expected: str, actual: object) -> None:

    if _same_path(expected, actual):
        report.ok(subject)
    else:
        report.drift("lineage", subject, expected, actual)


def _check_release(launch: Launch, report: Report) -> None:

    release = launch.config.release
    if not release:
        return
    prepare, seal = release["prepare"], release["seal"]
    root = launch.path(prepare["output"])
    record = _load_json(root / "private" / "preparation.json", report, "release preparation record")
    if record is not None:
        for key in ("common_substrate", "python_source_root"):
            if key in prepare:
                expected = launch.expand(prepare[key])
                actual = _dig(record, "scaffolding", "bundle_generation", "compiler_source", key)
                report.expect("release " + key, expected, actual)
        _expect_path(report, "release source_run", launch.expand(prepare["run_dir"]), record.get("source_run"))
        if "kernel_timing_selection" in prepare:
            found = _dig(record, "scaffolding", "capability_contract", "kernel_timing_selection", "sha256")
            report.expect(
                "release kernel timing selection sha256", prepare["kernel_timing_selection"]["sha256"], found, "hash"
            )
        if "phase1_policy_descriptor" in prepare:
            found = _dig(record, "phase1_policy", "sha256")
            report.expect("release phase1 policy sha256", prepare["phase1_policy_descriptor"]["sha256"], found, "hash")
        if "selected_variants" in prepare:
            found = _dig(record, "scaffolding", "bundle_generation", "selection", "variants")
            report.expect("release selected variants", list(prepare["selected_variants"]), found)
        if prepare["generated_only"]:
            report.expect("release assembly mode", "generated_only", _dig(record, "assembly", "mode"))
        if prepare.get("capsule_scope") == "accelerator":
            report.expect("release capsule scope", "accelerator", _dig(record, "assembly", "capsule_scope", "kind"))
    sealed = _load_json(launch.path(seal["seal"]), report, "release seal")
    if sealed is not None:
        report.expect("release seal reviewed_by", seal["reviewed_by"], _dig(sealed, "review", "reviewed_by"))
        report.expect("release seal review note", seal["review_note"], _dig(sealed, "review", "note"))
    descriptor = launch.path(seal["descriptor"])
    if descriptor.is_file():
        report.ok("release descriptor")
    else:
        report.drift("missing_file", "release descriptor", str(descriptor), None)
    bundle = _load_yaml(launch.path(launch.document["inputs"]["bundle_manifest"]), report, "bundle manifest")
    if bundle is not None:
        report.expect("bundle manifest bundle_id", launch.document["identity"]["bundle"], bundle.get("bundle_id"))


def _path_values(launch: Launch) -> list[tuple[str, str]]:
    from merlin.development.backend.session.launch_config import Prepend

    values = [("argv", word) for word in launch.argv]
    for name, value in [*launch.environment.items(), *launch.launch_environment.items()]:
        text = value.prefix if isinstance(value, Prepend) else value
        values += [(f"environment {name}", part) for part in text.split(":")]
    return [(where, value) for where, value in values if value.startswith("/")]


def _check_exclusions(launch: Launch, report: Report) -> None:
    from merlin.development.backend.session.launch_config import Report

    exclusions = launch.document.get("exclusions") or ()
    repo = launch.path(launch.document["inputs"]["repo"])
    excluded = [(entry["id"], repo / relative) for entry in exclusions for relative in entry["paths"]]
    for where, value in _path_values(launch):
        for name, root in excluded:
            if _within(value, root):
                report.drift("exclusion", f"{name} named by {where}", "unreferenced", value)
    bundle = _load_yaml(launch.path(launch.document["inputs"]["bundle_manifest"]), Report(), "bundle manifest")
    if bundle is not None:
        for key in ("allowed", "host_inputs"):
            for row in bundle.get(key) or ():
                raw = row.get("path") if isinstance(row, dict) else None
                if not isinstance(raw, str):
                    report.drift("unreadable", f"bundle {key} row", "a path", row)
                    continue
                granted = Path(raw) if Path(raw).is_absolute() else repo / raw
                for name, root in excluded:
                    if _within(granted, root) or _within(root, granted):
                        report.drift("exclusion", f"{name} reachable from bundle {key}", "unreachable", raw)
        report.ok("bundle grants exclude operator-only paths")
    digests = {digest: entry["id"] for entry in exclusions for digest in entry.get("content_sha256") or ()}
    digests[launch.config.sha256] = "launch config"
    if launch.config.release_sha256:
        digests[launch.config.release_sha256] = "release config"
    if launch.document.get("errata"):
        digests[launch.document["errata"]["sha256"]] = "operator errata"
    if launch.config.release:
        payload = launch.path(launch.config.release["prepare"]["output"]) / "payload"
        if payload.is_dir():
            scanned = 0
            for path in sorted(payload.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    scanned += 1
                    owner = digests.get(sha256_file(path))
                    if owner:
                        report.drift(
                            "exclusion", f"{owner} bytes in release payload", "absent", str(path.relative_to(payload))
                        )
            report.ok(f"release payload content ({scanned} files)")


def read_process(pid: int) -> tuple[list[str], dict[str, str], set[int]]:
    """A live process's exact argv, initial environment and CPU affinity."""
    base = Path("/proc") / str(pid)
    argv = [part.decode("utf-8", "surrogateescape") for part in (base / "cmdline").read_bytes().split(b"\0")]
    if argv and argv[-1] == "":
        argv.pop()
    environment = {}
    for item in (base / "environ").read_bytes().split(b"\0"):
        name, equals, value = item.partition(b"=")
        if equals:
            environment[name.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
    return argv, environment, set(os.sched_getaffinity(pid))


def _check_process(launch: Launch, pid: int, report: Report, reader=read_process) -> None:

    try:
        argv, environment, cpus = reader(pid)
    except OSError as exc:
        report.drift("process", f"controller {pid}", "readable /proc entry", f"{type(exc).__name__}: {exc}")
        return
    _compare_argv(launch, argv, f"controller {pid} argv", report, template=False)
    expected = {**launch.environment, **launch.launch_environment}
    policy = launch.document.get("environment_policy") or {}
    _compare_environment(expected, environment, f"controller {pid} environment", report, policy, live=True)
    report.expect(f"controller {pid} affinity", sorted(launch.manager_cpus), sorted(cpus), "affinity")


def verify(
    launch: Launch,
    *,
    plan: str | Path | None = None,
    environment_file: str | Path | None = None,
    controller_pid: int | None = None,
    reader=read_process,
) -> Report:
    """Every drift between the config and the prepared inputs (and, optionally, a live controller)."""
    from merlin.development.backend.session.launch_config import Report

    report = Report()
    launcher = launch.document["launcher"]
    environment_file = Path(environment_file) if environment_file else launch.path(launcher["environment_file"])
    _check_pins(launch, report)
    _check_environment_file(launch, environment_file, report)
    selected = _check_plan(launch, Path(plan) if plan else launch.path(launcher["plan"]), report)
    _check_numerics(launch, selected, report)
    _check_release(launch, report)
    _check_exclusions(launch, report)
    if controller_pid is not None:
        _check_process(launch, controller_pid, report, reader)
    return report


def report_record(launch: Launch, report: Report) -> dict:
    from merlin.development.backend.session.launch_config import REPORT_SCHEMA

    return {
        "schema": REPORT_SCHEMA,
        "id": launch.document["id"],
        "config": {"path": str(launch.config.path), "sha256": launch.config.sha256},
        "overrides": launch.overrides,
        "run_id": launch.run_id,
        "status": "pass" if not report.drifts else "drift",
        "checked": len(report.checked),
        "drifts": [drift.record() for drift in report.drifts],
        "passed": list(report.checked),
    }
