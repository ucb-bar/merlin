"""Validate an operator-selected package for the Chipyard readiness probe."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def select_reference_backend(raw: str | None, *, target: str) -> tuple[Path, dict]:
    if not raw:
        raise ValueError("Chipyard readiness requires --reference-backend /absolute/package/path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("--reference-backend must be an absolute package path")
    path = path.resolve(strict=True)
    manifest_path = path / "manifest.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"selected reference backend has no manifest.yaml: {path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"selected reference backend has an invalid manifest: {path}")
    if manifest.get("target") != target:
        raise ValueError(f"selected reference backend targets {manifest.get('target')!r}, expected {target!r}")
    if manifest.get("artifact_type") != "mlir_oot_target_backend":
        raise ValueError("selected reference backend is not an MLIR OOT target backend")
    if manifest.get("integrity_exempt") is not False:
        raise ValueError("selected reference backend must declare integrity_exempt: false")
    if manifest.get("language") not in ("cpp", "python"):
        raise ValueError("selected reference backend must declare language cpp or python")
    if not isinstance(manifest.get("entrypoints"), dict) or not manifest["entrypoints"].get("tool"):
        raise ValueError("selected reference backend has no tool entrypoint")
    return path, manifest


def select_probe_inputs(
    *,
    target: str,
    root: str | None,
    screen: str | None,
    timing: str | None,
    incorrect_backend: str | None,
    prohibited_backend: str | None,
    prohibited_probe: str | None,
    timeout_s: int | None = None,
    timing_output: str | None = None,
    resource_root: Path | None = None,
    required: bool = False,
) -> dict | None:
    """Select operator-only probes; their contents still pass through the ordinary grader."""
    selectors = (root, screen, timing, incorrect_backend, prohibited_backend, prohibited_probe)
    if (
        not any(value is not None for value in selectors)
        and timeout_s is None
        and timing_output is None
        and not required
    ):
        return None
    if not all(isinstance(value, str) and value for value in selectors):
        raise ValueError("explicit oracle probes require all root, name and negative-package selectors")
    if type(timeout_s) is not int or not 0 < timeout_s <= 600:
        raise ValueError("explicit oracle probes require --probe-timeout-s between 1 and 600")
    if not isinstance(timing_output, str) or not timing_output or resource_root is None:
        raise ValueError("explicit oracle probes require --oracle-timing-output")
    output = Path(timing_output).expanduser()
    verify_timing_output(output, resource_root=resource_root)
    path = Path(root).expanduser()
    if not path.is_absolute():
        raise ValueError("--probe-capsules-root must be an absolute path")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("selected probe root is not a directory")
    names = (screen, timing, prohibited_probe)
    if any(
        name in (".", "..", "all") or Path(name).name != name or "," in name or any(c.isspace() for c in name)
        for name in names
    ):
        raise ValueError("probe names must each select one direct capsule member")
    pins = {}
    policies = []
    for name in names:
        member = path / name / "capsule.yaml"
        if not member.is_file() or member.resolve().parent.parent != path:
            raise ValueError("selected probe capsule is absent or outside its root")
        data = member.read_bytes()
        capsule = yaml.safe_load(data)
        if not isinstance(capsule, dict) or capsule.get("name") != name:
            raise ValueError("selected probe capsule name differs from its member")
        policy = capsule
        for key in ("performance", "arms", "candidate", "instruction_policy"):
            policy = policy.get(key) if isinstance(policy, dict) else None
        if not isinstance(policy, dict):
            raise ValueError("each explicit probe must declare its prohibited instruction roles")
        roles = policy.get("prohibited_instruction_roles")
        if (
            not isinstance(roles, list)
            or not roles
            or any(not isinstance(role, str) or not role.strip() or "," in role for role in roles)
            or len(set(roles)) != len(roles)
        ):
            raise ValueError("each explicit probe must declare its prohibited instruction roles")
        policies.append(tuple(sorted(roles)))
        pins[member] = hashlib.sha256(data).hexdigest()
    if len(set(policies)) != 1:
        raise ValueError("explicit probe capsules select different prohibited instruction roles")
    incorrect, _ = select_reference_backend(incorrect_backend, target=target)
    prohibited, _ = select_reference_backend(prohibited_backend, target=target)
    for package in (incorrect, prohibited):
        member = package / "manifest.yaml"
        pins[member] = hashlib.sha256(member.read_bytes()).hexdigest()
    return {
        "root": path,
        "screen": screen,
        "timing": timing,
        "incorrect_backend": incorrect,
        "prohibited_backend": prohibited,
        "prohibited_probe": prohibited_probe,
        "timeout_s": timeout_s,
        "timing_output": output,
        "resource_root": resource_root,
        "pins": pins,
    }


def verify_timing_output(path: Path, *, resource_root: Path) -> None:
    if (
        not path.is_absolute()
        or path.exists()
        or any(part.is_symlink() for part in (path, *path.parents))
        or not path.parent.is_dir()
        or path.resolve().is_relative_to(resource_root.resolve())
    ):
        raise ValueError("oracle timing output must be a fresh absolute artifact outside descriptor resources")


def verify_probe_inputs(selection: dict) -> None:
    for member, digest in selection["pins"].items():
        if not member.is_file() or hashlib.sha256(member.read_bytes()).hexdigest() != digest:
            raise ValueError("selected probe capsule or package manifest changed")


def probe_row(report: dict, *, capsule: str, sim: str, returncodes: tuple[int, ...] = (0, 1)) -> dict:
    """Require the actual child report to name exactly the requested single observation."""
    rows = report.get("per_capsule") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or "error" in report
        or type(report.get("n_capsules")) is not int
        or report["n_capsules"] != 1
        or report.get("sim") != sim
        or type(report.get("_readiness_returncode")) is not int
        or report["_readiness_returncode"] not in returncodes
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], dict)
        or rows[0].get("capsule") != capsule
        or type(rows[0].get("pass")) is not bool
        or type(report.get("n_passed")) is not int
        or report["n_passed"] != int(rows[0]["pass"])
        or type(report.get("all_pass")) is not bool
        or report["_readiness_returncode"] != (0 if report["all_pass"] else 1)
    ):
        raise ValueError("oracle probe has no complete requested child observation")
    numeric = rows[0].get("numeric")
    if (
        rows[0]["pass"]
        and numeric is not None
        and (
            not isinstance(numeric, dict) or numeric.get("status") != "pass" or numeric.get("missing_outputs", []) != []
        )
    ):
        raise ValueError("passed oracle probe reports incomplete numerical outputs")
    return rows[0]


def rejected_probe(report: dict, *, capsule: str, sim: str, category: str) -> bool:
    row = probe_row(report, capsule=capsule, sim=sim, returncodes=(1,))
    failure = row.get("failure") or {}
    if (
        report.get("all_pass") is not False
        or type(report.get("n_passed")) is not int
        or report["n_passed"] != 0
        or row["pass"] is not False
        or not isinstance(failure, dict)
        or failure.get("category") != category
    ):
        return False
    if category == "PROHIBITED_INSTRUCTION":
        hits = failure.get("prohibited_instructions")
        return failure.get("plane") == "instruction_policy" and isinstance(hits, dict) and bool(hits)
    numeric = row.get("numeric") or {}
    return (
        category == "FUNCTIONAL_MISMATCH"
        and isinstance(numeric, dict)
        and numeric.get("status") == "fail"
        and type(numeric.get("mismatch_count")) is int
        and numeric["mismatch_count"] > 0
        and numeric.get("missing_outputs") == []
        and row.get("barrier_tier") == "L3"
        and row.get("barrier_status") == "fail"
    )
