"""Build-artifact observation on a retained copy, never on a published payload.

This is not numerical certification, dependency isolation, or portable-build proof.
Commands are explicitly requested trusted-host execution using the existing runtime.
"""

import shutil
from pathlib import Path
from typing import Any

from merlin.common.jsonio import write_pretty_json

from .package_records import payload_inventory


def verify_export(repo: Path, stage: Path, *, timeout: int) -> dict[str, Any]:
    from .package_runtime import build_package, integrity_scan, load_package

    if type(timeout) is not int or timeout <= 0:
        raise ValueError("verification timeout must be a positive integer")

    receipt: dict[str, Any] = {
        "version": 1,
        "status": "fail",
        "scope": "build-artifact-presence",
        "numerical_certification": "not-performed",
        "external_dependency_closure": "not-attested",
        "timeout_per_step_s": timeout,
    }
    try:
        before = payload_inventory(repo)
        receipt["exported_input"] = before
        parent = stage / "verification"
        parent.mkdir()
        copied = parent / "package"
        shutil.copytree(repo, copied)
        receipt["execution_path"] = str(copied)
        if payload_inventory(copied) != before:
            raise ValueError("export changed while making verification copy")
        pkg = load_package(copied)
        if not pkg.tool.resolve().is_relative_to(copied.resolve()):
            raise ValueError("verification tool escapes package copy")
        integrity_scan(pkg)
        receipt["build_invoked"] = bool(pkg.manifest.get("build"))
        build_package(pkg, timeout=timeout)
        if not pkg.tool.is_file():
            raise ValueError("declared tool absent after build")
        receipt["built_execution"] = payload_inventory(copied)
        if payload_inventory(repo) != before:
            raise ValueError("published payload changed during verification")
        receipt["status"] = "pass"
    except Exception as exc:
        receipt["error"] = {"type": type(exc).__name__, "detail": str(exc)}
    write_pretty_json(stage / "build_verification.json", receipt)
    return receipt
