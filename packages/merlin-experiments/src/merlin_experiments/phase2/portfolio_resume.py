"""Retained portfolio verifier selection over independently sealed source ownership.

Launch records locate inputs; checkpoint evidence owns their identities. This module
builds frozen Python commands and retains the existing bounded subprocess supervisor.
It never substitutes current scientific interpretation for an archived verifier.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from merlin_experiments import frozen_python, source_snapshot

from . import contracts as C
from . import host_policy as HP

MODULE = "merlin_experiments.phase2.portfolio_resume"
SCHEMA = "merlin.portfolio-checkpoint-dispatch.v1"


@dataclass(frozen=True)
class NativeVerifierLayout:
    """Historical defaults supplied only by the native compatibility edge."""

    controller_relative: str
    python_roots: tuple[str, ...]
    module: str

    def __post_init__(self):
        for value in (self.controller_relative, *self.python_roots):
            path = PurePosixPath(value)
            if not value or path.is_absolute() or ".." in path.parts or str(path) != value or value == ".":
                raise ValueError("native verifier layout must contain canonical relative paths")
        if not self.module or any(not part.isidentifier() for part in self.module.split(".")):
            raise ValueError("invalid native verifier module")


@dataclass(frozen=True)
class ResolvedCheckpoint:
    snapshot: Path
    argv: tuple[str, ...]
    environment: dict[str, str]
    document: dict[str, Any]

    def command(self) -> list[str]:
        return frozen_python.python_command(self.snapshot, self.argv, verifier_source=Path(source_snapshot.__file__))


def _contained(path: str | Path, snapshot: Path, *, directory: bool = False) -> Path:
    if not isinstance(path, (str, Path)):
        raise ValueError("retained verifier input must name a path")
    value = Path(path)
    if (
        not value.is_absolute()
        or value.resolve() != value
        or value.is_symlink()
        or not value.is_relative_to(snapshot)
        or not (value.is_dir() if directory else value.is_file())
    ):
        raise ValueError("retained verifier input is linked, absent, noncanonical or outside its snapshot")
    return value


def _module_source(snapshot: Path, receipt: dict) -> Path:
    suffix = MODULE.replace(".", "/") + ".py"
    members = [snapshot / root / suffix for root in receipt["python_roots"] if f"{root}/{suffix}" in receipt["files"]]
    if len(members) != 1:
        raise ValueError("retained installed verifier lacks unique sealed module ownership")
    return _contained(members[0], snapshot)


def installed_dispatch(
    *,
    snapshot: Path,
    controller_source: Path,
    contract_root: Path,
    compiler_shared_source_root: Path,
) -> dict:
    """Record explicit archived selections for a new run; never rewrite old launch records."""
    snapshot = Path(snapshot).resolve()
    receipt = source_snapshot.verify(snapshot)
    return {
        "schema": SCHEMA,
        "kind": "installed",
        "module": MODULE,
        "source": str(_module_source(snapshot, receipt)),
        "controller_source": str(_contained(Path(controller_source).resolve(), snapshot)),
        "contract_root": str(_contained(Path(contract_root).resolve(), snapshot, directory=True)),
        "compiler_shared_source_root": str(
            _contained(Path(compiler_shared_source_root).resolve(), snapshot, directory=True)
        ),
    }


def _installed_inputs(document: dict, launch: dict) -> tuple[Path, dict, dict]:
    selection = launch["checkpoint_verifier"]
    if (
        not isinstance(selection, dict)
        or set(selection)
        != {"schema", "kind", "module", "source", "controller_source", "contract_root", "compiler_shared_source_root"}
        or selection.get("schema") != SCHEMA
        or selection.get("kind") != "installed"
        or selection.get("module") != MODULE
    ):
        raise ValueError("unsupported retained checkpoint verifier dispatch")
    recorded_snapshot = document.get("source_snapshot")
    if (
        not isinstance(recorded_snapshot, str)
        or not recorded_snapshot
        or launch.get("source_snapshot") != recorded_snapshot
    ):
        raise ValueError("retained checkpoint source snapshot differs from launch locator")
    snapshot = Path(recorded_snapshot)
    if not snapshot.is_absolute() or snapshot.resolve() != snapshot or snapshot.is_symlink():
        raise ValueError("retained checkpoint source snapshot differs from launch locator")
    receipt = source_snapshot.verify(snapshot)
    if C.document_sha256(receipt["files"]) != document.get("source_snapshot_files_sha256"):
        raise ValueError("retained checkpoint source snapshot membership differs")
    module = _module_source(snapshot, receipt)
    if selection["source"] != str(module):
        raise ValueError("retained verifier module differs from sealed ownership")
    policy = document.get("host_verification_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("schema") != HP.SCHEMA
        or not isinstance(policy.get("identities"), dict)
        or not isinstance(policy.get("sources"), dict)
    ):
        raise ValueError("installed retained verifier requires its recorded host ownership policy")
    identities, sources = policy["identities"], policy["sources"]

    def identity(role: str, path: Path):
        path = _contained(path, snapshot)
        relative = path.relative_to(snapshot).as_posix()
        digest = C.sha256_file(path)
        if (
            identities.get(role) != str(path)
            or sources.get(str(path)) != digest
            or receipt["files"].get(relative) != digest
        ):
            raise ValueError(f"retained verifier {role} ownership or bytes differ")

    identity("python/" + MODULE, module)
    identity("controller/global", Path(selection["controller_source"]))
    contract = _contained(selection["contract_root"], snapshot, directory=True)
    for relative in HP.RESOURCE_FILES:
        identity("resource/contract/" + relative, contract / relative)
    shared = _contained(selection["compiler_shared_source_root"], snapshot, directory=True)
    dependencies = document.get("compiler_dependencies")
    if not isinstance(dependencies, dict) or dependencies.get("shared_source_root") != str(shared):
        raise ValueError("retained compiler shared-source selection differs from checkpoint")
    if shared.relative_to(snapshot).as_posix() not in receipt["directories"]:
        raise ValueError("retained compiler shared-source root is outside sealed membership")
    return snapshot, selection, receipt


def resolve(original_path: Path, *, native_layout: NativeVerifierLayout | None = None) -> ResolvedCheckpoint:
    """Resolve only an explicitly recorded installed verifier or an explicit native layout."""
    original_path = Path(original_path).absolute()
    original = C.mapping_file(original_path)
    launch = C.mapping_file(original_path.parent.parent / "launch.json")
    if "checkpoint_verifier" in launch:
        snapshot, selection, receipt = _installed_inputs(original, launch)
        argv = (sys.executable, "-m", MODULE, "--checkpoint", str(original_path))
        roots = (*receipt["python_roots"], *receipt["legacy_roots"])
        environment = source_snapshot.provider_environment(snapshot, receipt)
    else:
        if native_layout is None:
            raise ValueError("historical checkpoint requires an explicit native verifier layout")
        snapshot = Path(launch["source_snapshot"])
        controller = snapshot / native_layout.controller_relative
        policy = original["host_verification_policy"]
        if policy.get("schema") in (HP.V2_SCHEMA, HP.SCHEMA) and policy.get("identities", {}).get(
            "controller/global"
        ) != str(controller.resolve()):
            raise ValueError("original checkpoint controller ownership differs from archived native dispatch")
        if policy["sources"].get(str(controller.resolve())) != C.sha256_file(controller):
            raise ValueError("original checkpoint verifier implementation changed")
        code = (
            "import sys; from pathlib import Path; import " + native_layout.module + " as G; "
            "G.consume_global_candidate(Path(sys.argv[1])); print('original_checkpoint_verified')"
        )
        argv = (sys.executable, "-c", code, str(original_path))
        roots, environment = native_layout.python_roots, {}
    return ResolvedCheckpoint(
        snapshot,
        argv,
        {
            **os.environ,
            **environment,
            "MERLIN_REPO_ROOT": str(snapshot),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(str(snapshot / item) for item in roots),
        },
        original,
    )


def verify(original_path: Path, *, native_layout: NativeVerifierLayout | None = None) -> dict[str, Any]:
    """Execute only the selected archived verifier using the existing frozen transport."""
    selected = resolve(original_path, native_layout=native_layout)
    checked = subprocess.run(
        selected.command(),
        env=selected.environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if checked.returncode != 0:
        raise ValueError("original immutable checkpoint no longer verifies: " + checked.stderr[-2000:])
    return selected.document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    # The fixed archived entrypoint refuses invocation without its guarded source context.
    from merlin_experiments.frozen_python import active_source_identity

    if active_source_identity() is None:
        raise ValueError("installed retained checkpoint consumption requires frozen Python execution")
    path = args.checkpoint.absolute()
    document = C.mapping_file(path)
    launch = C.mapping_file(path.parent.parent / "launch.json")
    snapshot, selection, _receipt = _installed_inputs(document, launch)
    if Path(__file__).resolve() != Path(selection["source"]):
        raise ValueError("current installed verifier cannot replace the archived verifier")
    from .portfolio_checkpoint import CheckpointVerificationContext, consume_global_candidate

    consume_global_candidate(
        path,
        context=CheckpointVerificationContext(
            host_policy=HP.build_record(
                controller_source=Path(selection["controller_source"]),
                contract_root=Path(selection["contract_root"]),
            ),
            compiler_shared_source_root=Path(selection["compiler_shared_source_root"]),
        ),
    )
    print(json.dumps({"original_checkpoint_verified": True, "source_snapshot": str(snapshot)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
