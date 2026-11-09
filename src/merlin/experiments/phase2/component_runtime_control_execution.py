"""Actual source-sealed subprocess execution of private runtime controls.

This trusted control compiler contains no target ISA or workload implementation.
Author-generated candidates must use their ordinary isolated executor instead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record
from merlin.targetgen import package_runtime

from .contracts import StageGateError, exact_tree_record


def _plain(path):
    path = Path(path).absolute()
    if path.resolve() != path or any(part.is_symlink() for part in (path, *path.parents)):
        raise StageGateError("private control executor requires canonical unlinked paths")
    return path


@dataclass(frozen=True)
class PrivateRuntimeControlExecutor:
    candidate: Path
    evidence_root: Path
    candidate_sha256: str

    def build_package(self, pkg, *, timeout=1800):
        if (
            pkg.directory != self.candidate
            or pkg.manifest.get("build")
            or exact_tree_record(self.candidate)["sha256"] != self.candidate_sha256
        ):
            raise StageGateError("private primitive compiler differs from its prepared source")

    def run_entrypoint(
        self,
        pkg,
        name,
        input_mlir,
        output_json=None,
        *,
        timeout=600,
        write_bytecode=False,
        artifact_profile=None,
        invocation_directory=None,
    ):
        self.build_package(pkg)
        if (
            write_bytecode
            or artifact_profile is not None
            or invocation_directory is None
            or not _plain(input_mlir).is_relative_to(self.evidence_root)
            or not _plain(invocation_directory).is_relative_to(self.evidence_root)
            or output_json is not None
            and not _plain(output_json).is_relative_to(self.evidence_root)
        ):
            raise StageGateError("private control invocation escapes its prepared evidence owner")
        argv = package_runtime._resolve_argv(pkg, name, input_mlir, output_json)
        if argv[0] != "python3":
            raise StageGateError("private primitive compiler has an unprepared executable")
        # Resolving a venv interpreter symlink changes its import environment.
        # Preserve the actually selected invocation path; the recorder pins the
        # underlying executable bytes separately. No runtime authority follows.
        argv = [sys.executable, "-I", "-B", *argv[1:]]
        result = invocation_record.run(
            argv,
            directory=invocation_directory,
            stage=name,
            inputs=(Path(input_mlir),),
            outputs=(Path(output_json),) if output_json is not None else (),
            dependencies=tuple(path for path in self.candidate.rglob("*") if path.is_file()),
            cwd=self.candidate,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        self.build_package(pkg)
        return result
