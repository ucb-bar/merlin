"""Independent runtime authority shared by functional and performance phases.

Runtime support is qualified from public hardware/tool sources and actual
execution controls. It is separate from the newly authored compiler, its frozen
optimization baseline, and the private handwritten final reference. Constructors
and serialized reports cannot issue this live capability.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from weakref import WeakKeyDictionary

from merlin.development.backend.optimization import contracts as C
from merlin.development.backend.optimization.components.experiment import RuntimeGrant
from merlin.development.backend.optimization.contracts import StageGateError, document_sha256, sha256_file

# ---- component_runtime_authority ---------------------------------------------------------------

_ISSUED = WeakKeyDictionary()

_ROLES = ("grade", "stage_verifier", "feature_provider", "cca_provider", "rtl_executor")


def _qualification_types():
    from merlin.development.backend.optimization.controls.measurement_qualification import (
        IndependentMeasurementQualification,
    )
    from merlin.development.backend.optimization.controls.runtime_qualification import (
        IndependentRuntimeQualification,
    )

    return IndependentRuntimeQualification, IndependentMeasurementQualification


@dataclass(frozen=True)
class IndependentRuntimeServices:
    """Actual ordinary grading and observation methods of independent support."""

    grade: Callable
    stage_verifier: Callable
    feature_provider: Callable | None = None
    cca_provider: Callable | None = None
    rtl_executor: Callable | None = None


def _callback_identity(callback):
    original = callback
    while isinstance(callback, partial):
        callback = callback.func
    owner = inspect.getsourcefile(callback) if callable(callback) else None
    if owner is None:
        raise StageGateError("independent runtime callback lacks an inspected source owner")
    path = Path(owner).resolve()
    function = callback.__func__ if inspect.ismethod(callback) else callback
    return (
        original,
        getattr(function, "__code__", None),
        path,
        sha256_file(path),
        getattr(function, "__module__", None),
        getattr(function, "__qualname__", None),
    )


@dataclass(frozen=True, eq=False)
class IndependentComponentRuntime:
    """Issued only after the fixed independent runtime qualifier executes."""

    qualification: object
    services: IndependentRuntimeServices
    hardware_intake: object
    target_descriptor: Path
    source_pins: tuple[tuple[Path, str], ...]
    qualified_roles: tuple[str, ...]
    qualification_sha256: str
    callbacks: tuple[tuple[str, tuple], ...]

    @property
    def sha256(self):
        return document_sha256(
            {
                "schema": "merlin.independent_component_runtime.v1",
                "qualification_sha256": self.qualification_sha256,
                "hardware_intake_sha256": self.hardware_intake.sha256,
                "target_descriptor_sha256": sha256_file(self.target_descriptor),
                "qualified_roles": self.qualified_roles,
                "source_pins": [(str(path), digest) for path, digest in self.source_pins],
                "callback_sources": [
                    (role, str(identity[2]), identity[3], identity[4], identity[5]) for role, identity in self.callbacks
                ],
            }
        )

    @property
    def grader(self):
        return self.services.grade

    @property
    def stage_verifier(self):
        return self.services.stage_verifier

    def verify(self, *, required_roles=()):
        admitted = _ISSUED.get(self)
        if admitted is None:
            raise StageGateError("component runtime requires independently evaluated live authority")
        if type(self.qualification) not in _qualification_types():
            raise StageGateError("component runtime qualification is not the fixed independent owner")
        self.qualification.verify()
        self.hardware_intake.verify()
        if (
            self.qualification.services is not self.services
            or self.qualification.hardware_intake is not self.hardware_intake
            or self.qualification.target_descriptor != self.target_descriptor
            or self.qualification.receipt_sha256 != self.qualification_sha256
            or tuple(self.qualification.source_pins) != self.source_pins
            or tuple(self.qualification.qualified_roles) != self.qualified_roles
        ):
            raise StageGateError("independent runtime selection changed after its evaluated controls")
        if type(self.services) is not IndependentRuntimeServices:
            raise StageGateError("independent runtime service contract changed")
        for path, digest in self.source_pins:
            if path.is_symlink() or path.resolve() != path or not path.is_file() or sha256_file(path) != digest:
                raise StageGateError("independent runtime source/tool bytes changed")
        for role, identity in self.callbacks:
            if _callback_identity(getattr(self.services, role)) != identity:
                raise StageGateError("independent runtime callback implementation changed")
        if admitted != self.sha256:
            raise StageGateError("independent runtime derivation/control identity changed")
        if not set(required_roles) <= set(self.qualified_roles):
            raise StageGateError(
                "independent runtime has unqualified requested roles: "
                + ",".join(sorted(set(required_roles) - set(self.qualified_roles)))
            )
        return self.sha256


def admit_independent_component_runtime(qualification):
    """Connect a live qualified public-source runtime; metadata alone refuses.

    The fixed qualifier owns independent RTL semantic derivation, source/build
    correspondence, namespace isolation and actual positive/negative execution
    controls. Structural intake does not discharge those runtime obligations.
    The qualifier may certify functional roles before timing/CCA roles are ready.
    """
    try:
        from merlin.development.testgen.intake.rtl import IndependentHardwareIntake

    except ImportError as error:
        raise StageGateError("independent runtime semantic/execution qualifier is unavailable") from error
    if type(qualification) not in _qualification_types():
        raise StageGateError("independent runtime admission requires a live evaluated qualifier")
    qualification.verify()
    if type(qualification.hardware_intake) is not IndependentHardwareIntake:
        raise StageGateError("independent runtime requires actual separately issued RTL intake")
    qualification.hardware_intake.verify()
    services = qualification.services
    roles = tuple(qualification.qualified_roles)
    if (
        type(services) is not IndependentRuntimeServices
        or len(roles) != len(set(roles))
        or not {"grade", "stage_verifier"} <= set(roles)
        or not set(roles) <= set(_ROLES)
    ):
        raise StageGateError("independent runtime qualification lacks ordinary grading and semantic witness roles")
    pins = tuple((Path(path), digest) for path, digest in qualification.source_pins)
    if not pins or len(dict(pins)) != len(pins):
        raise StageGateError("independent runtime needs complete distinct source/tool membership")
    callbacks = tuple((role, _callback_identity(getattr(services, role))) for role in roles)
    for _role, identity in callbacks:
        if (identity[2], identity[3]) not in pins:
            raise StageGateError("independent runtime callback source is absent from qualified membership")
    runtime = IndependentComponentRuntime(
        qualification,
        services,
        qualification.hardware_intake,
        Path(qualification.target_descriptor),
        pins,
        roles,
        qualification.receipt_sha256,
        callbacks,
    )
    _ISSUED[runtime] = runtime.sha256
    runtime.verify(required_roles=("grade", "stage_verifier"))
    return runtime


def require_independent_runtime(runtime, *, required_roles, target_descriptor):
    if type(runtime) is not IndependentComponentRuntime:
        raise StageGateError("component evaluation requires independent runtime/semantic measurement authority")
    runtime.verify(required_roles=required_roles)
    if runtime.target_descriptor != Path(target_descriptor):
        raise StageGateError("independent component runtime targets another descriptor")
    return runtime


# ---- component_runtime -------------------------------------------------------------------------
# Inventory explicit reviewed system runtime files for component launch inputs.
#
# This preparation command establishes file membership and byte identities only.
# It does not establish namespace support, compiler correctness or model readiness.


def _dependency_relocations(rows):
    """Validate explicit coordinate projections, without granting a source tree."""
    if not isinstance(rows, tuple):
        raise C.StageGateError("runtime dependency relocations require a closed immutable selection")
    selected = []
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 2 or not isinstance(row[0], Path):
            raise C.StageGateError("runtime dependency relocation requires an explicit source/destination pair")
        source, destination = row
        try:
            canonical = source.resolve(strict=True)
        except OSError as exc:
            raise C.StageGateError("runtime dependency relocation source is unavailable") from exc
        if (
            not source.is_absolute()
            or canonical != source
            or not source.is_dir()
            or any(path.is_symlink() for path in (source, *source.parents))
        ):
            raise C.StageGateError("runtime dependency relocation source is indirect or unavailable")
        RuntimeGrant.verify_destination(destination)
        target = PurePosixPath(destination)
        if any(
            source.is_relative_to(other_source)
            or other_source.is_relative_to(source)
            or target.is_relative_to(other_target)
            or other_target.is_relative_to(target)
            for other_source, other_target in selected
        ):
            raise C.StageGateError("runtime dependency relocation prefixes overlap or collide")
        selected.append((source, target))
    return tuple(selected)


def inventory_runtime(
    *,
    files: tuple[tuple[Path, str], ...],
    trees: tuple[tuple[Path, str], ...],
    executables: tuple[tuple[Path, str], ...],
    dependency_relocations: tuple[tuple[Path, str], ...] = (),
) -> tuple[RuntimeGrant, ...]:
    """Pin declared files and the actual dynamic-loader dependencies of trusted tools.

    Callers review the input executable owners before running ``ldd``. Extra
    resources (Python modules, certificates, simulator data) must be declared
    explicitly. Optional reviewed prefix projections apply only to dependencies
    actually reported by those trusted executables; they never inventory a SDK
    tree or establish independence. Relative dependency names are preserved so
    the explicitly relocated tool layout can still resolve its original RPATH.
    The launch's actual readiness probes detect incomplete closure.
    """
    relocations = _dependency_relocations(dependency_relocations)
    index: dict[str, tuple[Path, str]] = {}

    def add(source, destination):
        source = Path(source).resolve(strict=True)
        RuntimeGrant.verify_destination(destination)
        identity = source, C.sha256_file(source)
        previous = index.get(destination)
        if previous is not None and previous != identity:
            raise C.StageGateError("runtime inventory has conflicting destination identities")
        index[destination] = identity

    for source, destination in files:
        add(source, destination)
    for root, destination in trees:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise C.StageGateError("runtime tree is not an explicit reviewed directory")
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path, str(Path(destination) / path.relative_to(root)))
    for executable, destination in executables:
        add(executable, destination)
        result = subprocess.run(
            ["ldd", str(Path(executable).resolve(strict=True))],
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        # ldd's ordinary diagnostic for a trusted statically linked tool is
        # valid. Any unresolved dynamic dependency remains an inventory error.
        if "not found" in result.stdout:
            raise C.StageGateError("trusted runtime executable has an unresolved dynamic dependency")
        if result.returncode and not any(
            text in result.stderr + result.stdout for text in ("not a dynamic executable", "statically linked")
        ):
            raise C.StageGateError("trusted runtime executable dependency inspection failed")
        for line in result.stdout.splitlines():
            # ldd names dependency paths as whitespace-delimited absolute
            # tokens; no target/tool names or pattern parsing are needed.
            for token in line.split():
                if token.startswith("/") and Path(token).is_file():
                    # Resolve parent coordinates while keeping the requested
                    # alias basename, which the dynamic loader still needs.
                    selected_path = Path(token).parent.resolve(strict=True) / Path(token).name
                    lexical_path = Path(os.path.normpath(token))
                    destination = str(lexical_path)
                    for source_root, destination_root in relocations:
                        if selected_path.is_relative_to(source_root) or lexical_path.is_relative_to(source_root):
                            if not selected_path.is_relative_to(source_root) or not selected_path.resolve(
                                strict=True
                            ).is_relative_to(source_root):
                                raise C.StageGateError(
                                    "projected runtime dependency escaped its selected source prefix"
                                )
                            destination = str(destination_root / selected_path.relative_to(source_root))
                    add(Path(token), destination)
    # No grant is constructed until the entire selected inventory has been
    # checked for identity and file/directory destination collisions.
    destinations = sorted(index)
    for destination in destinations:
        if any(str(parent) in index for parent in PurePosixPath(destination).parents):
            raise C.StageGateError("runtime inventory destinations collide as file and directory")
    grants = tuple(RuntimeGrant(index[name][0], name, index[name][1]) for name in destinations)
    for grant in grants:
        grant.verify()
    return grants


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("file", "tree", "executable", "dependency-relocation"):
        parser.add_argument("--" + flag, action="append", default=[], metavar="SOURCE=DESTINATION")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def values(name):
        result = []
        for value in getattr(args, name):
            source, separator, destination = value.partition("=")
            if not separator or not Path(source).is_absolute() or not Path(destination).is_absolute():
                parser.error("runtime membership requires absolute SOURCE=DESTINATION")
            result.append((Path(source), destination))
        return tuple(result)

    if args.output.exists() or args.output.is_symlink():
        parser.error("runtime inventory destination must be fresh")
    grants = inventory_runtime(
        files=values("file"),
        trees=values("tree"),
        executables=values("executable"),
        dependency_relocations=values("dependency_relocation"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            [{"source": str(row.source), "destination": row.destination, "sha256": row.sha256} for row in grants],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
