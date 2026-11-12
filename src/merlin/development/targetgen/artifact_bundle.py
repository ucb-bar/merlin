"""Profile-bound structural artifact bundles; no executor or certification.

Target providers own artifact ABIs, memory aliasing/capabilities and the proof
that actual synchronization realizes declared dependencies. File and graph
checks do not establish those facts or authorize execution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.digest import is_sha256, sha256_file
from merlin.development.targetgen.contract import schemas


class BundleError(ValueError):
    """A bundle is incomplete or makes an unsupported structural claim."""


@dataclass(frozen=True)
class SelectedArtifactProfile:
    """Caller-selected immutable profile identity, independently of the bundle."""

    path: Path
    name: str
    sha256: str
    bytes: int

    @classmethod
    def capture(cls, path: Path) -> SelectedArtifactProfile:
        path = Path(path).resolve()
        payload = path.read_bytes()
        try:
            data = yaml.safe_load(payload)
        except yaml.YAMLError as exc:
            raise BundleError("selected profile cannot be parsed") from exc
        if not isinstance(data, dict) or type(data.get("name")) is not str or not data["name"]:
            raise BundleError("selected profile requires a nonempty name")
        from merlin.common.digest import sha256_bytes

        return cls(path, data["name"], sha256_bytes(payload), len(payload))

    def validate(self) -> None:
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or type(self.name) is not str
            or not self.name
            or not is_sha256(self.sha256)
            or type(self.bytes) is not int
            or self.bytes <= 0
            or SelectedArtifactProfile.capture(self.path) != self
        ):
            raise BundleError("selected profile identity changed or is invalid")


def require_profile(profile: SelectedArtifactProfile) -> None:
    if type(profile) is not SelectedArtifactProfile:
        raise BundleError("structural artifact admission requires a typed selected profile")
    profile.validate()


def verify_bundle(
    document: dict[str, Any],
    root: Path,
    *,
    profile: SelectedArtifactProfile,
    contract: Path | None = None,
) -> None:
    """Check current profile/files and the complete declared buffer dependency DAG.

    Buffer IDs describe complete logical buffers. Partial writes, physical
    aliases, memory accessibility, entry ABIs and actual ordering are not proved
    by this structural carrier. There is currently no v0.2 execution adapter.
    """
    require_profile(profile)
    schemas.validate(document, "artifact_bundle", contract=contract)
    if document["profile"] != {"name": profile.name, "sha256": profile.sha256}:
        raise BundleError("bundle profile differs from the independently selected profile")
    root = Path(root).resolve()
    if not root.is_dir():
        raise BundleError("bundle root is not a directory")
    artifacts: set[str] = set()
    retained: list[tuple[Path, Path, str]] = []
    for artifact in document["artifacts"]:
        identity = artifact["id"]
        if identity in artifacts:
            raise BundleError(f"duplicate artifact {identity!r}")
        artifacts.add(identity)
        relative = Path(artifact["path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise BundleError(f"artifact {identity!r} has an unsafe path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise BundleError(f"artifact {identity!r} is missing or escapes the bundle")
        if sha256_file(path) != artifact["sha256"]:
            raise BundleError(f"artifact {identity!r} digest differs from its file")
        retained.append((relative, path, artifact["sha256"]))
    buffers: dict[str, bool] = {}
    for buffer in document["buffers"]:
        identity = buffer["id"]
        if identity in buffers:
            raise BundleError(f"duplicate buffer {identity!r}")
        buffers[identity] = buffer["initialized"]
    seen: dict[str, set[str]] = {}
    previous: list[dict[str, Any]] = []
    for step in document["steps"]:
        identity = step["id"]
        if identity in seen:
            raise BundleError(f"duplicate step {identity!r}")
        if step["artifact"] not in artifacts:
            raise BundleError(f"step {identity!r} names an absent artifact")
        after = step["after"]
        unknown = set(after) - seen.keys()
        if unknown:
            raise BundleError(f"step {identity!r} has unknown or forward dependencies {sorted(unknown)}")
        touched = set(step["reads"] + step["writes"])
        if touched - buffers.keys():
            raise BundleError(f"step {identity!r} names absent buffers {sorted(touched - buffers.keys())}")
        predecessors = set(after)
        for name in after:
            predecessors.update(seen[name])
        for buffer in step["reads"]:
            writers = [prior for prior in previous if buffer in prior["writes"]]
            if not buffers[buffer] and not writers:
                raise BundleError(f"step {identity!r} reads uninitialized buffer {buffer!r}")
        for prior in previous:
            hazard = (set(prior["writes"]) & touched) | (set(prior["reads"]) & set(step["writes"]))
            if hazard and prior["id"] not in predecessors:
                raise BundleError(
                    f"steps {prior['id']!r} and {identity!r} access {sorted(hazard)} without an order edge"
                )
        seen[identity] = predecessors
        previous.append(step)
    require_profile(profile)
    for relative, path, digest in retained:
        if (root / relative).resolve() != path or sha256_file(path) != digest:
            raise BundleError("artifact changed during structural verification")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify_bundle(
            json.loads(args.bundle.read_text()),
            args.bundle.parent,
            profile=SelectedArtifactProfile.capture(args.profile),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
