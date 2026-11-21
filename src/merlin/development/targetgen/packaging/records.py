"""Exact package payload inventories and external publication bookkeeping.

An inventory identifies bytes, not an executed certification input. Historical
receipts without producer binding never acquire authority through this module.
The filesystem must be trusted and stable during inventory/publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.digest import canonical_json, canonical_sha256, is_sha256, sha256_file, write_pretty_json
from merlin.common.paths import env, repo_root
from merlin.target.contract.interfaces import abi as schemas

# ---- package_records ---------------------------------------------------------------------------


def component(value: str) -> str:
    if not value or value in {".", "..", ".publication"} or Path(value).name != value or "\\" in value:
        raise ValueError(f"expected a single path component: {value!r}")
    return value


def safe_path(path: Path) -> Path:
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink path component: {path}")
    return path


def payload_inventory(root: Path) -> dict[str, Any]:
    """Bind all regular members, empty directories and executable permission bits."""
    root = safe_path(root)
    if not root.is_dir():
        raise ValueError(f"payload is not a directory: {root}")
    members = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        entry: dict[str, Any] = {"path": path.relative_to(root).as_posix()}
        if stat.S_ISDIR(mode):
            entry.update(kind="directory", executable=mode & 0o111)
        elif stat.S_ISREG(mode):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            entry.update(kind="file", executable=mode & 0o111, sha256=digest)
        else:
            raise ValueError(f"non-regular payload member: {path}")
        members.append(entry)
    body = {"version": 1, "root_executable": root.stat().st_mode & 0o111, "members": members}
    return {**body, "sha256": canonical_sha256(body)}


def record_path(package: Path) -> Path:
    component(package.name)
    component(package.parent.name)
    return safe_path(package.parent / ".publication" / f"{package.name}.json")


def valid_inventory(value: Any) -> bool:
    """Validate the exact inventory vocabulary emitted by :func:`payload_inventory`."""
    if not isinstance(value, dict) or set(value) != {"version", "root_executable", "members", "sha256"}:
        return False
    if type(value["version"]) is not int or value["version"] != 1 or not isinstance(value["members"], list):
        return False

    def mode(bits):
        return type(bits) is int and bits >= 0 and bits & ~0o111 == 0

    def digest(text):
        return isinstance(text, str) and len(text) == 64 and all(c in "0123456789abcdef" for c in text)

    if not mode(value["root_executable"]) or not digest(value["sha256"]):
        return False
    seen = {}
    for member in value["members"]:
        if not isinstance(member, dict):
            return False
        kind = member.get("kind")
        if not isinstance(kind, str):
            return False
        expected = {"path", "kind", "executable"} | ({"sha256"} if kind == "file" else set())
        if kind not in {"file", "directory"} or set(member) != expected or not mode(member["executable"]):
            return False
        name = member["path"]
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            return False
        path = Path(name)
        if path.is_absolute() or path.as_posix() != name or ".." in path.parts or name == "." or name in seen:
            return False
        if any(parent.as_posix() != "." and seen.get(parent.as_posix()) != "directory" for parent in path.parents):
            return False
        if kind == "file" and not digest(member["sha256"]):
            return False
        seen[name] = kind
    if list(seen) != sorted(seen, key=Path):
        return False
    return value["sha256"] == canonical_sha256({k: v for k, v in value.items() if k != "sha256"})


def bound_inputs(identity: Any, payload: dict[str, Any], compiler_package_id: Any) -> bool:
    """Validate a producer's scoped input commitment; never invent absent evidence."""
    if not isinstance(identity, dict):
        return False
    if (
        type(identity.get("version")) is not int
        or identity.get("version") != 1
        or identity.get("scope") != "package-payload"
        or identity.get("external_dependency_closure") != "not-attested"
        or identity.get("compiler_package_id") != compiler_package_id
        or identity.get("source") != payload
        or not isinstance(identity.get("environment"), dict)
        or not isinstance(identity.get("execution_path"), str)
    ):
        return False
    execution = identity.get("execution")
    if not valid_inventory(execution):
        return False
    tool = identity.get("tool")
    if not isinstance(tool, str) or not tool or Path(tool).is_absolute() or ".." in Path(tool).parts:
        return False
    return any(
        isinstance(member, dict) and member.get("path") == tool and member.get("kind") == "file"
        for member in execution["members"]
    )


def read_record(package: Path) -> dict[str, Any] | None:
    safe_path(package)
    path = record_path(package)
    if not path.exists():
        return None
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"publication record is not a regular file: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or type(data.get("version")) is not int or data.get("version") != 1:
        raise ValueError(f"invalid publication record: {path}")
    if data.get("target") != package.parent.name or data.get("package_slot") != package.name:
        raise ValueError(f"publication record identity mismatch: {path}")
    if not isinstance(data.get("publication"), dict):
        raise ValueError(f"invalid publication metadata: {path}")
    if data.get("payload") != payload_inventory(package):
        raise ValueError(f"publication payload drift: {package}")
    return data


def write_record(package: Path, manifest: dict[str, Any], publication: dict[str, Any], **metadata: Any) -> None:
    path = record_path(package)
    # Ordinary bookkeeping preserves installation provenance. A replacement
    # installation supplies its own metadata and must not inherit old claims.
    if not metadata and path.exists():
        previous = read_record(package)
        metadata = {k: v for k, v in previous.items() if k not in {"publication", "payload"}}
    document = {
        **metadata,
        "version": 1,
        "target": package.parent.name,
        "package_slot": package.name,
        "compiler_package_id": manifest.get("package_id"),
        "payload": payload_inventory(package),
        "publication": publication,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_json(document) + b"\n")
    os.replace(temporary, path)


# ---- artifact_bundle ---------------------------------------------------------------------------
# Profile-bound structural artifact bundles; no executor or certification.
#
# Target providers own artifact ABIs, memory aliasing/capabilities and the proof
# that actual synchronization realizes declared dependencies. File and graph
# checks do not establish those facts or authorize execution.


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


# ---- artifact_scale ----------------------------------------------------------------------------
# A program is as long as its commands, not as long as its data.
#
# An emitted artifact that grows with the payload is a lowering defect the harness used to find the
# hard way. Measured on the first campaign graded against a model's own layers: a candidate lowered a
# convolution by unrolling its host gather at compile time, one ``getelementptr``/``store`` pair per
# element, and emitted 71 MB to 768 MB of LLVM-dialect text per layer, against a median of 109 KB
# and a maximum of 12.5 MB over the 97 graded capsules. The grader then parsed that text for its
# trace decode and audits, inside the agent's single self-check slot, for 76 minutes; every other
# self-check the agent asked for in that time was starved, and nothing told it why.
#
# So the size is checked where the artifact arrives, before anything reads it twice, and the refusal
# says what the usual cause is. The limit is a statement about what this harness will analyse, not
# about any target: it sits five times above the largest artifact a passing program has needed, and
# ``MERLIN_MAX_ARTIFACT_BYTES`` moves it for a corpus whose honest programs are larger.

LIMIT_ENV = "MERLIN_MAX_ARTIFACT_BYTES"
#: Five times the largest artifact any capsule of the graded corpus has needed (12.5 MB).
DEFAULT_LIMIT_BYTES = 64 * 1024 * 1024


def limit_bytes() -> int:
    raw = os.environ.get(LIMIT_ENV, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_LIMIT_BYTES


def refusal(artifact_text: str, *, elements: int | None = None) -> str | None:
    """Why this emitted artifact is not analysed, or ``None`` when it is an ordinary size."""
    size, limit = len(artifact_text), limit_bytes()
    if size <= limit:
        return None
    per_element = f", about {size // elements} bytes for each of its {elements} payload elements" if elements else ""
    return (
        f"the emitted artifact is {size / 2**20:.0f} MiB{per_element}; this harness analyses at most "
        f"{limit / 2**20:.0f} MiB. A program this long has almost always been unrolled over its DATA "
        f"(one load or store per element, a gather written out at compile time). Emit a loop, or the "
        f"unit's own command for the movement, so the program grows with its commands and not with "
        f"its payload."
    )


# ---- publication_verification ------------------------------------------------------------------
# Build-artifact observation on a retained copy, never on a published payload.
#
# This is not numerical certification, dependency isolation, or portable-build proof.
# Commands are explicitly requested trusted-host execution using the existing runtime.


def verify_export(repo: Path, stage: Path, *, timeout: int) -> dict[str, Any]:
    from merlin.development.targetgen.packaging.loading import build_package, integrity_scan, load_package

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


# ---- provenance --------------------------------------------------------------------------------
# What a target-evolution result rests on, captured as data before the result is produced.
#
# An evolution experiment claims "we reused N% of a CERTIFIED parent compiler". That claim is only
# meaningful next to an exact statement of which parent, built from which hardware sources, with which
# toolchain, graded on which simulator binary. Two runs a week apart can disagree entirely because a
# sibling RTL checkout moved branch, and without this record the disagreement is unattributable.
#
# Three rules this module exists to obey:
#
# * **Ask, never assume.** Every field is read from the thing itself -- ``git rev-parse`` in the RTL
#   checkout, ``--version`` from the compiler, ``stat`` on the simulator binary, the champion's own
#   manifest. Nothing is a literal here.
# * **Degrade honestly.** A source that cannot be resolved records ``{"available": False, "reason": ...}``
#   and the run continues. It never raises and it never silently omits the key, because an absent key
#   reads as "not applicable" when what happened was "we could not tell".
# * **Name no target.** Which RTL trees, which simulator configs and which dtype strategy matter is the
#   caller's (i.e. the experiment descriptor's) knowledge; this module takes them as parameters.
#
# Paths to sibling checkouts are resolved from the environment (``MERLIN_CHIPYARD`` and friends) rather
# than hardcoded, so a record made on one machine states where its facts came from without embedding
# one developer's filesystem in the library.

_SHA_LEN = 12


def content_sha(obj: Any) -> str:
    """Stable content hash of a JSON-able record.

    Deliberately the same recipe as ``targetgen.rtl_check_runner._facts_sha`` -- sorted-keys JSON, so
    a dict that means the same thing hashes the same regardless of construction order. This is the
    hash an invalidation decision may rest on, so it hashes CONTENT, not identity (contrast
    ``publish._fingerprint``, which composes ids and therefore cannot notice that the facts moved
    under a stable package id).
    """
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:_SHA_LEN]


def _git(root: Path, *args: str) -> str | None:
    """One ``git`` query in ``root``, or None when it fails for any reason."""
    try:
        out = subprocess.run(("git", "-C", str(root), *args), capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_provenance(root: str | Path) -> dict[str, Any]:
    """``{available, path, head, branch, dirty}`` for any git checkout.

    ``dirty`` is load-bearing for an experiment record: a result produced from a tree with
    uncommitted changes is not reproducible from its sha, and saying so is cheaper than discovering
    it later.
    """
    from merlin.common import provenance as PROV
    from merlin.common.provenance_claims import UNKNOWN

    return PROV.scoped_observation(f"git_provenance:{Path(root)}", lambda: _git_provenance_now(root))


def _git_provenance_now(root: str | Path) -> dict[str, Any]:
    p = Path(root)
    if not p.is_dir():
        return {"available": False, "reason": f"not a directory: {p}"}
    head = _git(p, "rev-parse", "HEAD")
    if head is None:
        return {"available": False, "reason": f"not a git checkout: {p}"}
    status = _git(p, "status", "--porcelain")
    return {
        "available": True,
        "path": str(p),
        "head": head,
        "branch": _git(p, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def _submodule_pins(root: Path) -> dict[str, str]:
    """``{submodule path: sha}`` parsed structurally from ``git submodule status``.

    Lines are ``[+-U]?<sha> <path> (<describe>)``; split on whitespace rather than pattern-matching,
    and drop the leading state marker by character test so an out-of-date (``+``) or uninitialized
    (``-``) submodule is still recorded with its sha.
    """
    out = _git(root, "submodule", "status")
    pins: dict[str, str] = {}
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        sha = parts[0]
        if sha and not sha[0].isalnum():  # strip the +/-/U state marker
            sha = sha[1:]
        pins[parts[1]] = sha
    return pins


# --- merlin's own revision ---------------------------------------------------------------------------
#: The file a source export (``git archive`` of one commit, e.g. ``fresh_campaign export``) writes at its
#: root to say which commit it is. An export has no ``.git`` of its own.
SOURCE_RECORD = ".merlin-source-revision.json"
SOURCE_RECORD_SCHEMA = "merlin.source_revision.v1"


def _full_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) in (40, 64) and all(c in "0123456789abcdef" for c in value)


def write_source_record(root: str | Path, revision: str) -> Path:
    """Record at ``root`` (an export of ``revision``) which commit its bytes are. Returns the file."""
    if not _full_sha(revision):
        raise ValueError(f"not a full commit sha: {revision!r}")
    path = Path(root) / SOURCE_RECORD
    path.write_text(json.dumps({"schema": SOURCE_RECORD_SCHEMA, "revision": revision}, sort_keys=True) + "\n")
    return path


def _recorded_revision(root: Path) -> tuple[bool, str | None]:
    """``(record present, revision or None)`` from ``root``'s own source record."""
    path = root / SOURCE_RECORD
    if not path.is_file():
        return False, None
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return True, None
    revision = doc.get("revision") if isinstance(doc, dict) and doc.get("schema") == SOURCE_RECORD_SCHEMA else None
    return True, (revision if _full_sha(revision) else None)


def _own_checkout(root: Path) -> bool:
    """Whether ``root`` is itself the top of a git checkout, not a directory inside someone else's.

    ``git -C <dir>`` walks up to the nearest enclosing ``.git``. For a frozen export placed under a live
    worktree that is the WRONG repository: its HEAD moves with every commit on the enclosing branch while
    the export's bytes do not, so a revision read that way describes a tree nobody ran.
    """
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None:
        return False
    try:
        return Path(top).resolve() == root.resolve()
    except OSError:
        return False


def source_revision(root: str | Path | None = None) -> str:
    """The commit merlin's own code at ``root`` (default: :func:`repo_root`) was taken from, or UNKNOWN.

    An export's own :data:`SOURCE_RECORD` first; else ``HEAD`` of a checkout whose top IS ``root``;
    else UNKNOWN. Never an enclosing repository's ``HEAD`` (see :func:`_own_checkout`), and a malformed
    record is UNKNOWN rather than a fallback to git -- a record that cannot be read is not evidence that
    some other tree is the right one.
    """
    from merlin.common import provenance as PROV
    from merlin.common.provenance_claims import UNKNOWN

    root = Path(root) if root is not None else repo_root()
    present, revision = _recorded_revision(root)
    if present:
        return revision or UNKNOWN
    if _own_checkout(root):
        return git_provenance(root).get("head") or UNKNOWN
    return UNKNOWN


def repo_provenance() -> dict[str, Any]:
    """Merlin's own state: head, branch, dirty, and every submodule pin.

    An export (no ``.git`` of its own) is described by its source record, never by the checkout that
    happens to enclose it.
    """
    root = repo_root()
    present, revision = _recorded_revision(root)
    if present:
        if revision is None:
            return {"available": False, "reason": f"unreadable source record: {root / SOURCE_RECORD}"}
        # `git archive` of a committed tree; the exporter refuses uncommitted edits.
        return {
            "available": True,
            "path": str(root),
            "head": revision,
            "branch": None,
            "dirty": False,
            "source": SOURCE_RECORD,
        }
    if not _own_checkout(root):
        return {"available": False, "reason": f"not a git checkout of its own and no {SOURCE_RECORD}: {root}"}
    rec = git_provenance(root)
    if rec.get("available"):
        rec["submodules"] = _submodule_pins(root)
    return rec


def _tool_version(exe: Path, flag: str = "--version") -> str | None:
    """First line of ``exe --version``, or None. Used for compilers, where the first line carries
    the version and the rest is build configuration."""
    try:
        out = subprocess.run((str(exe), flag), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def toolchain_provenance() -> dict[str, Any]:
    """The compiler the binaries were built with, asked for its own version.

    Resolved through ``llvmlower.toolchain`` so the recorded path is the path that will actually be
    used (``MERLIN_CLANG`` overrides an in-repo install, and a stale override is exactly the kind of
    thing this record exists to expose).
    """
    rec: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        from merlin.compiler.lower.model import toolchain as tc
    except Exception as exc:  # noqa: BLE001
        rec["clang"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        return rec
    for label, getter in (("clang", tc.clang), ("mlir_translate", tc.mlir_translate)):
        try:
            path = getter()
        except Exception as exc:  # noqa: BLE001
            rec[label] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        if not Path(path).is_file():
            rec[label] = {"available": False, "reason": f"missing: {path}"}
            continue
        rec[label] = {"available": True, "path": str(path), "version": _tool_version(Path(path))}
    return rec


def rtl_provenance(sources: Mapping[str, str | Path]) -> dict[str, Any]:
    """Per-name git state of the hardware sources a target's facts were derived from.

    ``sources`` maps a caller-chosen label to a checkout path (typically assembled from an
    environment root plus a contract-declared subpath, so no tree name is baked in here). A hardware
    fact is only as pinned as the tree it was read from: an RTL checkout that switched branch is the
    single most common reason two "identical" runs disagree.
    """
    return {name: git_provenance(path) for name, path in sources.items()}


def simulator_provenance(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """``{label: {available, path, bytes, mtime, sha}}`` for grading binaries.

    A prebuilt simulator is an opaque artifact whose source config may no longer exist in the
    checkout that produced it, so identity has to come from the binary: size, mtime and a content
    hash of those two plus the name. Hashing the whole (tens of MB) binary is not worth the IO for a
    provenance line, and is noted here so the ``sha`` is not mistaken for a full digest.
    """
    out: dict[str, Any] = {}
    for label, path in paths.items():
        p = Path(path)
        if not p.is_file():
            out[label] = {"available": False, "reason": f"not built: {p}"}
            continue
        st = p.stat()
        stamp = {"name": p.name, "bytes": st.st_size, "mtime": int(st.st_mtime)}
        out[label] = {
            "available": True,
            "path": str(p),
            **stamp,
            "sha": content_sha(stamp),
            "sha_covers": "name+size+mtime, not file contents",
        }
    return out


def parent_provenance(target: str, *, dtype_strategy: str | None = None) -> dict[str, Any]:
    """The certified parent package an evolution run claims to have reused.

    Resolved through ``publish.select_champion`` -- the same selector the compiler itself uses -- so
    the recorded parent cannot drift from the compiled one. Records the publish fingerprint AND a
    content hash of the package's own codegen artifacts, because the fingerprint is a composition of
    ids (package_id + merlin sha + cert run) and so cannot notice an edited schedule under a stable
    id.
    """
    try:
        from merlin.development.targetgen.packaging import champion_selection as publish
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    try:
        sel = publish.select_champion(target, dtype_strategy=dtype_strategy)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"no champion for {target!r}: {exc}"}
    if sel is None:
        return {"available": False, "reason": f"no champion for {target!r}"}
    pub = (sel.manifest or {}).get("publication", {}) or {}
    files: dict[str, str] = {}
    pkg_dir = Path(sel.package_dir)
    for f in sorted(pkg_dir.rglob("*")):
        if f.is_file():
            files[str(f.relative_to(pkg_dir))] = hashlib.sha256(f.read_bytes()).hexdigest()[:_SHA_LEN]
    return {
        "available": True,
        "target": sel.target,
        "package_id": sel.package_id,
        "package_dir": str(pkg_dir),
        "family": sel.family,
        "layout_kind": sel.layout_kind,
        "status": sel.status,
        "version": sel.version,
        "fingerprint": pub.get("fingerprint"),
        "certification": pub.get("certification"),
        "certified_by_run": pub.get("certified_by_run"),
        "champion": pub.get("champion"),
        "artifact_shas": files,
        "content_sha": content_sha(files),
    }


def _env_root(var: str, subpath: str | None) -> str | None:
    """``$<var>/<subpath>`` when the variable is set, else None. Keeps sibling-checkout locations in
    the environment rather than in the library."""
    root = env(var)
    if not root:
        return None
    return str(Path(root) / subpath) if subpath else root


def record(
    target: str,
    *,
    dtype_strategy: str | None = None,
    rtl_sources: Mapping[str, str | Path] | None = None,
    simulators: Mapping[str, str | Path] | None = None,
    env_rtl_sources: Sequence[tuple[str, str, str | None]] = (),
    notes: str = "",
) -> dict[str, Any]:
    """The whole provenance record for one evolution run, with its own content hash.

    ``env_rtl_sources`` is a convenience for the common case: ``(label, ENV_VAR, subpath)`` triples
    resolved against the environment, so a descriptor can say "the hardware tree is
    ``$MERLIN_CHIPYARD/<its declared subpath>``" without the library knowing either half.

    The returned ``provenance_sha`` covers everything above it, so an experiment product can be
    compared to another by one field, and a mismatch points at the sub-record that moved.
    """
    sources: dict[str, str | Path] = dict(rtl_sources or {})
    for label, var, subpath in env_rtl_sources:
        resolved = _env_root(var, subpath)
        sources[label] = resolved if resolved else f"<unset: ${var}>"

    rec: dict[str, Any] = {
        "target": target,
        "merlin": repo_provenance(),
        "toolchain": toolchain_provenance(),
        "rtl": rtl_provenance(sources),
        "simulators": simulator_provenance(simulators or {}),
        "parent": parent_provenance(target, dtype_strategy=dtype_strategy),
        "notes": notes,
    }
    rec["provenance_sha"] = content_sha(rec)
    return rec


# --- which registry pins a target's results depend on ----------------------------------------------
#: The target-contract key naming the `src/merlin/resources/contracts/hardware_pins.yaml` entries a target's results
#: rest on. A target with no external RTL dependency legitimately declares none.
PINS_CONTRACT_KEY = "hardware_pins"


def declared_pins(target: str | None) -> tuple[str, ...]:
    """Pin names ``target``'s contract declares. Empty when it declares none or cannot be resolved.

    Deliberately not an error: a pure-simulation or host target has no external checkout to pin, and
    refusing would block provenance recording for exactly the targets that need it least.
    """
    if not target:
        return ()
    names: list[str] = []
    try:
        from merlin.target.registry.identity import resolve

        contract = resolve(target).load_contract() or {}
        names.extend(str(n) for n in (contract.get(PINS_CONTRACT_KEY) or ()))
    except Exception:  # noqa: BLE001 -- an unresolvable target contract states nothing
        pass
    # THE REGISTRY IS THE DURABLE HALF. A target contract is the natural place to declare a dependency,
    # but only one target's contract is a tracked source file -- the others are GENERATED under `out/`,
    # so a declaration added there is lost on the next regeneration. Measured before this: three of four
    # targets declared no pins at all, so `toolchain_shas` returned merlin's own commit and nothing else,
    # and every hardware-tier verdict they produced was recorded against no RTL revision. The pin
    # registry is tracked and reviewed, so a pin naming its targets there cannot be regenerated away.
    try:
        from merlin.common.provenance import load_pins

        for name, pin in load_pins().items():
            if target in pin.targets and name not in names:
                names.append(str(name))
    except Exception:  # noqa: BLE001 -- an unreadable registry adds nothing
        pass
    return tuple(names)


def toolchain_shas(target: str | None = None) -> dict[str, str]:
    """``{name: sha}`` for merlin plus every hardware pin ``target`` declares.

    Replaces a helper that named one target's checkouts directly (a chipyard root and a generator
    subdirectory), which meant every OTHER target's run record carried merlin's sha and nothing else --
    unattributable, and silently so. The shape is unchanged so existing run records stay comparable;
    what changed is that the entries come from the target's own declaration.

    A pin whose checkout is absent or unreadable is recorded as UNKNOWN rather than omitted: a missing
    key reads as "this run had no such dependency" when it means "nobody could tell".

    THIS IS A SHA, NOT A CITATION. The value here is the revision the checkout is ON, and it says
    nothing about whether the bytes that were read are that revision's. A report that prints it beside
    a verdict is spelling "pinned" for a tree that may not be; pair it with :func:`toolchain_citations`,
    which is the form :mod:`merlin.common.provenance` licenses a claim to use.
    """
    from merlin.common import provenance as PROV
    from merlin.common.provenance_claims import UNKNOWN

    # merlin's entry is the revision its OWN code was taken from (:func:`source_revision`): a frozen export
    # carries a source record, and an enclosing worktree's moving HEAD is never read in its place.
    shas = {"merlin": source_revision()}
    for name in declared_pins(target):
        try:
            shas[name] = PROV.verify(name).observed.commit or UNKNOWN
        except Exception:  # noqa: BLE001 -- unknown/unreadable pin is UNKNOWN, not absent
            shas[name] = UNKNOWN
    return shas


def toolchain_citations(target: str | None = None) -> dict[str, str]:
    """``{name: citation}`` for every hardware pin ``target`` declares — the spelling a claim may use.

    The sibling of :func:`toolchain_shas`, and the one a verdict-claiming report must carry. A bare sha
    cannot distinguish "measured on this commit" from "measured on this commit plus a handful of
    uncommitted lines", and both shapes have shipped in this repo's artifacts under 40 hex characters
    that look identical.

    Recorded as UNKNOWN, never omitted, for the same reason ``toolchain_shas`` does: an absent key reads
    as "no such dependency" when it means "nobody could tell".
    """
    from merlin.common import provenance as PROV
    from merlin.common.provenance_claims import UNKNOWN

    out: dict[str, str] = {}
    for name in declared_pins(target):
        try:
            out[name] = PROV.citation(name)
        except Exception:  # noqa: BLE001 -- unknown/unreadable pin is UNKNOWN, not absent
            out[name] = UNKNOWN
    return out


__all__ = [
    "SOURCE_RECORD",
    "content_sha",
    "declared_pins",
    "git_provenance",
    "parent_provenance",
    "record",
    "repo_provenance",
    "rtl_provenance",
    "simulator_provenance",
    "source_revision",
    "toolchain_provenance",
    "toolchain_shas",
    "write_source_record",
]
