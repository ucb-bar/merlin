"""Controller receipt protocol, source custody and detached issuance."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from merlin.common import digest as _mdigest
from merlin.common import jsonio as _mjson

CONTROLLER_ID = "merlin.compare.paper_measurement_controller_v2"

_ISSUANCE_KIND = "paper_controller_issuance_v1"

_HEX = frozenset("0123456789abcdef")

_REGISTRY_ADAPTERS = {
    "merlin_compile_v1": "merlin_compile",
    "executorch_v1": "executorch",
    "unit_test_v1": "unit_test",
}

_PRODUCTION_BUILD_ARGV = ["{tool}", "-O2", "-std=c11", "{source:runner}", "{source:model_object}", "-o", "{output}"]

_EXECUTORCH_PRODUCTION_BUILD_ARGV = ["verify_executorch_sealed_session", "{source:model_object}", "{output}"]

_ISSUED_RECEIPTS: dict[Path, tuple[str, str, str]] = {}

_sha_bytes = _mdigest.sha256_bytes

_FRAME_MAGIC = b"MRLNFRM1"


def _issuance_root(receipt_path: Path) -> Path:
    """Return the controller-owned root beside, but outside, a result bundle.

    Primary timing authenticity is rooted here rather than in mutually self-authored files in the
    bundle.  The detached Ed25519 signature lets a fresh verifier distinguish an original live
    observation from a bundle whose hashes were consistently rewritten.  Replay remains useful
    reproducibility evidence, but it is deliberately not the authenticity mechanism.
    """
    return receipt_path.parent.parent / ".paper-controller-issuance-v1"


def _openssl(*argv: str, timeout: float = 30) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(["/usr/bin/openssl", *argv], capture_output=True, timeout=timeout, check=False)
    if completed.returncode:
        raise ValueError(
            "controller issuance signature operation failed: " + completed.stderr.decode(errors="replace")[-500:]
        )
    return completed


@contextmanager
def _ephemeral_issuance_key(receipt_path: Path, identifier: str):
    """Create one issuance key and retain only its public half.

    The private key is placed in a mode-0700 temporary directory outside both the result bundle
    and issuance ledger.  The context removes it on success and on every exception path.  A key is
    never shared between runs, so a later process cannot sign another entry for this issuance.
    """
    entries = _issuance_root(receipt_path) / "entries"
    entries.mkdir(parents=True, exist_ok=True)
    public_path = entries / f"{identifier}.pub.pem"
    if public_path.exists():
        raise ValueError("controller issuance identifier was already used")
    with tempfile.TemporaryDirectory(prefix="merlin-paper-ephemeral-signing-") as temporary:
        os.chmod(temporary, 0o700)
        private = Path(temporary) / "private.pem"
        _openssl("genpkey", "-algorithm", "ED25519", "-out", str(private))
        os.chmod(private, 0o600)
        _openssl("pkey", "-in", str(private), "-pubout", "-out", str(public_path))
        try:
            yield private, public_path
        except BaseException:
            public_path.unlink(missing_ok=True)
            raise
        else:
            os.chmod(public_path, 0o444)
            _fsync(public_path)


def _issue_receipt(
    receipt_path: Path,
    receipt: Mapping[str, Any],
    raw_sha: str,
    contract_sha: str,
    *,
    private_key: Path,
    public_key: Path,
) -> None:
    issuance = receipt["issuance"]
    root = _issuance_root(receipt_path)
    entries = root / "entries"
    entries.mkdir(parents=True, exist_ok=True)
    identifier = str(issuance["id"])
    entry_path = entries / f"{identifier}.json"
    signature_path = entries / f"{identifier}.sig"
    if entry_path.exists() or signature_path.exists():
        raise ValueError("controller issuance identifier was already used")
    if _sha_file(public_key) != issuance["public_key_sha256"] or public_key != entries / f"{identifier}.pub.pem":
        raise ValueError("ephemeral issuance public key differs from receipt")
    entry = {
        "schema_version": 1,
        "kind": _ISSUANCE_KIND,
        "issuance_id": identifier,
        "created_at": datetime.now().astimezone().isoformat(),
        "receipt_sha256": _sha_file(receipt_path),
        "raw_measurement_sha256": raw_sha,
        "contract_sha256": contract_sha,
        "run_id": receipt["run_id"],
        "command_sha256": receipt["command_sha256"],
    }
    _write(entry_path, entry)
    _openssl(
        "pkeyutl", "-sign", "-inkey", str(private_key), "-rawin", "-in", str(entry_path), "-out", str(signature_path)
    )
    os.chmod(entry_path, 0o444)
    os.chmod(signature_path, 0o444)
    _fsync(signature_path)


def _validate_issuance(receipt_path: Path, receipt: Mapping[str, Any], raw_sha: str, contract_sha: str) -> str:
    issuance = _closed(
        receipt["issuance"],
        {
            "id",
            "kind",
            "public_key_sha256",
        },
        "controller issuance reference",
    )
    try:
        identifier = str(uuid.UUID(str(issuance["id"])))
    except ValueError as error:
        raise ValueError("controller issuance identifier is invalid") from error
    root = _issuance_root(receipt_path)
    entry_path = root / "entries" / f"{identifier}.json"
    signature_path = root / "entries" / f"{identifier}.sig"
    public = root / "entries" / f"{identifier}.pub.pem"
    if (
        issuance["kind"] != _ISSUANCE_KIND
        or not public.is_file()
        or public.is_symlink()
        or _sha_file(public) != issuance["public_key_sha256"]
        or not entry_path.is_file()
        or entry_path.is_symlink()
        or not signature_path.is_file()
        or signature_path.is_symlink()
    ):
        raise ValueError("controller immutable issuance root differs")
    _openssl(
        "pkeyutl",
        "-verify",
        "-pubin",
        "-inkey",
        str(public),
        "-rawin",
        "-in",
        str(entry_path),
        "-sigfile",
        str(signature_path),
    )
    entry = _closed(
        json.loads(entry_path.read_text(encoding="utf-8")),
        {
            "schema_version",
            "kind",
            "issuance_id",
            "created_at",
            "receipt_sha256",
            "raw_measurement_sha256",
            "contract_sha256",
            "run_id",
            "command_sha256",
        },
        "controller immutable issuance entry",
    )
    try:
        created = datetime.fromisoformat(str(entry["created_at"]))
    except ValueError as error:
        raise ValueError("controller issuance timestamp is invalid") from error
    if (
        entry["schema_version"] != 1
        or entry["kind"] != _ISSUANCE_KIND
        or entry["issuance_id"] != identifier
        or created.tzinfo is None
        or entry["receipt_sha256"] != _sha_file(receipt_path)
        or entry["raw_measurement_sha256"] != raw_sha
        or entry["contract_sha256"] != contract_sha
        or entry["run_id"] != receipt["run_id"]
        or entry["command_sha256"] != receipt["command_sha256"]
    ):
        raise ValueError("controller immutable issuance entry does not bind primary observation")
    return _canonical_sha(
        {
            "issuance_id": identifier,
            "public_key_sha256": _sha_file(public),
            "entry_sha256": _sha_file(entry_path),
            "signature_sha256": _sha_file(signature_path),
        }
    )


def issuance_fingerprint(receipt_path: str | Path) -> str:
    """Return the external-notary value for one intact controller issuance.

    The caller must retain/notarize this value outside the same-user mutable receipt tree.  It is
    intentionally not embedded as an authority in the receipt itself.
    """
    receipt_path = Path(receipt_path).resolve()
    receipt = yaml.safe_load(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("controller receipt must be a mapping")
    raw_path = _bound_file(receipt_path.parent, receipt.get("raw_measurement"), "raw measurement")
    contract_path = _bound_file(receipt_path.parent, receipt.get("contract"), "measurement contract")
    return _validate_issuance(receipt_path, receipt, _sha_file(raw_path), _sha_file(contract_path))


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical_sha(value: object) -> str:
    return _mjson.canonical_sha256(value, allow_nan=True)


def _is_sha(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in _HEX for character in text)


def _closed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    extra, missing = sorted(set(value) - fields), sorted(fields - set(value))
    if extra or missing:
        raise ValueError(f"{label} is closed; unrecognized={extra} missing={missing}")
    return value


def _allowed(value: object, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    extra = sorted(set(value) - fields)
    if extra:
        raise ValueError(f"{label} is closed; unrecognized={extra}")
    return value


def _contained(root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a contained relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the contract root") from error
    if not path.is_file():
        raise ValueError(f"{label} is absent")
    return path


def _bound_file(root: Path, value: object, label: str) -> Path:
    ref = _closed(value, {"path", "sha256"}, label)
    path = _contained(root, ref["path"], label)
    if not _is_sha(ref["sha256"]) or _sha_file(path) != ref["sha256"]:
        raise ValueError(f"{label} digest differs")
    return path


def _fsync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    _fsync(path)


_IMPLEMENTATION_FILES = (
    "paper_measurement_controller.py",
    "paper_measurement_protocol.py",
    "paper_measurement_study.py",
    "paper_measurement_contract.py",
    "paper_measurement_build.py",
    "paper_measurement_process.py",
    "paper_measurement_execution.py",
    "paper_measurement_verification.py",
)


def entry_source() -> Path:
    """The executable Python entry source bound by the original receipt field."""
    return Path(__file__).with_name("paper_measurement_controller.py")


def implementation_paths() -> tuple[Path, ...]:
    """Exact current owners of the decomposed controller implementation."""
    return tuple(Path(__file__).with_name(name) for name in _IMPLEMENTATION_FILES)


def retain_controller_sources(output: Path) -> dict[str, str]:
    """Retain every implementation member alongside the separate Python entry copy."""
    directory = output / "controller_sources"
    directory.mkdir()
    members = []
    for source in implementation_paths():
        target = directory / source.name
        shutil.copy2(source, target)
        _fsync(target)
        members.append({"path": source.name, "sha256": _sha_file(target)})
    manifest = directory / "manifest.json"
    _write(manifest, {"schema_version": 1, "members": members})
    return {"path": manifest.relative_to(output).as_posix(), "sha256": _sha_file(manifest)}


def validate_controller_sources(root: Path, reference: object) -> None:
    """Require exact retained membership and bytes from this controller version."""
    manifest = _bound_file(root, reference, "controller source closure")
    if manifest != (root / "controller_sources/manifest.json").resolve():
        raise ValueError("controller source closure has a foreign manifest path")
    document = _closed(json.loads(manifest.read_bytes()), {"schema_version", "members"}, "controller source closure")
    expected = [{"path": source.name, "sha256": _sha_file(source)} for source in implementation_paths()]
    if document != {"schema_version": 1, "members": expected}:
        raise ValueError("receipt was not produced by this complete trusted controller version")
    names = {"manifest.json", *(row["path"] for row in expected)}
    if {path.name for path in manifest.parent.iterdir()} != names:
        raise ValueError("controller source closure has changed membership")
    for row, source in zip(expected, implementation_paths(), strict=True):
        retained = _bound_file(manifest.parent, row, "controller source member")
        if retained.is_symlink() or retained.read_bytes() != source.read_bytes():
            raise ValueError("retained controller implementation differs from its trusted source")
