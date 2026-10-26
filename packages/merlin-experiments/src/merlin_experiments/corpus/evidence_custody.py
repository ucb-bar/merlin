"""Custody of a sealed Phase 0 derivation: list what it pins, verify it, seal it, repair it with a receipt.

A reviewed derivation is evidence later runs re-read by path and digest, and nothing protected it:
on 2026-10-24 a disk cleanup deleted ``materialized/*/pytorch-opset.json`` from every application of
a reviewed derivation and two ``frontend-trace.json`` beside them. The loss surfaced only a week
later, when a regeneration re-verified the bytes. The repair was a byte-identical copy from the
capture each file was materialized from, digest-checked by hand, into a new location -- because no
tool could put a file back into sealed evidence and say so.

This module is that tool, in three operations over the same pin set:

* **verify** lists every pinned member and reports each one ``missing`` or ``tampered`` by name;
* **seal** records every member's digest beside the derivation, clears write bits and asks for the
  immutable attribute, recording whether the host granted it;
* **repair** restores missing members ONLY from a source whose digest equals the pin, refuses when any
  present member is tampered, and writes a schema'd receipt beside the derivation: what was missing,
  where each byte came from, the digests, when, by whom and why. It never overwrites, never deletes
  and never unlocks; an immutable or read-only destination is reported, not forced.

The pins are derived from the derivation's own records, the same ones its consumers check: the
evidence manifest, the synthesis profile's ``materialized_capture`` selections, each materialized
capture's producer receipt, and the framework catalog and frontend trace its ``meta.json`` binds
(exactly the members ``phase0.requirements`` copies in). A derivation sealed by this module also
pins every other member through its seal record.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..spec import SpecError

SEAL_SCHEMA = "merlin.evidence_seal.v1"
REPAIR_SCHEMA = "merlin.evidence_repair.v1"
SEAL_SUFFIX = ".evidence-seal.json"
REPAIR_INFIX = ".repair-receipt-"
_RECEIPT = "capture_receipt.json"
_META = "meta.json"
# meta.json rows that bind a sibling file the materializer copies beside the capture.
_META_MEMBERS = (("framework_catalog", "pytorch-opset.json"), ("frontend_trace", "frontend-trace.json"))
_SELECTION_PAIRS = (("path", "capture_sha256"), ("loader_path", "loader_sha256"), ("profile_path", "profile_sha256"))


@dataclass
class Pin:
    sha256: str
    size: int | None = None
    pinned_by: list[str] = field(default_factory=list)
    origins: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _member(value: str) -> str | None:
    parts = Path(value).parts
    if not value or Path(value).is_absolute() or ".." in parts or not parts:
        return None
    return Path(value).as_posix()


def seal_record_path(root: Path) -> Path:
    return root.parent / f"{root.name}{SEAL_SUFFIX}"


def derivation_root(path: Path) -> Path:
    """The derivation directory: the path itself or its ``derivation/`` child.

    It is recognized by its ``derivation.json`` or, when that very file is what was deleted, by the
    seal record beside it -- a damaged derivation must still be found to be reported.
    """
    path = Path(os.path.abspath(path))
    for candidate in (path, path / "derivation"):
        if (candidate / "derivation.json").is_file() or seal_record_path(candidate).is_file():
            return candidate
    raise SpecError(f"no Phase 0 derivation (derivation.json or {SEAL_SUFFIX}) at {path}")


class _Pins:
    def __init__(self) -> None:
        self.rows: dict[str, Pin] = {}
        self.conflicts: list[dict] = []

    def add(self, member: str | None, sha256: object, record: str, *, size=None, origin=None) -> None:
        if member is None or not _is_digest(sha256):
            return
        row = self.rows.get(member)
        if row is None:
            row = self.rows[member] = Pin(sha256)
        elif row.sha256 != sha256:
            self.conflicts.append({"member": member, "pins": [row.sha256, sha256], "records": [*row.pinned_by, record]})
            return
        if record not in row.pinned_by:
            row.pinned_by.append(record)
        if size is not None and row.size is None:
            row.size = size
        if origin and origin not in row.origins:
            row.origins.append(origin)


def _read(path: Path):
    raw = path.read_bytes()
    return json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)


def collect_pins(root: Path) -> _Pins:
    """Every member the derivation's own records pin, with the records that pin it."""
    pins = _Pins()
    manifest = root / "evidence" / "evidence-manifest.json"
    if manifest.is_file():
        record = manifest.relative_to(root).as_posix()
        for name, row in (_read(manifest).get("artifacts") or {}).items():
            if isinstance(row, dict) and _member(name):
                pins.add(f"evidence/{_member(name)}", row.get("sha256"), record, size=row.get("size_bytes"))
    synthesis = root / "synthesis.yaml"
    if synthesis.is_file():
        for entry in (_read(synthesis) or {}).get("capsules") or []:
            selection = entry.get("materialized_capture") if isinstance(entry, dict) else None
            if not isinstance(selection, dict):
                continue
            for path_key, digest_key in _SELECTION_PAIRS:
                if isinstance(selection.get(path_key), str):
                    pins.add(_member(selection[path_key]), selection.get(digest_key), "synthesis.yaml")
            capture = _member(selection.get("path") or "")
            if capture:
                pins.add(str(Path(capture).parent / _RECEIPT), selection.get("receipt_sha256"), "synthesis.yaml")
    for receipt in sorted(root.rglob(_RECEIPT)):
        if receipt.is_symlink():
            continue
        record = receipt.relative_to(root).as_posix()
        directory = receipt.parent.relative_to(root)
        for name, row in (_read(receipt).get("artifacts") or {}).items():
            if isinstance(row, dict) and _member(name):
                pins.add((directory / name).as_posix(), row.get("sha256"), record, size=row.get("bytes"))
        meta = receipt.parent / _META
        if meta.is_file() and not meta.is_symlink():
            document = _read(meta)
            for key, name in _META_MEMBERS:
                row = document.get(key) if isinstance(document, dict) else None
                if isinstance(row, dict) and row.get("sha256"):
                    origin = row.get("path") if isinstance(row.get("path"), str) else None
                    origin = origin if origin and Path(origin).is_absolute() else None
                    pins.add(
                        (directory / name).as_posix(), row["sha256"], (directory / _META).as_posix(), origin=origin
                    )
    sealed = seal_record_path(root)
    if sealed.is_file():
        record = sealed.name
        document = _read(sealed)
        if document.get("schema") != SEAL_SCHEMA:
            raise SpecError(f"unsupported evidence seal record: {sealed}")
        for name, row in (document.get("members") or {}).items():
            pins.add(_member(name), (row or {}).get("sha256"), record, size=(row or {}).get("bytes"))
    return pins


def verify(path: Path) -> dict:
    """Every pinned member, each classified present, ``missing`` or ``tampered`` -- by name."""
    root = derivation_root(path)
    pins = collect_pins(root)
    missing, tampered = [], []
    for member, pin in sorted(pins.rows.items()):
        target = root / member
        if target.is_symlink() or not target.is_file():
            missing.append({"member": member, "sha256": pin.sha256, "pinned_by": pin.pinned_by})
            continue
        observed = _sha256(target)
        if observed != pin.sha256:
            tampered.append(
                {"member": member, "expected": pin.sha256, "observed": observed, "pinned_by": pin.pinned_by}
            )
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink()}
    sealed = seal_record_path(root)
    status = "intact" if not (missing or tampered or pins.conflicts) else "damaged"
    return {
        "schema": "merlin.evidence_verification.v1",
        "root": str(root),
        "status": status,
        "pinned_members": len(pins.rows),
        "missing": missing,
        "tampered": tampered,
        "conflicting_pins": pins.conflicts,
        "unpinned_members": sorted(present - set(pins.rows)),
        "sealed": sealed.is_file(),
        "repair_receipts": sorted(p.name for p in root.parent.glob(f"{root.name}{REPAIR_INFIX}*.json")),
    }


def describe(report: dict) -> str:
    lines = [f"{report['status']}: {report['pinned_members']} pinned member(s) under {report['root']}"]
    for row in report["missing"]:
        lines.append(f"  missing   {row['member']}  (pinned {row['sha256'][:12]} by {', '.join(row['pinned_by'])})")
    for row in report["tampered"]:
        lines.append(f"  tampered  {row['member']}  (expected {row['expected'][:12]}, found {row['observed'][:12]})")
    for row in report["conflicting_pins"]:
        lines.append(f"  conflict  {row['member']}  ({' vs '.join(p[:12] for p in row['pins'])})")
    return "\n".join(lines)


def _identity(by: str | None) -> str:
    chosen = (by or "").strip()
    if not chosen:
        raise SpecError("custody records need an explicit operator identity (--by)")
    return chosen


def _tool() -> dict:
    from merlin.common.paths import repo_root

    try:
        from merlin.common.artifacts import git_sha7

        sha = git_sha7(repo_root())
    except Exception:  # noqa: BLE001 -- provenance is best-effort; the record still binds the bytes
        sha = None
    return {"module": __name__, "git_sha7": sha, "user": getpass.getuser(), "pid": os.getpid()}


def _write_record(path: Path, document: dict, schema: str) -> None:
    """Create ``path`` exclusively and read-only; a custody record is never overwritten."""
    from merlin.common.schemas import validate_or_raise

    validate_or_raise(document, schema)
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _writable_directory(directory: Path) -> str | None:
    """Why a file cannot be added to ``directory`` without forcing anything, or ``None``."""
    from merlin.common.immutability import is_immutable

    if not directory.is_dir() or directory.is_symlink():
        return f"{directory} is not an ordinary directory"
    if is_immutable(directory):
        return f"{directory} carries the immutable attribute (chattr +i); an operator must unlock it"
    if not os.access(directory, os.W_OK | os.X_OK):
        return f"{directory} is read-only; this tool does not change its mode"
    return None


def seal(path: Path, *, by: str, note: str, immutable: str | None = None) -> dict:
    """Pin every member, then make the derivation read-only and (where permitted) immutable."""
    from merlin.common.immutability import seal_tree

    root = derivation_root(path)
    who = _identity(by)
    if not note.strip():
        raise SpecError("sealing needs a note saying what was reviewed")
    record = seal_record_path(root)
    if record.exists() or record.is_symlink():
        raise SpecError(f"derivation is already sealed: {record}")
    report = verify(root)
    if report["status"] != "intact":
        raise SpecError("refusing to seal damaged evidence; repair it first:\n" + describe(report))
    members = {}
    for member in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        members[member.relative_to(root).as_posix()] = {"sha256": _sha256(member), "bytes": member.stat().st_size}
    immutability = seal_tree([root], mode=immutable)
    document = {
        "schema": SEAL_SCHEMA,
        "root": str(root),
        "protects": [str(root)],
        "at": datetime.now(UTC).isoformat(),
        "by": who,
        "note": note,
        "tool": _tool(),
        "pinned_by_records": report["pinned_members"],
        "members": members,
        "immutability": immutability,
    }
    _write_record(record, document, "evidence_seal")
    record_lock = seal_tree([record], mode=immutable)["immutable"]
    return {"seal": str(record), "members": len(members), "immutability": immutability, "record_lock": record_lock}


def _candidates(root: Path, member: str, pin: Pin, sources: list[Path]) -> list[Path]:
    """Where a missing member's bytes may be: mirrors of the derivation, then the recorded origins."""
    out: list[Path] = []
    relative = Path(member)
    for source in sources:
        source = Path(os.path.abspath(source))
        if source.is_file():
            out.append(source)
            continue
        out += [source / relative, source / root.name / relative]
        out += sorted(source.rglob(relative.name))
    out += [Path(origin) for origin in pin.origins]
    return list(dict.fromkeys(out))


def _source_for(root: Path, member: str, pin: Pin, sources: list[Path]) -> tuple[Path | None, list[dict]]:
    considered = []
    for candidate in _candidates(root, member, pin, sources):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if Path(os.path.abspath(candidate)).is_relative_to(root):
            continue  # never restore a member from another name inside the damaged tree
        if pin.size is not None and candidate.stat().st_size != pin.size:
            considered.append({"path": str(candidate), "result": "size differs from pin"})
            continue
        digest = _sha256(candidate)
        if digest == pin.sha256:
            return candidate, considered
        considered.append({"path": str(candidate), "result": f"digest {digest[:12]} differs from pin"})
    return None, considered


def _restore(source: Path, destination: Path, pin: Pin, mode: int) -> str:
    """Copy into a private temporary name, verify, then link into place without overwriting."""
    temporary = destination.with_name(f".{destination.name}.repair-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as stream, source.open("rb") as origin:
            for block in iter(lambda: origin.read(1 << 20), b""):
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if digest.hexdigest() != pin.sha256:
            raise SpecError(f"source changed while copying {source}; nothing was restored at {destination}")
        os.chmod(temporary, mode)
        os.link(temporary, destination)  # fails if the name appeared meanwhile: never overwrite
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def repair(
    path: Path,
    *,
    sources: list[Path],
    by: str,
    reason: str,
    approval: str | None = None,
    apply: bool = False,
) -> dict:
    """Restore every missing pinned member from a digest-equal source, or refuse and change nothing."""
    root = derivation_root(path)
    who = _identity(by)
    if not reason.strip():
        raise SpecError("a repair needs a reason")
    before = verify(root)
    if before["tampered"] or before["conflicting_pins"]:
        raise SpecError("refusing to repair: present members differ from their pins\n" + describe(before))
    pins = collect_pins(root).rows
    plan, unresolved, blocked = [], [], []
    for row in before["missing"]:
        member = row["member"]
        source, considered = _source_for(root, member, pins[member], sources)
        if source is None:
            unresolved.append({"member": member, "sha256": row["sha256"], "considered": considered})
            continue
        reason_blocked = _writable_directory((root / member).parent)
        if reason_blocked:
            blocked.append({"member": member, "reason": reason_blocked})
        plan.append(
            {
                "member": member,
                "sha256": row["sha256"],
                "bytes": pins[member].size,
                "source": str(source),
                "pinned_by": row["pinned_by"],
            }
        )
    result = {
        "root": str(root),
        "missing": len(before["missing"]),
        "plan": plan,
        "unresolved": unresolved,
        "blocked": blocked,
        "applied": False,
    }
    if not before["missing"]:
        result["status"] = "intact"
        return result
    if unresolved or blocked:
        result["status"] = "refused"
        return result
    if not apply:
        result["status"] = "planned"
        return result
    restored, failure = [], None
    try:
        for row in plan:
            destination = root / row["member"]
            anchor = root / row["pinned_by"][0] if (root / row["pinned_by"][0]).is_file() else destination.parent
            mode = stat.S_IMODE(anchor.stat().st_mode) & ~0o111
            digest = _restore(Path(row["source"]), destination, pins[row["member"]], mode)
            restored.append({**row, "source_sha256": row["sha256"], "restored_sha256": digest})
    except (OSError, SpecError) as error:
        # Whatever was already restored is recorded below: a receipt is written for every change.
        failure = f"{type(error).__name__}: {error}"
    after = verify(root)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    receipt = root.parent / f"{root.name}{REPAIR_INFIX}{stamp}.json"
    _write_record(
        receipt,
        {
            "schema": REPAIR_SCHEMA,
            "root": str(root),
            "protects": [str(root)],
            "at": datetime.now(UTC).isoformat(),
            "by": who,
            "reason": reason,
            "approval": approval,
            "tool": _tool(),
            "policy": "restore only members whose pinned digest equals the source's; never overwrite, delete "
            "or change the mode of existing members",
            "before": {k: before[k] for k in ("status", "pinned_members", "missing", "tampered")},
            "restored": restored,
            "failure": failure,
            "after": {k: after[k] for k in ("status", "pinned_members", "missing", "tampered")},
        },
        "evidence_repair",
    )
    result.update(
        status="repaired" if after["status"] == "intact" and failure is None else "partially_repaired",
        applied=True,
        receipt=str(receipt),
        after=after["status"],
        failure=failure,
        restored=restored,
    )
    return result


def add_parser(operations) -> None:
    """``merlin-experiment corpus evidence {verify,seal,repair}``."""
    evidence = operations.add_parser("evidence", help="verify, seal or repair-with-receipt a Phase 0 derivation")
    verbs = evidence.add_subparsers(dest="evidence_operation", required=True)
    check = verbs.add_parser("verify", help="list pinned members; report each missing or tampered one by name")
    check.add_argument("derivation", type=Path)
    sealing = verbs.add_parser("seal", help="pin every member, then make the tree read-only and immutable")
    sealing.add_argument("derivation", type=Path)
    sealing.add_argument("--by", required=True, help="operator identity recorded in the seal")
    sealing.add_argument("--note", required=True, help="what was reviewed")
    sealing.add_argument("--immutable", choices=("attempt", "sudo", "off"), help="chattr +i attempt (default: env)")
    fix = verbs.add_parser("repair", help="restore missing members from digest-equal sources and write a receipt")
    fix.add_argument("derivation", type=Path)
    fix.add_argument("--source", action="append", type=Path, default=[], help="tree or file holding the bytes")
    fix.add_argument("--by", required=True, help="operator identity recorded in the receipt")
    fix.add_argument("--reason", required=True)
    fix.add_argument("--approval", help="who approved modifying sealed evidence, and where")
    fix.add_argument("--apply", action="store_true", help="restore; without it nothing is written")


def run(args) -> tuple[dict, int]:
    if args.evidence_operation == "verify":
        report = verify(args.derivation)
        return report, 0 if report["status"] == "intact" else 1
    if args.evidence_operation == "seal":
        return seal(args.derivation, by=args.by, note=args.note, immutable=args.immutable), 0
    result = repair(
        args.derivation, sources=args.source, by=args.by, reason=args.reason, approval=args.approval, apply=args.apply
    )
    return result, 0 if result["status"] in {"intact", "planned", "repaired"} else 1
