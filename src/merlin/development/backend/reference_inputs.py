"""Operator-reviewed positive reference selection, separate from compiler grants.

Only individually pinned files enter staging. A repository checkout or archive
is a container, never permission to copy its neighbours or its complete tree.
Permission identities come from the operator's review; this helper cannot grant
permission or establish that an archive was publicly published.
Explicit user-permitted raw evidence can contain inert binary artifacts; it
does not grant imports, execution, compiler seeds or timing authority.
"""

from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

MAX_FILE_BYTES = 8 << 20
MAX_TOTAL_BYTES = 64 << 20
RAW_MAX_FILE_BYTES = 128 << 20
RAW_MAX_TOTAL_BYTES = 512 << 20
OPAQUE_MAX_FILE_BYTES = 2 << 30
OPAQUE_MAX_TOTAL_BYTES = 4 << 30


def _hash(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("reference identity requires an exact SHA256")
    return value


def _relative(value):
    if (
        type(value) is not str
        or not value
        or len(value) > 1024
        or "\\" in value
        or any(char.isspace() for char in value)
    ):
        raise ValueError("reference file requires a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(p in (".", "..", ".git", "__pycache__") for p in path.parts)
        or value == "reference-selection.json"
    ):
        raise ValueError("unsafe reference file path")
    return path


def _ordinary(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("reference container must be canonical and unlinked")
    return path


def _opaque_digest(path, size):
    if (
        type(size) is not int
        or not 0 < size <= OPAQUE_MAX_FILE_BYTES
        or not path.is_file()
        or path.stat().st_size != size
    ):
        raise ValueError("opaque reference container requires matching bounded size")
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(remaining, 1 << 20))
            if not chunk:
                raise ValueError("opaque reference container changed size")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise ValueError("opaque reference container changed size")
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceFile:
    member: str
    destination: str
    sha256: str
    size_bytes: int | None = None

    def __post_init__(self):
        _relative(self.member)
        _relative(self.destination)
        _hash(self.sha256)
        if self.size_bytes is not None and (type(self.size_bytes) is not int or self.size_bytes <= 0):
            raise ValueError("reference member size pin must be a positive integer")


@dataclass(frozen=True)
class ReferenceOrigin:
    name: str
    kind: str
    container: Path
    identity: str
    license: str
    files: tuple[ReferenceFile, ...]
    public_url: str = ""
    user_authorization: str = ""
    allowed_purpose: str = "reference_reading"
    content_kind: str = "source_text"

    def __post_init__(self):
        if type(self.name) is not str or not self.name.isidentifier() or len(self.name) > 128:
            raise ValueError("reference origin requires a bounded identifier")
        if self.kind not in ("public_repository", "user_permitted_archive"):
            raise ValueError("reference origin is not positively permitted; private archives are excluded")
        if self.allowed_purpose != "reference_reading":
            raise ValueError("reference origins grant reading only, not compiler seeds or performance authority")
        if self.content_kind not in ("source_text", "raw_evidence", "opaque_container"):
            raise ValueError("unsupported reference content kind")
        if self.content_kind in ("raw_evidence", "opaque_container") and self.kind != "user_permitted_archive":
            raise ValueError("raw reference evidence requires explicit user-permitted archive authorization")
        _ordinary(self.container)
        if not isinstance(self.license, str) or not self.license.strip() or len(self.license) > 4096:
            raise ValueError("reviewed reference license identity required")
        if (
            type(self.files) is not tuple
            or not 1 <= len(self.files) <= 256
            or any(type(row) is not ReferenceFile for row in self.files)
            or len({r.member for r in self.files}) != len(self.files)
            or len({r.destination for r in self.files}) != len(self.files)
        ):
            raise ValueError("reference origin requires unique individually pinned files")
        if self.kind == "public_repository":
            url = urlsplit(self.public_url)
            if (
                url.scheme != "https"
                or not url.netloc
                or url.username
                or url.password
                or url.fragment
                or any(char.isspace() for char in self.public_url)
                or type(self.identity) is not str
                or len(self.identity) not in (40, 64)
                or any(c not in "0123456789abcdef" for c in self.identity)
                or self.user_authorization
            ):
                raise ValueError("public repository reference requires an HTTPS origin and exact revision")
        else:
            _hash(self.identity)
            if (
                type(self.user_authorization) is not str
                or not self.user_authorization.strip()
                or len(self.user_authorization) > 4096
                or self.public_url
            ):
                raise ValueError("archive reference requires explicit user authorization, not a public-origin claim")


def stage_references(origins, destination, *, review_id):
    """Validate the complete selection before writing a fresh bounded directory.

    The receipt records public source or explicit archive permission separately.
    Nothing here imports reference code, selects schedules or adds author APIs.
    Caller-owned sandbox staging can mount the returned directory read-only.
    """
    if (
        type(origins) is not tuple
        or not 1 <= len(origins) <= 16
        or any(type(origin) is not ReferenceOrigin for origin in origins)
        or len({origin.name for origin in origins}) != len(origins)
        or type(review_id) is not str
        or not review_id.strip()
        or len(review_id) > 4096
    ):
        raise ValueError("reference staging requires an explicit bounded reviewed selection")
    destination = _ordinary(destination)
    if destination.exists():
        raise ValueError("reference staging requires a fresh destination")
    payloads, records, total, source_total = {}, [], 0, 0
    opaque_payloads, inline_total = {}, 0
    opaque_selected = any(o.content_kind == "opaque_container" for o in origins)
    inline_limit = RAW_MAX_TOTAL_BYTES if any(o.content_kind == "raw_evidence" for o in origins) else MAX_TOTAL_BYTES
    total_limit = OPAQUE_MAX_TOTAL_BYTES if opaque_selected else inline_limit
    readonly = set()
    for origin in origins:
        origin.__post_init__()
        container = _ordinary(origin.container)
        if container.is_relative_to(destination) or destination.is_relative_to(container):
            raise ValueError("reference source and staging destination must not overlap")
        archive = None
        if origin.kind == "user_permitted_archive":
            # Hash streaming; a large archive never becomes an in-memory copy.
            if origin.content_kind == "opaque_container":
                digest = _opaque_digest(container, origin.files[0].size_bytes)
                if digest != origin.identity:
                    raise ValueError("reference archive differs from its authorized identity")
            else:
                with container.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != origin.identity:
                        raise ValueError("reference archive differs from its authorized identity")
                archive = zipfile.ZipFile(container)
        elif not container.is_dir():
            raise ValueError("public repository reference requires an explicit source directory")
        try:
            rows = []
            raw = origin.content_kind == "raw_evidence"
            opaque = origin.content_kind == "opaque_container"
            if opaque and (len(origin.files) != 1 or origin.files[0].member != container.name):
                raise ValueError("opaque reference selects exactly its original container filename")
            file_limit = RAW_MAX_FILE_BYTES if raw else MAX_FILE_BYTES
            for member in origin.files:
                member.__post_init__()
                if opaque:
                    size = container.stat().st_size
                    if (
                        size != member.size_bytes
                        or not 0 < size <= OPAQUE_MAX_FILE_BYTES
                        or member.sha256 != origin.identity
                    ):
                        raise ValueError("opaque reference container requires matching size and content pins")
                    total += size
                    if total > total_limit:
                        raise ValueError("opaque reference selection exceeds total byte bound")
                    name = origin.name + "/" + member.destination
                    opaque_payloads[name] = (container, member)
                    rows.append({"member": member.member, "path": name, "sha256": member.sha256, "bytes": size})
                    continue
                if archive is not None:
                    matches = [entry for entry in archive.infolist() if entry.filename == member.member]
                    if (
                        len(matches) != 1
                        or matches[0].is_dir()
                        or matches[0].file_size > file_limit
                        or stat.S_ISLNK(matches[0].external_attr >> 16)
                    ):
                        raise ValueError("reference archive member is ambiguous, indirect or oversized")
                    with archive.open(matches[0]) as stream:
                        payload = stream.read(file_limit + 1)
                else:
                    path = _ordinary(container / member.member)
                    if not path.is_file() or path.stat().st_size > file_limit:
                        raise ValueError("reference source member is absent or oversized")
                    payload = path.read_bytes()
                total += len(payload)
                inline_total += len(payload)
                source_total += 0 if raw else len(payload)
                if (
                    len(payload) > file_limit
                    or member.size_bytes is not None
                    and len(payload) != member.size_bytes
                    or total > total_limit
                    or inline_total > inline_limit
                    or source_total > MAX_TOTAL_BYTES
                    or hashlib.sha256(payload).hexdigest() != member.sha256
                ):
                    raise ValueError("reference source bytes differ from their pin or exceed staging bounds")
                if not raw:
                    try:
                        decoded = payload.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError("reference member must be UTF-8 source text") from error
                    if "\x00" in decoded:
                        raise ValueError("reference member contains a binary payload")
                name = origin.name + "/" + member.destination
                payloads[name] = payload
                if raw:
                    readonly.add(name)
                rows.append({"member": member.member, "path": name, "sha256": member.sha256, "bytes": len(payload)})
            records.append(
                {
                    "name": origin.name,
                    "kind": origin.kind,
                    "identity": origin.identity,
                    "license": origin.license,
                    "public_url": origin.public_url,
                    "user_authorization": origin.user_authorization,
                    "allowed_purpose": origin.allowed_purpose,
                    "files": rows,
                }
            )
            if raw or opaque:
                records[-1]["content_kind"] = origin.content_kind
        finally:
            if archive is not None:
                archive.close()
    # Detect file/directory collisions before creating any output.
    names = set(payloads) | set(opaque_payloads)
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            raise ValueError("reference destinations overlap as a file and directory")
    record = {
        "schema": "merlin.reviewed_reference_inputs.v1",
        "review_id": review_id,
        "origins": records,
        "scope": "reference reading only; no implementation, import, execution or timing grant",
    }
    encoded = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        if name in readonly:
            path.chmod(0o444)
    for name, (container, member) in opaque_payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Own the copy before changing its mode. Never chmod a user archive's
        # inode through a hardlink; later arm staging may hardlink this copy.
        with container.open("rb") as source, path.open("xb") as target:
            remaining = member.size_bytes
            while remaining:
                chunk = source.read(min(remaining, 1 << 20))
                if not chunk:
                    raise ValueError("opaque reference changed during bounded streaming copy")
                target.write(chunk)
                remaining -= len(chunk)
            if source.read(1):
                raise ValueError("opaque reference changed during bounded streaming copy")
        with path.open("rb") as stream:
            if (
                path.stat().st_size != member.size_bytes
                or hashlib.file_digest(stream, "sha256").hexdigest() != member.sha256
            ):
                raise ValueError("opaque reference changed during bounded streaming copy")
        path.chmod(0o444)
    for origin in origins:
        if origin.kind == "user_permitted_archive":
            if origin.content_kind == "opaque_container":
                if _opaque_digest(_ordinary(origin.container), origin.files[0].size_bytes) != origin.identity:
                    raise ValueError("reference archive changed during staging")
            else:
                with _ordinary(origin.container).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != origin.identity:
                        raise ValueError("reference archive changed during staging")
        else:
            for member in origin.files:
                with _ordinary(origin.container / member.member).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != member.sha256:
                        raise ValueError("reference source changed during staging")
    (destination / "reference-selection.json").write_bytes(encoded)
    return {"root": str(destination), "manifest_sha256": hashlib.sha256(encoded).hexdigest(), **record}
