"""Compare an existing engine export to an independently selected config blob.

No image/tag/daemon changes, downloads, extraction or saved-JSON admission. The
caller owns the config's actual public-source provenance; this check joins exact
config bytes and every uncompressed filesystem layer, not a displayed image tag.
It issues no compiler, runtime, physical, service or isolation capability.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def _hash_stream(stream):
    result = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        result.update(chunk)
    return result.hexdigest()


def image_root_masks(archive: Path) -> tuple[str, ...]:
    """Derive empty scopes from actual verified layer names, without extraction.

    Only directory roots and aliases into directory roots are supported. Direct
    root files or whiteout semantics refuse; names never grant executable/data
    access. All image directories except private kernel/tmp scopes are hidden.
    """
    entries = {}
    with tarfile.open(archive) as source:
        (image,) = json.load(source.extractfile("manifest.json"))
        for layer in image["Layers"]:
            with tarfile.open(fileobj=source.extractfile(layer), mode="r|") as filesystem:
                for member in filesystem:
                    name = PurePosixPath(member.name)
                    if name.is_absolute() or ".." in name.parts or any(part.startswith(".wh.") for part in name.parts):
                        raise ValueError("unsupported image root path or overlay deletion semantics")
                    if not name.parts:
                        continue
                    root = "/" + name.parts[0]
                    entries.setdefault(root, ("directory", ""))
                    if len(name.parts) == 1:
                        if member.isdir():
                            entries[root] = ("directory", "")
                        elif member.issym():
                            entries[root] = ("alias", member.linkname)
                        else:
                            raise ValueError("image contains an ungrantable root file")
    directories = {name for name, (kind, _) in entries.items() if kind == "directory"}
    for _, (kind, target) in entries.items():
        if kind == "alias":
            alias = PurePosixPath(target)
            if ".." in alias.parts or not alias.parts:
                raise ValueError("unsupported image root alias")
            first = alias.parts[1] if alias.is_absolute() else alias.parts[0]
            if "/" + first not in directories:
                raise ValueError("image root alias has no original directory owner")
    return tuple(sorted(directories - {"/dev", "/proc", "/sys", "/tmp"}))


def verify_image_export(*, configuration: Path, archive: Path, config_sha256: str) -> dict:
    """Reopen the original independent bytes and exact selected engine export."""
    if (
        len(config_sha256) != 64
        or any(char not in "0123456789abcdef" for char in config_sha256)
        or configuration.is_symlink()
        or archive.is_symlink()
    ):
        raise ValueError("image selection needs plain files and an exact config SHA")
    payload = configuration.read_bytes()
    if len(payload) > 65536 or hashlib.sha256(payload).hexdigest() != config_sha256:
        raise ValueError("independent image configuration changed")
    original = json.loads(payload)
    if original.get("rootfs", {}).get("type") != "layers":
        raise ValueError("unsupported independent image filesystem grammar")
    layers = original["rootfs"].get("diff_ids")
    if not isinstance(layers, list) or not layers or len(layers) > 64:
        raise ValueError("independent image must declare its complete bounded layer roster")
    observed = []
    with tarfile.open(archive) as source:
        index = {}
        for row in source:
            if len(index) >= 4096 or row.name in index:
                raise ValueError("image archive has duplicate or excessive members")
            index[row.name] = row

        def read(name, limit):
            row = index.get(name)
            if row is None or not row.isfile() or row.size > limit:
                raise ValueError("image export has missing or unsupported selected member")
            return source.extractfile(row)

        manifest = json.load(read("manifest.json", 65536))
        if not isinstance(manifest, list) or len(manifest) != 1:
            raise ValueError("image export must contain exactly the selected image")
        image = manifest[0]
        if read(image["Config"], 65536).read() != payload:
            raise ValueError("engine image config differs from independent original bytes")
        selected = image.get("Layers")
        if not isinstance(selected, list) or len(selected) != len(layers):
            raise ValueError("image export omitted original layers")
        for member, expected in zip(selected, layers, strict=True):
            actual = "sha256:" + _hash_stream(read(member, 2 * 1024**3))
            if actual != expected:
                raise ValueError("engine filesystem layer differs from independent config")
            observed.append({"member": member, "sha256": actual[7:]})
    with archive.open("rb") as stream:
        archive_sha256 = _hash_stream(stream)
    return {
        "image_id": "sha256:" + config_sha256,
        "configuration_sha256": config_sha256,
        "archive_sha256": archive_sha256,
        "layers": observed,
        "scope": "exact supplied config and exported layer byte correspondence; no source or runtime authority",
    }
