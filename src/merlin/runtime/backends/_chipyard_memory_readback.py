"""Closed, opt-in oracle memory-export arguments for the chipyard RoCC engines.

Transport preparation, not output admission: the trusted host still proves the ELF symbols, the
writable layout, the selected engine and the complete decoded values. The two transports are
properties of the SIMULATORS, not of any accelerator -- spike's HTIF signature dump and the
Merlin-built GSIM harness's coherent region dump -- so one implementation serves every target the
generic chipyard RoCC backend serves.
"""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_TRANSPORTS = {"spike": "htif_signature_v1", "gsim": "gsim_coherent_dump_v1"}
_PACKET_POLICY = "coherent_packet_v1"
_PACKET_TRANSPORT = "gsim_coherent_packet_v1"
_SCHEMA = "oracle_memory_readback_v1"
_REGION_KEYS = frozenset({"base", "bytes"})
_REQUEST_KEYS = frozenset({"schema", "transport", "output_path", "regions", "elf_sha256", "regions_path"})
#: Bounds on what one export may ask the simulator to write: a policy of this transport, not a fact
#: about any device.
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
_MAX_REGIONS = 1024
_PACKET_HEADER_BYTES = 8


def memory_readback_transport(simulator: str, *, policy_transport: str | None = None) -> str | None:
    """The transport this backend implements for ``simulator`` (and the optional policy), or None."""
    if policy_transport == _PACKET_POLICY:
        return {"spike": _TRANSPORTS["spike"], "gsim": _PACKET_TRANSPORT}.get(simulator)
    if policy_transport is not None:
        return None
    return _TRANSPORTS.get(simulator)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _absolute(value: object, *, name: str) -> Path:
    if type(value) is not str or not value or not Path(value).is_absolute():
        raise ValueError(f"memory readback {name} must be an absolute path")
    path = Path(value)
    if str(path) != value or path != path.resolve(strict=False):
        raise ValueError(f"memory readback {name} must be canonical")
    return path


def _request_digest(request: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("memory readback request is not JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MemoryReadbackCommand:
    """Preflighted fixed simulator arguments; revalidate before and after the run."""

    argv_suffix: tuple[str, ...]
    elf: Path
    elf_sha256: str
    output_path: Path
    regions_path: Path | None
    regions_text: bytes | None
    request: Mapping[str, object]
    request_digest: str

    def revalidate(self, *, completed: bool = False) -> None:
        if _request_digest(self.request) != self.request_digest or not _regular(self.elf):
            raise ValueError("memory readback request or ELF identity changed")
        if _sha256(self.elf) != self.elf_sha256:
            raise ValueError("memory readback ELF bytes changed")
        if self.regions_path is not None:
            if not _regular(self.regions_path) or self.regions_path.read_bytes() != self.regions_text:
                raise ValueError("memory readback region manifest changed")
        if completed:
            if not _regular(self.output_path) or self.output_path.stat().st_size <= 0:
                raise ValueError("memory readback produced no regular complete-path output")
        elif self.output_path.exists() or self.output_path.is_symlink():
            raise ValueError("memory readback output path is not fresh")


def _regions(regions: object, *, simulator: str, packet: bool) -> list[str]:
    if type(regions) is not list or not 0 < len(regions) <= _MAX_REGIONS:
        raise ValueError("memory readback needs a bounded nonempty region roster")
    if simulator == "spike" and len(regions) != 1:
        raise ValueError("HTIF signature requires exactly one contiguous region")
    if packet and len(regions) != 2:
        raise ValueError("coherent packet requires exactly two role-ordered regions")
    preceding_end, total, lines = -1, 0, []
    for row in regions:
        if type(row) is not dict or row.keys() != _REGION_KEYS:
            raise ValueError("memory readback region is malformed")
        base, size = row["base"], row["bytes"]
        if (
            type(base) is not int
            or type(size) is not int
            or base < 0
            or size <= 0
            or base >= 1 << 64
            or size > (1 << 64) - base
        ):
            raise ValueError("memory readback region is out of bounded address range")
        if not packet and base < preceding_end:
            raise ValueError("memory readback regions overlap or are unordered")
        preceding_end = base + size
        total += size
        if total > _MAX_OUTPUT_BYTES + (_PACKET_HEADER_BYTES if packet else 0):
            raise ValueError("memory readback exceeds the bounded export size")
        lines.append(f"{base:#x} {size}\n")
    if packet:
        header, arena = regions
        disjoint = (
            header["base"] + _PACKET_HEADER_BYTES <= arena["base"] or arena["base"] + arena["bytes"] <= header["base"]
        )
        if (
            header["bytes"] != _PACKET_HEADER_BYTES
            or header["base"] % _PACKET_HEADER_BYTES
            or arena["bytes"] > _MAX_OUTPUT_BYTES
            or not disjoint
        ):
            raise ValueError("coherent packet requires disjoint aligned u64 metadata and one bounded byte arena")
    return lines


def prepare_memory_readback(elf: str | Path, simulator: str, request: Mapping[str, object]) -> MemoryReadbackCommand:
    """Validate a host-selected transport; never infer an output region."""
    if type(request) is not dict or request.keys() != _REQUEST_KEYS:
        raise ValueError("memory readback request has an unknown or missing field")
    if simulator not in _TRANSPORTS:
        raise ValueError("memory readback simulator is unsupported")
    packet = simulator == "gsim" and request["transport"] == _PACKET_TRANSPORT
    if request["schema"] != _SCHEMA or (request["transport"] != memory_readback_transport(simulator) and not packet):
        raise ValueError("memory readback transport does not match the simulator")
    expected_sha = request["elf_sha256"]
    if (
        type(expected_sha) is not str
        or len(expected_sha) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha)
    ):
        raise ValueError("memory readback requires a full lowercase ELF digest")
    elf_path = Path(elf)
    if not elf_path.is_absolute() or not _regular(elf_path):
        raise ValueError("memory readback ELF must be an absolute regular file")
    output = _absolute(request["output_path"], name="output_path")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("memory readback output parent must be an existing real directory")
    regions = request["regions"]
    lines = _regions(regions, simulator=simulator, packet=packet)
    manifest: Path | None = None
    manifest_bytes: bytes | None = None
    if simulator == "spike":
        from merlin.runtime.backends.spike_model import declared_memory

        span = declared_memory(elf_path)
        if span is not None:
            start, nbytes = span
            if (
                type(start) is not int
                or type(nbytes) is not int
                or nbytes <= 0
                or any(row["base"] < start or row["base"] + row["bytes"] > start + nbytes for row in regions)
            ):
                raise ValueError("memory readback region exceeds the ELF-declared simulator memory")
        if request["regions_path"] is not None:
            raise ValueError("HTIF signature has no region manifest")
        suffix: tuple[str, ...] = (f"+signature={output}", "+signature-granularity=1")
    else:
        manifest = _absolute(request["regions_path"], name="regions_path")
        manifest_bytes = "".join(lines).encode("ascii")
        if manifest == output or not _regular(manifest) or manifest.read_bytes() != manifest_bytes:
            raise ValueError("GSIM region manifest is not exact or regular")
        mode = "+dump-mode=coherent-packet" if packet else "+dump-mode=coherent"
        suffix = (f"+dump-regions={manifest}", f"+dump-out={output}", mode)
    result = MemoryReadbackCommand(
        suffix, elf_path, expected_sha, output, manifest, manifest_bytes, request, _request_digest(request)
    )
    result.revalidate()
    return result
