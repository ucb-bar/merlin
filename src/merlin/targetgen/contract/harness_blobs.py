"""Stage renderer-supplied constant operands beside a runner-owned C harness.

The target renderer owns tensor layout and bytes. The runner owns filenames,
assembly, and linking so no target-specific source is needed in the core.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path


def _symbol_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and ("A" <= value[0] <= "Z" or "a" <= value[0] <= "z" or value[0] == "_")
        and all(
            "A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9" or char == "_"
            for char in value[1:]
        )
    )


def _payload_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and ("A" <= value[0] <= "Z" or "a" <= value[0] <= "z" or "0" <= value[0] <= "9" or value[0] == "_")
        and all(
            "A" <= char <= "Z" or "a" <= char <= "z" or "0" <= char <= "9" or char in "_.-"
            for char in value[1:]
        )
        and value.endswith(".bin")
        and Path(value).name == value
    )


def render_blob_asm(symbol: str, payload_name: str, *, align: int, elems: int, section: str = ".rodata") -> str:
    """Define an aligned C-visible symbol from exact binary input bytes."""
    if not _symbol_name(symbol):
        raise ValueError(f"invalid harness blob symbol {symbol!r}")
    if not _payload_name(payload_name):
        raise ValueError(f"invalid harness blob payload name {payload_name!r}")
    if (type(align) is not int or type(elems) is not int or
            align < 1 or align & (align - 1) or elems < 1):
        raise ValueError("harness blob alignment must be a power of two and element count positive")
    if section not in (".data", ".rodata"):
        raise ValueError("harness blob needs an explicit initialized data section")
    return (
        f"  .section {section}\n"
        f"  .balign {align}\n"
        f"  .globl {symbol}\n"
        f"{symbol}:\n"
        f'  .incbin "{payload_name}"\n'
        f"  .size {symbol}, . - {symbol}\n"
        f"  /* {elems} elements */\n"
    )


def stage_harness_blobs(workdir: Path, blobs: Mapping[str, Mapping]) -> tuple[Path, ...]:
    """Validate and materialize a renderer's sidecars for the current build."""
    sources = []
    records = []
    extended = False
    for symbol, spec in sorted(blobs.items()):
        from merlin.llvmlower.weight_payload import WeightPayload

        if (not isinstance(spec, Mapping)
                or set(spec) not in ({"bytes", "align", "elems"}, {"bytes", "align", "elems", "section"})):
            raise ValueError(f"invalid harness blob declaration for {symbol!r}")
        payload = spec["bytes"]
        align, elems = spec["align"], spec["elems"]
        lazy = type(payload) is WeightPayload
        if (not (type(payload) is bytes or lazy) or not len(payload) or
                type(align) is not int or type(elems) is not int or
                elems < 1 or len(payload) % elems):
            raise ValueError(f"invalid harness blob bytes/extent for {symbol!r}")
        name = f"harness_blob_{symbol}.bin"
        section = spec.get("section", ".rodata")
        asm = render_blob_asm(symbol, name, align=align, elems=elems, section=section)
        blob_path = workdir / name
        source = workdir / f"harness_blob_{symbol}.S"
        if blob_path.is_symlink() or source.is_symlink():
            raise ValueError("harness blob output path is a symlink")
        if lazy:
            expected = hashlib.sha256()
            with tempfile.TemporaryDirectory(dir=workdir, prefix="pending-blob-") as temporary:
                pending = Path(temporary) / name
                with pending.open("wb") as stream:
                    for chunk in payload.chunks():
                        expected.update(chunk)
                        stream.write(chunk)
                with pending.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if pending.stat().st_size != len(payload) or actual != expected.hexdigest():
                    raise ValueError("realized harness blob differs from complete source bytes")
                pending.replace(blob_path)
            digest = actual
        else:
            blob_path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
        source.write_text(asm, encoding="utf-8")
        sources.append(source)
        records.append({
            "symbol": symbol,
            "file": name,
            "assembly": source.name,
            "object": source.with_suffix(".o").name,
            "sha256": digest,
            "bytes": len(payload),
            "alignment": align,
            "elements": elems,
            **({"storage": "streamed" if lazy else "bytes", "section": section} if lazy or "section" in spec else {}),
        })
        extended |= lazy or "section" in spec
    receipt = workdir / "harness_blobs.json"
    if receipt.is_symlink():
        raise ValueError("harness blob receipt path is a symlink")
    if records:
        receipt.write_text(json.dumps(
            {"schema": "merlin.harness_blobs.v2" if extended else "merlin.harness_blobs.v1", "blobs": records},
            sort_keys=True, indent=2) + "\n", encoding="utf-8")
    else:
        receipt.unlink(missing_ok=True)
    return tuple(sources)
