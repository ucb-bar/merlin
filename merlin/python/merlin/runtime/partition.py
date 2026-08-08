"""Carve one shared DRAM into per-image windows, from a map someone else can write.

Why this exists: on a chip where every core shares one DDR with no separate banks, running an
independent image per core means each has to be *linked* for its own slice — otherwise two images both
place their weights blob and activation arena at the same addresses and quietly overwrite each other.
The failure is not a crash: it is one model reading another's activations, producing plausible wrong
numbers. That was raised as the biggest problem with such a board, and the mechanism to answer it was
already here — ``spike_model._layout`` takes the window as ``(dram_base, dram_bytes)`` parameters and
fails closed when a model does not fit. What was missing is a way to *say* what the windows are.

So this module is deliberately thin: it turns a description into windows, validates them hard, and
leaves the placing to the existing layout code. Two ways to describe them:

* **equal split** — ``equal_partitions(n)``, the "one core each" case;
* **explicit map** — a JSON file, for when the slices differ (a model with big weights next to two
  small ones) or when some region is reserved for something else entirely.

The JSON is a list of windows, sizes in bytes or with a ``K``/``M``/``G`` suffix::

    {
      "dram_base": "0x80000000",
      "dram_bytes": "512M",
      "partitions": [
        {"name": "big",   "bytes": "256M"},
        {"name": "small", "bytes": "128M"},
        {"name": "spare", "bytes": "128M", "reserved": true}
      ]
    }

Sizes, not addresses, by default: bases are computed by packing in order, so a map cannot describe
overlapping windows by accident. An explicit ``base`` is allowed for a region fixed by something
outside our control, and is then *checked* against its neighbours rather than trusted.

**What this is not.** A window is a linking convention, not isolation. Nothing in the hardware stops a
stray pointer in one image from writing another's window; images produced by this repo respect their
window by construction (a static layout, no allocation outside the arena), but a partition map cannot
make that true of arbitrary code. Enforcement needs RISC-V PMP, which is a separate mechanism and is
disabled in the board configs this ships against. Any report generated here says so, because a
partition map that reads as a safety guarantee is worse than none.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Suffix multipliers for human-written sizes. Deliberately powers of two: a "256M" window that meant
#: 256e6 would silently be 6% smaller than the linker script's 256 MiB and the mismatch would land as
#: an overlap at the far end of the region.
_SUFFIX = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


class PartitionError(RuntimeError):
    """A partition map is unusable. Never repaired silently — a wrong window overlaps another."""


def parse_size(value: "int | str") -> int:
    """Bytes from an int, a decimal/hex string, or one with a K/M/G suffix.

    Accepts what a human writes in a config file (``"256M"``, ``"0x10000000"``, ``268435456``) and
    rejects anything ambiguous rather than guessing.
    """
    if isinstance(value, bool):                      # bool is an int subclass; never a size
        raise PartitionError(f"not a size: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise PartitionError(f"negative size: {value}")
        return value
    if not isinstance(value, str):
        raise PartitionError(f"not a size: {value!r}")
    text = value.strip().replace("_", "")
    if not text:
        raise PartitionError("empty size")
    mult = 1
    if text[-1:].lower() in _SUFFIX:
        mult = _SUFFIX[text[-1].lower()]
        text = text[:-1].strip()
    try:
        n = int(text, 16) if text[:2].lower() == "0x" else int(text, 10)
    except ValueError as exc:
        raise PartitionError(f"cannot read size {value!r}") from exc
    if n < 0:
        raise PartitionError(f"negative size: {value!r}")
    return n * mult


@dataclass(frozen=True)
class Partition:
    """One window of DRAM an image can be linked for."""

    name: str
    base: int
    size: int
    reserved: bool = False        # not ours to use; carved out so nothing else is placed there

    @property
    def end(self) -> int:
        return self.base + self.size

    def as_build_kwargs(self) -> dict:
        """The two arguments that place an image inside this window.

        Exactly the parameters ``spike_model.build`` / ``zephyr_model.build_app`` already take, so a
        partition is a pair of build arguments rather than a new code path — and the existing
        fail-closed check ("does not fit") applies unchanged.
        """
        if self.reserved:
            raise PartitionError(f"partition {self.name!r} is reserved; nothing may be built into it")
        return {"dram_base": self.base, "dram_bytes": self.size}

    def to_dict(self) -> dict:
        return {"name": self.name, "base": hex(self.base), "size_bytes": self.size,
                "size_mb": round(self.size / 2 ** 20, 1), "end": hex(self.end),
                "reserved": self.reserved}


@dataclass(frozen=True)
class PartitionMap:
    """A validated carve-up of one DRAM region."""

    dram_base: int
    dram_bytes: int
    partitions: tuple[Partition, ...]

    @property
    def end(self) -> int:
        return self.dram_base + self.dram_bytes

    def get(self, name: str) -> Partition:
        for p in self.partitions:
            if p.name == name:
                return p
        raise PartitionError(f"no partition named {name!r} "
                             f"(have {[p.name for p in self.partitions]})")

    def usable(self) -> tuple[Partition, ...]:
        return tuple(p for p in self.partitions if not p.reserved)

    def to_dict(self) -> dict:
        return {"dram_base": hex(self.dram_base), "dram_bytes": self.dram_bytes,
                "dram_mb": round(self.dram_bytes / 2 ** 20, 1),
                "partitions": [p.to_dict() for p in self.partitions],
                "unallocated_bytes": self.dram_bytes - sum(p.size for p in self.partitions),
                "note": ("Windows are a LINKING convention, not isolation: nothing in the hardware "
                         "prevents one image from writing another's window. Enforcement would need "
                         "RISC-V PMP, which is a separate mechanism.")}


def _validate(dram_base: int, dram_bytes: int, parts: list[Partition],
              align: int) -> PartitionMap:
    """Reject any map that could produce two images sharing an address."""
    if dram_bytes <= 0:
        raise PartitionError("dram_bytes must be positive")
    if not parts:
        raise PartitionError("a partition map needs at least one partition")
    seen: set[str] = set()
    for p in parts:
        if not p.name:
            raise PartitionError("every partition needs a name (it is how a build selects one)")
        if p.name in seen:
            raise PartitionError(f"duplicate partition name {p.name!r}")
        seen.add(p.name)
        if p.size <= 0:
            raise PartitionError(f"partition {p.name!r} has size {p.size}")
        if p.base % align:
            raise PartitionError(
                f"partition {p.name!r} base {hex(p.base)} is not {align}-byte aligned; an unaligned "
                "window breaks the image's own alignment assumptions")
        if p.base < dram_base or p.end > dram_base + dram_bytes:
            raise PartitionError(
                f"partition {p.name!r} [{hex(p.base)}, {hex(p.end)}) falls outside the region "
                f"[{hex(dram_base)}, {hex(dram_base + dram_bytes)}) — an image linked for it would "
                "address memory the chip does not have")
    ordered = sorted(parts, key=lambda p: p.base)
    for a, b in zip(ordered, ordered[1:]):
        if a.end > b.base:
            raise PartitionError(
                f"partitions {a.name!r} and {b.name!r} OVERLAP: [{hex(a.base)}, {hex(a.end)}) meets "
                f"[{hex(b.base)}, {hex(b.end)}). Two images placed here would silently overwrite each "
                "other's weights and activations — wrong numbers, not a crash.")
    return PartitionMap(dram_base=dram_base, dram_bytes=dram_bytes, partitions=tuple(ordered))


def equal_partitions(n: int, *, dram_base: int, dram_bytes: int,
                     align: int = 1 << 20, names: "list[str] | None" = None) -> PartitionMap:
    """``n`` equal windows — the "one independent image per core" case.

    Each window is truncated DOWN to ``align``, so the slices never grow into one another; the
    remainder is left unallocated rather than handed to the last partition, which keeps every window
    the same size and therefore keeps a per-core image interchangeable between them.
    """
    if n < 1:
        raise PartitionError(f"need at least one partition, got {n}")
    slice_size = (dram_bytes // n) & ~(align - 1)
    if slice_size <= 0:
        raise PartitionError(
            f"{dram_bytes / 2**20:.0f} MB does not divide into {n} windows of at least "
            f"{align / 2**20:.0f} MB")
    labels = names or [f"core{i}" for i in range(n)]
    if len(labels) != n:
        raise PartitionError(f"got {len(labels)} names for {n} partitions")
    parts = [Partition(name=labels[i], base=dram_base + i * slice_size, size=slice_size)
             for i in range(n)]
    return _validate(dram_base, dram_bytes, parts, align)


def load_partition_map(path: "str | Path", *, dram_base: int | None = None,
                       dram_bytes: int | None = None, align: int = 1 << 20) -> PartitionMap:
    """Read a JSON partition map. ``dram_base``/``dram_bytes`` override the file's own values.

    Sizes pack in order unless a partition states its own ``base``; an explicit base is validated
    against its neighbours rather than trusted, so a hand-written map cannot introduce an overlap.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except OSError as exc:
        raise PartitionError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PartitionError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise PartitionError(f"{path}: expected an object at the top level")

    base = dram_base if dram_base is not None else parse_size(doc.get("dram_base", 0x80000000))
    size = dram_bytes if dram_bytes is not None else parse_size(doc.get("dram_bytes", 0))
    if size == 0:
        raise PartitionError(f"{path}: dram_bytes is required (or pass it explicitly)")

    raw = doc.get("partitions")
    if not isinstance(raw, list) or not raw:
        raise PartitionError(f"{path}: 'partitions' must be a non-empty list")
    parts: list[Partition] = []
    cursor = base
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PartitionError(f"{path}: partition {i} is not an object")
        unknown = set(item) - {"name", "bytes", "size", "base", "reserved"}
        if unknown:
            # A typo'd key would otherwise be ignored and the partition would silently take its
            # default size -- which is the overlap this whole module exists to prevent.
            raise PartitionError(f"{path}: partition {i} has unknown key(s) {sorted(unknown)}")
        if "bytes" not in item and "size" not in item:
            raise PartitionError(f"{path}: partition {i} needs 'bytes'")
        psize = parse_size(item.get("bytes", item.get("size")))
        pbase = parse_size(item["base"]) if "base" in item else cursor
        if pbase % align:
            pbase = (pbase + align - 1) & ~(align - 1)
        parts.append(Partition(name=str(item.get("name", f"part{i}")), base=pbase, size=psize,
                               reserved=bool(item.get("reserved", False))))
        cursor = (pbase + psize + align - 1) & ~(align - 1)
    return _validate(base, size, parts, align)


def main(argv: "list[str] | None" = None) -> int:
    """``python -m merlin.runtime.partition`` — show the windows a map or an equal split produces."""
    import argparse

    from .boards import board as _board

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--board", default=None, help="take dram_base/dram_bytes from this descriptor")
    ap.add_argument("--map", default=None, help="a JSON partition map")
    ap.add_argument("--equal", type=int, default=None, help="split into N equal windows instead")
    ap.add_argument("--dram-base", default=None)
    ap.add_argument("--dram-bytes", default=None)
    a = ap.parse_args(argv)

    base = parse_size(a.dram_base) if a.dram_base else None
    size = parse_size(a.dram_bytes) if a.dram_bytes else None
    if a.board:
        brd = _board(a.board)
        base = base if base is not None else brd.dram_base
        size = size if size is not None else brd.dram_bytes
    if a.map:
        pm = load_partition_map(a.map, dram_base=base, dram_bytes=size)
    elif a.equal:
        if base is None or size is None:
            print("--equal needs --board or --dram-base/--dram-bytes")
            return 2
        pm = equal_partitions(a.equal, dram_base=base, dram_bytes=size)
    else:
        print("pass --map or --equal")
        return 2
    print(json.dumps(pm.to_dict(), indent=2))
    return 0


if __name__ == "__main__":            # pragma: no cover
    raise SystemExit(main())
