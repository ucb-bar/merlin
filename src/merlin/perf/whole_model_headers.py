"""The vendor parameter header a whole-model program is built against: asserted by content for the
machine, copied into the build's harness, and recorded by what the program actually read.

Split out of :mod:`merlin.perf.whole_model_build` (which re-exports every name here) to keep that
module under the repository's module-size limit.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _build_error(message: str) -> Exception:
    """The builder's own refusal type (imported at call time: that module imports this one)."""
    from .whole_model_build import WholeModelBuildError

    return WholeModelBuildError(message)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def machine_header(machine: str, header: str | Path, header_sha256: str | None = None) -> dict[str, Any]:
    """The parameter header a program for ``machine`` is built against, ASSERTED by content, or a refusal.

    A program is compiled for one machine, and which machine is decided by the vendor parameter header
    it includes: two designs of one target can differ in a readout the header either declares or does
    not, and an ELF built against the wrong one runs and reads back zeros. So the header is never
    whatever sits in some checkout. ``machine`` names an entry of the hardware registry
    (``src/merlin/resources/contracts/hardware_pins.yaml``); the ABI header digest it declares -- on itself or on an
    entry it is ``built_from`` -- is what ``header``'s bytes must hash to. A machine the registry
    declares no ABI header for is buildable only when the caller asserts one (``header_sha256``), and
    the record then says the assertion is the caller's, with the registry's own notes beside it.
    """
    import yaml

    from merlin.common import provenance as PROV

    registry = yaml.safe_load(PROV.pins_path().read_text(encoding="utf-8")) or {}
    entries = {**(registry.get("pins") or {}), **(registry.get("artifacts") or {})}
    if machine not in entries:
        raise _build_error(f"{machine!r} is not an entry of the hardware registry; name the machine built for")
    declared, declared_by, seen, frontier = None, None, set(), [machine]
    while frontier and declared is None:
        name = frontier.pop(0)
        if name in seen or name not in entries:
            continue
        seen.add(name)
        entry = entries[name] or {}
        if entry.get("abi_header_sha256"):
            declared, declared_by = str(entry["abi_header_sha256"]), name
        frontier.extend(str(n) for n in entry.get("built_from") or ())
    path = Path(header)
    if not path.is_file():
        raise _build_error(f"the parameter header {path} does not exist")
    actual = _sha256(path)
    if declared is None and not header_sha256:
        raise _build_error(
            f"the registry declares no ABI header for {machine!r} (nor for anything it is built from), so "
            f"which header is correct for it is UNKNOWN; assert one explicitly with its sha256"
        )
    for expected, who in ((declared, f"the registry ({declared_by})"), (header_sha256, "the caller")):
        if expected and actual != expected:
            raise _build_error(
                f"{path} hashes to {actual[:12]}, and {who} declares {expected[:12]} for {machine!r}; "
                f"building against it would describe a different machine"
            )
    record = {"machine": machine, "header": str(path.resolve()), "sha256": actual}
    if declared is not None:
        record.update({"declared_by": declared_by, "status": "registry_declared"})
    else:
        entry = entries[machine] or {}
        record.update(
            {
                "status": "caller_asserted",
                "registry_notes": [str(g) for g in entry.get("gaps") or ()] or entry.get("notes"),
            }
        )
    return record


def _with_header(recipe, header: Path, into: Path, overrides: Sequence[Path] = ()):
    """The target's recipe over a copy of its harness tree in which ``header`` replaces its namesake.

    Only the header moves (and each of ``overrides``, likewise by namesake); everything else in the
    tree is the target's own, byte for byte. Each namesake has to exist exactly once in the tree -- a
    file the tree does not include would change nothing, and one it holds twice would leave which one
    is read to the include order.
    """
    import shutil

    root = Path(os.path.commonpath([str(p) for p in recipe.include_roots]))
    placed = []
    for replacement in (Path(header), *(Path(o) for o in overrides)):
        matches = [p for p in root.rglob(replacement.name) if p.is_file()]
        if len(matches) != 1:
            raise _build_error(
                f"the harness tree {root} holds {len(matches)} file(s) named {replacement.name!r}; "
                f"a replacement must replace exactly one"
            )
        placed.append((replacement, matches[0].relative_to(root)))
    if len({rel for _r, rel in placed}) != len(placed):
        raise _build_error("two replacements name the same harness file")
    if into.exists():
        shutil.rmtree(into)
    shutil.copytree(root, into, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    # The copy keeps the source's modes, and a frozen source snapshot is read-only: its directories
    # are ours, so open them (a read-only copy would block both the replacement and the next build's
    # removal of the copy), and replace each namesake by unlinking it first rather than writing
    # through a file the copy cannot open.
    for directory in (into, *(p for p in into.rglob("*") if p.is_dir() and not p.is_symlink())):
        directory.chmod(directory.stat().st_mode | 0o700)
    for replacement, rel in placed:
        target = into / rel
        target.unlink()
        shutil.copyfile(replacement, target)

    def move(path: Path) -> Path:
        path = Path(path)
        try:
            return into / path.relative_to(root)
        except ValueError:
            return path

    return dataclasses.replace(
        recipe,
        include_roots=tuple(move(p) for p in recipe.include_roots),
        support_sources=tuple(move(p) for p in recipe.support_sources),
        link_script=move(recipe.link_script),
    )


def _headers_read(receipt: Mapping[str, Any], program: Path) -> dict[str, str]:
    """Every file the preprocessor READ to compile the program, with its digest.

    Asked of the compiler (``-M``) rather than assumed from a header's name: which parameter header a
    build saw is whichever one the include path resolved, and a copy of the vendor tree that differs by
    one line is a different machine.
    """
    done = subprocess.run(
        [str(receipt["compiler"]), *receipt["flags"], *receipt.get("includes", ()), "-M", str(program)],
        capture_output=True,
        text=True,
        cwd=str(program.parent),
    )
    if done.returncode != 0:
        return {"UNKNOWN": f"the compiler could not list its inputs: {done.stderr[-300:]}"}
    tokens = done.stdout.replace("\\\n", " ").split()
    files = [t for t in tokens[1:] if t != "\\"]
    return {str(Path(program.parent, f).resolve()): _sha256(Path(program.parent, f)) for f in files}


def machine_harts(machine: str) -> list[dict[str, Any]]:
    """The harts the hardware registry declares for ``machine`` (``[{hart, isa, unit, vlen?}]``), in
    hart order; empty when it declares none. Read on the entry itself -- a hart layout is a property of
    the elaborated design, not of anything it is built from."""
    import yaml

    from merlin.common import provenance as PROV

    registry = yaml.safe_load(PROV.pins_path().read_text(encoding="utf-8")) or {}
    entries = {**(registry.get("pins") or {}), **(registry.get("artifacts") or {})}
    rows = []
    for row in (entries.get(machine) or {}).get("harts") or ():
        if not isinstance(row, Mapping) or not str(row.get("isa") or "") or row.get("hart") is None:
            raise _build_error(f"{machine!r} declares a hart without an index and an ISA: {row!r}")
        rows.append(
            {"hart": int(row["hart"]), "isa": str(row["isa"]), "unit": bool(row.get("unit"))}
            | ({"vlen": int(row["vlen"])} if row.get("vlen") is not None else {})
        )
    return sorted(rows, key=lambda r: r["hart"])
