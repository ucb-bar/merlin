"""Where a whole-model program's outputs live in memory, and a run graded from a dump of it.

Split out of :mod:`merlin.perf.whole_model_build` (which re-exports both names): the build writes the
map beside the ELF it linked, and a ``host_dump`` reader (:mod:`merlin.perf.whole_model_gsim`) grades a
run from the bytes the map names. Every address and size is read off the linked ELF's own symbol table,
never computed; nothing here names a target.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from merlin.perf.whole_model_build import LocalReference

__all__ = ["MEMORY_MAP_SCHEMA", "grade_memory", "memory_map"]


def _build():
    # The build module imports this one at load; reach back lazily for its error type and digest.
    from merlin.perf import whole_model_build

    return whole_model_build


#: v2 adds, per group, ``inputs``: the buffers ANOTHER group produced that this group reads, by symbol,
#: address and size -- what a dump must contain for the group to be graded LOCALLY -- and, at the top,
#: ``dump``: the union a reader must capture. An exact group is graded on its reference recomputed from
#: those inputs (``grade_memory(..., local=LocalReference)``); the chained oracle digest is information.
MEMORY_MAP_SCHEMA = "whole_model_memory_map_v2"


def _symbols(elf: Path) -> dict[str, tuple[int, int]]:
    """``{symbol: (address, size in bytes)}`` for every sized symbol the linked ELF defines."""
    from merlin.llvmlower import toolchain

    listed = subprocess.run(
        [str(toolchain.nm()), "--print-size", "--defined-only", "--radix=d", str(elf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    found: dict[str, tuple[int, int]] = {}
    for line in listed.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            found[parts[3]] = (int(parts[0]), int(parts[1]))
    return found


def memory_map(elf: str | Path, model: Mapping[str, Any], buffer: Mapping[str, Any]) -> dict[str, Any]:
    """Where each group's committed output lives in the linked program, for a HOST-SIDE check.

    One row per device group, in program order: the C symbol, its address and size read off the ELF's
    own symbol table (never computed), the element width that size implies, and how the group is
    graded. A tolerance-declared group (``bound_lsb``) also names its two operand buffers and the facts
    its contract grades on, so a reader of a memory dump can recompute its reference exactly as the
    contract states it (see :func:`grade_memory`). Everything is little-endian two's-complement.
    """
    symbols = _symbols(Path(elf))
    stated = {int(r["group"]): r for r in (buffer.get("whole_program") or {}).get("per_group") or ()}

    def place(symbol: str, elements: int) -> dict[str, Any]:
        if symbol not in symbols:
            raise _build().WholeModelBuildError(f"the linked program defines no sized symbol {symbol!r}")
        address, size = symbols[symbol]
        if elements <= 0 or size % elements:
            raise _build().WholeModelBuildError(
                f"{symbol!r} holds {size} bytes, not a whole number of {elements} elements"
            )
        return {
            "symbol": symbol,
            "address": address,
            "bytes": size,
            "elements": elements,
            "element_bytes": size // elements,
        }

    sizes = {str(b["name"]): int(b["elements"]) for b in model.get("buffers") or ()}
    produced = {str(r["operands"]["dst"]): int(g) for g, r in stated.items() if "dst" in (r.get("operands") or {})}
    rows = []
    for step in model["steps"]:
        group = int(step["group"])
        row = {"group": group, "kind": step["kind"], **place(str(step["out"]), sizes[str(step["out"])])}
        # A FUSED REGION reads what its members read from OUTSIDE it: its internal members' outputs
        # are its own, and a dump holds nothing a reader may grade them from.
        members = [int(m["group"]) for m in step["members"]] if step["kind"] == "region" else [group]
        row["inputs"] = [
            {**place(name, sizes[name]), "produced_by": produced[name]}
            for member in members
            for key, name in ((stated.get(member) or {}).get("operands") or {}).items()
            if key != "dst" and name in produced and produced[name] not in members and name in sizes
        ]
        entry = (stated.get(group) or {}).get("entry") or {}
        if step["kind"] == "region":
            # GRADED AT ITS BOUNDARY: each internal member recomputed from the region's inputs, the
            # boundary from those -- as an exact group, or within the bound its own entry declares.
            row.update({"compare": "region", "members": members})
            boundary = step["members"][-1]
            if entry.get("bound_lsb") is not None:
                row["boundary"] = {
                    "compare": "bounded",
                    "bound_lsb": int(entry["bound_lsb"]),
                    "lhs_scale": float(entry["lhs_scale"]),
                    "rhs_scale": float(entry["rhs_scale"]),
                    "relu": "relu" in (entry.get("epilogue") or ()),
                    "lhs": str(boundary["lhs"]),
                    "rhs": str(boundary["rhs"]),
                }
            else:
                row["boundary"] = {"compare": "exact"}
        elif step["kind"] == "fused":
            # A merged group is graded on the core (its producer's reference needs the contraction);
            # a host-side reader of a dump has no statement of it, and says so rather than guessing.
            row.update({"compare": "merged", "graded_as": int(step["graded_as"])})
        elif entry.get("bound_lsb") is not None:
            row.update(
                {
                    "compare": "bounded",
                    "bound_lsb": int(entry["bound_lsb"]),
                    "lhs_scale": float(entry["lhs_scale"]),
                    "rhs_scale": float(entry["rhs_scale"]),
                    "relu": "relu" in (entry.get("epilogue") or ()),
                    "lhs": place(str(step["lhs"]), sizes[str(step["lhs"])]),
                    "rhs": place(str(step["rhs"]), sizes[str(step["rhs"])]),
                }
            )
        else:
            row["compare"] = "exact"
        rows.append(row)
    dump = sorted({r["symbol"] for r in rows} | {i["symbol"] for r in rows for i in r["inputs"]})
    return {
        "schema": MEMORY_MAP_SCHEMA,
        "elf": str(elf),
        "elf_sha256": _build()._sha256(elf),
        "byte_order": "little",
        "signed": True,
        "reference": "local: each exact group against its reference recomputed from its dumped inputs",
        "dump": {"symbols": dump, "bytes": sum(symbols[name][1] for name in dump)},
        "groups": rows,
    }


def grade_memory(
    read, layout: Mapping[str, Any], oracle: Mapping[str, Any], *, local: LocalReference | None = None
) -> dict[str, Any]:
    """A run's outputs, read from memory, against the reference -- the host-side half of ``host_dump``.

    ``read(address, size) -> bytes`` returns the program's memory after the run (a dump, a simulator's
    backing store). Every group is graded LOCALLY, on its own arithmetic:

    * a 'bounded' group is recomputed from the two operand buffers the device actually read, as the
      RESIDUAL_ADD contract states it -- ``sat(roundeven(f32(lhs)*lhs_scale + f32(rhs)*rhs_scale))`` in
      single precision, then the activation -- and every element must lie within ``bound_lsb``;
    * an 'exact' group is recomputed by ``local`` (:class:`LocalReference`) from the buffers its row
      lists as ``inputs``, read from the same memory, and must equal the device's output exactly.
      Without ``local`` an exact group cannot be graded and is reported ``unverified`` -- never agreed.

    The chained oracle digest of each exact group is reported under ``chained`` as INFORMATION: below a
    legitimate bounded difference upstream it differs for a correct run.

    A row may carry its group's EXACTNESS CONTRACT (``exactness``, :meth:`merlin.perf.exactness.Exactness.
    to_dict`, put there by whoever wrote the map): the group is then held to exactly that contract --
    exact, its op's own declared bound, or the bound its form's contract states -- instead of the
    comparison its op implies, and ``contracts`` says which was applied to each group.  ``evidence``
    holds each graded group's own numbers (largest difference, elements that differ, elements).
    """
    import numpy as np

    from merlin.perf import exactness as EX
    from merlin.perf.layer_bench import reference as ref

    evidence: dict[str, dict[str, int]] = {}
    contracts: dict[str, str] = {}

    def passes(group: str, row: Mapping[str, Any], got, want, *, builtin_ok: bool) -> bool:
        """Whether ``got`` passes ``row``'s contract against ``want``; records the numbers and the label."""
        delta = np.abs(got.astype(np.int64) - want.astype(np.int64)) if got.shape == want.shape else None
        if delta is not None:
            evidence[group] = {
                "max_abs": int(delta.max()) if delta.size else 0,
                "mismatches": int((delta != 0).sum()),
                "elements": int(delta.size),
            }
        declared = row.get("exactness")
        if not declared:
            contracts[group] = "exact" if row.get("compare") == "exact" else f"bounded(<={row.get('bound_lsb')} LSB)"
            return builtin_ok
        exactness = EX.Exactness.from_dict(declared)
        contracts[group] = exactness.label()
        if delta is None:
            return False
        return bool(EX.judge(exactness, **evidence[group])["passed"])

    def values(place):
        raw = bytes(read(int(place["address"]), int(place["bytes"])))
        return np.frombuffer(raw, dtype=f"<i{int(place['element_bytes'])}").astype(np.int64)

    def bounded(got, lhs, rhs, spec, element_bytes):
        """``(max_abs, over, want)``: ``got`` against the RESIDUAL_ADD contract's reference of ``lhs``, ``rhs``."""
        total = lhs.astype(np.float32) * np.float32(spec["lhs_scale"]) + rhs.astype(np.float32) * np.float32(
            spec["rhs_scale"]
        )
        info = np.iinfo(f"i{int(element_bytes)}")
        want = np.clip(np.rint(total), info.min, info.max).astype(np.int64)
        if spec["relu"]:
            want = np.maximum(want, 0)
        worst = int(np.abs(got - want).max()) if got.size else 0
        return worst, int((np.abs(got - want) > int(spec["bound_lsb"])).sum()), want

    agree, disagree, unverified = [], [], []
    chained: dict[str, list] = {"agree": [], "disagree": []}
    for row in layout["groups"]:
        group = str(row["group"])
        if row["compare"] == "region":
            # A FUSED REGION, AT ITS BOUNDARY: its internal members recomputed (LocalReference) from the
            # region's inputs as the dump holds them, then the boundary from those. Never graded on an
            # internal member's own buffer, which the region's kernel is not required to write.
            if local is None:
                unverified.append({"group": group, "why": "a region is graded on its local reference; none was given"})
                continue
            held = {i["symbol"]: values(i) for i in row.get("inputs") or ()}
            try:
                for member in row["members"][:-1]:
                    _commands, dst = local.slices[int(member)]
                    held[dst] = local.expected(int(member), {n: held[n] for n in local.inputs_of(int(member))})
                got = values(row)
                spec = row["boundary"]
                if spec["compare"] == "bounded":
                    lhs, rhs = held[spec["lhs"]], held[spec["rhs"]]
                    worst, over, want_b = bounded(got, lhs, rhs, spec, row["element_bytes"])
                    graded_as = {**spec, "exactness": row.get("exactness")}
                    if passes(group, graded_as, got, want_b, builtin_ok=over == 0):
                        agree.append(group)
                    else:
                        disagree.append({"group": group, "max_abs": worst, "over": over})
                    continue
                want = local.expected(int(row["group"]), {n: held[n] for n in local.inputs_of(int(row["group"]))})
            except (KeyError, _build().WholeModelBuildError) as why:
                unverified.append({"group": group, "why": f"the region's reference could not be formed: {why}"})
                continue
            wrong = np.flatnonzero(want != got) if want.shape == got.shape else np.arange(max(got.size, 1))
            graded_as = {"compare": "exact", "exactness": row.get("exactness")}
            if passes(group, graded_as, got, want, builtin_ok=wrong.size == 0):
                agree.append(group)
            else:
                disagree.append(
                    {"group": group, "mismatches": int(wrong.size), "of": int(got.size), "first": int(wrong[0])}
                    if wrong.size
                    else {"group": group, "mismatches": 0, "of": int(got.size), "first": None}
                )
            continue
        if row["compare"] == "merged":
            unverified.append(
                {"group": str(row.get("graded_as", group)), "why": "a merged group is graded on the core"}
            )
            continue
        got = values(row)
        if row["compare"] == "bounded":
            lhs, rhs = values(row["lhs"]), values(row["rhs"])
            total = lhs.astype(np.float32) * np.float32(row["lhs_scale"]) + rhs.astype(np.float32) * np.float32(
                row["rhs_scale"]
            )
            info = np.iinfo(f"i{int(row['element_bytes'])}")
            want = np.clip(np.rint(total), info.min, info.max).astype(np.int64)
            if row["relu"]:
                want = np.maximum(want, 0)
            worst = int(np.abs(got - want).max()) if got.size else 0
            over = int((np.abs(got - want) > int(row["bound_lsb"])).sum())
            if passes(group, row, got, want, builtin_ok=over == 0):
                agree.append(group)
            else:
                disagree.append({"group": group, "max_abs": worst, "over": over})
            continue
        chain = (oracle.get("groups") or {}).get(group) or {}
        digest = ref.fnv1a64_words(np.ascontiguousarray(got, dtype="<i8").tobytes()) & ref.DIGEST_MASK
        (chained["agree"] if chain and int(chain["fnv1a"]) == int(digest) else chained["disagree"]).append(group)
        if local is None:
            unverified.append({"group": group, "why": "no local reference was given"})
            continue
        try:
            want = local.expected(int(row["group"]), {i["symbol"]: values(i) for i in row.get("inputs") or ()})
        except _build().WholeModelBuildError as why:
            unverified.append({"group": group, "why": str(why)})
            continue
        if want.shape != got.shape:
            disagree.append({"group": group, "why": f"reference has {want.size} elements, device {got.size}"})
            continue
        wrong = np.flatnonzero(want != got)
        if passes(group, row, got, want, builtin_ok=wrong.size == 0):
            agree.append(group)
        else:
            # The same row as ever; its largest difference and the contract applied are in ``evidence`` and
            # ``contracts`` (a contract stricter than the op can fail a group with nothing mismatched).
            disagree.append(
                {"group": group, "mismatches": int(wrong.size), "of": int(got.size), "first": int(wrong[0])}
                if wrong.size
                else {"group": group, "mismatches": 0, "of": int(got.size), "first": None}
            )
    return {
        "gate": "local",
        "agree": agree,
        "disagree": disagree,
        "unverified": unverified,
        "quotable": not disagree and not unverified,
        "chained": {"note": "information only", **chained},
        "evidence": evidence,
        "contracts": contracts,
    }
