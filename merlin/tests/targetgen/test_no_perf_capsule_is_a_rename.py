"""A declared performance axis must reach the emitted interface, or the sweep measured one point twice.

A perf family's `fit_axes` is the axis its claim is fitted over, and `gate.capacity` can demand more
than one level of it (PC: `at_least_two_separation_regimes`). Nothing checked that two levels produce
two PROGRAMS. Measured before this test existed, on the materialized gemmini corpus:

* `PC00_k64` and `PC01_k128` had BYTE-IDENTICAL interfaces, differing only in the capsule name and a
  prose `source_reference` that claimed K=64 and K=128. So PC's two "separation regimes" were one
  regime with two labels, and a paired differential across them would have measured the same program
  twice and reported it as a two-level result.
* `PL00_k16`/`PL02_k32` and `PL01_k16`/`PL03_k32` were identical in the same way -- PL's whole K axis
  was inert.

The cause was one silent default: `expand_sweeps` resolves an `axes:` entry to the BARE name (`K`),
while `build_resident_reuse`/`build_attention_qk` read `entry["K_tiles"]` only, so `entry.get(
"K_tiles", 1)` returned 1 and both points emitted the tile edge. Every other builder already read both
spellings. A default that stands in for a DECLARED value is the failure here; see the repo rule that an
inert lever must be proven by a measured emitted-code delta, not by its declaration.

This is the corpus-level invariant rather than a builder unit test on purpose: the same silent default
can reappear in any builder, and only the emitted bytes show whether a declared axis survived.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

import pytest
import yaml

from merlin.common.paths import merlin_dir

_PERF = merlin_dir() / "contract" / "capsules" / "_perf"


def _capsules() -> list[dict]:
    out = []
    if not _PERF.is_dir():
        return out
    for d in sorted(p for p in _PERF.iterdir() if p.is_dir()):
        cap, iface = d / "capsule.yaml", d / "capsule.interface.mlir"
        if not (cap.is_file() and iface.is_file()):
            continue
        try:
            doc = yaml.safe_load(cap.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        out.append({"name": d.name, "doc": doc,
                    "iface_sha": sha256(iface.read_bytes()).hexdigest()})
    return out


def _consumes_a_contraction_axis(doc: dict) -> bool:
    """Does this capsule's operation have a contraction depth at all?

    Derived from the operand ROLES the capsule already declares, not from an op-name list. An operand
    with role ``weight`` is the right-hand side of a contraction, so its K is an extent of the program
    and a sweep over K must change it. An elementwise epilogue like a bias add declares ``src``/``bias``
    and no weight: its operands are `X[M,N]` and `B[N]`, so K is genuinely not one of its extents and
    two K levels of it ARE the same program. Distinguishing these two cases is the whole job here --
    failing both would make the guard unusable, and failing neither is what let the defect ship.
    """
    return any(str((i or {}).get("role") or "") == "weight" for i in (doc.get("inputs") or []))


def test_a_declared_axis_reaches_the_program_of_every_op_that_has_it():
    """The defect: two levels of a fit axis emitting one program, for an op that HAS that axis."""
    caps = _capsules()
    if not caps:
        pytest.skip("no materialized performance corpus")
    by_family: dict[str, list[dict]] = defaultdict(list)
    for c in caps:
        fam = ((c["doc"].get("performance") or {}).get("family") or "")
        if fam:
            by_family[fam].append(c)
    assert by_family, "no capsule declares a performance family, so this test established nothing"

    collisions, redundant = {}, {}
    for fam, members in sorted(by_family.items()):
        seen: dict[str, list[dict]] = defaultdict(list)
        for c in members:
            seen[c["iface_sha"]].append(c)
        for sha, group in seen.items():
            if len(group) < 2:
                continue
            names = sorted(c["name"] for c in group)
            if any(_consumes_a_contraction_axis(c["doc"]) for c in group):
                collisions[f"{fam}:{sha[:12]}"] = names
            else:
                redundant[f"{fam}:{sha[:12]}"] = names
    assert not collisions, (
        "capsules of a contraction op in one performance family emit byte-identical interfaces, so a "
        f"declared axis never reached the program: {collisions}")

    # Reported, not failed. An axis-independent part shared by two groups is a true fact about the op,
    # but it is still the SAME program measured twice, which costs oracle time on a substrate whose
    # cost is set by output size. Kept visible so the redundancy is a decision rather than an accident.
    if redundant:
        print(f"\n[redundant] axis-independent capsules sharing one program: {redundant}")


def test_a_family_whose_gate_demands_two_levels_has_two_distinct_programs():
    """The capacity clause is about EVIDENCE, so it must be satisfied by programs, not by names."""
    caps = _capsules()
    if not caps:
        pytest.skip("no materialized performance corpus")
    checked = 0
    for fam in sorted({((c["doc"].get("performance") or {}).get("family") or "") for c in caps} - {""}):
        members = [c for c in caps
                   if ((c["doc"].get("performance") or {}).get("family")) == fam]
        capacity = str((((members[0]["doc"].get("performance") or {}).get("gate")) or {})
                       .get("capacity") or "")
        # "at least two <something>" is the shape of every capacity clause that demands levels; read it
        # by its leading tokens rather than matching a family name, so a new family is covered too.
        tokens = capacity.replace("-", "_").split("_")
        if tokens[:3] != ["at", "least", "two"]:
            continue
        distinct = {c["iface_sha"] for c in members}
        assert len(distinct) >= 2, (
            f"{fam} declares gate.capacity {capacity!r} but its {len(members)} capsule(s) emit "
            f"{len(distinct)} distinct program(s); the levels are labels, not measurements")
        checked += 1
    if not checked:
        pytest.skip("no family declares an at-least-two capacity clause")
