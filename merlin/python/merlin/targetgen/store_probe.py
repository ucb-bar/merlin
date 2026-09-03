"""Measure the extent at which a target's backend DECLINES, and report the capacity that predicts it.

WHY A MEASUREMENT RATHER THAN A DECLARATION. A target's on-chip operand store decides the whole
memory-mapping axis -- which residency regimes a capsule reaches, how deep an accumulation the
corpus can demand, and whether the perf ladder has residency bands to fit against. Where the RTL
facts carry a memory list the store is DERIVED and none of this is needed. Where they do not, the
tempting move is to pick a plausible SRAM and declare its size, and that move has already been made
and refuted once: on the device this module was written for, the matrix register file (65,536 B,
matching the shipped ISA model's 64 registers x 32 depth x 32 B) was the obvious candidate, and a
32x256x256 layer needing 73,728 elements RUNS UNBLOCKED on its cosim. Declaring it would have made
the runtime split layers the device handles -- which on a bf16 accumulator also changes the
reduction order, so the wrong number is not merely conservative, it changes the arithmetic.

The honest question is not "which SRAM is it" but "where does this backend actually stop", and that
is answerable by running the backend. This module asks it: grow one contraction's working set until
the target's own program oracle stops producing a correct result, and report the bracket. A store
whose capacity falls inside that bracket is the one that predicts the device's behaviour; a store
whose capacity does not is refuted, however plausible its name.

WHAT IT DOES NOT DO. It does not declare the store. The bracket is evidence and it is written down;
which SRAM to name remains a reviewed decision, for the same reason `capability_derive.reconcile`
audits rather than authors -- a derivation bug that silently moved a capacity would move every
residency verdict downstream of it with nobody noticing.

TARGET-AGNOSTIC: the sweep, the oracle call and the verdict are all keyed on the target parameter;
nothing here names a device, a store or a size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProbePoint:
    """One measured extent: what was asked, and what the backend did with it."""

    m: int
    k: int
    n: int
    elements: int                      # the working set in operand elements (lhs + weight)
    ran: bool
    detail: str = ""
    seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"m": self.m, "k": self.k, "n": self.n, "elements": self.elements,
                "ran": self.ran, "detail": self.detail[:300], "seconds": self.seconds}


@dataclass
class DeclineBracket:
    """Where the backend stopped, as an interval, plus every point that produced it."""

    target: str
    dtype: str
    largest_ran: int | None = None      # elements
    smallest_declined: int | None = None
    points: list[ProbePoint] = field(default_factory=list)
    unavailable: str = ""

    @property
    def decided(self) -> bool:
        return self.largest_ran is not None and self.smallest_declined is not None

    def predicts(self, capacity_elements: int) -> bool | None:
        """Whether a store of ``capacity_elements`` predicts the measured boundary.

        ``None`` when the sweep did not bracket a boundary -- an undecided probe must not be read as
        refuting anything, which is the difference between "we measured no decline" and "it declines
        above every size we tried"."""
        if not self.decided:
            return None
        return int(self.largest_ran) <= int(capacity_elements) < int(self.smallest_declined)

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "dtype": self.dtype,
                "largest_ran_elements": self.largest_ran,
                "smallest_declined_elements": self.smallest_declined,
                "decided": self.decided, "unavailable": self.unavailable,
                "points": [p.to_dict() for p in self.points],
                "method": ("one contraction's working set grown until the target's own program oracle "
                           "stops producing a correct result; the bracket is (largest that ran, "
                           "smallest that declined] in operand elements")}


#: Oracle tiers cheapest-first. A probe grades at the cheapest tier that RUNS: the question is whether
#: the device completed the layer, and the cycle-exact tier answers a different one at far greater cost.
_TIER_ORDER = ("L0", "L1", "L2", "L3", "L4", "L5")


def working_set_elements(m: int, k: int, n: int) -> int:
    """Operand elements a single contraction asks the store to hold: ``A[m,k]`` plus ``W[k,n]``.

    The OUTPUT is deliberately absent. It is drained to the accumulator rather than held in the
    operand store, and counting it would move the boundary by a term that is not competing for the
    space being measured."""
    return m * k + k * n


def ladder(edge: int, *, floor_elements: int, ceiling_elements: int) -> list[tuple[int, int, int]]:
    """Shapes to try, coarsest first, every extent a whole multiple of ``edge``.

    ``m`` is pinned at one tile and the square edge is swept, so the working set grows with the SQUARE
    of one varied extent and a handful of points spans orders of magnitude. Pinning m also keeps the
    written output small, which keeps each point cheap: what is being measured is what the store
    HOLDS, not what the device writes."""
    out: list[tuple[int, int, int]] = []
    d = int(edge)
    while True:
        w = working_set_elements(edge, d, d)
        if w > ceiling_elements:
            break
        if w >= floor_elements:
            out.append((int(edge), d, d))
        d *= 2
    return out


def probe(target: str, *, dtype: str, edge: int, package_dir, model_ext: str = "",
          floor_elements: int = 4096, ceiling_elements: int = 1 << 24,
          timeout: int = 1800, workroot=None, runs_root=None) -> DeclineBracket:
    """Run the ladder against ``target``'s own capsule path and bracket the decline.

    THE CAPSULE PATH, not a bespoke one. What has to be measured is where the backend stops holding a
    layer, and the backend only ever sees a layer as a capsule: the same builder, the same golden, the
    same oracle adapters the graded corpus uses. A separate harness would measure a different program
    and answer a different question -- and the model's own validation programs are fixed-shape, so
    there is no parameterized program to sweep anyway.

    Each point is graded at the LOOP tier. That is the tier that answers "did the device complete this
    and produce the right values", which is the question; the cycle-exact tier answers how long it took,
    which is not. A point that raises, times out, or grades anything but a pass is a DECLINE with its
    reason recorded -- never a silent skip, because "we could not ask" and "it declined" are different
    facts and only one is about the hardware.
    """
    import sys
    import tempfile
    import time

    out = DeclineBracket(target=target, dtype=dtype)
    try:
        from merlin.common.paths import merlin_dir
        caps = merlin_dir() / "contract" / "capsules"
        if str(caps) not in sys.path:
            sys.path.insert(0, str(caps))
        import generate_corpus as GC                                   # noqa: PLC0415
        from merlin.targetgen import capsule_runner as CR              # noqa: PLC0415
        from merlin.targetgen.target_experiment import load_target_experiment
    except Exception as exc:                    # noqa: BLE001 — cannot ask: undecided, with the reason
        out.unavailable = f"{type(exc).__name__}: {exc}"
        return out

    pkg = Path(package_dir)
    if not (pkg / "manifest.yaml").is_file():
        out.unavailable = f"no backend package at {pkg} (needs manifest.yaml); nothing to run against"
        return out

    try:
        te = load_target_experiment(GC._descriptor_for(target))
        binding = GC.CS.derive_binding(te, _datapath_of(target))
        adapters = CR.oracle_adapters(target, te.sim_via) or {}
    except Exception as exc:                    # noqa: BLE001
        out.unavailable = f"{type(exc).__name__}: {exc}"
        return out
    if not adapters:
        out.unavailable = f"{target!r} resolves no oracle adapter, so no point can be graded"
        return out

    shapes = ladder(edge, floor_elements=floor_elements, ceiling_elements=ceiling_elements)
    if not shapes:
        out.unavailable = (f"no shape between {floor_elements} and {ceiling_elements} elements at edge "
                           f"{edge}: widen the range rather than reading this as a decline")
        return out

    root = Path(workroot) if workroot is not None else Path(tempfile.mkdtemp(prefix="store_probe_"))
    root.mkdir(parents=True, exist_ok=True)
    runs = Path(runs_root) if runs_root is not None else (root / "runs")
    for (m, k, n) in shapes:
        elems = working_set_elements(m, k, n)
        started = time.time()
        ok, detail = False, ""
        try:
            entry = {"cat": "isa", "kind": "isa", "name": f"PROBE_store_m{m}k{k}n{n}",
                     "op": "matmul", "operand_dtype": dtype, "out": "Y0", "lhs": "A0", "weight": "W",
                     "source_role": "derived_sweep", "label": "public", "modes": {},
                     "source_reference": ("operand-store decline probe: this shape asks the device to "
                                          f"hold {elems} operand elements"),
                     "M": m, "K": k, "N": n}
            built = GC._write_capsule(entry, binding, root / f"m{m}k{k}n{n}")
            import yaml                                                # noqa: PLC0415
            cap = yaml.safe_load((Path(built) / "capsule.yaml").read_text(encoding="utf-8"))
            cap["__dir__"] = str(built)
            res = CR.run_capsule(cap, pkg, runs_root=runs, oracle_adapters=adapters,
                                 target=target, timeout=timeout)
            tiers = (res or {}).get("tiers") or {}
            # THE CHEAPEST TIER THAT ACTUALLY GRADED, in the target's own declared order. Taking the
            # alphabetically-first tier read `L0: skipped` as a decline on a float datapath, where L0
            # and L1 are the INTEGER reference and are skipped by design -- so the probe reported the
            # device refusing 6,144 elements when nothing had asked it anything. A skipped or
            # unavailable tier is not evidence about the hardware; only a tier that ran is.
            ok, detail = False, "no tier produced a verdict"
            for _t in sorted(adapters, key=lambda t: _TIER_ORDER.index(str(t))
                             if str(t) in _TIER_ORDER else len(_TIER_ORDER)):
                status = str(((tiers.get(_t) or {})).get("status") or "")
                if status in ("", "skipped", "unavailable", "not_run", "inapplicable"):
                    continue
                ok = status == "pass"
                detail = "" if ok else f"tier {_t}: {status}"
                break
        except Exception as exc:                # noqa: BLE001 — a decline is an answer; record it
            ok, detail = False, f"{type(exc).__name__}: {str(exc)[:200]}"
        pt = ProbePoint(m=m, k=k, n=n, elements=elems, ran=ok, detail=detail,
                        seconds=round(time.time() - started, 2))
        out.points.append(pt)
        if ok:
            out.largest_ran = elems if out.largest_ran is None else max(out.largest_ran, elems)
        else:
            out.smallest_declined = (elems if out.smallest_declined is None
                                     else min(out.smallest_declined, elems))
            break                                # the first decline brackets it; larger is not news
    return out


def _datapath_of(target: str) -> dict:
    """The target profile's ``datapath`` block, which pins the operand/accumulate dtypes and tiers."""
    import yaml

    from merlin.common.paths import merlin_dir
    path = merlin_dir() / "contract" / "capsules" / "profiles" / f"{target}.yaml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return (doc or {}).get("datapath") or {}


def capacity_candidates(bracket: DeclineBracket, candidates: dict) -> dict:
    """``{name: verdict}`` for each candidate capacity, where a verdict is True / False / None.

    ``None`` everywhere means the sweep decided nothing, and the caller must say so rather than
    reporting every candidate as unrefuted."""
    return {str(name): bracket.predicts(int(elems)) for name, elems in (candidates or {}).items()}


def elements_for_bytes(nbytes: int, dtype: str) -> int | None:
    """A byte capacity in elements of ``dtype``, or ``None`` when the width is unknown."""
    try:
        from merlin.targetgen.address_space import element_bits
        bits = element_bits(dtype)
    except Exception:                            # noqa: BLE001
        return None
    return (int(nbytes) * 8) // int(bits) if bits else None


def summarize(bracket: DeclineBracket, candidates: dict | None = None) -> str:
    """A human-readable verdict, including what it does NOT establish."""
    lines = [f"decline probe — {bracket.target} @ {bracket.dtype}"]
    for p in bracket.points:
        lines.append(f"  {'ran ' if p.ran else 'DECL'} m{p.m} k{p.k} n{p.n}  "
                     f"{p.elements:>10,} elements  {p.seconds}s  {p.detail[:90]}")
    if bracket.unavailable:
        lines.append(f"  UNDECIDED: {bracket.unavailable}")
    elif bracket.decided:
        lines.append(f"  boundary in ({bracket.largest_ran:,}, {bracket.smallest_declined:,}] elements")
    elif bracket.smallest_declined is not None:
        lines.append(f"  UNDECIDED: the FIRST point tried already declined "
                     f"({bracket.smallest_declined:,} elements), so nothing brackets the boundary from "
                     f"below. Either the store is smaller than every point, or -- far likelier at this "
                     f"size -- the decline is not about capacity at all and its reason above says so")
    else:
        lines.append("  UNDECIDED: no decline was reached; the store is larger than every point tried, "
                     "which refutes any candidate below the largest that ran and establishes no upper "
                     "bound")
    for name, verdict in (capacity_candidates(bracket, candidates or {})).items():
        lines.append(f"  candidate {name}: "
                     + ("PREDICTS the boundary" if verdict else
                        "REFUTED" if verdict is False else "undecided"))
    return "\n".join(lines)
