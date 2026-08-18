"""Constrained-random semantic fuzzing (Phase D2) — generate random *legal* op-graphs from a target's
declared capability grammar, to differential-test the frozen compiler far beyond the hand corpus.

Where :mod:`merlin.targetgen.capability_probes` enumerates the closure deterministically, the fuzzer
samples it: each program is a random chain drawn from the declared families with in-closure dtypes and
shapes (a contraction followed by legal elementwise/normalization/attention epilogues), so the
generalization set can grow to thousands of programs instead of a few dozen capsules. Every emitted
region is eligible by construction — the frozen compiler is then differential-tested (CPU reference vs
accelerator) on each, and the fraction it lowers correctly is a sampled Acceleratable Region Recall.

Deterministic per seed (``random.Random(seed)``): the same (contract, seed) yields the same corpus, so a
fuzz finding is reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.eligibility import RegionDescriptor

# In-closure shape pool (structural corners; the fuzzer picks among them so shapes stay legal but varied).
_DIMS = [1, 7, 16, 17, 32, 64, 127, 128, 256, 4096]
# families that make a legal EPILOGUE after a contraction (per-element / row-wise ops)
_EPILOGUE = ("elementwise_map", "reduction", "normalization", "softmax")


@dataclass(frozen=True)
class FuzzProgram:
    name: str
    seed: int
    regions: tuple[RegionDescriptor, ...]


def _pick_dtype(rng: random.Random, cap: SemanticCapability):
    return rng.choice(list(cap.dtypes)) if cap.dtypes else None


def fuzz_program(seed: int, cap_map: dict[str, SemanticCapability], *, max_len: int = 5) -> FuzzProgram:
    """One random legal program: a head op from a declared family (a contraction when available) plus a
    random run of legal epilogues drawn from declared families. All regions are in-closure/eligible."""
    rng = random.Random(seed)
    fams = list(cap_map)
    regions: list[RegionDescriptor] = []

    head_fam = "contraction" if "contraction" in cap_map else rng.choice(fams)
    cap = cap_map[head_fam]
    dt = _pick_dtype(rng, cap)
    contractionish = head_fam in ("contraction", "attention")
    m, k, n = (rng.choice(_DIMS), rng.choice(_DIMS), rng.choice(_DIMS))
    regions.append(RegionDescriptor(
        source=f"fuzz{seed}/{head_fam}0", family=head_fam, in_dtype=dt,
        weight_dtype=(dt if contractionish else None),
        m=m, k=(k if contractionish else None), n=(n if contractionish else None), rank=2))

    epilogue_fams = [f for f in _EPILOGUE if f in cap_map]
    for i in range(rng.randint(0, max_len - 1)):
        if not epilogue_fams:
            break
        fam = rng.choice(epilogue_fams)
        ecap = cap_map[fam]
        edt = _pick_dtype(rng, ecap)
        regions.append(RegionDescriptor(
            source=f"fuzz{seed}/{fam}{i + 1}", family=fam, in_dtype=edt, m=n, rank=2))
    return FuzzProgram(name=f"fuzz_{seed}", seed=seed, regions=tuple(regions))


def fuzz_corpus(cap_map: dict[str, SemanticCapability], n: int, *, base_seed: int = 0,
                max_len: int = 5) -> list[FuzzProgram]:
    """``n`` deterministic random programs (seeds ``base_seed .. base_seed+n-1``)."""
    return [fuzz_program(base_seed + i, cap_map, max_len=max_len) for i in range(n)]
