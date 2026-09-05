"""EVERY cost fit in `cert_cost` withholds its source list, not just the one that leaked.

The narrow regression is covered next door in `test_no_holdout_names_in_tracked_specs`: `CostFit`
published a `sources` list -- the run file each sample came from, e.g.

    /scratch/.../rb_gemsg1/grading_hidden/runs/.../H0_matmul_hidden/capsule_result.json

-- and the conformance spec writer embeds that dict into a TRACKED file every arm reads, so ten
holdout names reached the granted tree and `verify_no_cheat` check 7 failed on it.

This is the other half. `cert_cost` defines TWO fits over the same certification history, both
carrying the same provenance strings, and only one of them was patched: `CycleCostFit.to_dict` still
emitted its sources. It has no tracked-artifact writer TODAY, which is exactly the state `CostFit` was
in until the conformance spec started embedding it -- the hazard is a fit class shipping a provenance
list at all, not one particular caller.

So the classes are DISCOVERED from the module rather than listed here. A third fit added later is
covered the day it appears, instead of being covered only if whoever adds it remembers this file.
"""
from __future__ import annotations

import dataclasses

import pytest

from merlin.targetgen import cert_cost as CC

#: A source string of exactly the shape that leaked, so a regression is caught by the same substrings
#: the gate looks for rather than by a stand-in that could not have leaked anything.
_LEAKY_SOURCE = ("/scratch/u/repo/out/runs/t/capsule-bench/rb/grading_hidden/runs/"
                 "t-capsule-bench/H0_matmul_hidden/capsule_result.json#cycle_accurate_tier:L3")


def _fit_classes() -> list[type]:
    """Every dataclass in `cert_cost` that carries a `sources` field -- i.e. every class that COULD
    publish a holdout name. Derived, so this test cannot silently stop covering a class."""
    out = []
    for obj in vars(CC).values():
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        if any(f.name == "sources" for f in dataclasses.fields(obj)):
            out.append(obj)
    return out


def _instantiate(cls: type):
    """A fit of `cls` with one leaky source, filling every other field with a harmless placeholder.

    Built from the dataclass's own fields so a class that gains or renames one does not quietly fall
    out of coverage by failing to construct.
    """
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name == "sources":
            kwargs[f.name] = (_LEAKY_SOURCE,)
        elif f.name == "target":
            kwargs[f.name] = "t"
        elif f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            continue
        elif f.type in ("int", int):
            kwargs[f.name] = 1
        elif f.type in ("float", float):
            kwargs[f.name] = 1.0
        else:
            kwargs[f.name] = "x"
    return cls(**kwargs)


def test_the_module_actually_defines_fits_to_check():
    """"Found nothing" and "looked at nothing" print the same. If the discovery above ever returns an
    empty list, every test below passes vacuously -- so the discovery itself is asserted."""
    assert len(_fit_classes()) >= 2, (
        "expected at least the shape fit and the cycle fit; a discovery that finds none would make "
        "the rest of this file a check that cannot fail")


@pytest.mark.parametrize("cls", _fit_classes(), ids=lambda c: c.__name__)
def test_to_dict_withholds_the_source_list(cls):
    """The default serialization is the one that reaches tracked artifacts, so it is the one that must
    be safe."""
    rendered = repr(_instantiate(cls).to_dict())
    assert "sources" not in rendered, (
        f"{cls.__name__}.to_dict() published its source list; those entries name holdout capsules and "
        f"local absolute paths, and a dict like this is embedded verbatim in a tracked contract file")
    assert "H0_matmul_hidden" not in rendered, "a holdout name reached the default serialization"
    assert "/scratch/" not in rendered, "a local absolute path reached the default serialization"


@pytest.mark.parametrize("cls", _fit_classes(), ids=lambda c: c.__name__)
def test_the_counts_that_make_the_fit_auditable_survive(cls):
    """Withholding provenance must not turn a fit into an unfalsifiable number: how many samples it
    rests on is what lets a reader judge it."""
    assert "n_samples" in _instantiate(cls).to_dict()


@pytest.mark.parametrize("cls", _fit_classes(), ids=lambda c: c.__name__)
def test_provenance_is_still_reachable_for_a_local_diagnostic(cls):
    """Withheld, not deleted. Somebody auditing a fit on their own machine must still see which runs it
    rests on -- the rule is about what lands in a tracked file, not about hiding evidence."""
    assert _instantiate(cls).to_dict(with_sources=True)["sources"] == [_LEAKY_SOURCE]
