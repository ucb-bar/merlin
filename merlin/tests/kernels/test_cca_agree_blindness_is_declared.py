"""What `cca_agree` cannot see, written down so it stops being a surprise.

`cca_agree` is the function an expressiveness measurement would naturally reach for: give it an expert
kernel's lifted description and ours, and it reports where they differ. It iterates a HARDCODED facet
tuple, and two facets are not in it. Anything those two describe -- fences, copy/compute overlap,
engine-to-engine bytes -- is not compared at all, and the report says "agree".

This file does not fix that. `cca_agree` is load-bearing for an existing backend's bijection suite, and
silently widening what it compares would change what those green tests assert without anyone deciding
to. What it does instead is make the blindness a declared, checked fact: the assertion below FAILS the
day someone widens the tuple, and its message names the fix. That is the difference between a known
limitation and a latent one.

The sibling module `cca_compare.uncomparable_axes` exists for the same reason at the axis level -- its
own docstring puts it best: trading visible noise for invisible blindness is the wrong trade.
"""

from __future__ import annotations

import inspect
import re

import pytest

from merlin.kernels import cca, cca_compare

pytestmark = pytest.mark.target("gemmini")

#: The facets `cca_agree` does not look at, as measured. Not a wish list: a change to either side of
#: this equality is a change to what an agreement report means.
KNOWN_BLIND = {"communication", "coverage"}


def _iterated_facets() -> tuple[str, ...]:
    """The facet names `cca_agree` actually loops over, read from its source.

    Read rather than imported because the tuple is a literal inside the function -- which is precisely
    the property this file exists to record. If it ever becomes a module-level name or a reflection,
    this helper stops finding it and the test fails, which is the correct outcome: the thing being
    described has changed.
    """
    match = re.search(r"for facet in \(([^)]*)\)", inspect.getsource(cca.cca_agree))
    assert match, (
        "cca_agree no longer loops over a literal facet tuple. If it now reflects over the facets, "
        "delete this file: the blindness it declares is gone."
    )
    return tuple(part.strip().strip('"').strip("'") for part in match.group(1).split(",") if part.strip())


def test_cca_agree_sees_fewer_facets_than_the_cca_has():
    """The declaration itself, and the tripwire on it.

    `cca_compare._facet_names()` is reflected over the dataclass, so it is the real universe. When
    someone widens `cca_agree` to match, this goes red and says so -- a declared gap that heals must
    announce that it healed, or the declaration outlives the problem and starts misinforming.
    """
    seen = set(_iterated_facets())
    universe = set(cca_compare._facet_names())
    blind = universe - seen
    assert blind == KNOWN_BLIND, (
        f"the set of facets cca_agree cannot see changed: {sorted(blind)} (was {sorted(KNOWN_BLIND)}). "
        "If it SHRANK, cca_agree was widened -- narrow KNOWN_BLIND, and check whether the bijection "
        "suite's expectations moved with it. If it GREW, a new facet is invisible to every agreement "
        "report the moment it is populated."
    )


def test_the_blind_facets_are_not_empty_of_content():
    """A facet with no fields would be blindness that cost nothing. These are not that.

    Three of the capability axes the scheduling-IR coverage register declares name fields on the
    communication facet, so a measurement built on `cca_agree` would report agreement on axes it never
    looked at -- the exact shape of a number that cannot fail.
    """
    import dataclasses

    for name in sorted(KNOWN_BLIND):
        # Found by matching the dataclass, not by assuming the class is the field name capitalised --
        # the naming convention holds today and is not a thing this test should depend on.
        cls = next(
            (c for c in vars(cca).values() if dataclasses.is_dataclass(c) and c.__name__.lower().startswith(name)),
            None,
        )
        assert cls is not None, f"no facet class found for {name!r}"
        fields = [f.name for f in dataclasses.fields(cls)]
        assert fields, f"{name} has no fields; its absence from cca_agree would cost nothing"


def test_a_field_populated_on_one_side_only_is_skipped_silently():
    """The second half of the blindness, and the one that is easier to miss.

    Even for a facet it DOES iterate, `cca_agree` compares only fields populated on both sides. A field
    one side never filled is not reported as a disagreement, and does not appear in `compared_fields`
    either -- so it is absent from the numerator AND the denominator, which is how a coverage ratio
    reads high because the hard cases were quietly dropped.
    """
    a = cca.CCA(op="m", backend=["t"], memory=cca.MemoryFacet(banks_used=4))
    b = cca.CCA(op="m", backend=["t"], memory=cca.MemoryFacet(banks_used=None))
    report = cca.cca_agree(a, b)
    assert report.agree, "a one-sided field was reported as a disagreement"
    assert not any("banks_used" in f for f in report.compared_fields), (
        "a one-sided field appeared in compared_fields; if that changed, the denominator of every "
        "agreement report changed with it"
    )
