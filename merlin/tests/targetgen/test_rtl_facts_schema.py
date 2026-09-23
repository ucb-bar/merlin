"""The facts artifact has a declared shape, and the declaration discriminates.

Forty-odd modules read an RTL-facts body, and until this schema existed none of them could say what one
is: each recognised the shape it wanted by reaching for a key, so the three shapes on disk were told
apart by a key each. The risk a schema of this kind carries is that it converges on the INTERSECTION of
what the artifacts already have -- a document that validates everything and therefore says nothing --
which is why the first test here is that the discriminator can reject.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract.schemas import ContractViolation, validate
from merlin.targetgen.rtl.facts import family_of, validate_facts

pytestmark = pytest.mark.target("gemmini", "gemmini_universal", "muon")


def _artifacts() -> list[tuple[str, dict]]:
    """Every facts artifact in this checkout with a NON-EMPTY body.

    An empty body is a legal artifact and a distinct state -- the extractor ran and grounded nothing --
    but it exercises no family's shape, so including it would let the suite pass over nothing.
    """
    root = repo_root()
    out: list[tuple[str, dict]] = []
    for pattern in (
        "merlin/targets/*/contracts/rtl_facts/facts.json",
        "out/artifacts/cache/rtl_introspect/*/facts.json",
    ):
        for path in sorted(glob.glob(str(root / pattern))):
            try:
                doc = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict) and (doc.get("facts") or {}):
                out.append((Path(path).parent.parent.parent.name or path, doc))
    return out


def test_the_family_discriminator_rejects_a_body_of_another_family():
    """The test that stops this schema being the intersection of what the tree already has.

    A thread geometry is not a statically extracted body and a spatial tile is neither, so each family
    must refuse the other two. If this passes with the branches collapsed into one definition, the
    schema validates every artifact and distinguishes none, which is the failure mode a schema written
    against three existing files falls into by default.
    """
    simt_body = {"schema_version": "simt-facts/v0", "facts": {"target": "t", "simt": {"lanes_per_warp": 16}}}
    spatial_body = {"schema_version": "spatial-facts/v0", "facts": {"target": "t", "spatial": {"tile_dim": {}}}}
    static_body = {"schema_version": "2.0", "facts": {"target": "t", "arrays": [{"name": "mesh"}]}}

    # Each body validates under its OWN family...
    validate({**simt_body, "family": "simt_config"}, "rtl_facts")
    validate({**spatial_body, "family": "opu"}, "rtl_facts")
    validate({**static_body, "family": "circt_static"}, "rtl_facts")

    # ...and is refused under a family whose shape it does not have. Only the two bodies carrying a
    # distinguishing block can make the cross-claim: a statically extracted body is characterised by
    # what it does NOT carry, so claiming a richer family for it is not a contradiction the schema sees.
    for body, foreign in ((simt_body, "circt_static"), (spatial_body, "circt_static"), (simt_body, "opu")):
        with pytest.raises(ContractViolation):
            validate({**body, "family": foreign}, "rtl_facts")


def test_every_artifact_in_this_checkout_resolves_a_family_and_validates_under_it():
    artifacts = _artifacts()
    assert artifacts, "no facts artifact with a body was found; this suite would assert nothing"
    problems = {name: validate_facts(doc) for name, doc in artifacts}
    assert not any(problems.values()), (
        f"artifacts that do not validate for their family: { {k: v for k, v in problems.items() if v} }"
    )


def test_the_artifact_set_covers_more_than_one_family():
    """Anti-vacuity: a suite that only ever sees one shape cannot notice a discriminator that collapsed."""
    families = {family_of(doc)[0] for _, doc in _artifacts()}
    assert len(families) >= 2, (
        f"only one family is present in this checkout ({families}); the per-family branches are untested"
    )


def test_a_family_that_cannot_be_decided_is_refused_rather_than_guessed():
    """Fail closed. Inferring a family from the keys present is the sniffing this replaces, and it
    answers confidently for an artifact that grounded nothing."""
    mystery = {"schema_version": "2.0", "facts": {"target": "t", "arrays": [{"name": "mesh"}]}}
    family, basis = family_of(mystery)
    assert family is None, "a family was inferred from the body's shape"
    assert "schema_version" in basis
    assert validate_facts(mystery) == [f"family is undecidable: {basis}"]


def test_a_targets_own_declaration_is_not_read_as_the_artifacts():
    """A target declaring a family says what it EXPECTS, not what the artifact IS.

    Taking the target's declaration as the artifact's own would assert the very agreement that is worth
    checking -- and the recorded failure it guards is exactly that: a spatial tile's artifact written by
    an extractor that cannot see it won a cache lookup forever, because nothing compared the two.
    """
    mystery = {"schema_version": "2.0", "facts": {"target": "gemmini", "arrays": [{"name": "mesh"}]}}
    family, basis = family_of(mystery, target="gemmini")
    assert family is None
    assert "DECLARES" in basis and "would assert the very agreement" in basis


def test_the_writer_stamps_the_family_so_a_regenerated_artifact_carries_it():
    """The stamp has to be produced, not applied by hand to the artifacts that happen to exist.

    RTL facts are generated and untracked: every clone regenerates them. A family resolved only from a
    hand-edited file is a fact about one working copy, so the writer -- the only code that knows the
    answer without inferring it -- records it at write time.
    """
    from merlin.targetgen.rtl.facts import _stamp_family

    fresh = _stamp_family(
        {"schema_version": "2.0", "facts": {"target": "t", "arrays": [{"name": "m"}]}}, "circt_static"
    )
    assert family_of(fresh) == ("circt_static", "stamped on the artifact")
    assert validate_facts(fresh) == []


def test_an_existing_stamp_is_never_overwritten_by_the_writer():
    """A hand-promoted artifact declares its own family, and a regeneration that clobbered that
    declaration would replace a human's statement with the extractor's assumption about itself."""
    from merlin.targetgen.rtl.facts import _stamp_family

    assert _stamp_family({"family": "simt_config", "facts": {"x": 1}}, "circt_static")["family"] == "simt_config"


def test_a_stamped_family_is_believed_over_everything_else():
    """The artifact's own stamp is the most direct thing it can say, so it wins -- including for the one
    hand-promoted artifact in this checkout, whose generator is prose and which no extractor claims."""
    doc = {"schema_version": "2.0", "family": "simt_config", "facts": {"target": "t", "simt": {"cores": 1}}}
    assert family_of(doc) == ("simt_config", "stamped on the artifact")
    assert validate_facts(doc) == []
