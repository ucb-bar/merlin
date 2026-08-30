"""The performance corpus: a capsule that measures a schedule must say which schedule it measures.

These capsules live in their own ``perf/`` category rather than under a new capsule ``kind`` or a new
``op``, because both of those enums are closed and test-pinned -- a ``kind`` is a grading contract and
an ``op`` is only expressible where a builder and a golden exist for it -- while a category is a
directory name and the capsule schema is ``additionalProperties: true``. What holds the level together
is therefore DATA (a ``performance:`` block), and data with no reader goes stale silently. This is the
reader.

Target-agnostic by construction: it walks whichever corpora have a ``perf/`` category, so a second
target's performance corpus is checked by the same file without an edit.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.perf import comparand as C
from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.contract.schemas import validate_capsule

CAPSULES = repo_root() / "merlin" / "contract" / "capsules"
PERF_ROOTS = sorted(p for p in CAPSULES.glob("*/perf") if p.is_dir())


def _load(root):
    return discover_capsules(root, labels=None)


pytestmark = pytest.mark.skipif(not PERF_ROOTS, reason="DID NOT RUN: no target ships a perf corpus")


@pytest.mark.parametrize("root", PERF_ROOTS, ids=lambda p: p.parent.name)
class TestThePerfCorpusIsWellFormed:
    def test_every_capsule_validates_against_the_contract(self, root):
        caps = _load(root)
        assert caps, f"{root} holds capsule dirs but none loaded"
        for cap in caps:
            c = dict(cap)
            c.pop("__dir__", None)
            validate_capsule(c)

    def test_every_capsule_declares_the_level_it_exercises(self, root):
        """Without this the perf corpus and the functional corpus are indistinguishable once generated,
        and the level -- the whole organizing principle -- exists only in the profile that made them."""
        for cap in _load(root):
            perf = cap.get("performance")
            assert isinstance(perf, dict), f"{cap['name']} carries no performance block"
            assert perf.get("level"), f"{cap['name']} declares no level"
            assert perf.get("lever"), f"{cap['name']} names no lever its cycle count can see"

    def test_no_perf_capsule_is_labelled_public(self, root):
        """A perf capsule is not a functional conformance claim. Growing `public_passed`'s denominator
        with capsules authored to measure a schedule corrupts the one number that gets quoted."""
        for cap in _load(root):
            assert cap.get("label") != "public", cap["name"]

    def test_every_capsule_declares_shapes_and_dtypes_for_every_operand(self, root):
        """This is what makes `useful_bytes` derivable rather than hand-entered -- the open half of the
        transfer-amplification ratio, whose measured side has always been real and whose useful side
        came from operand sizes typed in by hand."""
        from merlin.perf import preflight as PF
        from merlin.targetgen import capsule_dram as DR
        for cap in _load(root):
            specs = PF.operands_from_declaration(cap["inputs"],
                                                 element_bytes_of=lambda d: DR.dtype_bits(d) / 8)
            assert specs and all(s.nbytes > 0 for s in specs), cap["name"]

    def test_a_reduction_sweep_carries_at_least_four_points(self, root):
        """Two points fit a rate and an intercept and leave nothing over. Four is the smallest set that
        fits both AND leaves a residual -- which is where the k-chain accumulate excess lives (a naive
        254 against a measured 284, and a two-point fit has nowhere to put the 30)."""
        caps = _load(root)
        tile_level = [c for c in caps if (c.get("performance") or {}).get("level", "").startswith("L1")]
        if not tile_level:
            pytest.skip(f"DID NOT RUN: {root.parent.name} declares no L1 tile level")
        depths = set()
        for c in tile_level:
            # The reduction depth is the contracted axis of the LEFT operand, and which operand that is
            # comes from the capsule's own declaration. Reading "the second dimension of the first 2-D
            # tensor" instead reports the parallel extent on half of them (the capsule lists operands in
            # the frontend's order, not a fixed lhs-then-rhs one), and matching the two shapes
            # structurally is ambiguous on a square-ish tile, where both pairings agree.
            attrs = (c.get("operation") or {}).get("attributes") or {}
            lhs_name = attrs.get("lhs") or (attrs.get("arg_order") or [None])[0]
            shapes = {t["name"]: t["shape"] for t in c["inputs"]}
            lhs = shapes.get(lhs_name)
            assert lhs and len(lhs) == 2, (c["name"], lhs_name, shapes)
            depths.add(lhs[1])
        assert len(depths) >= 4, f"{root.parent.name} L1 sweeps only {sorted(depths)}"


@pytest.mark.parametrize("root", PERF_ROOTS, ids=lambda p: p.parent.name)
class TestTheComparisonGroupsAreUsable:
    def test_every_group_has_one_fused_member_and_at_least_one_part(self, root):
        groups = C.declared_groups(_load(root))
        if not groups:
            pytest.skip(f"DID NOT RUN: {root.parent.name} declares no comparison group")
        for name, roles in groups.items():
            assert len(roles.get(C.FUSED) or []) == 1, (name, roles)
            assert len(roles.get(C.PART) or []) >= 1, (name, roles)
            assert not roles.get("unspecified"), (
                f"{name} has a member declaring the group as a bare name with no role; which capsule "
                f"is the fused implementation would then have to be guessed")

    def test_every_member_of_a_group_sits_at_the_same_shape(self, root):
        """The comparand subtracts these numbers from one another, so a shape difference between the
        fused capsule and its parts would be read as a fusion saving."""
        caps = {c["name"]: c for c in _load(root)}
        groups = C.declared_groups(list(caps.values()))
        if not groups:
            pytest.skip(f"DID NOT RUN: {root.parent.name} declares no comparison group")
        for name, roles in groups.items():
            fused = caps[roles[C.FUSED][0]]
            parts = [caps[n] for n in roles[C.PART]]
            # A fused implementation replaces its parts, so every operand SHAPE it consumes must be one
            # its parts consume too. A shape the fused capsule has and no part has means the two sides
            # are not doing the same work, and the comparand's subtraction would report that difference
            # as a fusion saving. (An equality is wrong in the other direction: the parts additionally
            # carry the intermediate the fusion is there to eliminate.)
            part_shapes = {tuple(t["shape"]) for c in parts for t in c["inputs"]}
            fused_shapes = {tuple(t["shape"]) for t in fused["inputs"]}
            assert fused_shapes <= part_shapes, (name, sorted(fused_shapes - part_shapes))

    def test_the_comparand_resolves_once_every_member_has_a_count(self, root):
        """End to end on the real declarations, with synthetic counts: no oracle is run here. What is
        under test is that the corpus's groups are shaped so the arithmetic can happen at all."""
        caps = _load(root)
        groups = C.declared_groups(caps)
        if not groups:
            pytest.skip(f"DID NOT RUN: {root.parent.name} declares no comparison group")
        cycles = {c["name"]: {"L3": 1000} for c in caps}
        out = C.fusion_comparands(caps, cycles, submission="pkg", ladder=("L0", "L1", "L3", "L4"))
        for name, res in out.items():
            assert res["status"] == "resolved", (name, res["reason"])
            assert res["fused_cycles"] == 1000
            assert res["sum_of_parts"] == 1000 * len(res["parts"])
            # No adapter populates TierResult.toolchain, so nothing is attributable to a submission yet
            # and the arithmetic must refuse to be quoted rather than quietly claiming provenance.
            assert res["citable"] is False


def test_the_generated_capsules_match_their_profile_declaration():
    """A generated capsule must carry the intent its profile entry declared. Three of the four writers
    build the capsule dict themselves, so a key handled in only one of them vanishes from most of the
    corpus -- which is how the ``comparison_group`` field came to be declared and consumed by nothing."""
    for root in PERF_ROOTS:
        profile = CAPSULES / "profiles" / f"{root.parent.name}.yaml"
        if not profile.is_file():
            continue
        doc = yaml.safe_load(profile.read_text(encoding="utf-8")) or {}
        declared_levels = {(s.get("base") or {}).get("performance", {}).get("level")
                           for s in (doc.get("sweeps") or [])}
        declared_levels.discard(None)
        on_disk = {(c.get("performance") or {}).get("level") for c in _load(root)}
        assert on_disk <= declared_levels or not declared_levels, (on_disk, declared_levels)
