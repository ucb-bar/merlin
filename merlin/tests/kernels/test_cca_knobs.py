"""The typed knob layer over the four CCA surfaces — ``kernels.knobs``.

WHAT THESE TESTS ARE FOR. The optimization phase is allowed exactly one mutation surface, so the
properties that matter are not "the knobs exist" but:

  * every domain is DERIVED from the target's own sources, and the derivation matches what those
    sources actually say (asserted against real values, not a fixture);
  * a quantity that cannot be derived comes out UNKNOWN and is REPORTED, never given a plausible
    default — the failure this repo keeps re-learning, most recently as a check that could not run
    and reported success;
  * ``allowed`` / ``refused`` / ``undeterminable`` never collapse into two states;
  * a change that lies on NO declared surface is REFUSED and the refusal NAMES what was outside.

Target names appear here deliberately: a test is the "genuine edge where that target is legitimately
the subject", and the point of these assertions is that shared code produced the right answer for two
targets with different silicon and a third whose facts are empty.
"""
from __future__ import annotations

import pytest

from merlin.kernels import knobs as K
from merlin.targetgen import address_space as AS
from merlin.targetgen import memory_regime as MR


# ---- fixtures over REAL targets ---------------------------------------------------------------
#
# Skipped rather than faked when a target's artifact is unavailable in this checkout: a knob-derivation
# test that silently passes against a synthesized address space would assert nothing about derivation,
# which is the only thing it is here to assert.

def _require_derived(target: str):
    space = AS.derive_address_space(target)
    if space.stores_status != AS.DERIVED:
        pytest.skip(f"{target}: no derived on-chip stores in this checkout "
                    f"(stores_status={space.stores_status})")
    return space


@pytest.fixture(scope="module")
def array_target() -> str:
    """A target with a derivable array + operand store."""
    _require_derived("gemmini")
    return "gemmini"


@pytest.fixture(scope="module")
def inv(array_target):
    return K.derive_knobs(array_target)


# ---- 1. the surfaces are closed -----------------------------------------------------------------

def test_exactly_four_surfaces_and_every_knob_is_on_one(inv):
    assert K.SURFACES == (K.TILE, K.LAYER, K.PLACEMENT, K.GLOBAL)
    assert len(K.SURFACES) == 4
    for knob in inv.knobs:
        assert knob.surface in K.SURFACES
    # Each surface actually carries knobs: a surface with none is a surface the phase cannot use, and
    # would make "one mutation surface" a smaller claim than it reads as.
    for surface in K.SURFACES:
        assert inv.by_surface(surface), f"surface {surface} exposes no knob"


def test_every_declared_surface_maps_to_cca_facets():
    """The surface tags are the knob-side of facets that EXIST on the CCA, so an attributed cost on a
    facet can be routed to a surface. A tag naming a facet the CCA does not have would make that
    routing silently empty."""
    from merlin.kernels import cca as CCA
    facet_fields = set(CCA.CCA.__dataclass_fields__)
    for surface, facets in K.SURFACE_FACETS.items():
        assert surface in K.SURFACES
        for facet in facets:
            assert facet in facet_fields, f"{surface} names facet {facet!r} that CCA does not carry"


def test_a_knob_cannot_be_minted_on_an_undeclared_surface():
    with pytest.raises(ValueError) as e:
        K.Knob("x.y", "peephole", K.value_set((1,), source="t"), controls="c", default=1)
    assert "peephole" in str(e.value)


def test_a_knob_with_a_known_domain_may_not_omit_a_default_silently():
    """A missing default must be an explicit ADMISSION. Otherwise "the compiler starts here" and
    "nobody said where the compiler starts" are the same object."""
    with pytest.raises(ValueError) as e:
        K.Knob("x.y", K.TILE, K.value_set((1, 2), source="t"), controls="c")
    assert "explicit admission" in str(e.value)


# ---- 2. domains are DERIVED, and match the source they cite -------------------------------------

def test_tile_extents_come_from_the_targets_own_array_geometry(array_target, inv):
    space = AS.derive_address_space(array_target)
    for axis, extent in (("rows", space.array_rows), ("cols", space.array_cols)):
        knob = inv.get(f"tile.extent_{axis}")
        assert knob.domain.kind == K.RANGE
        assert (knob.domain.lo, knob.domain.hi) == (1, extent), (
            f"tile.extent_{axis} must be bounded by the array's own {axis} extent")
        assert knob.default == extent          # tiling TO the array edge is the derived default
        assert "derive_address_space" in knob.domain.source


def test_reduction_and_layer_residency_are_bounded_by_the_operand_store(array_target, inv):
    _store, capacity = MR.operand_store(array_target)
    assert capacity, "the fixture guarantees a derived store"
    for name in ("tile.reduction_rows", "layer.operand_residency_rows"):
        knob = inv.get(name)
        assert (knob.domain.lo, knob.domain.hi) == (1, capacity)
        # The measured abort was a schedule that asked for exactly capacity+1 rows; the bound must
        # refuse it rather than round it away.
        assert knob.admits(capacity) is True
        assert knob.admits(capacity + 1) is False
        # ... and neither carries an invented starting point.
        assert knob.default is None and knob.why_no_default


def test_stage_depth_is_bounded_by_the_stores_own_bank_count(array_target, inv):
    store, _cap = MR.operand_store(array_target)
    knob = inv.get("tile.stage_depth")
    assert (knob.domain.lo, knob.domain.hi) == (1, store.banks)
    assert knob.admits(store.banks + 1) is False


def test_stage_depth_tightens_to_one_when_the_working_set_only_fits_once(array_target):
    """The regime is the second bound, and it is the regime's own documented meaning: ``fits_double``
    is the ONLY regime in which staging is possible, so a working set that fits once must not license
    a depth the hardware cannot provide."""
    _store, capacity = MR.operand_store(array_target)
    single = K.derive_knobs(array_target, working_set=(capacity - 1, capacity - 1))
    assert MR.classify(capacity - 1, capacity - 1, capacity) == MR.FITS_SINGLE
    assert single.get("tile.stage_depth").domain.hi == 1

    doubles = K.derive_knobs(array_target, working_set=(capacity // 8, capacity // 8))
    assert MR.classify(capacity // 8, capacity // 8, capacity) == MR.FITS_DOUBLE
    assert doubles.get("tile.stage_depth").domain.hi > 1
    assert "classify" in doubles.get("tile.stage_depth").domain.source


def test_loop_order_axes_are_derived_not_listed(array_target, inv):
    """Six orders here because the array declares two extents and the endpoint licenses an accumulate
    role — not because 'm,n,k' was written down."""
    knob = inv.get("tile.loop_order")
    assert knob.domain.kind == K.SET
    for order in knob.domain.values:
        assert set(order) == {"rows", "cols", "reduction"}
    assert len(knob.domain.values) == 6            # 3! permutations of the three DERIVED axes
    assert len(set(knob.domain.values)) == 6


def test_placement_sites_and_transfers_come_from_declared_compute_units(array_target, inv):
    from merlin.targetgen import target_registry as tr
    from merlin.targetgen.compute_units import compute_units
    units = compute_units(tr.load_contract(array_target) or {})
    sites = set(inv.get("placement.site").domain.values)
    for u in units:
        if u.kind != "scalar":
            assert u.name in sites
    # A->H and H->A must both be expressible, and so must A->A.
    transfers = set(inv.get("placement.transfer").domain.values)
    assert len(transfers) == len(sites) ** 2
    engine = next(n for n in sites if n != "host")
    assert ("host", engine) in transfers and (engine, "host") in transfers
    assert (engine, engine) in transfers


def test_loop_offload_and_dispatch_group_key_on_the_targets_own_role_census(array_target, inv):
    """A device whose ISA licenses a hardware-loop role gets the lever; one that does not gets a
    one-value domain — the lever exists in the vocabulary and has no room on that silicon. Read off
    roles, never off an opcode name."""
    from merlin.kernels import endpoints as EP
    roles = set()
    for e in EP.endpoints_for(array_target):
        roles.update(getattr(e, "roles", {}) or {})
    offload = inv.get("placement.loop_offload")
    if "loop_descriptor" in roles:
        assert offload.domain.values == (False, True)
        # ... and then the group extent is set by the offloaded nest, which nothing bounds.
        assert not inv.get("placement.dispatch_group").determinable
    else:
        assert offload.domain.values == (False,)
        assert inv.get("placement.dispatch_group").domain.values == (1,)

    # No dma role on this endpoint -> movement cannot be pipelined, and the knob says so with a
    # single-value domain rather than by being absent.
    moved = inv.get("placement.pipeline_movement")
    assert moved.domain.values == ((False, True) if "dma" in roles else (False,))


def test_global_formats_are_the_union_over_declared_units(array_target, inv):
    from merlin.targetgen import target_registry as tr
    from merlin.targetgen.compute_units import compute_units
    declared = {d for u in compute_units(tr.load_contract(array_target) or {}) for d in u.dtypes}
    assert set(inv.get("global.element_format").domain.values) == declared
    # A single declared format IS the default (the silicon accepts nothing else); several is a
    # schedule choice we decline to invent.
    knob = inv.get("global.element_format")
    if len(declared) == 1:
        assert knob.default == next(iter(declared))
    else:
        assert knob.default is None and knob.why_no_default


def test_accumulate_format_falls_back_to_the_widest_row_store(array_target, inv):
    """Derived, and derived the same way ``memory_regime`` picks the operand store — by row WIDTH, not
    by name. A target declaring its accumulator only in RTL facts still gets the knob."""
    space = AS.derive_address_space(array_target)
    knob = inv.get("global.accumulate_format")
    if len(space.stores) >= 2:
        widest = max(space.stores, key=lambda s: s.row_bytes)
        assert widest.element_dtype in knob.domain.values


def test_partition_count_collapses_on_a_single_engine_target(array_target, inv):
    """``[1..1]`` is a real answer that must not be confused with UNKNOWN: the lever exists and this
    device has no room for it."""
    knob = inv.get("global.partition_count")
    assert knob.determinable
    assert knob.domain.lo == 1
    assert knob.admits(1) is True
    assert knob.admits(knob.domain.hi + 1) is False


def test_a_second_target_with_different_silicon_derives_a_different_inventory(array_target):
    """The de-overfit check: same code, no target literal in it, two different answers."""
    other = "muon"
    if not AS.derive_address_space(other).array_rows:
        pytest.skip(f"{other}: no array geometry in this checkout")
    a, b = K.derive_knobs(array_target), K.derive_knobs(other)
    assert a.names() == b.names(), "the knob VOCABULARY is target-agnostic"
    assert (set(a.get("global.element_format").domain.values)
            != set(b.get("global.element_format").domain.values))
    # The other target declares two engines, so partitioning is a lever there and pinned here.
    assert b.get("global.partition_count").domain.hi > a.get("global.partition_count").domain.hi


# ---- 3. UNKNOWN is reported, never defaulted ----------------------------------------------------

def test_an_underivable_domain_is_unknown_with_a_reason_and_no_default(array_target):
    """Requirement 1's teeth. Fusion depth is a property of the op graph, not of any target, so with
    no program supplied it must come out UNKNOWN — with the reason attached, and with NO default."""
    knob = K.derive_knobs(array_target).get("layer.fusion_depth")
    assert not knob.determinable
    assert knob.domain.kind == K.UNKNOWN_DOMAIN
    assert knob.default is None
    assert "op graph" in knob.domain.why_unknown
    assert knob.admits(2) is None                  # not False — nothing was checked


def test_every_unknown_knob_carries_a_reason_and_is_still_enumerable(array_target):
    inv = K.derive_knobs(array_target)
    for knob in inv.unknowns:
        assert knob.domain.why_unknown, f"{knob.name} is UNKNOWN with no reason"
        assert knob.default is None, f"{knob.name} is UNKNOWN and yet carries a default"
        # Still in the inventory: dropping it would read as "this target has no such lever", which is
        # the opposite of "we could not bound it".
        assert knob.name in inv.names()


def test_a_target_whose_facts_are_empty_yields_unknowns_not_an_empty_inventory():
    """A target with an EMPTY facts artifact must still enumerate the full knob vocabulary, with the
    underivable domains marked UNKNOWN and the reason carried. An empty inventory would read as
    'this target exposes no levers'."""
    inv = K.derive_knobs("k1_cpu")
    assert len(inv.knobs) > 0
    assert inv.unknowns, "a target with no derivable facts must report unknowns"
    assert inv.get("tile.extent_rows").domain.kind == K.UNKNOWN_DOMAIN
    assert inv.notes, "the reasons must be reported on the inventory, not swallowed"


def test_stores_absent_and_stores_unknown_are_not_conflated():
    """A device that declares NO on-chip store is a fact about the device; an artifact we could not
    read is a fact about our extraction. Both make the capacity knobs UNKNOWN, and the note must still
    tell them apart — otherwise a scalar core and a broken toolchain read identically."""
    absent = K.derive_knobs("muon")
    unreadable = K.derive_knobs("k1_cpu")
    if absent.notes.get("stores_status") != AS.ABSENT:
        pytest.skip("no target with ABSENT stores in this checkout")
    assert absent.notes["stores_status"] == AS.ABSENT
    assert unreadable.notes["stores_status"] == AS.UNKNOWN
    assert absent.notes["stores_status"] != unreadable.notes["stores_status"]


# ---- 4. ENFORCEMENT: three states, and the falsifier --------------------------------------------

def test_a_mutation_on_a_declared_surface_is_allowed(array_target, inv):
    v = K.check_mutation({"tile.extent_rows": 8, "layer.prepack": True}, inv)
    assert v.state == K.ALLOWED and v.allowed
    assert v.reason() == ""
    assert set(v.surfaces_touched()) == {K.TILE, K.LAYER}


def test_falsifier_a_mutation_outside_every_declared_surface_is_refused_naming_it(inv):
    """THE FALSIFIER. A change that is not on any of the four surfaces — here a pass flag the phase
    might otherwise reach directly — must be REFUSED, and the refusal must NAME what was outside.
    Without this, a score can move for a reason nobody can attribute, which is the entire failure this
    layer exists to prevent."""
    v = K.check_mutation({"llvm.unroll_threshold": 512}, inv)
    assert v.state == K.REFUSED
    assert not v.allowed
    refusals = v.refusals()
    assert len(refusals) == 1
    outside = refusals[0].outside
    assert "llvm.unroll_threshold" in outside                 # names WHAT was outside
    assert "no declared CCA surface exposes" in outside       # ... and WHY
    assert "llvm.unroll_threshold" in v.reason()


def test_a_value_outside_a_derived_domain_is_refused_naming_the_domain(array_target, inv):
    _store, capacity = MR.operand_store(array_target)
    v = K.check_mutation({"tile.reduction_rows": capacity + 1}, inv)
    assert v.state == K.REFUSED
    outside = v.refusals()[0].outside
    assert str(capacity) in outside                      # the derived bound is quoted back
    assert "memory_regime.operand_store" in outside      # ... with the source that produced it


def test_an_underivable_knob_is_undeterminable_and_never_reported_as_allowed(array_target, inv):
    v = K.check_mutation({"layer.fusion_depth": 3}, inv)
    assert v.state == K.UNDETERMINABLE
    assert not v.allowed, "undeterminable must never read as success"
    assert v.state != K.REFUSED, "and must never collapse into refused either"
    assert v.undeterminable()[0].surface == K.LAYER
    assert "could not be derived" in v.undeterminable()[0].outside


def test_refusal_dominates_undeterminable_in_a_mixed_proposal(inv):
    """A definite violation is not softened by an unrelated unknown."""
    v = K.check_mutation({"layer.fusion_depth": 3, "some.invented.flag": 1}, inv)
    assert v.state == K.REFUSED
    assert len(v.refusals()) == 1 and len(v.undeterminable()) == 1


def test_the_three_states_are_distinct_objects(inv):
    assert len({K.ALLOWED, K.REFUSED, K.UNDETERMINABLE}) == 3


def test_an_empty_proposal_is_allowed_with_no_findings(inv):
    v = K.check_mutation({}, inv)
    assert v.state == K.ALLOWED and v.findings == ()


def test_a_boolean_knob_does_not_admit_an_integer(inv):
    """``False in (0, 1)`` is True in Python. A boolean 'did it apply' flag silently admitting an int
    is how a flag becomes a count."""
    assert inv.get("layer.prepack").admits(True) is True
    assert inv.get("layer.prepack").admits(1) is False
    assert inv.get("tile.extent_rows").admits(True) is False


# ---- 5. attribution -----------------------------------------------------------------------------

def test_the_ledger_records_the_knob_a_mutation_is_attributed_to(inv):
    led = K.AttributionLedger(inv)
    v = led.record({"tile.extent_rows": 8}, note="round 1")
    assert v.allowed
    assert len(led.entries) == 1
    e = led.entries[0]
    assert (e.knob, e.surface, e.after) == ("tile.extent_rows", K.TILE, 8)
    assert e.before == inv.get("tile.extent_rows").default   # the derived starting point


def test_a_refused_mutation_is_never_given_an_attribution(inv):
    """Attributing a score change to a knob a change never touched is the exact failure this layer
    prevents; a refused change must leave the ledger empty."""
    led = K.AttributionLedger(inv)
    v = led.record({"llvm.unroll_threshold": 512})
    assert v.state == K.REFUSED
    assert led.entries == []
    assert len(led.rejected) == 1


def test_an_unmeasured_entry_reports_unmeasured_not_zero(inv):
    """The recurring bug class, in this layer's own terms: a check that could not run must not report
    success. Nothing here compiles or runs anything, so every entry is born unmeasured and the summary
    SAYS SO rather than summing a delta of zero."""
    led = K.AttributionLedger(inv)
    led.record({"tile.extent_rows": 8, "placement.loop_offload": True})
    summary = led.by_surface()
    assert summary["status"] == "unmeasured"
    assert summary["n_measured"] == 0
    assert "requires compiling and running both sides" in summary["why"]
    assert summary["by_surface"][K.TILE]["n_unmeasured"] == 1
    assert summary["by_surface"][K.TILE]["measured_delta"] == 0.0
    # measured_delta 0.0 with n_measured 0 must be readable as "nothing ran", not "no effect".
    assert summary["by_surface"][K.TILE]["n_measured"] == 0


def test_an_observed_entry_becomes_measured_and_lands_on_its_surface(inv):
    led = K.AttributionLedger(inv)
    led.record({"tile.extent_rows": 8, "placement.loop_offload": True})
    assert led.observe("placement.loop_offload", metric="cycles",
                       metric_before=1000.0, metric_after=650.0) is True
    summary = led.by_surface()
    assert summary["status"] == "measured"
    assert summary["by_surface"][K.PLACEMENT]["n_measured"] == 1
    assert summary["by_surface"][K.PLACEMENT]["measured_delta"] == -350.0
    # The tile change still has no measurement, and the summary keeps the two apart — which is what
    # lets a caller answer "no point tuning a tile for 3% if 35% is dispatch".
    assert summary["by_surface"][K.TILE]["n_measured"] == 0
    assert summary["by_surface"][K.TILE]["n_unmeasured"] == 1


def test_a_measurement_with_nothing_to_attribute_it_to_is_dropped_not_invented(inv):
    led = K.AttributionLedger(inv)
    assert led.observe("tile.extent_rows", metric="cycles",
                       metric_before=1.0, metric_after=2.0) is False
    assert led.entries == []


def test_surface_of_answers_which_surface_an_attributed_cost_belongs_to(inv):
    assert inv.surface_of("tile.stage_depth") == K.TILE
    assert inv.surface_of("placement.transfer") == K.PLACEMENT
    assert inv.surface_of("global.element_format") == K.GLOBAL
    assert inv.surface_of("nothing.here") is None


def test_the_inventory_serializes_with_its_unknowns_visible(inv):
    d = inv.to_dict()
    assert d["target"] == inv.target
    assert len(d["knobs"]) == len(inv.knobs)
    assert d["unknown_knobs"] == [k.name for k in inv.unknowns]
    for knob in d["knobs"]:
        assert knob["domain"]["source"] or knob["domain"]["kind"] == K.UNKNOWN_DOMAIN


def test_a_zero_working_set_does_not_tighten_a_hardware_bound(array_target):
    """A zero live set is the ABSENCE of a program, not a program that fits twice — and
    ``memory_regime.classify`` reads it as ``fits_double``. Tightening on it would report a staging
    verdict for something nobody sized."""
    store, _cap = MR.operand_store(array_target)
    inv = K.derive_knobs(array_target, working_set=(0, 0))
    assert inv.get("tile.stage_depth").domain.hi == store.banks    # the bank bound, unmodified
    assert "classify" not in inv.get("tile.stage_depth").domain.source
    assert not inv.get("layer.fusion_depth").determinable


def test_knob_names_are_unique_across_every_surface(array_target):
    for ws in (None, (8, 8)):
        names = K.derive_knobs(array_target, working_set=ws).names()
        assert len(names) == len(set(names))
