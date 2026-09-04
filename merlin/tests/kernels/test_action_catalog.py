"""R4: CCA divergences -> typed compiler-action catalog (FLAG/HEURISTIC/PASS/KNOB)."""
from __future__ import annotations

from merlin.common.paths import repo_root
from merlin.kernels import action_catalog as ac
from merlin.kernels import cca, cca_compare


def _pair():
    ours = cca.CCA(op="matmul", backend=["rvv"],
                   compute=cca.ComputeFacet(op="matmul", contraction_form="mul_add",
                                            widening=False, epilogue="none"),
                   vector=cca.VectorFacet(sew=32, lmul=2.0, vl_strategy="vsetivli_fixed"),
                   provenance={"level": "asm", "source": "ours"})
    expert = cca.CCA(op="matmul", backend=["rvv"],
                     compute=cca.ComputeFacet(op="matmul", contraction_form="fused_fma",
                                              widening=False, epilogue="none"),
                     vector=cca.VectorFacet(sew=32, lmul=4.0, vl_strategy="vsetvl_loop"),
                     provenance={"level": "asm", "source": "xnnpack_rvv_gemm"})
    return expert, ours


def test_divergences_route_to_typed_actions():
    expert, ours = _pair()
    divs = cca_compare.compare(expert, ours, evidence=["xnnpack_rvv_gemm"])
    actions, unrouted = ac.build_catalog(divs)
    assert unrouted == []
    by_axis = {a.divergence_axis: a for a in actions}
    cf = by_axis["compute.contraction_form"]
    # Evidence-driven, RESOLVED: knob/flag attempts measured no-op -> demoted to PASS; the PASS
    # was then implemented + certified (impr_rvv_v5: vfmacc=8065, correct) -> re-promoted forkable.
    assert cf.action_class == "PASS" and cf.forkable_now
    assert cf.target_seam == "impr_features:fused_vfmacc_contraction"
    assert by_axis["vector.lmul"].action_class == "KNOB" and by_axis["vector.lmul"].forkable_now
    vl = by_axis["vector.vl_strategy"]
    assert vl.action_class == "PASS" and not vl.forkable_now            # deferred work-item


def test_dtype_axes_route_to_dtype_strategy_knob():
    # WS-C: the accumulate-width and element-width axes close via the existing dtype_strategy knob
    # (the datapath lever), a distinct axis from `widening`. Both are forkable KNOB registrations.
    for axis, expert, ours in (("compute.accumulator_dtype", "i32", "f32"), ("vector.sew", 8, 32)):
        d = cca_compare.Divergence(axis=axis, expert=expert, ours=ours, backend="rvv", evidence=["x"])
        a = ac.route(d)
        assert a is not None, axis
        assert a.action_class == "KNOB" and a.forkable_now, axis
        assert "dtype_strategy" in a.target_seam, axis


def test_evidence_carried_through():
    expert, ours = _pair()
    divs = cca_compare.compare(expert, ours, evidence=["xnnpack_rvv_gemm", "openblas_rvv_gemm"])
    actions, _ = ac.build_catalog(divs)
    assert all("xnnpack_rvv_gemm" in a.evidence for a in actions)


def test_no_divergence_when_equal():
    expert, _ = _pair()
    assert cca_compare.compare(expert, expert) == []


def test_shape_conditional_optimization():
    # the SAME divergence gets a DIFFERENT optimization depending on the shape regime (small vs large)
    from merlin.kernels.cca_compare import Divergence
    rb = Divergence("compute.register_block", (7, None), (1, None), "rvv")
    mr = Divergence("compute.mr_adapts_to_m", True, False, "rvv")
    # large square: register-block MR=7 applies; the small-M clamp does NOT
    assert ac.route_for_shape(rb, "matmul", 256, 256, 256) is not None
    assert ac.route_for_shape(mr, "matmul", 256, 256, 256) is None
    # small-M (M=1 decode): the MR-clamp applies; the big register-block does NOT
    assert ac.route_for_shape(mr, "matmul", 1, 256, 256) is not None
    assert ac.route_for_shape(rb, "matmul", 1, 256, 256) is None
    # a shape-agnostic action (empty shape_regimes) applies in every regime
    cf = ac.route(Divergence("compute.contraction_form", "fused_fma", "mul_add", "rvv"))
    assert ac.applies_to_shape(cf, "skinny") and ac.applies_to_shape(cf, "square_large")


def test_seam_location_classifies_prefixes():
    # the "which file do I edit" map: a pass: seam needs new code; the others edit existing seams.
    assert ac.seam_location("pass:fuse-requant-narrowing-store")["needs_new_code"] is True
    assert ac.seam_location("impr_features:fused_vfmacc_contraction")["needs_new_code"] is False
    assert ac.seam_location("schedule:dtype_strategy")["needs_new_code"] is False
    assert "impr_features.py" in ac.seam_location("impr_features:x")["seam_file"]


def test_escalation_ladder_is_monotone_with_seam_files():
    # accumulator_resident is the multi-rung axis: PASS (impr feature) then CODEGEN (new pass).
    ladder = ac.escalation_ladder("compute.accumulator_resident")
    classes = [step["action_class"] for step in ladder]
    assert classes == ["PASS", "CODEGEN"]
    order = [ac._CLASS_ORDER[c] for c in classes]
    assert order == sorted(order)                    # weakest -> strongest
    assert all(step["seam_file"] for step in ladder)  # every rung names a file to edit
    assert ladder[-1]["needs_new_code"] is True       # the CODEGEN rung is the new-pass work-item


def test_accumulator_residency_routes_to_deferred_pass():
    # expert keeps the accumulator resident, ours does not -> a PASS action at the impr feature
    # seam. forkable_now is HONEST: the transform-dialect feature does not yet fully close it (still
    # spills the accumulator per K-tile), so it is a deferred work-item, not a green fork.
    d = cca_compare.Divergence(axis="compute.accumulator_resident", expert=True, ours=False,
                               backend="rvv", evidence=["openblas_rvv_gemm"])
    a = ac.route(d)
    assert a is not None and a.action_class == "PASS"
    assert a.target_seam == "impr_features:accumulator_resident_microkernel"
    assert a.forkable_now is False                       # deferred: transform path doesn't close it


def test_accumulator_residency_codegen_when_ours_unknown():
    # when ours can't even be judged (no fma loop), route to the dedicated micro-kernel CODEGEN
    # closer (the intrinsic_microkernel ceiling target), also a deferred work-item.
    d = cca_compare.Divergence(axis="compute.accumulator_resident", expert=True, ours=None,
                               backend="rvv")
    a = ac.route(d)
    assert a is not None and a.action_class == "CODEGEN" and a.forkable_now is False


def test_vl_nr_routes_to_forkable_heuristic():
    # NR=vsetvlmax (VL-adaptive output tile + N-tail) is expressible today (the N-tail-safe feature
    # vectorizes small-N attention), so it is a forkable HEURISTIC.
    d = cca_compare.Divergence(axis="compute.nr_is_vsetvlmax", expert=True, ours=False,
                               backend="rvv")
    a = ac.route(d)
    assert a is not None and a.action_class == "HEURISTIC" and a.forkable_now is True
    assert "vsetvlmax" in a.target_seam


def test_mtail_routes_to_forkable_heuristic():
    # MR=min(MR,M) (matmul M-tail clamp) is the M-side analog of nr_is_vsetvlmax: a forkable
    # HEURISTIC (the accumulator_resident_mtail feature vectorizes the M=1 token-decode matmul).
    d = cca_compare.Divergence(axis="compute.mr_adapts_to_m", expert=True, ours=False,
                               backend="rvv")
    a = ac.route(d)
    assert a is not None and a.action_class == "HEURISTIC" and a.forkable_now is True
    assert "MR=min(MR,M)" in a.target_seam
    assert "accumulator_resident_mtail" in a.target_seam


def test_reduction_form_routes_to_forkable_vectorize_reduction_pass():
    # compute.reduction_form was a bijection ORPHAN (a lever with no route). It now routes to the
    # vectorize_reduction PASS (vfredusum/vredsum), forkable now (a registered impr feature).
    d = cca_compare.Divergence(axis="compute.reduction_form", expert="vredsum_tree", ours="none",
                               backend="rvv")
    a = ac.route(d)
    assert a is not None and a.action_class == "PASS" and a.forkable_now is True
    assert a.target_seam == "impr_features:vectorize_reduction"
    assert a.intended_facet == {"compute.reduction_form": "vredsum_tree"}
    # no route when we already vectorize the reduction (ours matches a real reduction form)
    assert ac.route(cca_compare.Divergence(axis="compute.reduction_form", expert="vredsum_tree",
                                           ours="vredsum_tree", backend="rvv")) is None


def test_reduction_form_no_longer_a_bijection_orphan():
    from merlin.kernels.cca_contract import check_bijection
    r = check_bijection("rvv")
    assert "compute.reduction_form" not in r.orphan_fields
    assert r.unexpected().clean          # the ratchet stays green (only allowlisted gaps remain)


def test_unrouted_reported_not_dropped():
    # an axis with no route is returned as unrouted, never silently dropped
    d = cca_compare.Divergence(axis="compute.made_up_axis", expert="x", ours="y", backend="rvv")
    actions, unrouted = ac.build_catalog([d])
    assert actions == [] and len(unrouted) == 1


# --- intended-vs-achieved + escalation ladder (the closed-loop methodology fix) ---

def test_routes_carry_machine_readable_intended_facet():
    # register_block: the promise is "reach the EXPERT's MR" (derived from the divergence, not a const)
    d = cca_compare.Divergence(axis="compute.register_block", expert=(7, ("vsetvlmax", 4)),
                               ours=None, backend="rvv")
    a = ac.route(d)
    assert a.intended_facet == {"compute.register_block": 7}
    # accumulator_resident: the cheapest matching class is PASS, promising resident=True
    d2 = cca_compare.Divergence(axis="compute.accumulator_resident", expert=True, ours=False,
                                backend="rvv")
    a2 = ac.route(d2)
    assert a2.action_class == "PASS" and a2.intended_facet == {"compute.accumulator_resident": True}


def test_achieved_residual_detects_unmet_promise():
    a = ac.route(cca_compare.Divergence(axis="compute.accumulator_resident", expert=True,
                                        ours=False, backend="rvv"))
    bad = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(accumulator_resident=False))
    ok = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(accumulator_resident=True))
    assert ac.achieved_residual(a, bad) == ["compute.accumulator_resident"]
    assert ac.achieved_residual(a, ok) == []
    # register_block: emitted MR must be >= promised MR
    arb = ac.route(cca_compare.Divergence(axis="compute.register_block", expert=(7, None),
                                          ours=None, backend="rvv"))
    mr4 = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(register_block=(4, None)))
    mr7 = cca.CCA(op="matmul", backend=["rvv"], compute=cca.ComputeFacet(register_block=(7, None)))
    assert ac.achieved_residual(arb, mr4) == ["compute.register_block"]
    assert ac.achieved_residual(arb, mr7) == []


def test_escalation_ladder_walks_up_classes():
    # PASS was insufficient for accumulator_resident -> escalate to the CODEGEN microkernel route
    d = cca_compare.Divergence(axis="compute.accumulator_resident", expert=True, ours=False,
                               backend="rvv")
    esc = ac.route_escalated(d, prior_class="PASS")
    assert esc is not None and esc.action_class == "CODEGEN"
    # CODEGEN is the top of the ladder -> nothing stronger
    assert ac.route_escalated(d, prior_class="CODEGEN") is None


class TestPromisesAreDerivedFromTheExpert:
    """A route that exists to close an axis toward the expert is promising the expert's value on that
    axis. Deriving that promise is what lets the emitted-code audit confirm the action landed WITHOUT
    running anything -- which is the only gate on the ladder that costs a build instead of hardware.
    """

    def _div(self, axis, expert, ours=None):
        from merlin.kernels.cca_compare import Divergence
        return Divergence(axis=axis, expert=expert, ours=ours, backend="rvv", evidence=[])

    def test_a_match_the_expert_axis_gets_a_promise_without_being_named(self):
        """The derivation used to name two axes as literals, leaving every other one unverifiable."""
        from merlin.kernels.action_catalog import route
        a = route(self._div("vector.sew", 8))
        assert a is not None and a.intended_facet == {"vector.sew": 8}

    def test_an_absent_expert_value_yields_NO_promise(self):
        """'The expert does not exhibit this property' names no target to reach, so there is nothing
        to promise. Fail closed rather than promising None, which anything would satisfy."""
        from merlin.kernels.action_catalog import _promised_value
        assert _promised_value(None) is None

    def test_a_tuple_promise_collapses_to_what_the_audit_actually_compares(self):
        """_facet_value reads register_block as its MR, so promising the whole tuple would compare a
        tuple against an int and never match."""
        from merlin.kernels.action_catalog import _promised_value
        assert _promised_value((7, 4)) == 7

    def test_direction_is_declared_not_inferred_from_the_value_type(self):
        """A bigger register block is better; a bigger vector.sew is worse (wider elements, fewer
        lanes). Both are ints, so type cannot decide the comparison.

        The promise is per ROUTE, not per axis: an axis can carry both a widening and a narrowing
        route, and collapsing them into `{r.axis: r}` hides whichever is declared second. That is
        not hypothetical -- `compute.register_block` now has both, because XNNPACK's int8 ukernel is
        1x4v (MR=1) while its f32 kernel is 7x4v, so "more blocking is better" holds in one
        direction and is exactly backwards in the other.
        """
        from merlin.kernels.action_catalog import _RVV_ROUTES, route

        def _rb(expert_mr, ours_mr):
            return route(self._div("compute.register_block",
                                   (expert_mr, ("vsetvlmax", 4.0)), (ours_mr, ("vsetvlmax", 4.0))))

        # RAISING toward an expert above us keeps the promise by exceeding it.
        up = _rb(7, 1)
        assert up.promise_comparison == "at_least" and "raise" in up.change
        # LOWERING toward an expert below us does not: overshooting downward IS the regression, and
        # `at_least` would certify our slower config as a kept promise.
        down = _rb(1, 4)
        assert down.promise_comparison == "exact" and "lower" in down.change
        assert all(r.promise_comparison == "exact"
                   for r in _RVV_ROUTES if r.axis == "vector.sew")

    def test_at_least_credits_exceeding_the_expert(self):
        from merlin.kernels.action_catalog import CompilerAction, achieved_residual
        from merlin.kernels.cca import CCA, ComputeFacet
        act = CompilerAction(divergence_axis="compute.register_block", action_class="PASS",
                             target_seam="s", change="c", forkable_now=True, expected_effect="e",
                             backend="rvv", intended_facet={"compute.register_block": 4},
                             promise_comparison="at_least")
        cca = CCA(op="matmul", backend="rvv", compute=ComputeFacet(register_block=(8, 4)))
        assert achieved_residual(act, cca) == []          # 8 >= 4 keeps the promise

    def test_at_least_still_reports_falling_short(self):
        from merlin.kernels.action_catalog import CompilerAction, achieved_residual
        from merlin.kernels.cca import CCA, ComputeFacet
        act = CompilerAction(divergence_axis="compute.register_block", action_class="PASS",
                             target_seam="s", change="c", forkable_now=True, expected_effect="e",
                             backend="rvv", intended_facet={"compute.register_block": 8},
                             promise_comparison="at_least")
        cca = CCA(op="matmul", backend="rvv", compute=ComputeFacet(register_block=(4, 4)))
        assert achieved_residual(act, cca) == ["compute.register_block"]

    def test_exact_does_not_credit_merely_exceeding(self):
        """Default semantics must stay exact -- a wider SEW than the expert is worse, not better."""
        from merlin.kernels.action_catalog import CompilerAction, achieved_residual
        from merlin.kernels.cca import CCA, VectorFacet
        act = CompilerAction(divergence_axis="vector.sew", action_class="KNOB", target_seam="s",
                             change="c", forkable_now=True, expected_effect="e", backend="rvv",
                             intended_facet={"vector.sew": 8})
        cca = CCA(op="matmul", backend="rvv", vector=VectorFacet(sew=32))
        assert achieved_residual(act, cca) == ["vector.sew"]


class TestAnUnverifiableActionIsNotAnAchievedOne:
    """The emitted-code audit is the one gate that needs no hardware. A vacuous pass there is the most
    expensive kind of wrong: it certifies, for free, something nothing checked."""

    def _action(self, facet):
        from merlin.kernels.action_catalog import CompilerAction
        return CompilerAction(divergence_axis="compute.widening", action_class="KNOB",
                              target_seam="schedule:x", change="c", forkable_now=True,
                              expected_effect="e", backend="rvv", intended_facet=facet)

    def test_no_promise_is_not_reported_as_closed(self):
        from merlin.kernels.search_step import make_step
        step = make_step(self._action(None), None, correctness_ok=True, speedup=1.2)
        assert step.achieved is False and step.promise_checkable is False
        assert "UNVERIFIED" in step.to_line()

    def test_an_unverifiable_step_has_nothing_to_escalate_toward(self):
        """The gap is a missing promise in the catalog, not a weak lever in the compiler, so it must
        not be pushed up the FLAG->KNOB->...->CODEGEN ladder."""
        from merlin.kernels.search_step import make_step
        assert make_step(self._action(None), None, correctness_ok=True, speedup=None).residual == []

    def test_a_kept_promise_still_reads_as_closed(self):
        from merlin.kernels.search_step import make_step
        from merlin.kernels.cca import CCA, ComputeFacet
        cca = CCA(op="matmul", backend="rvv", compute=ComputeFacet(widening=True))
        step = make_step(self._action({"compute.widening": True}), cca,
                         correctness_ok=True, speedup=1.2)
        assert step.achieved is True and step.promise_checkable is True
        assert "closed" in step.to_line()


# ---------------------------------------------------------------------------------------
# Seam honesty. A route's `target_seam` is not documentation: `mining/fork_from_action`
# splits an `impr_features:` seam and puts the resulting string STRAIGHT into
# `compiler_features`. So a seam naming something that does not exist mints a fork that
# dies with "unknown impr feature" rather than running -- and it looks, from the outside,
# exactly like the lever not working. That is what `coverage.unclaimed_op_classes` did:
# its seam read `per_op_register_block` while the sentinel is `perop_register_block`, and
# it also carried `forkable_now=False`, so the dead name was never exercised.
# ---------------------------------------------------------------------------------------

#: Names a seam may resolve to that are NOT registered ImprFeatures. `perop_register_block` is a
#: SENTINEL: `zephyr_model.prepare_for_lowering` derives the per-op block table from the prepared IR
#: and swaps the sentinel for the concrete `ensure_perop_block(...)` feature before lowering, so it is
#: deliberately absent from the registry -- if it reached `normalize` it SHOULD raise.
_SEAM_SENTINELS = {"perop_register_block"}


def _impr_seam_feature(seam: str) -> str | None:
    return seam.split(":", 1)[1].split()[0] if seam.startswith("impr_features:") else None


def test_every_impr_features_seam_names_something_that_exists():
    """Applies to EVERY route, forkable or not: a seam that names nothing is a broken reference
    whether or not anything currently follows it."""
    from merlin.kernels import action_catalog as ac
    from merlin.llvmlower import impr_features as F

    known = set(F.known()) | _SEAM_SENTINELS
    dead = []
    for r in ac._RVV_ROUTES:
        feat = _impr_seam_feature(r.target_seam)
        if feat is not None and feat not in known:
            dead.append((r.axis, feat, r.forkable_now))
    assert not dead, f"routes name non-existent impr features: {dead}"


def test_a_forkable_seam_mints_a_fork_the_lowering_can_actually_resolve():
    """`forkable_now=True` is a promise that the beam can mint this fork and it will build. Assert the
    proposer produces a real feature override (not a demoted work-item) for every such seam."""
    from merlin.kernels import action_catalog as ac
    from merlin.mining.fork_from_action import action_to_fork

    broken = []
    for r in ac._RVV_ROUTES:
        feat = _impr_seam_feature(r.target_seam)
        if feat is None or not r.forkable_now:
            continue
        # `action_to_fork` consumes a CompilerAction; build the one this route would produce. Only the
        # seam/flag/axis matter to the mapping, so a minimal stand-in is honest here.
        action = ac.CompilerAction(
            divergence_axis=r.axis, action_class=r.action_class, target_seam=r.target_seam,
            change=r.change, forkable_now=r.forkable_now, expected_effect=r.expected_effect,
            backend="rvv", evidence=(), intended_facet=r.intended_facet)
        fork = action_to_fork(action, {})
        if not fork.forkable or fork.overrides.get("compiler_features") != [feat]:
            broken.append((r.axis, feat, fork.forkable, fork.overrides))
    assert not broken, f"forkable seams that do not mint a usable fork: {broken}"


def test_per_op_register_block_is_forkable_because_it_is_wired():
    """The flag flipped only after checking the machinery, so pin what makes it true: the sentinel is
    consumed by the whole-model preparation, which derives + tags + swaps in the concrete feature."""
    from merlin.kernels import action_catalog as ac
    from merlin.llvmlower.impr_features import PEROP_BLOCK_NAME

    route = next(r for r in ac._RVV_ROUTES
                 if r.axis == "coverage.unclaimed_op_classes")
    assert route.forkable_now is True
    assert _impr_seam_feature(route.target_seam) == PEROP_BLOCK_NAME
    src = (repo_root() / "merlin/python/merlin/runtime/backends/zephyr_model.py").read_text()
    assert "if PEROP_BLOCK_NAME in features:" in src           # the sentinel IS consumed
    assert "ensure_perop_block(table, _PEROP_KC)" in src       # ...and swapped for the real feature


# ---------------------------------------------------------------------------------------
# Escalation on an "eliminate X" axis. The guard used to be `bool(divergence.expert)`,
# which silently disabled escalation on every axis whose TARGET is the absence of
# something -- the expert's value there is (), 0 or False, all falsy. Three axes were
# affected, and one of them is the largest measured lever in the profile.
# ---------------------------------------------------------------------------------------

def _div(axis, ours, expert):
    from merlin.kernels.cca_compare import Divergence
    return Divergence(axis=axis, backend="rvv", ours=ours, expert=expert)


def test_escalation_works_when_the_target_is_the_absence_of_something():
    """envelope.runtime_calls' cheapest rung is the SELF-copy erase, which cannot remove a copy whose
    source and destination differ. Measured on small_llama int8 AFTER that erase runs, the residual copy
    family is 38.59% of real model work over 24 call sites -- so the ladder MUST be able to move past
    the erase. With a truthiness guard it never could, because an expert GEMM's runtime_calls is ()."""
    d = _div("envelope.runtime_calls", ("memcpy", "memrefCopy", "memset"), ())
    cheap = ac.route(d)
    assert cheap.action_class == "PASS" and cheap.forkable_now is True
    assert cheap.intended_facet == {"envelope.runtime_calls": ()}, "the promise must be checkable"

    up = ac.route_escalated(d, cheap.action_class)
    assert up is not None, "a falsy target must not disable escalation"
    assert up.action_class == "CODEGEN" and up.forkable_now is False
    assert "store-once" in up.target_seam or "STORE C ONCE" in up.change
    assert ac.route_escalated(d, "CODEGEN") is None, "and it must terminate"


def test_a_falsy_target_is_a_target_but_an_unknown_one_is_not():
    """(), 0 and False are legitimate targets -- "call nothing", "zero calls in the loop". None means
    the axis was not lifted, so there is genuinely nothing to escalate toward and it must still block."""
    assert ac.route_escalated(_div("envelope.runtime_calls", ("memcpy",), ()), "PASS") is not None
    assert ac.route_escalated(_div("envelope.runtime_calls", ("memcpy",), None), "PASS") is None


def test_no_route_promises_a_target_it_cannot_express():
    """Every route on an elimination axis must carry, or derive, a promise -- otherwise the escalation
    it feeds has nothing to check and `achieved_residual` returns [] for any emitted code at all."""
    for axis, ours, expert in (("envelope.runtime_calls", ("memcpy",), ()),
                               ("coverage.unclaimed_op_classes", ("linalg.batch_matmul",), ())):
        a = ac.route(_div(axis, ours, expert))
        if a is None:
            continue
        assert a.intended_facet, f"{axis} routed to an action with no checkable promise"
        assert axis in a.intended_facet


# ------------------------------------------------- the ladder must terminate in something ACTIONABLE

def test_every_blocked_route_says_WHERE_the_fix_goes():
    """The escalation ladder was terminating in prose.

    A route with ``forkable_now=False`` means no knob or feature expresses the fix and new code must
    be written -- and that is where mining.pass_slot takes over. But the slot needs a MODULE to
    overlay, and of the six blocked routes exactly one (``pass:llvmlower/act_poly.py``) named a module
    that exists; the rest were labels like ``pass:tile-epilogue-store-once``. So the leaf could not act
    on them, including the largest measured lever left (the residual rank-generic copies, 38.59% of
    real work on small_llama int8). The catalog said what to fix without saying where.

    A blocked route is acceptable when ANY of these holds:
      * its seam resolves to a module the slot can overlay; or
      * SEAMS_NEEDING_A_NEW_MODULE declares why there is none; or
      * a stronger rung on the same axis satisfies one of the above (the ladder escalates to it, so
        that rung is where the work-item lives).
    """
    from merlin.kernels import action_catalog as ac

    def _ok(route) -> bool:
        return (ac.seam_module(route.target_seam) is not None
                or ac.seam_needs_new_module(route.target_seam) is not None)

    order = ac._CLASS_ORDER
    unactionable = []
    for backend, routes in ac._ROUTES.items():
        for r in routes:
            if r.forkable_now or _ok(r):
                continue
            stronger = [x for x in routes
                        if x.axis == r.axis
                        and order.get(x.action_class, -1) > order.get(r.action_class, -1)
                        and _ok(x)]
            if not stronger:
                unactionable.append(f"{backend}:{r.action_class}:{r.axis} seam={r.target_seam!r}")
    assert not unactionable, (
        "these blocked routes name neither a module the pass slot can overlay nor a declared reason "
        "there is none, so the ladder dead-ends in prose:\n  " + "\n  ".join(unactionable))


def test_seam_module_resolves_only_a_pass_seam_whose_file_really_exists():
    """No guessing: a seam that does not resolve returns None, and the caller consults the declared
    reasons. A `pass:` label, a feature seam and a schedule seam are all correctly not modules."""
    from merlin.kernels import action_catalog as ac
    assert ac.seam_module("pass:llvmlower/act_poly.py (extend coverage)") == "merlin.llvmlower.act_poly"
    assert ac.seam_module("pass:llvmlower/does_not_exist.py") is None
    assert ac.seam_module("pass:tile-epilogue-store-once (a label)") is None
    assert ac.seam_module("impr_features:erase_self_copy") is None
    assert ac.seam_module("schedule:vector_sizes") is None
    assert ac.seam_module("") is None
    assert ac.seam_module("no-colon-at-all") is None


def test_every_declared_new_module_reason_belongs_to_a_real_blocked_seam():
    """The declaration list may only shrink as passes get written. An entry naming a seam no route
    carries is stale bookkeeping that would let a future dead-end pass the gate above."""
    from merlin.kernels import action_catalog as ac
    tokens = set()
    for routes in ac._ROUTES.values():
        for r in routes:
            kind, sep, rest = (r.target_seam or "").partition(":")
            if sep:
                tokens.add(rest.strip().split(" ", 1)[0].strip())
    stale = sorted(set(ac.SEAMS_NEEDING_A_NEW_MODULE) - tokens)
    assert not stale, f"declared reasons for seams no route carries: {stale}"
