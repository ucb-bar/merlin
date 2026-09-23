"""Two programs that are not the same program may not be scored against each other.

WHY THIS EXISTS. The ledger already refused to score across DESIGNS and that refusal is right. It
had an exact twin nobody had written: three programs in this project carry whole-model numbers for
one network, and a ratio between two of them travelled for weeks as a compiler result. It is not
one -- the identity is a property of the lowering and of the capture it came from, so no compiler
edit moves a candidate between identities and the "gap" is not one.

Every negative case below is a mutation of a positive one, so a refusal that has stopped refusing
is visible rather than merely absent.
"""

from __future__ import annotations

import pytest

from merlin.perf import program_identity as P
from merlin.perf.target_reference import (
    ProgramIdentityError,
    ReferenceError,
    estimate_cycles,
    find_reference,
    load_ledger,
    load_program_identities,
    load_references,
    score_against_reference,
    structural_gap,
)

# A two-identity roster with the shape the real one has: the two share some facts and differ on
# others, so "states enough facts" and "states the DISTINGUISHING facts" are not the same thing.
_ROSTER = {
    "pointer_entry_static": {
        "facts": {
            "entry": "pointer_arguments",
            "parameters": "abi_arguments",
            "quantization_scheme": "static",
            "cycle_scope": "window",
        }
    },
    "pointer_entry_dynamic": {
        "facts": {
            "entry": "pointer_arguments",
            "parameters": "abi_arguments",
            "quantization_scheme": "dynamic",
            "cycle_scope": "window",
        }
    },
    "closed_image": {
        "facts": {
            "entry": "no_arguments",
            "parameters": "linked_blobs",
            "quantization_scheme": "static",
            "cycle_scope": "summed_brackets",
        }
    },
    "UNKNOWN": {"description": "declares no facts, so nothing resolves to it"},
}


class TestItResolvesOnTwoAgreeingFacts:
    def test_enough_distinguishing_facts_name_the_program(self):
        got = P.resolve({"entry": "no_arguments", "parameters": "linked_blobs"}, _ROSTER)
        assert got.resolved and got["name"] == "closed_image"
        assert set(got["confirmed_by"]) == {"entry", "parameters"}

    def test_it_reports_which_declared_facts_went_unconfirmed(self):
        """Weaker evidence is allowed and must be VISIBLE -- the design-identity rule."""
        got = P.resolve({"entry": "no_arguments", "parameters": "linked_blobs"}, _ROSTER)
        assert set(got["unconfirmed"]) == {"quantization_scheme", "cycle_scope"}

    def test_a_fact_no_identity_declares_is_not_an_error(self):
        """The roster says what DISTINGUISHES programs, not everything one can say about itself."""
        got = P.resolve(
            {"entry": "no_arguments", "parameters": "linked_blobs", "compiler_revision": "abc123"},
            _ROSTER,
        )
        assert got.resolved and got["name"] == "closed_image"


class TestItRefusesRatherThanGuessing:
    def test_one_fact_never_names_a_program(self):
        """THE MUTATION THAT MATTERS. An argument count is a number two unrelated lowerings can
        share, and reading one as a program is how this got misidentified in the first place."""
        got = P.resolve({"entry": "pointer_arguments"}, _ROSTER)
        assert not got.resolved
        assert "2 agreeing facts are required" in got["reason"] or "agreeing facts" in got["reason"]

    def test_facts_that_match_two_identities_are_reported_not_picked(self):
        """`entry` + `parameters` are shared by the two pointer-entry programs. Picking one would
        make a single fact bundle name two programs, which is the confusion this module ends."""
        got = P.resolve({"entry": "pointer_arguments", "parameters": "abi_arguments"}, _ROSTER)
        assert not got.resolved
        assert "cannot name two programs" in got["reason"]
        assert "pointer_entry_static" in got["reason"] and "pointer_entry_dynamic" in got["reason"]

    def test_one_disagreeing_fact_disqualifies_an_identity_however_many_agree(self):
        got = P.resolve(
            {"entry": "pointer_arguments", "parameters": "abi_arguments", "quantization_scheme": "mystery"},
            _ROSTER,
        )
        assert not got.resolved
        assert "mystery" in got["reason"], "the refusal should name the fact that disagreed"

    def test_a_null_fact_is_not_a_statement(self):
        """Present-and-null is the honest way to say "I do not know this about myself"; folding it
        into a statement would let a program resolve on evidence it never gave."""
        got = P.resolve({"entry": "no_arguments", "parameters": None}, _ROSTER)
        assert not got.resolved

    def test_an_identity_with_no_facts_is_unreachable(self):
        """`UNKNOWN` is a label an entry may DECLARE and nothing may RESOLVE to by accident."""
        assert all(P.resolve(f, _ROSTER)["name"] != "UNKNOWN" for f in (_ROSTER["closed_image"]["facts"], {}))

    def test_an_empty_roster_refuses_instead_of_accepting_anything(self):
        got = P.resolve({"entry": "no_arguments", "parameters": "linked_blobs"}, {})
        assert not got.resolved and "no program identities" in got["reason"]


class TestComparability:
    def test_the_same_name_on_both_sides_is_comparable(self):
        verdict = P.comparability(P.ProgramIdentity(name="a"), P.ProgramIdentity(name="a"))
        assert verdict.ok

    def test_different_names_are_refused_and_the_reason_names_both(self):
        verdict = P.comparability(P.ProgramIdentity(name="a"), P.ProgramIdentity(name="b"))
        assert not verdict.ok
        assert "'a'" in verdict["reason"] and "'b'" in verdict["reason"]
        assert "no compiler edit moves a candidate between them" in verdict["reason"]

    def test_an_unknown_candidate_is_refused_not_assumed_to_match(self):
        """THREE STATES, NEVER TWO. "We could not tell" must not collapse into "they match"."""
        unknown = P.ProgramIdentity(name=None, reason="nothing was stated")
        verdict = P.comparability(unknown, P.ProgramIdentity(name="b"))
        assert not verdict.ok and "nothing was stated" in verdict["reason"]

    def test_an_unknown_reference_is_refused_too(self):
        unknown = P.ProgramIdentity(name=None, reason="the reference recorded no program")
        verdict = P.comparability(P.ProgramIdentity(name="a"), unknown)
        assert not verdict.ok and "the reference recorded no program" in verdict["reason"]

    def test_the_raising_form_raises_with_the_same_reason(self):
        with pytest.raises(P.ProgramIdentityError) as caught:
            P.require_comparable(P.ProgramIdentity(name="a"), P.ProgramIdentity(name="b"))
        assert "not comparable across program identities" in str(caught.value)

    def test_a_reference_that_declares_no_identity_is_unknown_not_wildcard(self):
        got = P.identity_of_reference({"model": "m", "design": "d"})
        assert not got.resolved and "states no `program_identity`" in got["reason"]

    def test_two_entries_declaring_unknown_are_not_thereby_the_same_program(self):
        """THE MUTATION THAT WOULD MAKE HONESTY THE LOOPHOLE. `UNKNOWN` is a string, and a resolver
        that matched it against itself would make the most honest declaration in the file the most
        permissive thing in it: two numbers nobody identified would be freely comparable."""
        left = P.identity_of_reference({"program_identity": P.UNKNOWN_IDENTITY})
        right = P.identity_of_reference({"program_identity": P.UNKNOWN_IDENTITY})
        assert not left.resolved and not right.resolved
        assert not P.comparability(left, right).ok

    def test_a_candidate_stating_unknown_is_also_refused(self):
        with pytest.raises(ProgramIdentityError):
            structural_gap(_buffer(), _REFS["same_identity"], program_identity=P.UNKNOWN_IDENTITY)


# --------------------------------------------------------------------------------------------
# The scorer itself. These are the call sites where a cross-identity comparison does its damage.
# --------------------------------------------------------------------------------------------

_REFS = {
    "same_identity": {
        "model": "m",
        "design": "d",
        "device": "dev",
        "status": "achieved",
        "program_identity": "pointer_entry_static",
        "measured": {
            "whole_model_cycles": 1000,
            "cycles_per_host_operation_anchor": 2.0,
            "host_share_lower_bound": 0.9,
        },
        "emitted_structure": {"commands": 2, "mesh_regions": 54, "kernel_abi_args": 393},
    },
}


def _buffer(mesh=1, args=325):
    return {
        "commands": [{"opcode": "COMMIT"}, {"opcode": "COMMIT"}],
        "kernel_abi": {"args": [{}] * args},
        "params": {"mesh_regions": [{}] * mesh, "host_lane_regions": [], "weight_prepack_recipes": []},
    }


class TestTheScorerRefusesAcrossIdentities:
    def test_a_matching_identity_scores_normally(self):
        gap = structural_gap(
            _buffer(mesh=54, args=393), _REFS["same_identity"], program_identity="pointer_entry_static"
        )
        assert gap["matches"] is True

    def test_a_different_identity_raises_instead_of_reporting_a_gap(self):
        """THE CENTRAL MUTATION, and the real incident in miniature. A candidate compiled from a
        capsule whose interface is float emits ONE mesh region against a reference's 54. Reported
        as a gap that reads `mesh_regions: -53`, it is indistinguishable from a schedule the
        compiler failed to find -- and it is in fact the distance between two different programs,
        closable only by re-capturing the model."""
        with pytest.raises(ProgramIdentityError) as caught:
            structural_gap(_buffer(), _REFS["same_identity"], program_identity="pointer_entry_dynamic")
        assert "not comparable across program identities" in str(caught.value)

    def test_an_unstated_candidate_identity_raises_rather_than_scoring(self):
        """Silence is not a match. The caller knows what its own candidate is; leaving it unstated
        is a decision not to say, and a -53 emitted under it would be quoted exactly the same."""
        with pytest.raises(ProgramIdentityError) as caught:
            structural_gap(_buffer(), _REFS["same_identity"])
        assert "stated no `program_identity`" in str(caught.value)

    def test_a_cycle_estimate_across_identities_returns_a_refusal_not_a_number(self):
        """estimate_cycles RECORDS rather than raises, deliberately: its result is embedded in a
        loop that wraps analysis in try/except, so a raise there would become silence."""
        record = estimate_cycles(500, _REFS["same_identity"], program_identity="closed_image")
        assert record["status"] == "not_comparable_across_program_identities"
        assert "estimated_whole_model_cycles" not in record
        assert record["candidate_program_identity"] == "closed_image"
        assert record["reference_program_identity"] == "pointer_entry_static"

    def test_a_matching_identity_still_estimates(self):
        record = estimate_cycles(500, _REFS["same_identity"], program_identity="pointer_entry_static")
        assert record["status"] == "estimated"

    def test_find_reference_refuses_an_identity_the_ledger_does_not_carry(self):
        with pytest.raises(ReferenceError) as caught:
            find_reference("m", "d", references=_REFS, program_identity="closed_image")
        assert "not comparable across program identities" in str(caught.value)

    def test_the_whole_record_states_both_identities_including_when_unstated(self):
        record = score_against_reference(
            _buffer(mesh=54, args=393),
            model="m",
            design="d",
            references=_REFS,
            program_identity="pointer_entry_static",
        )
        assert record["program_identity"] == {
            "candidate": "pointer_entry_static",
            "reference": "pointer_entry_static",
            "confirmed_by": "stated by the caller",
        }


# --------------------------------------------------------------------------------------------
# THE SHIPPED LEDGER. These are the gates that fire on the next person's edit.
# --------------------------------------------------------------------------------------------


class TestTheShippedLedger:
    def test_it_declares_a_roster_and_every_declared_identity_has_facts_or_is_unknown(self):
        roster = load_program_identities()
        assert roster, "the ledger must declare `program_identities` for any entry to name one"
        for name, entry in roster.items():
            if name == P.UNKNOWN_IDENTITY:
                assert not entry.get("facts"), "UNKNOWN must declare no facts, so nothing resolves to it"
                continue
            facts = entry.get("facts") or {}
            assert len(facts) >= P.MIN_AGREEING_FACTS, f"{name} declares {len(facts)} fact(s); it cannot be resolved"
            assert entry.get("description"), f"{name} declares no description"

    def test_every_reference_states_its_program_identity(self):
        """THE MUTATION THAT FIRES ON THE NEXT EDIT. A reference with no identity is a number whose
        program is unrecorded, and the only thing anyone will do with it is compare it to
        something."""
        missing = [name for name, entry in load_references().items() if not entry.get("program_identity")]
        assert not missing, (
            f"these references state no `program_identity`: {missing}. State which program produced "
            "the number, or state UNKNOWN -- an unstated identity is not a third option"
        )

    def test_every_stated_identity_is_one_the_roster_declares(self):
        roster = set(load_program_identities())
        unknown = {
            name: entry["program_identity"]
            for name, entry in load_references().items()
            if entry.get("program_identity") and entry["program_identity"] not in roster
        }
        assert not unknown, f"references name identities the roster does not declare: {unknown}"

    def test_every_reference_states_its_device(self):
        """A configuration name is not a device: two registered bitstreams elaborate the same
        `config` onto the same board and are different silicon."""
        missing = [name for name, entry in load_references().items() if not entry.get("device")]
        assert not missing, f"these references state no `device`: {missing}"

    def test_the_ambiguous_design_string_is_the_reason_device_is_required(self):
        """Not a style rule -- a measured collision. If this ever stops holding, `design` became
        sufficient again and the extra key can go; until then it may not."""
        from merlin.common.provenance import load_artifacts
        from merlin.perf.design_identity import BITSTREAM_ROLE

        by_config: dict[str, list[str]] = {}
        for name, artifact in load_artifacts().items():
            if artifact.role == BITSTREAM_ROLE and artifact.config:
                by_config.setdefault(artifact.config, []).append(name)
        shared = {config: names for config, names in by_config.items() if len(names) > 1}
        assert shared, (
            "no configuration string is claimed by two registered bitstreams any more. If that is "
            "real, this test and the `device` key it justifies should be revisited together"
        )

    def test_a_device_a_measured_entry_names_is_a_registered_bitstream(self):
        from merlin.common.provenance import load_artifacts

        registered = set(load_artifacts())
        stray = {
            name: entry["device"]
            for name, entry in load_references().items()
            if entry.get("device") not in registered and entry.get("device") != P.UNKNOWN_IDENTITY
        }
        assert not stray, f"references name devices the pin registry does not declare: {stray}"

    def test_the_design_string_alone_no_longer_selects_a_reference(self):
        """The live proof that the collision is not hypothetical: `resnet50` now has measured
        entries on two devices under one `design`, and a lookup keyed on `design` must refuse."""
        with pytest.raises(ReferenceError) as caught:
            find_reference("resnet50", "FireSimGemminiRocketConfig")
        assert "does not pin a device" in str(caught.value)

    def test_a_device_alone_is_still_not_enough_when_it_holds_two_programs(self):
        """THE HAZARD THIS CAUGHT, on shipped data. One device now holds both our compiler's output
        and the vendor's own benchmark, and the vendor entry is an order of magnitude smaller --
        so `min` over cycles hands the authoring loop the program least like its candidate. The
        fastest number is exactly the one most likely to be a different program."""
        with pytest.raises(ReferenceError) as caught:
            find_reference("resnet50", "FireSimGemminiRocketConfig", device="firesim_gemmini_rocket_u250")
        assert "more than one program identity" in str(caught.value)

    def test_naming_both_the_device_and_the_program_resolves_it_again(self):
        name, entry = find_reference(
            "resnet50",
            "FireSimGemminiRocketConfig",
            device="firesim_gemmini_rocket_u250",
            program_identity="static_quantized_recapture_entry",
        )
        assert entry["device"] == "firesim_gemmini_rocket_u250"
        assert entry["program_identity"] == "static_quantized_recapture_entry", name

    def test_the_vendor_benchmark_is_reachable_only_by_naming_it(self):
        """It is a legitimate measured point and must stay readable. What it may not be is the
        default answer to "what should this compiler be aiming at"."""
        _, entry = find_reference(
            "resnet50",
            "FireSimGemminiRocketConfig",
            device="firesim_gemmini_rocket_u250",
            program_identity="vendor_reference_c_program",
        )
        batch = entry["measured"]["batch"]
        assert batch == 4
        # The published figure is the batch total divided by the batch and ROUNDED -- 96,142,906 / 4
        # is 24,035,726.5. Asserted to within the rounding rather than exactly, because the point is
        # that the per-image number stays reconcilable with the run that produced it, and a figure
        # that could no longer be divided back would have lost its provenance.
        assert abs(entry["measured"]["whole_model_cycles"] * batch - entry["measured"]["measured_batch_cycles"]) < batch

    def test_the_two_devices_hold_different_program_identities_and_cannot_be_ratioed(self):
        """The finding this whole file exists for, asserted against the shipped data: the fast
        number and the slow number are neither the same device nor the same program."""
        references = load_references()
        fast = references["resnet50_int8_compute_group_image_job728"]
        slow = references["resnet50_w8a8_firesim_q535"]
        assert fast["device"] != slow["device"]
        assert fast["program_identity"] != slow["program_identity"]
        assert not P.comparability(P.identity_of_reference(fast), P.identity_of_reference(slow)).ok, (
            "a ratio between these two is the unsound comparison; it must not be licensed"
        )

    def test_the_roster_lives_in_the_ledger_not_in_code(self):
        """The roster is DATA. A concern list that lived in code drifted once already."""
        assert P.ROSTER_KEY in load_ledger()

    def test_the_authoring_loop_is_not_yet_wired_and_that_is_recorded_here(self):
        """THE KNOWN HOLE, asserted so it cannot be forgotten.

        `run_global_perf_experiment.reference_gap` still calls `estimate_cycles` without stating
        which program its candidate is, so the loop keeps receiving an estimate against a
        reference it may not share an identity with. The one-line fix is written and was backed
        out of this change for an unrelated reason: that file is 6,874 lines and not yet
        ruff-formatted, the format gate judges the whole staged file, and reformatting 6,531 lines
        of a module another session has checked out is the exact in-flight-diff collision that
        policy exists to prevent.

        When the loop is wired -- pass `program_identity=` at that call site, then make this test
        the positive one that its stated identity is in the roster -- this assertion flips and
        says so."""
        from merlin.common.paths import merlin_dir

        script = merlin_dir() / "experiments/gemmini_perf_bench/scripts/run_global_perf_experiment.py"
        text = script.read_text()
        assert "estimate_cycles(" in text, "the loop no longer scores against a reference at all"
        if "program_identity=" in text:
            stated = [
                line.split("=", 1)[1].strip().strip('"')
                for line in text.splitlines()
                if line.strip().startswith("candidate_identity =")
            ]
            assert stated, "the loop passes program_identity= but states no candidate identity to pass"
            roster = set(load_program_identities())
            assert set(stated) <= roster, f"the loop states {stated}, which the roster does not declare"
