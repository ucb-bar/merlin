"""Lane and warp occupancy on a SIMT cluster, derived from the machine's ELABORATED design.

A busy bit cannot see divergence: a warp running one lane of sixteen is 100% busy for the whole issue.
These tests hold the derivation (the dimensions are read out of CIRCT's own output as a mask width, a
warp-slot count and an instance-path count — never typed, and never taken from a cycle model), the
three states, and — the point of the module — that a missing trace refuses instead of returning a
flattering 1.0.

The fixtures below are SYNTHETIC and deliberately spelled unlike any shipped design (``laneEnableMask``
in a ``SchedUnit``, not the real ``threadMasks`` in a ``WarpScheduler``), so a passing test cannot be
passing because a name was hardcoded.
"""
from __future__ import annotations

import textwrap

import pytest

from merlin.perf import simt_occupancy as SO
from merlin.targetgen.rtl import simt_geometry as SG

# A CIRCT HW-dialect elaboration: one scheduler holding an 8-slot table of 16-bit lane masks (CIRCT
# lowers a FIRRTL aggregate register into one register per index, which is the shape read back here),
# one core instantiating it, one cluster instantiating two cores.
#
# The attribute dictionary on each register carries a `: ui64` of its own — kept because reading the
# LAST colon outright, rather than the last one at bracket depth zero, turns this into a 64-lane
# machine.
_HW_SLOTS = "\n".join(
    f"    %laneEnableMask_{i} = seq.firreg %x clock %clock reset sync %reset, %c-1_i16 "
    f"{{firrtl.random_init_start = {i * 16} : ui64}} : i16"
    for i in range(8))
_HW = textwrap.dedent("""\
    module {{
      hw.module private @SchedUnit(in %clock : !seq.clock, in %reset : i1) {{
    {slots}
        hw.output
      }}
      hw.module private @CoreUnit(in %clock : !seq.clock, in %reset : i1) {{
        hw.instance "sched" @SchedUnit(clock: %clock: !seq.clock, reset: %reset: i1) -> ()
        hw.output
      }}
      hw.module @Cluster(in %clock : !seq.clock, in %reset : i1) {{
        hw.instance "c0" @CoreUnit(clock: %clock: !seq.clock, reset: %reset: i1) -> ()
        hw.instance "c1" @CoreUnit(clock: %clock: !seq.clock, reset: %reset: i1) -> ()
        hw.output
      }}
    }}
    """).format(slots=_HW_SLOTS)

# The same machine as elaborated FIRRTL: the mask table is still one aggregate, so the slot count is
# the vector dimension rather than a lowered index run.
_FIR = textwrap.dedent("""\
    circuit TestTop :
      module SchedUnit :
        input clock : Clock
        input reset : UInt<1>
        regreset laneEnableMasks : UInt<16>[8], clock, reset, _init @[Sched.scala 33:28]
      module CoreUnit :
        input clock : Clock
        inst sched of SchedUnit @[Core.scala 12:18]
      module TestTop :
        input clock : Clock
        inst c0 of CoreUnit @[Top.scala 4:9]
        inst c1 of CoreUnit @[Top.scala 5:9]
    """)


class TestDerivationFromCirct:
    def test_the_hw_dialect_gives_all_three_dimensions(self):
        g = SG.derive_from_text(_HW)
        assert (g.lane_width, g.warps_per_core, g.cores) == (16, 8, 2)
        assert g.dialect == "hw"

    def test_the_firrtl_of_the_same_machine_agrees(self):
        g = SG.derive_from_text(_FIR)
        assert (g.lane_width, g.warps_per_core, g.cores) == (16, 8, 2)
        assert g.dialect == "firrtl"

    def test_every_number_names_the_structure_it_came_from(self):
        # A lane count that does not name its source cannot be checked against the run it prices.
        g = SG.derive_from_text(_HW)
        assert "laneEnableMask" in g.keys["lane"] and "i16" in g.keys["lane"]
        assert "8" in g.keys["warp"]
        assert "SchedUnit" in g.keys["core"] and "Cluster" in g.keys["core"]

    def test_a_differently_spelled_design_resolves_the_same_way(self):
        # The whole point of factoring the role out of the identifier: a second SIMT machine that
        # writes `thread_mask` in a `WarpSlots` module must not report UNKNOWN.
        text = textwrap.dedent("""\
            circuit T :
              module WarpSlots :
                regreset thread_mask : UInt<32>[4], clock, reset, _i
              module T :
                inst w of WarpSlots
            """)
        g = SG.derive_from_text(text)
        assert (g.lane_width, g.warps_per_core, g.cores) == (32, 4, 1)

    def test_the_derived_denominators_multiply_out(self):
        g = SG.derive_from_text(_HW)
        assert g.threads_per_core == 128            # 16 lanes x 8 warps
        assert g.lane_slots_per_cycle == 256        # x 2 cores

    def test_cores_are_instance_PATHS_not_module_definitions(self):
        # CIRCT deduplicates module bodies: the identical second core is one definition and two
        # paths. Counting definitions would report a dual-core cluster as single-core.
        g = SG.derive_from_text(_FIR)
        assert g.cores == 2

    def test_a_uniquified_second_core_is_a_second_core_not_a_contradiction(self):
        # A pre-dedup FIRRTL circuit uniquifies a module per instance, so a two-core cluster
        # elaborates as `SchedUnit` and `SchedUnit_1` — identical structures under different names.
        text = textwrap.dedent("""\
            circuit T :
              module SchedUnit :
                regreset laneMasks : UInt<16>[8], clock, reset, _i
              module SchedUnit_1 :
                regreset laneMasks : UInt<16>[8], clock, reset, _i
              module T :
                inst a of SchedUnit
                inst b of SchedUnit_1
            """)
        g = SG.derive_from_text(text)
        assert (g.lane_width, g.warps_per_core, g.cores) == (16, 8, 2)

    def test_a_slotless_lane_mask_is_not_the_warp_table(self):
        # The FP unit's per-request `laneMask` is 4 bits wide on the real design. Reading it as the
        # warp table would report a sixteen-lane machine as four lanes wide.
        text = textwrap.dedent("""\
            circuit T :
              module Fpu :
                regreset req_laneMask : UInt<4>, clock, reset, _i
              module T :
                inst f of Fpu
            """)
        g = SG.derive_from_text(text)
        assert g.lane_width is None
        assert "no slot dimension" in g.unread["lane"]

    def test_two_differently_shaped_tables_resolve_to_nothing(self):
        text = textwrap.dedent("""\
            circuit T :
              module A :
                regreset laneMasks : UInt<16>[8], clock, reset, _i
              module B :
                regreset threadMasks : UInt<32>[4], clock, reset, _i
              module T :
                inst a of A
                inst b of B
            """)
        g = SG.derive_from_text(text)
        assert g.lane_width is None
        assert "lane" in g.ambiguous and "different shapes" in g.unread["lane"]

    def test_a_gap_in_the_lowered_slot_indices_refuses(self):
        # A table read as three slots because index 1 was missed is a warp denominator that is
        # quietly a third too small.
        hw = ("module {\n  hw.module @T(in %clock : !seq.clock) {\n"
              "    %laneMask_0 = seq.firreg %x clock %clock : i16\n"
              "    %laneMask_2 = seq.firreg %x clock %clock : i16\n"
              "    %laneMask_3 = seq.firreg %x clock %clock : i16\n"
              "    hw.output\n  }\n}\n")
        g = SG.derive_from_text(hw)
        assert g.lane_width is None

    def test_an_attribute_dicts_own_type_is_not_read_as_the_width(self):
        # `{firrtl.random_init_start = 0 : ui64} : i16` — taking the LAST colon outright reads ui64,
        # i.e. a 64-lane machine. The result type is the last colon at bracket depth zero.
        g = SG.derive_from_text(_HW)
        assert g.lane_width == 16

    def test_an_mlir_files_own_module_wrapper_is_not_a_firrtl_module(self):
        # Every `.hw.mlir` opens with a bare `module {`. Sniffing that as FIRRTL parses the whole
        # file in the wrong dialect and returns the honest-looking "declares no lanes".
        assert SG.sniff_dialect(_HW.splitlines()) == "hw"
        assert SG.sniff_dialect(_FIR.splitlines()) == "firrtl"

    def test_nothing_is_defaulted_when_a_design_declares_no_mask_table(self):
        # The hard rule: a lane count nobody elaborated is UNKNOWN. A plausible 16 substituted here
        # would price a run against a machine that was never built.
        g = SG.derive_from_text("circuit T :\n  module T :\n    input clock : Clock\n")
        assert g.lane_width is None and g.warps_per_core is None and g.cores is None
        assert g.threads_per_core is None and g.lane_slots_per_cycle is None

    def test_unreadable_text_is_not_a_crash(self):
        assert SG.derive_from_text("]]] nope").resolved() == ()


class TestThreeStates:
    def test_supplied_text_derives(self):
        got = SO.geometry_for_target("anything", artifact_text=_HW, cross_check=False)
        assert got["status"] == "derived"
        assert got["geometry"]["lane_width"] == 16

    def test_an_elaboration_with_no_simt_geometry_is_ABSENT(self, tmp_path):
        p = tmp_path / "design.fir"
        p.write_text("circuit T :\n  module T :\n    input clock : Clock\n", encoding="utf-8")
        got = SO.geometry_for_target("anything", artifact_path=p, cross_check=False)
        assert got["status"] == "absent"
        assert got["read"]                            # absent requires having read something

    def test_a_file_that_cannot_be_OPENED_is_UNAVAILABLE_not_absent(self, tmp_path):
        # The collapse this guards: a package emits a placeholder path when its toolchain env is
        # unset, and "we could not look" must never be reported as "the machine has no lanes".
        got = SO.geometry_for_target("anything", artifact_path=tmp_path / "nope.fir",
                                     cross_check=False)
        assert got["status"] == "unavailable"
        assert got["unreadable"]

    def test_a_contradictory_elaboration_is_UNAVAILABLE_not_absent(self, tmp_path):
        p = tmp_path / "design.fir"
        p.write_text(textwrap.dedent("""\
            circuit T :
              module A :
                regreset laneMasks : UInt<16>[8], clock, reset, _i
              module B :
                regreset threadMasks : UInt<32>[4], clock, reset, _i
              module T :
                inst a of A
                inst b of B
            """), encoding="utf-8")
        got = SO.geometry_for_target("anything", artifact_path=p, cross_check=False)
        assert got["status"] == "unavailable"
        assert "lane" in got["geometry"]["ambiguous"]

    def test_a_target_with_no_elaboration_anywhere_is_UNAVAILABLE(self):
        got = SO.geometry_for_target("no_such_target_at_all", cross_check=False)
        assert got["status"] == "unavailable"
        assert got["routes_tried"]                    # each route says why it did not answer

    def test_the_derived_geometry_names_the_file_it_came_from(self, tmp_path):
        p = tmp_path / "design.hw.mlir"
        p.write_text(_HW, encoding="utf-8")
        got = SO.geometry_for_target("anything", artifact_path=p, cross_check=False)
        assert got["source"] == str(p)
        assert got["geometry"]["source"] == str(p)


class TestTheModelIsNeverTheSource:
    """The violation this module was rewritten to undo: the denominator came from a cycle model."""

    def test_a_target_with_only_a_model_config_REFUSES(self):
        # There is no route from a `config.toml` to a geometry, by construction. A machine whose
        # elaboration is not on this host is UNKNOWN even when its simulator config is right there.
        got = SO.geometry_for_target("no_such_target_at_all", cross_check=False)
        assert got["status"] == "unavailable"
        assert "model" in got["why"] and "not a fallback" in got["why"].lower()

    def test_occupancy_refuses_rather_than_pricing_against_the_model(self):
        got = SO.occupancy_for_target("no_such_target_at_all",
                                      {SO.ACTIVE_LANE_CYCLES: 4, SO.WARP_ISSUES: 1})
        assert got["state"] == "unknown"
        assert got["geometry_status"] == "unavailable"
        assert "NOT a substitute" in got["why"]

    def test_the_cross_check_carries_no_number_a_caller_could_use_as_a_denominator(self, tmp_path):
        p = tmp_path / "model.toml"
        p.write_text("[engine]\nnum_lanes = 16\nnum_warps = 8\nnum_cores = 2\n", encoding="utf-8")
        cc = SO.model_cross_check("anything", config_path=p)
        assert cc["model_says"] == {"lane": 16, "warp": 8, "core": 2}
        # It reports what the MODEL says under a name that cannot be mistaken for the geometry.
        assert "lane_width" not in cc and "warps_per_core" not in cc and "cores" not in cc

    def test_a_cross_check_that_AGREES_is_reported_as_agreement(self, tmp_path):
        p = tmp_path / "model.toml"
        p.write_text("[engine]\nnum_lanes = 16\nnum_warps = 8\nnum_cores = 2\n", encoding="utf-8")
        derived = SG.derive_from_text(_HW).to_dict()
        cc = SO.model_cross_check("anything", derived=derived, config_path=p)
        assert cc["agrees"] == ["lane", "warp", "core"] and cc["disagrees"] == []

    def test_a_cross_check_that_DISAGREES_is_surfaced_not_swallowed(self, tmp_path):
        # A cross-check that disagrees is exactly what has to be visible: the model is then running a
        # machine that was never elaborated, and its cycles belong to THAT machine.
        p = tmp_path / "model.toml"
        p.write_text("[engine]\nnum_lanes = 16\nnum_warps = 8\nnum_cores = 4\n", encoding="utf-8")
        derived = SG.derive_from_text(_HW).to_dict()
        cc = SO.model_cross_check("anything", derived=derived, config_path=p)
        assert cc["disagrees"] == [{"role": "core", "rtl_derived": 2, "model_says": 4}]
        assert "belongs to THAT machine" in cc["why"]

    def test_an_unreadable_model_config_does_not_taint_the_derived_geometry(self, tmp_path):
        cc = SO.model_cross_check("anything", config_path=tmp_path / "absent.toml")
        assert cc["status"] == "unavailable"
        assert "stands on the RTL alone" in cc["note"]

    def test_a_ratio_key_in_the_model_config_is_not_read_as_a_count(self, tmp_path):
        p = tmp_path / "model.toml"
        p.write_text("[c]\nlanes_per_warp = 16\n", encoding="utf-8")
        cc = SO.model_cross_check("anything", config_path=p)
        assert cc["model_says"] == {"lane": None, "warp": None, "core": None}

    def test_a_flag_in_the_model_config_is_not_a_count(self, tmp_path):
        # bool is an int subclass: admitting one would make `warp_sync = true` a warp count of 1.
        p = tmp_path / "model.toml"
        p.write_text("[c]\nnum_lanes = 8\nwarp_sync = true\n", encoding="utf-8")
        cc = SO.model_cross_check("anything", config_path=p)
        assert cc["model_says"]["lane"] == 8 and cc["model_says"]["warp"] is None


class TestElaborationsThatDisagree:
    def test_two_elaborations_disagreeing_are_reported_side_by_side(self, tmp_path):
        # Two elaborated configurations of one generator are two machines. The preferred artifact is
        # authoritative and the other is quoted beside it — never resolved out of sight.
        one = tmp_path / "single.fir"
        one.write_text(textwrap.dedent("""\
            circuit T :
              module SchedUnit :
                regreset laneMasks : UInt<16>[8], clock, reset, _i
              module T :
                inst a of SchedUnit
            """), encoding="utf-8")
        got = SO.geometry_for_target("anything", artifact_path=one, cross_check=False)
        assert got["geometry"]["cores"] == 1
        assert got["corroboration"][0]["core"] == 1
        assert "contested" not in got               # one artifact cannot disagree with itself


def _geom(text: str = _HW):
    return SG.derive_from_text(text)


class TestLaneOccupancy:
    def test_a_fully_dense_kernel_occupies_every_lane(self):
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANE_CYCLES: 100 * 16, SO.WARP_ISSUES: 100}, _geom())
        assert got["lane"]["state"] == "measured"
        assert got["lane"]["value"] == pytest.approx(1.0)
        assert got["lane"]["divergence"] == pytest.approx(0.0)

    def test_a_fully_divergent_kernel_is_one_lane_of_sixteen(self):
        # The case a correctness gate cannot fail on: right answer, 1/16 of the datapath.
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANE_CYCLES: 100, SO.WARP_ISSUES: 100}, _geom())
        assert got["lane"]["value"] == pytest.approx(1.0 / 16)
        assert got["lane"]["divergence"] == pytest.approx(15.0 / 16)

    def test_per_issue_masks_reduce_to_the_same_figure(self):
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANES_PER_ISSUE: [16, 8, 4, 4]}, _geom())
        assert got["lane"]["value"] == pytest.approx(32 / 64)
        assert got["lane"]["warp_issues"] == 4

    def test_masks_that_contradict_a_pre_reduced_pair_refuse(self):
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANES_PER_ISSUE: [16, 16], SO.ACTIVE_LANE_CYCLES: 4}, _geom())
        assert got["lane"]["state"] == "unknown"
        assert "different runs" in got["lane"]["why"]

    def test_no_trace_refuses_rather_than_reporting_full_lanes(self):
        got = SO.occupancy_from_readings({}, _geom())
        assert got["state"] == "unknown"
        assert got["lane"]["value"] is None
        assert SO.ACTIVE_LANE_CYCLES in got["lane"]["why"]

    def test_no_issues_is_undefined_not_zero(self):
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANE_CYCLES: 0, SO.WARP_ISSUES: 0}, _geom())
        assert got["lane"]["state"] == "unknown"
        assert "undefined, not 0.0" in got["lane"]["why"]

    def test_a_reading_wider_than_the_derived_width_refuses(self):
        # Trace and geometry describing different machines is a REFUSAL, not a clamp to 1.0.
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANE_CYCLES: 10 * 32, SO.WARP_ISSUES: 10}, _geom())
        assert got["lane"]["state"] == "unknown"
        assert "different machines" in got["lane"]["why"]

    def test_no_lane_width_means_no_denominator(self):
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANE_CYCLES: 4, SO.WARP_ISSUES: 1},
            _geom("circuit T :\n  module T :\n    input clock : Clock\n"))
        assert got["lane"]["state"] == "unknown"
        assert "denominator" in got["lane"]["why"]


class TestWarpOccupancy:
    def test_residency_is_resident_over_declared(self):
        got = SO.occupancy_from_readings({SO.RESIDENT_WARPS: 2}, _geom())
        assert got["warp"]["value"] == pytest.approx(2 / 8)

    def test_an_unobserved_warp_count_refuses(self):
        got = SO.occupancy_from_readings({}, _geom())
        assert got["warp"]["state"] == "unknown"
        assert SO.RESIDENT_WARPS in got["warp"]["why"]

    def test_more_warps_than_a_core_holds_refuses(self):
        got = SO.occupancy_from_readings({SO.RESIDENT_WARPS: 99}, _geom())
        assert got["warp"]["state"] == "unknown"

    def test_no_derived_warps_per_core_means_no_denominator(self):
        got = SO.occupancy_from_readings({SO.RESIDENT_WARPS: 2},
                                         SO.SimtGeometry(lane_width=16))
        assert got["warp"]["state"] == "unknown"


class TestCombined:
    def test_the_product_needs_both_factors_measured(self):
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANE_CYCLES: 8 * 100, SO.WARP_ISSUES: 100, SO.RESIDENT_WARPS: 4}, _geom())
        assert got["simt"]["value"] == pytest.approx(0.5 * 0.5)

    def test_a_product_with_an_assumed_factor_is_refused(self):
        got = SO.occupancy_from_readings(
            {SO.ACTIVE_LANE_CYCLES: 8 * 100, SO.WARP_ISSUES: 100}, _geom())
        assert got["simt"]["state"] == "unknown"
        assert "assumption" in got["simt"]["why"]

    def test_the_headline_state_follows_the_LANE_figure(self):
        # A fully resident core running fully divergent warps is the case this exists to name; the
        # residency figure must not be able to call the result measured on its own.
        got = SO.occupancy_from_readings({SO.RESIDENT_WARPS: 8}, _geom())
        assert got["warp"]["state"] == "measured"
        assert got["state"] == "unknown"


class TestEndToEnd:
    def test_a_geometry_with_no_trace_reports_UNKNOWN_with_the_reason(self):
        got = SO.occupancy_for_target("anything", artifact_text=_HW)
        assert got["geometry_status"] == "derived"
        assert got["state"] == "unknown"
        assert got["lane"]["value"] is None

    def test_a_target_with_no_elaboration_has_no_denominator(self):
        got = SO.occupancy_for_target("no_such_target_at_all")
        assert got["state"] == "unknown"
        assert got["geometry_status"] == "unavailable"
        assert "no occupancy denominator" in got["why"]

    def test_readings_flow_through_to_a_figure(self):
        got = SO.occupancy_for_target(
            "anything", {SO.ACTIVE_LANES_PER_ISSUE: [16, 16, 8, 8], SO.RESIDENT_WARPS: 8},
            artifact_text=_HW)
        assert got["state"] == "measured"
        assert got["lane"]["value"] == pytest.approx(48 / 64)
        assert got["simt"]["value"] == pytest.approx(48 / 64)
