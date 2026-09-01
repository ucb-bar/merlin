"""Lane and warp occupancy on a SIMT cluster, derived from the machine's own declaration.

A busy bit cannot see divergence: a warp running one lane of sixteen is 100% busy for the whole issue.
These tests hold the derivation (the dimensions are factored out of the config's key names, never
typed), the three states, and — the point of the module — that a missing trace refuses instead of
returning a flattering 1.0.
"""
from __future__ import annotations

import textwrap

import pytest

from merlin.perf import simt_occupancy as SO

# A synthetic declaration in the shape a real one has, with key spellings chosen NOT to be a copy of
# any shipped config's, so a passing test cannot be passing because a spelling was hardcoded.
_DECL = textwrap.dedent("""\
    [engine]
    num_lanes = 16
    num_warps = 8
    num_cores = 2
    num_regs = 256
    start_pc = 0x10000000

    [mem]
    io_addr = 0xFF080000
    io_size = 64

    [sim]
    timeout = 1000000
    trace = false
    """)


class TestGeometryDerivation:
    def test_the_dimensions_are_factored_out_of_the_key_names(self):
        g = SO.derive_simt_geometry(_DECL)
        assert (g.lane_width, g.warps_per_core, g.cores) == (16, 8, 2)
        assert g.table == "engine"
        assert g.keys == {"lane": "num_lanes", "warp": "num_warps", "core": "num_cores"}

    def test_a_differently_spelled_config_resolves_the_same_way(self):
        # The whole point of factoring the role out of the key: a second SIMT machine that writes
        # `lane_width` / `warps` / `core_count` must not report UNKNOWN.
        g = SO.derive_simt_geometry("[cluster]\nlane_width = 32\nwarps = 4\ncore_count = 6\n")
        assert (g.lane_width, g.warps_per_core, g.cores) == (32, 4, 6)

    def test_a_geometry_at_the_document_root_is_found(self):
        g = SO.derive_simt_geometry("lanes = 8\nwarps = 2\n")
        assert (g.lane_width, g.warps_per_core, g.table) == (8, 2, "")

    def test_a_nested_table_is_walked(self):
        g = SO.derive_simt_geometry("[a]\nx = 1\n[a.b]\nnum_lanes = 4\nnum_warps = 3\n")
        assert (g.lane_width, g.table) == (4, "a.b")

    def test_the_derived_denominators_multiply_out(self):
        g = SO.derive_simt_geometry(_DECL)
        assert g.threads_per_core == 128            # 16 lanes x 8 warps
        assert g.lane_slots_per_cycle == 256        # x 2 cores

    def test_a_ratio_key_is_not_read_as_a_count(self):
        # `lanes_per_warp` names two dimensions at once: it is a ratio, and reading it as either one
        # is off by exactly the other. It must leave the geometry undeclared, not fill it in.
        g = SO.derive_simt_geometry("[c]\nlanes_per_warp = 16\n")
        assert g.lane_width is None and g.warps_per_core is None

    def test_a_flag_is_not_a_count(self):
        # bool is an int subclass: admitting one would make `warp_sync = true` a warp count of 1,
        # which reads as a fully occupied core.
        g = SO.derive_simt_geometry("[c]\nnum_lanes = 8\nwarp_sync = true\n")
        assert g.lane_width == 8 and g.warps_per_core is None

    def test_a_contradicting_declaration_resolves_to_nothing(self):
        g = SO.derive_simt_geometry("[c]\nnum_lanes = 16\nlane_count = 32\nnum_warps = 4\n")
        assert g.lane_width is None
        assert "lane" in g.ambiguous
        assert g.warps_per_core == 4                # the unambiguous dimension still resolves

    def test_two_tables_declaring_the_same_number_of_dimensions_refuse(self):
        # A tie means the file describes two machines and nothing here can say which one the cycles
        # belong to. Picking the first would publish a denominator nobody chose.
        two = "[a]\nnum_lanes = 16\nnum_warps = 8\n[b]\nnum_lanes = 32\nnum_warps = 4\n"
        assert SO.derive_simt_geometry(two).lane_width is None

    def test_a_declaration_with_no_simt_dimensions_is_empty(self):
        assert SO.derive_simt_geometry("[build]\njobs = 8\ntimeout = 5\n").resolved() == ()

    def test_unparseable_toml_is_not_a_crash(self):
        assert SO.derive_simt_geometry("[[[ nope").resolved() == ()

    def test_nothing_is_defaulted_when_a_dimension_is_undeclared(self):
        # The hard rule: a lane count nobody wrote down is UNKNOWN. A plausible 16 substituted here
        # would price a run against a machine that was never run.
        g = SO.derive_simt_geometry("[c]\nnum_warps = 8\nnum_cores = 2\n")
        assert g.lane_width is None
        assert g.threads_per_core is None and g.lane_slots_per_cycle is None


class TestThreeStates:
    def test_supplied_text_derives(self, tmp_path):
        got = SO.geometry_for_target("anything", config_text=_DECL)
        assert got["status"] == "derived"
        assert got["geometry"]["lane_width"] == 16

    def test_a_file_that_declares_no_geometry_is_ABSENT(self, tmp_path):
        p = tmp_path / "decl.toml"
        p.write_text("[build]\njobs = 8\n", encoding="utf-8")
        got = SO.geometry_for_target("anything", config_path=p)
        assert got["status"] == "absent"
        assert got["read"]                            # absent requires having read something

    def test_a_file_that_cannot_be_OPENED_is_UNAVAILABLE_not_absent(self, tmp_path):
        # The collapse this guards: a package emits a placeholder path when its toolchain env is
        # unset, and "we could not look" must never be reported as "the machine declares no lanes".
        got = SO.geometry_for_target("anything", config_path=tmp_path / "does_not_exist.toml")
        assert got["status"] == "unavailable"
        assert got["unreadable"]

    def test_an_ambiguous_declaration_is_UNAVAILABLE_not_absent(self, tmp_path):
        p = tmp_path / "decl.toml"
        p.write_text("[c]\nnum_lanes = 16\nlane_count = 32\n", encoding="utf-8")
        got = SO.geometry_for_target("anything", config_path=p)
        assert got["status"] == "unavailable"
        assert "lane" in got["geometry"]["ambiguous"]

    def test_a_target_with_no_declaration_anywhere_is_UNAVAILABLE(self):
        got = SO.geometry_for_target("no_such_target_at_all")
        assert got["status"] == "unavailable"
        assert got["routes_tried"]                    # each route says why it did not answer

    def test_the_derived_geometry_names_the_file_it_came_from(self, tmp_path):
        p = tmp_path / "decl.toml"
        p.write_text(_DECL, encoding="utf-8")
        got = SO.geometry_for_target("anything", config_path=p)
        assert got["source"] == str(p)
        assert got["geometry"]["source"] == str(p)


def _geom(text=_DECL):
    return SO.derive_simt_geometry(text)


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
            _geom("[c]\nnum_warps = 8\n"))
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

    def test_no_declared_warps_per_core_means_no_denominator(self):
        got = SO.occupancy_from_readings({SO.RESIDENT_WARPS: 2}, _geom("[c]\nnum_lanes = 16\n"))
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
        got = SO.occupancy_for_target("anything", config_text=_DECL)
        assert got["geometry_status"] == "derived"
        assert got["state"] == "unknown"
        assert got["lane"]["value"] is None

    def test_a_target_with_no_declaration_has_no_denominator(self):
        got = SO.occupancy_for_target("no_such_target_at_all")
        assert got["state"] == "unknown"
        assert got["geometry_status"] == "unavailable"
        assert "no occupancy denominator" in got["why"]

    def test_readings_flow_through_to_a_figure(self):
        got = SO.occupancy_for_target(
            "anything", {SO.ACTIVE_LANES_PER_ISSUE: [16, 16, 8, 8], SO.RESIDENT_WARPS: 8},
            config_text=_DECL)
        assert got["state"] == "measured"
        assert got["lane"]["value"] == pytest.approx(48 / 64)
        assert got["simt"]["value"] == pytest.approx(48 / 64)
