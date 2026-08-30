"""Joint occupancy: the three ways a per-cycle trace fabricates an overlap, and their fixes.

Each test here is a regression on a fabricated number that was actually measured and believed
before the rule existed. The fixtures are hand-built boolean vectors, so nothing depends on a
target, an engine or a signal name.
"""
from __future__ import annotations

from merlin.perf.occupancy import (
    align_offset,
    calibrate_state_idle,
    joint_counts,
    merge_engines,
    subsumed_columns,
)


def _cols(**kw):
    return {k: [c == "1" for c in v] for k, v in kw.items()}


class TestSubsumption:
    def test_component_signals_fold_into_their_aggregate(self):
        # An aggregate busy exactly when either half is: one unit, not three.
        hot = _cols(unit="111100", half_a="110000", half_b="001100")
        assert subsumed_columns(hot) == {"half_a": "unit", "half_b": "unit"}

    def test_a_merely_correlated_column_is_not_subsumed(self):
        # Overlapping most cycles is not containment; containment must hold on EVERY cycle.
        hot = _cols(a="111000", b="011100")
        assert subsumed_columns(hot) == {}

    def test_identical_columns_keep_exactly_one(self):
        hot = _cols(port="1100", view="1100")
        folded = subsumed_columns(hot, prefer=lambda a, b: a == "port")
        assert folded == {"view": "port"}

    def test_an_always_idle_column_is_not_subsumed(self):
        # It is unmeasured, not contained; folding it would hide that nothing was observed.
        hot = _cols(a="1100", dark="0000")
        assert "dark" not in subsumed_columns(hot)

    def test_fabricated_overlap_disappears(self):
        # The measured failure: a unit counted beside its own components reported overlap on every
        # cycle it was busy, in a program where nothing overlaps.
        hot = _cols(unit="111100", half_a="110000", half_b="001100")
        assert joint_counts(hot)["overlap_any"] == 0


class TestStateCalibration:
    def test_idle_value_is_derived_from_a_column_that_has_both_a_port_and_a_state(self):
        tr = {"xluBusy": list("001100"), "xlu_state": list("002100"), "vpu_state": list("110000")}
        out = calibrate_state_idle([tr], ["xlu_state", "vpu_state"], ["xluBusy"])
        assert out["idle_value"] == "0"
        assert out["paired_with"] == ["xluBusy"]
        assert out["paired_columns"] == ["xlu_state"]

    def test_a_nonzero_idle_encoding_is_derived_just_as_well(self):
        # Nothing here assumes zero means idle: the pairing decides, so a design that idles at 7
        # calibrates correctly and a hardcoded 0 would get it exactly backwards.
        tr = {"p": list("001100"), "s": list("770077")}
        assert calibrate_state_idle([tr], ["s"], ["p"])["idle_value"] == "7"

    def test_nothing_pairs_means_refusal_not_a_guess(self):
        tr = {"p": list("0011"), "s": list("0101")}
        out = calibrate_state_idle([tr], ["s"], ["p"])
        assert out["idle_value"] is None
        assert "no state column pairs" in out["detail"]

    def test_a_trace_that_leaves_the_state_constant_cannot_withdraw_the_calibration(self):
        # The measured failure: calibrating per trace dropped the busiest unit on exactly the
        # programs that never exercised the paired unit.
        rich = {"p": list("0110"), "s": list("0220"), "vpu": list("1100")}
        flat = {"p": list("0000"), "s": list("0000"), "vpu": list("1111")}
        out = calibrate_state_idle([rich, flat], ["s"], ["p"])
        assert out["idle_value"] == "0"
        assert out["checked_traces"] == 2

    def test_disagreeing_idle_values_refuse_rather_than_pick_one(self):
        a = {"p": list("0110"), "s": list("0220")}
        b = {"q": list("0110"), "t": list("9009")}       # idle 9 here, 0 there
        out = calibrate_state_idle([a, b], ["s", "t"], ["p", "q"])
        assert out["idle_value"] is None
        assert "disagree" in out["detail"]


class TestEngineMerge:
    def test_the_sampling_offset_is_derived_not_assumed(self):
        a = _cols(unit="011000")
        b = _cols(unit_view="110000")               # the same signal, sampled one cycle earlier
        shift, hits = align_offset(a, b)
        assert (shift, hits) == (-1, 1)

    def test_a_shared_unit_is_not_counted_twice(self):
        # Assuming a zero offset makes one unit look like two busy in adjacent cycles.
        a = _cols(unit="011000")
        b = _cols(unit_view="110000")
        merged, prov = merge_engines(a, b)
        assert prov["added"] == []
        assert joint_counts(merged)["overlap_any"] == 0

    def test_a_unit_only_the_second_instrument_can_see_is_admitted(self):
        a = _cols(unit="110000")
        b = _cols(unit_view="110000", hidden="001100")
        merged, prov = merge_engines(a, b)
        assert prov["added"] == ["hidden"]
        assert merged["hidden"] == [False, False, True, True, False, False]

    def test_an_aggregate_view_of_what_the_other_instrument_already_reports_is_folded(self):
        # The measured failure: a bus-valid signal beside the per-channel ports of the same bus
        # reported 6.8% overlap on a corpus where no two distinct units are ever busy together.
        a = _cols(ch0="110000", ch1="001100")
        b = _cols(bus_valid="010100")               # a strict subset of ch0 | ch1
        merged, prov = merge_engines(a, b)
        assert prov["folded"]["bus_valid"] == "<covered by the other instrument>"
        assert joint_counts(merged)["overlap_any"] == 0

    def test_a_hidden_unit_that_genuinely_overlaps_is_still_reported(self):
        # The folding rule must not be able to hide real overlap: `hidden` is not contained in
        # what the first instrument reports, and it is busy in a cycle `unit` is busy too.
        a = _cols(unit="110000")
        b = _cols(unit_view="110000", hidden="101000")
        merged, prov = merge_engines(a, b)
        assert prov["folded"]["unit_view"] == "unit"
        assert prov["added"] == ["hidden"]
        assert joint_counts(merged)["overlap_any"] == 1


class TestJointCounts:
    def test_idle_counts_only_cycles_with_nothing_busy(self):
        out = joint_counts(_cols(a="1100", b="0010"))
        assert out["idle_cycles"] == 1
        assert out["overlap_any"] == 0

    def test_across_kind_overlap_is_a_lower_bound_when_a_kind_is_undeclared(self):
        hot = _cols(a="1100", b="1100", c="1100")
        out = joint_counts(hot, kinds={"a": "compute", "b": "movement"})
        assert out["overlap_across_kinds_is_lower_bound"] is True
        assert out["undeclared_columns"] == ["c"]

    def test_overlap_across_kinds_needs_two_distinct_kinds(self):
        hot = _cols(a="1100", b="0110")          # distinct signals, both busy on cycle 1
        assert joint_counts(hot, kinds={"a": "compute", "b": "compute"})["overlap_across_kinds"] == 0
        assert joint_counts(hot, kinds={"a": "compute", "b": "movement"})["overlap_across_kinds"] == 1

    def test_busy_is_reported_for_every_column_including_the_subsumed_ones(self):
        # Folding decides what may claim to be a second unit; it never deletes a measurement.
        out = joint_counts(_cols(unit="1110", half="1100"))
        assert out["busy"] == {"unit": 3, "half": 2}
        assert out["joint_columns"] == ["unit"]


class TestDeclaredHierarchy:
    """A device whose engines NEST is the case derived containment gets wrong."""

    def test_a_contained_engine_is_not_folded_into_its_container(self):
        # An accelerator embedded in the cluster that drives it: the busy cycles nest exactly as a
        # sub-signal's would, but these are two engines and their concurrency is the measurement.
        hot = _cols(cluster="1111111100", embedded_pe="0011110000")
        units = {"cluster": "cluster", "embedded_pe": "embedded_pe"}
        assert subsumed_columns(hot, unit_of=units) == {}
        jc = joint_counts(hot, kinds={"cluster": "compute", "embedded_pe": "compute"},
                          unit_of=units)
        assert jc["joint_columns"] == ["cluster", "embedded_pe"]
        assert jc["overlap_any"] == 4          # the two microarchitectures running together

    def test_without_the_declaration_the_inner_engine_disappears(self):
        # The failure this exists to prevent, kept as a regression: undeclared, the inner engine is
        # folded away and the overlap between two engines reads zero.
        hot = _cols(cluster="1111111100", embedded_pe="0011110000")
        assert subsumed_columns(hot) == {"embedded_pe": "cluster"}
        assert joint_counts(hot)["overlap_any"] == 0

    def test_sub_signals_of_ONE_declared_unit_still_fold(self):
        # The declaration must not disable folding where folding is right: both halves belong to the
        # same declared unit, so they are one measurement, not three.
        hot = _cols(lsu="111100", load_half="110000", store_half="001100")
        units = {"lsu": "lsu", "load_half": "lsu", "store_half": "lsu"}
        assert subsumed_columns(hot, unit_of=units) == {"load_half": "lsu", "store_half": "lsu"}
        assert joint_counts(hot, unit_of=units)["overlap_any"] == 0

    def test_a_column_bound_to_no_unit_is_reported(self):
        hot = _cols(a="1100", b="0011")
        jc = joint_counts(hot, unit_of={"a": "a"})
        assert jc["unbound_columns"] == ["b"]


class TestDeclaredEngines:
    """The unit set comes from the target's own contract, not from this module."""

    CONTRACT = {"compute_units": [
        {"name": "cluster", "kind": "simt", "dtypes": ["float32"], "ops": ["matmul"],
         "contains": ["embedded_pe"]},
        {"name": "embedded_pe", "kind": "systolic", "dtypes": ["int8"], "ops": ["matmul"]},
    ]}

    def test_a_composed_device_yields_both_engines_and_the_containment(self):
        from merlin.perf.occupancy import declared_engines
        eng = declared_engines(self.CONTRACT)
        assert set(eng) == {"cluster", "embedded_pe"}
        assert eng["cluster"]["kind"] == "simt"
        assert eng["embedded_pe"]["kind"] == "systolic"
        assert eng["cluster"]["contains"] == ("embedded_pe",)

    def test_a_binding_to_an_undeclared_engine_raises(self):
        # The trace and the contract disagreeing about what the device is must be loud.
        from merlin.perf.occupancy import declared_engines, unit_bindings
        eng = declared_engines(self.CONTRACT)
        try:
            unit_bindings(["a"], {"a": "ghost_unit"}, eng)
        except ValueError as exc:
            assert "ghost_unit" in str(exc)
        else:
            raise AssertionError("expected a ValueError for an undeclared engine")

    def test_an_unbound_column_is_returned_not_folded(self):
        from merlin.perf.occupancy import declared_engines, unit_bindings
        eng = declared_engines(self.CONTRACT)
        unit_of, unbound = unit_bindings(["x", "y"], {"x": "cluster"}, eng)
        assert unit_of == {"x": "cluster"}
        assert unbound == ["y"]

    def test_end_to_end_the_contained_engine_survives(self):
        from merlin.perf.occupancy import declared_engines, unit_bindings
        eng = declared_engines(self.CONTRACT)
        hot = _cols(cluster_busy="1111111100", pe_busy="0011110000")
        unit_of, _ = unit_bindings(list(hot),
                                   {"cluster_busy": "cluster", "pe_busy": "embedded_pe"}, eng)
        jc = joint_counts(hot, unit_of=unit_of)
        assert set(jc["joint_columns"]) == {"cluster_busy", "pe_busy"}
        assert jc["overlap_any"] == 4


class TestGeneralisesAcrossRealTargets:
    """The anti-overfit proof: one code path, every shipped target's own declaration.

    The archetypes differ (systolic, simt, vector, spatial) and so do the topologies (flat, nested,
    sibling). If reading the engine set needed a branch per target, this module would be overfit to
    whichever one it was written against.
    """

    def _contract(self, name):
        import pytest
        import yaml
        from merlin.common.paths import merlin_dir
        p = merlin_dir() / "targets" / name / "contracts" / "target_contract.yaml"
        if not p.is_file():
            pytest.skip(f"{name} declares no target contract in this checkout")
        return yaml.safe_load(p.read_text())

    def test_every_declared_engine_has_a_kind(self):
        from merlin.perf.occupancy import declared_engines
        seen = set()
        for name in ("gemmini", "muon", "saturn", "toy_npu"):
            for engine, rec in declared_engines(self._contract(name)).items():
                assert rec["kind"], f"{name}.{engine} declares no kind"
                seen.add(rec["kind"])
        assert len(seen) >= 3, f"only saw kinds {sorted(seen)}; the corpus should span archetypes"

    def test_a_nested_heterogeneous_device_keeps_both_engines_separable(self):
        # The case derived containment gets wrong: an engine of one archetype inside another.
        from merlin.perf.occupancy import declared_engines, unit_bindings
        eng = declared_engines(self._contract("muon"))
        outer = [n for n, r in eng.items() if r["contains"]]
        assert outer, "muon should declare a composed engine"
        inner = eng[outer[0]]["contains"][0]
        assert eng[inner]["kind"] != eng[outer[0]]["kind"], "the two engines are different archetypes"
        hot = _cols(a="1111111100", b="0011110000")
        unit_of, _ = unit_bindings(list(hot), {"a": outer[0], "b": inner}, eng)
        assert joint_counts(hot, unit_of=unit_of)["overlap_any"] == 4

    def test_sibling_engines_are_separable_too(self):
        from merlin.perf.occupancy import declared_engines
        eng = declared_engines(self._contract("saturn"))
        assert len(eng) >= 2 and not any(r["contains"] for r in eng.values())
