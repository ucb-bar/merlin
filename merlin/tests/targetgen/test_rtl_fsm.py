"""The FSM inventory: derived from what synthesis DETECTED, not from what it chose to export."""
from __future__ import annotations

from merlin.targetgen.rtl.fsm import (
    FsmRegister,
    detected_registers,
    fsm_inventory,
    reset_values,
)

_LOG = r"""
Creating register for signal `\ExecuteController.\control_state' using process ...
Extracting FSM `\control_state' from module `\ExecuteController'.
Extracting FSM `\control_state' from module `\LoadController'.
Extracting FSM `\state' from module `\LoopMatmulStC'.
Warning: Regarding the user-specified fsm_encoding attribute on ExecuteController.control_state:
    Users of state reg look like FSM recoding might result in larger circuit.
Exporting FSM `$fsm$\control_state$1' from module `\LoadController' to file `x.kiss2'.
"""

_TABLE = """.i 4
.o 5
.p 10
.s 3
.r s0
--00 s0 s0 00100
--10 s0 s1 00101
"""


class TestDetection:
    def test_every_detected_machine_is_reported(self):
        assert detected_registers(_LOG) == [
            ("ExecuteController", "control_state"),
            ("LoadController", "control_state"),
            ("LoopMatmulStC", "state"),
        ]

    def test_an_export_line_is_not_a_detection(self):
        # Export and detection are different statements; only detection is the inventory.
        assert ("LoadController", "$fsm$\\control_state$1") not in detected_registers(_LOG)

    def test_a_log_with_no_detections_yields_nothing(self):
        assert detected_registers("nothing to see\nWarning: unrelated\n") == []


class TestInventory:
    def _dir(self, tmp_path, *, log=_LOG, tables=None):
        (tmp_path / "yosys.log").write_text(log)
        for name, text in (tables or {}).items():
            (tmp_path / name).write_text(text)
        return tmp_path

    def test_the_inventory_is_the_detected_set_not_the_exported_one(self, tmp_path):
        # The measured failure: a synthesis flow exported 3 of 15 and dropped the two controllers
        # whose concurrency was the entire point. Exporting is an optimisation decision.
        d = self._dir(tmp_path, tables={"LoadController.control_state.kiss2": _TABLE})
        inv = fsm_inventory("t", d)
        assert len(inv) == 3
        assert [f.qualified for f in inv] == ["ExecuteController.control_state",
                                              "LoadController.control_state",
                                              "LoopMatmulStC.state"]

    def test_an_exported_table_enriches_its_entry_and_the_rest_stay_unknown(self, tmp_path):
        d = self._dir(tmp_path, tables={"LoadController.control_state.kiss2": _TABLE})
        by = {f.qualified: f for f in fsm_inventory("t", d)}
        assert by["LoadController.control_state"].states == 3
        assert by["LoadController.control_state"].reset_state == "s0"
        assert by["LoadController.control_state"].exported is True
        # UNKNOWN, never zero: the machine exists whether or not a table was written for it.
        assert by["ExecuteController.control_state"].states is None
        assert by["ExecuteController.control_state"].exported is False

    def test_the_mangled_export_filename_joins_its_detected_entry(self, tmp_path):
        # Synthesis writes `<Module>-$fsm$\<register>$<id>.kiss2`; a curated export renames it. Both
        # spellings must join, or an exported table silently fails to enrich anything.
        d = self._dir(tmp_path, tables={"LoadController-$fsm$\\control_state$76885.kiss2": _TABLE})
        by = {f.qualified: f for f in fsm_inventory("t", d)}
        assert by["LoadController.control_state"].states == 3

    def test_no_extraction_is_unknown_engines_not_zero_engines(self, tmp_path):
        assert fsm_inventory("t", tmp_path / "absent") == []


class TestSignalMatching:
    def test_a_register_matches_an_instance_path_by_its_leaf(self):
        # Synthesis names the module CLASS; a state manifest names the INSTANCE path.
        f = FsmRegister(module="VectorFSM", register="state")
        assert f.matches_signal("vpu/core/fsm/state")
        assert f.matches_signal("vpu.core.fsm.state")

    def test_a_different_leaf_does_not_match(self):
        f = FsmRegister(module="LoadController", register="control_state")
        assert not f.matches_signal("load_controller/_control_state_T_1")

    def test_a_bare_leaf_with_no_instance_is_not_a_match(self):
        assert not FsmRegister(module="M", register="state").matches_signal("state")


class TestResetValues:
    def test_the_idle_encoding_is_observed_rather_than_mapped(self):
        # A table names states symbolically (s0) while a trace carries a numeric encoding. Rather
        # than invent a mapping, read what the register holds after reset.
        seen = {"a/state": 0, "b/state": 7}
        assert reset_values(lambda s: seen[s], seen) == {"a/state": "0", "b/state": "7"}

    def test_a_nonzero_reset_encoding_is_reported_as_it_is(self):
        # Nothing here assumes zero means idle; a design idling at 7 reports 7.
        assert reset_values(lambda s: 7, ["x/state"]) == {"x/state": "7"}
