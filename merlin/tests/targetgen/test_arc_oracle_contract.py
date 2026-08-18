"""What the arc oracle must be HANDED before it can grade: the operand values, the resident aliases,
and the target.

All three come from one artifact. The first agent runs on a command-buffer target whose RTL tier is the
arc model failed 27 of 28 capsules per arm with `L3 crash: 'data'` — a `KeyError` inside the backend,
classified `tool_crash`, while preflight reported "mlc arc oracle available". The module was indeed
importable; what was missing was every part of its input contract:

  * command-buffer leaves declare shape/dtype/role and NO values (the reference materializes them), but
    the backend indexes ``tensors[name]["data"]``;
  * a resident handle is produced by RES_PACK and is not in the tensor table, yet the backend resolves a
    matmul's operand names straight against that table;
  * the backend routes on a ``target`` argument with a default, so an unpassed target answered one
    accelerator's buffer on another's model.

These are structural tests over the translation, so they run with no mlc checkout present.
"""
from __future__ import annotations

import pytest

from merlin.runtime.reference import reference_outputs
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.rtl import mlc_bridge as B


def _cb():
    return {
        "abi_version": "0.1",
        "target": "some_target",
        "tensors": {
            "W": {"shape": [4, 2], "dtype": "i8", "role": "weight"},
            "A0": {"shape": [1, 4], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [1, 2], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
             "attributes": {"epilogue": [], "output_dtype": "i32"}},
        ],
    }


# --------------------------------------------------------------------------- operand values
def test_every_leaf_operand_carries_its_values():
    out = CR._cb_with_leaf_values(_cb())
    for name, n in (("W", 8), ("A0", 4)):
        assert len(out["tensors"][name]["data"]) == n


def test_the_values_are_the_ones_the_numeric_floor_used():
    """If the RTL tier invented its own stimulus it would disagree with L0/L1 and report a mismatch
    that is really a stimulus difference."""
    from merlin.runtime.commandbuffer import materialize_inputs

    out = CR._cb_with_leaf_values(_cb())
    leaves = materialize_inputs(_cb())
    assert out["tensors"]["A0"]["data"] == list(leaves["A0"].data)
    assert out["tensors"]["W"]["data"] == list(leaves["W"].data)


def test_the_original_buffer_is_not_mutated():
    cb = _cb()
    CR._cb_with_leaf_values(cb)
    assert "data" not in cb["tensors"]["W"], "the graded artifact must stay as the agent emitted it"
    assert "W_res" not in cb["tensors"]


def test_declared_values_are_left_alone():
    cb = _cb()
    cb["tensors"]["A0"]["data"] = [7, 7, 7, 7]
    assert CR._cb_with_leaf_values(cb)["tensors"]["A0"]["data"] == [7, 7, 7, 7]


# --------------------------------------------------------------------------- resident aliases
def test_a_resident_handle_becomes_an_alias_of_its_source():
    out = CR._cb_with_leaf_values(_cb())
    assert out["tensors"]["W_res"]["data"] == out["tensors"]["W"]["data"]
    assert out["tensors"]["W_res"]["shape"] == [4, 2]
    assert out["tensors"]["W_res"]["resident_of"] == "W"


def test_the_alias_matches_how_the_reference_reads_res_pack():
    """The reference treats RES_PACK as a layout alias and computes A0 @ W; the translation must not
    change what the buffer means."""
    ref = reference_outputs(_cb())
    assert list(ref) == ["Y0"] and len(ref["Y0"]) == 1


def test_a_handle_from_some_other_producer_is_not_invented():
    """Fail loudly instead of grading against a value this translation made up."""
    cb = _cb()
    cb["commands"][0] = {"opcode": "SOMETHING_ELSE", "operands": {"src": "W", "dst": "W_res"}}
    assert "W_res" not in CR._cb_with_leaf_values(cb)["tensors"]


def test_an_output_tensor_gets_no_values():
    out = CR._cb_with_leaf_values(_cb())
    assert "data" not in out["tensors"]["Y0"], "an output is produced, not supplied"


# --------------------------------------------------------------------------- the target is passed
def test_the_arc_run_refuses_to_fall_back_to_a_default_model(monkeypatch):
    monkeypatch.setattr(B, "require_mlc", lambda: None)
    with pytest.raises(ValueError, match="needs a target"):
        B.arc_run_command_buffer({"commands": []})


def test_the_target_reaches_the_backend(monkeypatch):
    seen = {}

    class _Ctx:
        def __enter__(self): return None
        def __exit__(self, *a): return False

    def _fake_module():
        import types
        m = types.ModuleType("mlc.runtime.backend")

        def run_command_buffer(cb, *, target="default_model", base=None, **_kw):
            seen["target"] = target
            return {"outputs": {}, "metrics": {}}
        m.run_command_buffer = run_command_buffer
        return m

    import sys
    import types
    for name in ("mlc", "mlc.runtime"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mlc.runtime.backend", _fake_module())
    monkeypatch.setattr(B, "require_mlc", lambda: None)
    monkeypatch.setattr(B, "_mlc_cwd", lambda: _Ctx())
    monkeypatch.setattr(B, "_arc_target", lambda t: t)

    B.arc_run_command_buffer({"commands": []}, "some_target")
    assert seen["target"] == "some_target"


def test_the_buffers_own_target_is_the_fallback(monkeypatch):
    seen = {}

    class _Ctx:
        def __enter__(self): return None
        def __exit__(self, *a): return False

    import sys
    import types
    m = types.ModuleType("mlc.runtime.backend")

    def run_command_buffer(cb, *, target="default_model", base=None, **_kw):
        seen["target"] = target
        return {"outputs": {}, "metrics": {}}
    m.run_command_buffer = run_command_buffer
    for name in ("mlc", "mlc.runtime"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mlc.runtime.backend", m)
    monkeypatch.setattr(B, "require_mlc", lambda: None)
    monkeypatch.setattr(B, "_mlc_cwd", lambda: _Ctx())
    monkeypatch.setattr(B, "_arc_target", lambda t: t)

    B.arc_run_command_buffer(_cb())
    assert seen["target"] == "some_target"


# --------------------------------------------------------------------------- an unmodeled epilogue
def _epi_cb(stage="acc_scale"):
    cb = _cb()
    for cmd in cb["commands"]:
        if cmd["opcode"] == "COMMIT":
            cmd["attributes"] = {"epilogue": [stage], "acc_scale": 0.5, "output_dtype": "i32"}
    return cb


class _Bridge:
    """A stand-in arc backend whose epilogue support is a switch."""

    def __init__(self, applies: bool):
        self.applies = applies
        self.runs = 0

    def arc_run_command_buffer(self, cb, target=None):
        self.runs += 1
        has_epi = any((c.get("attributes") or {}).get("epilogue")
                      for c in cb["commands"] if c["opcode"] == "COMMIT")
        val = 5 if (self.applies and has_epi) else 10
        return {"outputs": {"Y0": [[val, val]]}}


def test_an_ignored_epilogue_is_detected():
    b = _Bridge(applies=False)
    res = b.arc_run_command_buffer(_epi_cb(), "t")
    assert CR._epilogue_stages_ignored(_epi_cb(), res, "t", b) == {"acc_scale"}


def test_an_applied_epilogue_is_not_flagged():
    b = _Bridge(applies=True)
    res = b.arc_run_command_buffer(_epi_cb(), "t")
    assert CR._epilogue_stages_ignored(_epi_cb(), res, "t", b) == set()


def test_a_buffer_without_an_epilogue_pays_no_second_run():
    b = _Bridge(applies=False)
    assert CR._epilogue_stages_ignored(_cb(), {"outputs": {}}, "t", b) == set()
    assert b.runs == 0


def test_a_failing_probe_never_invents_a_gap():
    class _Broken(_Bridge):
        def arc_run_command_buffer(self, cb, target=None):
            raise RuntimeError("probe blew up")

    assert CR._epilogue_stages_ignored(_epi_cb(), {"outputs": {"Y0": [[1]]}}, "t",
                                       _Broken(applies=False)) == set()


def test_an_unmodeled_epilogue_reports_unavailable_not_fail(monkeypatch):
    """A tier that FAILED here would teach the agent its correct epilogue is wrong."""
    b = _Bridge(applies=False)
    import types

    from merlin.targetgen import rtl as _rtl
    fake = types.ModuleType("mlc_bridge")
    fake.arc_available = lambda t: True
    fake.arc_run_command_buffer = b.arc_run_command_buffer
    monkeypatch.setattr(_rtl, "mlc_bridge", fake)
    run = CR.mlc_arc_adapter("t")
    with pytest.raises(CR.OracleUnavailable, match="does not model the commit epilogue"):
        run(_epi_cb(), "", None, 5)


# --------------------------------------------------------------------------- the preflight answers
def test_the_preflight_probe_builds_a_buffer_from_the_derived_tile(monkeypatch):
    """No shape literal: the probe's geometry is the target's own tile edge."""
    seen = {}

    def _fake_adapter(target):
        def run(cb, llvm_text, workdir, timeout):
            seen["shape"] = cb["tensors"]["probe_a"]["shape"]
            from merlin.runtime.reference import reference_outputs
            return {"outputs": reference_outputs(cb)}
        return run

    monkeypatch.setattr(CR, "mlc_arc_adapter", _fake_adapter)
    monkeypatch.setattr("merlin.targetgen.corpus_spec._tile_dim", lambda t, c: 8)
    ok, why = CR._arc_answers_a_buffer("some_target")
    assert ok and "8x8" in why
    assert seen["shape"] == [8, 8]


def test_a_model_that_disagrees_with_the_reference_is_not_available(monkeypatch):
    def _fake_adapter(target):
        def run(cb, llvm_text, workdir, timeout):
            return {"outputs": {"probe_y": [[0]]}}
        return run

    monkeypatch.setattr(CR, "mlc_arc_adapter", _fake_adapter)
    monkeypatch.setattr("merlin.targetgen.corpus_spec._tile_dim", lambda t, c: 4)
    ok, why = CR._arc_answers_a_buffer("some_target")
    assert not ok and "disagreed with the reference" in why


def test_a_raising_model_is_not_available(monkeypatch):
    """The exact shape of the defect this probe exists for: importable, unusable."""
    def _fake_adapter(target):
        def run(cb, llvm_text, workdir, timeout):
            raise KeyError("data")
        return run

    monkeypatch.setattr(CR, "mlc_arc_adapter", _fake_adapter)
    monkeypatch.setattr("merlin.targetgen.corpus_spec._tile_dim", lambda t, c: 4)
    ok, why = CR._arc_answers_a_buffer("some_target")
    assert not ok and "KeyError" in why


def test_an_empty_answer_is_not_available(monkeypatch):
    def _fake_adapter(target):
        def run(cb, llvm_text, workdir, timeout):
            return {"outputs": {}}
        return run

    monkeypatch.setattr(CR, "mlc_arc_adapter", _fake_adapter)
    monkeypatch.setattr("merlin.targetgen.corpus_spec._tile_dim", lambda t, c: 4)
    ok, why = CR._arc_answers_a_buffer("some_target")
    assert not ok and "no outputs" in why
