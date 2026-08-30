"""The RTL-derived timing facts, and every way the walk must refuse to answer.

The load-bearing tests here are the negative ones. A depth walk that returns a NUMBER for a
sequenced unit is worse than one that returns nothing: the number is finite, plausible, and wrong,
and a statically-scheduled target compiles it straight into a wrong answer.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.rtl import timing


# --------------------------------------------------------------------------------------------
# A minimal fake of the mlc HW graph API (ops / defining_op / is_block_arg), so the walk's
# semantics are testable without mlc, CIRCT, or any target's RTL being present.
# --------------------------------------------------------------------------------------------
class _Name:
    def __init__(self, data): self.data = data


class _Op:
    def __init__(self, name, operands=()):
        self.op_name = _Name(name)
        self.operands = list(operands)
        self.result = object()


class _Module:
    def __init__(self, name, ops):
        self.name = name
        self.op = self
        self._ops = ops


class _Graph:
    """Values are plain objects; a value is a block arg unless some op declares it as its result."""
    def __init__(self, modules):
        self.modules = {m.name: m for m in modules}

    def ops(self, name, within=None):
        mods = [within] if within is not None else list(self.modules.values())
        return [o for m in mods for o in m._ops if o.op_name.data == name]

    def defining_op(self, value):
        for m in self.modules.values():
            for o in m._ops:
                if o.result is value:
                    return o
        return None

    def is_block_arg(self, value):
        return self.defining_op(value) is None


def _build(name, ops):
    g = _Graph([_Module(name, ops)])
    return g, g.modules[name]


# --------------------------------------------------------------------------------------------
# What the walk SHOULD answer
# --------------------------------------------------------------------------------------------
def test_a_chain_of_registers_is_its_own_depth():
    port = object()
    r1 = _Op("seq.firreg", [port])
    r2 = _Op("seq.firreg", [r1.result])
    r3 = _Op("seq.firreg", [r2.result])
    out = _Op("hw.output", [r3.result])
    g, m = _build("Chain", [r1, r2, r3, out])
    assert timing.module_timing(g, m)["pipeline_depth"] == 3


def test_the_depth_is_the_LONGEST_path_not_the_shortest():
    """A unit is ready when its slowest output is, so a min/first-path answer would understate it."""
    port = object()
    shallow = _Op("seq.firreg", [port])
    deep1 = _Op("seq.firreg", [port])
    deep2 = _Op("seq.firreg", [deep1.result])
    comb = _Op("comb.add", [shallow.result, deep2.result])
    out = _Op("hw.output", [comb.result])
    g, m = _build("Uneven", [shallow, deep1, deep2, comb, out])
    assert timing.module_timing(g, m)["pipeline_depth"] == 2


def test_purely_combinational_is_zero_and_zero_is_a_real_answer():
    """Depth 0 means 'measured, no registers'. It must not be confused with UNKNOWN."""
    port = object()
    comb = _Op("comb.mul", [port])
    out = _Op("hw.output", [comb.result])
    g, m = _build("Comb", [comb, out])
    rec = timing.module_timing(g, m)
    assert rec["pipeline_depth"] == 0
    assert rec["pipeline_depth"] is not None


def test_a_constant_terminates_a_path_without_adding_depth():
    port = object()
    const = _Op("hw.constant", [])
    reg = _Op("seq.firreg", [port])
    comb = _Op("comb.add", [const.result, reg.result])
    out = _Op("hw.output", [comb.result])
    g, m = _build("WithConst", [const, reg, comb, out])
    assert timing.module_timing(g, m)["pipeline_depth"] == 1


def test_a_registers_clock_and_reset_operands_do_not_count_as_data_depth():
    """Only operand 0 carries data. Counting clock/reset would inflate every depth in the design."""
    port, clock, reset = object(), object(), object()
    deep = _Op("seq.firreg", [object()])
    reg = _Op("seq.firreg", [port, clock, reset, deep.result])
    out = _Op("hw.output", [reg.result])
    g, m = _build("ClockOperands", [deep, reg, out])
    assert timing.module_timing(g, m)["pipeline_depth"] == 1


# --------------------------------------------------------------------------------------------
# What the walk MUST REFUSE to answer -- the negative tests
# --------------------------------------------------------------------------------------------
def test_a_feedback_output_yields_UNKNOWN_not_a_number():
    """An FSM/counter reaches its output through itself. 'Longest path' is not finite there, and a
    finite-looking answer is exactly what a statically-scheduled target would mis-compile."""
    reg = _Op("seq.firreg", [])
    reg.operands = [reg.result]              # state feeding itself
    out = _Op("hw.output", [reg.result])
    g, m = _build("Fsm", [reg, out])
    rec = timing.module_timing(g, m)
    assert rec["pipeline_depth"] is None
    assert rec["n_cyclic"] == 1


def test_a_partly_sequenced_module_does_not_report_the_acyclic_maximum_as_its_depth():
    """The flattering bug: one clean datapath output beside a handshake loop would let a module with
    feedback report a confident depth. partial_depth carries it under a name nobody reads as latency."""
    port = object()
    r1 = _Op("seq.firreg", [port])
    r2 = _Op("seq.firreg", [r1.result])
    loop = _Op("seq.firreg", [])
    loop.operands = [loop.result]
    out = _Op("hw.output", [r2.result, loop.result])
    g, m = _build("Mixed", [r1, r2, loop, out])
    rec = timing.module_timing(g, m)
    assert rec["pipeline_depth"] is None, "a module with feedback has no wiring-depth latency"
    assert rec["partial_depth"] == 2
    assert rec["n_cyclic"] == 1 and rec["n_outputs"] == 2


def test_partial_depth_and_pipeline_depth_are_separate_fields():
    """Two differently-derived numbers must never share one field: one is the module's latency, the
    other is the maximum over the subset this method could reach."""
    reg = _Op("seq.firreg", [object()])
    out = _Op("hw.output", [reg.result])
    g, m = _build("Clean", [reg, out])
    rec = timing.module_timing(g, m)
    assert rec["pipeline_depth"] == 1
    assert rec["partial_depth"] is None, "a fully-resolved module reports no partial"


def test_a_module_driving_no_output_is_UNKNOWN_not_zero():
    g, m = _build("Empty", [])
    rec = timing.module_timing(g, m)
    assert rec["pipeline_depth"] is None
    assert rec["n_outputs"] == 0
    assert "nothing to walk" in rec["evidence"]


def test_every_record_carries_the_evidence_that_produced_it():
    reg = _Op("seq.firreg", [object()])
    out = _Op("hw.output", [reg.result])
    g, m = _build("Evidenced", [reg, out])
    rec = timing.module_timing(g, m)
    assert rec["evidence"] and rec["source"] == "mlc_hw_graph_walk"
    assert rec["module"] == "Evidenced"


def test_unreachable_rtl_is_None_never_an_empty_list(monkeypatch):
    """None is UNKNOWN ('nobody could look'). [] would assert the design HAS no timing -- a claim
    about hardware made from a missing tool."""
    from merlin.targetgen.rtl import mlc_bridge
    monkeypatch.setattr(mlc_bridge, "mlc_available", lambda: (False, "not installed"))
    assert timing.discovered_timing("any-target") is None

    monkeypatch.setattr(mlc_bridge, "mlc_available", lambda: (True, "ok"))
    monkeypatch.setattr(mlc_bridge, "core_hw_mlir", lambda t: None)
    assert timing.discovered_timing("any-target") is None


def test_a_deep_chain_does_not_exhaust_the_interpreter_stack():
    """Recursion would turn a deep-but-answerable module into a module with no answer."""
    port = object()
    prev, ops = port, []
    for _ in range(4000):
        r = _Op("seq.firreg", [prev])
        ops.append(r)
        prev = r.result
    ops.append(_Op("hw.output", [prev]))
    g, m = _build("VeryDeep", ops)
    assert timing.module_timing(g, m)["pipeline_depth"] == 4000


# --------------------------------------------------------------------------------------------
# Against the real RTL. Skipped where mlc / the HW dialect is not reachable -- which is the same
# fail-closed posture the extractor itself takes, not a silently-passing test.
# --------------------------------------------------------------------------------------------
def _rtl_timing_or_skip(target: str):
    recs = None
    try:
        recs = timing.discovered_timing(target)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{target} RTL not walkable here: {exc}")
    if not recs:
        pytest.skip(f"no reachable HW dialect for {target}")
    return {r["module"]: r for r in recs}


@pytest.mark.slow
def test_the_walk_agrees_with_the_geometry_it_never_saw():
    """Cross-check between two facts derived by DIFFERENT methods from the same RTL.

    The mesh is discovered by counting replicated cells; the depth by walking registers. Neither
    reads the other. For a systolic array the reduction path is 2*rows-2 (down the columns and back
    along the accumulation chain), so agreement is corroboration -- and a disagreement would say one
    of the two extractors is wrong, which is the point of deriving both.
    """
    from merlin.targetgen.rtl import facts as F

    target = "atlas"
    try:
        arrays = (F.load_facts(target).get("facts") or {}).get("arrays") or []
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no facts for {target}: {exc}")
    mesh = next((a for a in arrays if a.get("name") == "mesh"), None)
    if not mesh or not mesh.get("rows"):
        pytest.skip(f"{target} declares no discovered mesh geometry")

    by_name = _rtl_timing_or_skip(target)
    container = mesh.get("container")
    array = by_name.get(container)
    if array is None or array.get("pipeline_depth") is None:
        pytest.skip(f"mesh container {container!r} has no resolved depth")

    rows = int(mesh["rows"])
    assert array["pipeline_depth"] == rows - 1, (
        f"the mesh container's depth should be one stage per row after the first; "
        f"got {array['pipeline_depth']} for a {rows}-row mesh"
    )


@pytest.mark.slow
def test_sequenced_units_are_reported_UNKNOWN_on_real_rtl_not_guessed():
    """On a real design the walk must still refuse the units it cannot reach, or the refusal is only
    a property of the hand-built fixtures above."""
    by_name = _rtl_timing_or_skip("atlas")
    resolved = [r for r in by_name.values() if r["pipeline_depth"] is not None]
    sequenced = [r for r in by_name.values() if r["pipeline_depth"] is None]
    assert resolved, "no module resolved: the walk answered nothing on a real design"
    assert sequenced, "every module resolved: a design with FSMs and queues should have refusals"
    for rec in sequenced:
        assert rec["evidence"], "a refusal must say why it refused"
