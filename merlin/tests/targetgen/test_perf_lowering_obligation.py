"""A performance member must EMIT the transformation its cycles are attributed to.

THE DEFECT THIS PINS. The performance task contract tells the agent that "a lever that does not change
the emitted code is inert by definition". The command-buffer ABI, meanwhile, lets a buffer DECLARE that
one of its tensors is derived from another (``params.im2col_recipes``), and the harness then performs
that convolution-window gather in Python before the program runs. The frozen submission lowers every
conv exactly that way: it declares the recipe and emits ``RES_PACK / MATMUL_RESIDENT / COMMIT``. So the
four conv performance members were measuring a plain matmul over a matrix somebody else built, the
windowing appeared nowhere in the emitted program, and by the contract's own rule im2col lowering was
inert BY DEFINITION and could never be a lever -- the agent would have been right to report it a null.

The mechanism is not deleted, because it has a real job: a derived activation must be materialized
IDENTICALLY for the reference, the simulator and the device, or the numeric comparison between them
compares three different stimuli. What is refused is using it to discharge the COMPILER's obligation.

SCOPE, STATED. This is a PERFORMANCE-member rule and nothing else. A functional conv capsule asks
whether the arithmetic is right, not how many cycles it took; a recipe there is answering a different
question and is deliberately left alone. That is asserted below, not merely written here.
"""
from __future__ import annotations

import copy

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.perf import lowering_obligation as LOB
from merlin.runtime.commandbuffer import (DERIVATION_RECIPE_KEYS, IM2COL_RECIPE_KEY, conv_im2col,
                                          harness_derived_tensors, materialize_inputs, operand_flow)
from merlin.runtime.tensor import Tensor

CAPSULES = merlin_dir() / "contract/capsules"
SHARED_PERF_TEMPLATE = CAPSULES / "profiles/_perf.yaml"

# The conv performance family, named by the FAMILY id its members declare, not by a target.
_CONV_PERF_FAMILY = "PV"


# ---------------------------------------------------------------------------------------------
# the three lowerings, as command buffers
# ---------------------------------------------------------------------------------------------
_IFM_SHAPE = [1, 8, 8, 16]
_KH = _KW = 3
_CI = 16
_CO = 16
_M = 6 * 6                                # (8 - 3 + 1)^2 output positions
_K = _KH * _KW * _CI


def _conv_capsule(**perf_overrides) -> dict:
    """A minimal PERFORMANCE member shaped like the shipped conv family (no target facts read)."""
    performance = {"family": _CONV_PERF_FAMILY, "lever": "window_reuse_amplification",
                   "claim": "PREDICTS", "member_class": "LAW"}
    performance.update(perf_overrides)
    return {"name": "PVxx_synthetic", "kind": "model_slice",
            "operation": {"op": "conv2d", "attributes": {"ifm": "IFM", "weight": "W", "out": "Y0"}},
            "performance": performance}


def _functional_conv_capsule() -> dict:
    """The same conv, asked as a CORRECTNESS question: no performance block, no cycle count."""
    cap = _conv_capsule()
    cap["name"] = "B3_synthetic_functional_conv"
    del cap["performance"]
    return cap


def _leaves() -> dict:
    return {"IFM": {"shape": _IFM_SHAPE, "dtype": "i8", "role": "input"},
            "W": {"shape": [_K, _CO], "dtype": "i8", "role": "weight"},
            "Y0": {"shape": [_M, _CO], "dtype": "i32", "role": "output"}}


def _recipe() -> dict:
    return {"target": "IFM_im2col", "source": "IFM", "kh": _KH, "kw": _KW, "ci": _CI,
            "stride": [1, 1], "padding": [0, 0, 0, 0], "dilation": [1, 1], "layout": "nhwc"}


def _buffer_declaring_a_recipe() -> dict:
    """THE LOWERING THAT IS NOT ONE, reproduced from the frozen submission's own emitter.

    ``gemmini_backend/conv.py`` builds ``{target: <ifm>_im2col, source: <ifm>, ...}`` and
    ``cmdbuf.py`` declares the derived activation as a ``role: input`` tensor, appends the recipe to
    ``params.im2col_recipes``, and then emits pack/matmul/commit over it. Not one command gathers a
    window.
    """
    tensors = _leaves()
    tensors["IFM_im2col"] = {"shape": [_M, _K], "dtype": "i8", "role": "input"}
    return {"abi_version": "0.1", "target": "synthetic", "tensors": tensors,
            "params": {IM2COL_RECIPE_KEY: [_recipe()]},
            "commands": [
                {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
                 "attributes": {"layout": "packed_conv_rhs"}},
                {"opcode": "MATMUL_RESIDENT",
                 "operands": {"lhs": "IFM_im2col", "rhs": "W_res", "dst": "conv_acc0"}},
                {"opcode": "COMMIT", "operands": {"src": "conv_acc0", "dst": "Y0"},
                 "attributes": {"epilogue": [], "output_dtype": "i32"}}]}


def _buffer_emitting_the_whole_operation() -> dict:
    """Honest lowering #1: emit the ABI's CONV2D and let the datapath do the windowing."""
    return {"abi_version": "0.1", "target": "synthetic", "tensors": _leaves(),
            "commands": [
                {"opcode": "CONV2D", "operands": {"ifm": "IFM", "weight": "W", "dst": "Y0"},
                 "attributes": {"kernel": [_KH, _KW, _CI, _CO], "stride": [1, 1],
                                "padding": [0, 0, 0, 0], "dilation": [1, 1], "layout": "nhwc",
                                "epilogue": [], "output_dtype": "i32"}}]}


def _buffer_emitting_an_explicit_gather() -> dict:
    """Honest lowering #2: a command PRODUCES the column matrix, then the contraction reads it."""
    cb = _buffer_declaring_a_recipe()
    cb.pop("params")
    cb["tensors"]["IFM_im2col"]["role"] = "output"       # produced, not handed in
    cb["commands"].insert(0, {"opcode": "GATHER_WINDOWS",
                              "operands": {"src": "IFM", "dst": "IFM_im2col"},
                              "attributes": {"kh": _KH, "kw": _KW, "ci": _CI, "stride": [1, 1],
                                             "padding": [0, 0, 0, 0], "dilation": [1, 1],
                                             "layout": "nhwc"}})
    return cb


# ---------------------------------------------------------------------------------------------
# MUTATION: the check must be able to fail, and must be able to pass
# ---------------------------------------------------------------------------------------------
def test_a_performance_member_that_declares_the_gather_away_is_violated():
    row = LOB.assess(_conv_capsule(), _buffer_declaring_a_recipe())
    assert row["status"] == LOB.VIOLATED, row
    assert row["operands"] == ["IFM_im2col"], row
    assert row["recipe_keys"] == [IM2COL_RECIPE_KEY], row
    # the message has to say what to do, not only that something is wrong
    assert "gather" in row["detail"] and "PRODUCES" in row["detail"], row["detail"]


def test_a_performance_member_that_emits_the_whole_operation_is_satisfied():
    row = LOB.assess(_conv_capsule(), _buffer_emitting_the_whole_operation())
    assert row["status"] == LOB.SATISFIED, row
    assert row["harness_derived"] == [], row


def test_a_performance_member_that_emits_an_explicit_gather_is_satisfied():
    row = LOB.assess(_conv_capsule(), _buffer_emitting_an_explicit_gather())
    assert row["status"] == LOB.SATISFIED, row


def test_a_declared_recipe_nothing_contracts_over_is_not_a_violation():
    """The rule is about what the program READS, not about the presence of a declaration.

    A recipe whose result no command consumes costs the program nothing and hides no lowering; failing
    it would be punishing a spelling. Stated as a test so the check cannot quietly widen into one.
    """
    cb = _buffer_emitting_the_whole_operation()
    cb["params"] = {IM2COL_RECIPE_KEY: [_recipe()]}
    cb["tensors"]["IFM_im2col"] = {"shape": [_M, _K], "dtype": "i8", "role": "input"}
    assert LOB.assess(_conv_capsule(), cb)["status"] == LOB.SATISFIED


# ---------------------------------------------------------------------------------------------
# SCOPE: functional conv capsules are a different question and are deliberately untouched
# ---------------------------------------------------------------------------------------------
def test_a_functional_conv_capsule_using_a_recipe_is_out_of_scope():
    """EXPLICIT: tightening a functional conv capsule is NOT part of this rule.

    A functional capsule asks whether the arithmetic is right. It is graded against an independent
    golden, no cycle count is attributed to its program, and a stimulus materialized by the harness
    cannot flatter that verdict. The same buffer that FAILS as a performance member is therefore
    `not_applicable` here -- and the assertion pairs the two so the scope is measured, not asserted in
    prose.
    """
    violating = _buffer_declaring_a_recipe()
    assert LOB.assess(_conv_capsule(), violating)["status"] == LOB.VIOLATED
    row = LOB.assess(_functional_conv_capsule(), violating)
    assert row["status"] == LOB.NOT_APPLICABLE, row
    assert "no performance block" in row["detail"]


def test_the_shipped_functional_conv_corpus_still_declares_no_obligation():
    """The scope statement, checked against the corpus rather than trusted.

    If a later edit sprinkled the obligation onto functional conv capsules, this fails and the scope
    decision gets re-made deliberately instead of drifting.
    """
    functional = sorted(CAPSULES.glob("layers/*conv*/capsule.yaml"))
    if not functional:
        pytest.skip("no functional conv capsules in this corpus")
    for path in functional:
        cap = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert not LOB.is_performance_member(cap), path
        assert LOB.declared_obligation(cap) is None, path


# ---------------------------------------------------------------------------------------------
# FAIL CLOSED
# ---------------------------------------------------------------------------------------------
def test_an_obligation_this_build_cannot_evaluate_is_refused_not_satisfied():
    cap = _conv_capsule(lowering_obligation={"id": "some_future_rule", "on_violation": "fail_member"})
    row = LOB.assess(cap, _buffer_emitting_the_whole_operation())
    assert row["status"] == LOB.UNENFORCEABLE, row
    assert row["status"] != LOB.SATISFIED
    assert LOB.OBLIGATION_ID in row["detail"]


def test_a_performance_member_with_no_buffer_at_all_is_refused():
    row = LOB.assess(_conv_capsule(), None)
    assert row["status"] == LOB.UNENFORCEABLE, row


def test_a_declined_compilation_is_not_a_violation():
    row = LOB.assess(_conv_capsule(), {"declined": {"reason": "unsupported op"}, "commands": []})
    assert row["status"] == LOB.NOT_APPLICABLE, row


def test_the_refusing_statuses_are_both_failures():
    """The caller gates on this set; leaving UNENFORCEABLE out would restore the silent pass."""
    assert LOB.REFUSING_STATUSES == frozenset({LOB.VIOLATED, LOB.UNENFORCEABLE})


# ---------------------------------------------------------------------------------------------
# the two readers of the derivation declaration cannot drift apart
# ---------------------------------------------------------------------------------------------
def test_every_recipe_kind_the_harness_materializes_is_visible_to_the_obligation():
    """``materialize_inputs`` honours exactly ``DERIVATION_RECIPE_KEYS`` -- no more, no less.

    A recipe kind honoured by the materializer but absent from that tuple would build an operand the
    obligation cannot see, which is the whole defect re-opened under a new key. Both halves are
    measured: each listed key is really honoured, and a key that is NOT listed is really ignored.
    """
    cb = _buffer_declaring_a_recipe()
    for key in DERIVATION_RECIPE_KEYS:
        probe = copy.deepcopy(cb)
        probe["params"] = {key: [_recipe()]}
        env = materialize_inputs(probe)
        expected = conv_im2col(env["IFM"], kh=_KH, kw=_KW, ci=_CI, stride=(1, 1),
                               padding=(0, 0, 0, 0), dilation=(1, 1))
        assert env["IFM_im2col"].data == expected.data, (
            f"params.{key} is declared a derivation key but the materializer did not honour it")
        assert set(harness_derived_tensors(probe)) == {"IFM_im2col"}

    unlisted = copy.deepcopy(cb)
    unlisted["params"] = {"a_recipe_kind_that_does_not_exist": [_recipe()]}
    env = materialize_inputs(unlisted)
    assert env["IFM_im2col"].data == Tensor.deterministic("IFM_im2col", (_M, _K), "i8").data, (
        "an unlisted params key must materialize nothing; otherwise the obligation is blind to it")
    assert harness_derived_tensors(unlisted) == {}


def test_operand_flow_recognises_a_write_only_by_the_abi_write_slot():
    written, referenced = operand_flow(_buffer_emitting_an_explicit_gather())
    assert "IFM_im2col" in written and {"IFM", "IFM_im2col"} <= referenced


def test_a_role_label_cannot_stand_in_for_a_command_that_writes_the_operand():
    """The adversarial spelling: declare the recipe AND label its result ``role: output``.

    Nothing was produced -- no command writes the tensor -- and taking the label as proof would hand
    the whole rule back. This is the shape the check has to survive, so it is measured.
    """
    cb = _buffer_declaring_a_recipe()
    cb["tensors"]["IFM_im2col"]["role"] = "output"
    row = LOB.assess(_conv_capsule(), cb)
    assert row["status"] == LOB.VIOLATED, row
    assert row["operands"] == ["IFM_im2col"], row


# ---------------------------------------------------------------------------------------------
# the shipped corpus states the obligation, and states the SAME one the shared template does
# ---------------------------------------------------------------------------------------------
def _shared_conv_family_declaration() -> dict:
    doc = yaml.safe_load(SHARED_PERF_TEMPLATE.read_text(encoding="utf-8"))
    sweeps = [s for s in (doc.get("sweeps") or []) if s.get("id") == _CONV_PERF_FAMILY]
    assert len(sweeps) == 1, f"expected exactly one {_CONV_PERF_FAMILY} sweep, got {len(sweeps)}"
    declaration = ((sweeps[0].get("base") or {}).get("performance") or {}).get(LOB.DECLARATION_KEY)
    assert isinstance(declaration, dict), (
        f"the shared performance template must declare the {_CONV_PERF_FAMILY} family's lowering "
        f"obligation; the capsules are generated from it, so a hand-edited capsule alone would be "
        f"erased by the next regeneration")
    return declaration


def _shipped_conv_performance_members() -> list:
    members = []
    for path in sorted(CAPSULES.glob("**/_perf/*/capsule.yaml")):
        cap = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (cap.get("performance") or {}).get("family") == _CONV_PERF_FAMILY:
            members.append((path, cap))
    return members


def test_the_shared_template_declares_the_obligation_this_build_implements():
    assert _shared_conv_family_declaration()["id"] == LOB.OBLIGATION_ID


def test_no_shipped_conv_member_declares_a_DIFFERENT_obligation():
    """The drift gate, and the reason it is shaped as a drift gate rather than a census.

    The capsules are GENERATED from the shared template, and the per-target corpus subtrees are
    regenerated independently, so at any moment one of them may legitimately predate a template
    change. Demanding every member carry the declaration would therefore fail on a mid-regeneration
    tree for a reason that says nothing about the rule.

    That is affordable ONLY because enforcement does not read the declaration: it is scoped by the
    presence of the member's ``performance`` block (see the runner tests below), so a member whose
    declaration has not caught up is still held to the obligation. What must never happen is a member
    declaring a DIFFERENT obligation from the template's, because that is the one state where the
    contract says two things at once -- and an id this build cannot evaluate is refused, not assumed.
    """
    declaration = _shared_conv_family_declaration()
    for path, cap in _shipped_conv_performance_members():
        carried = LOB.declared_obligation(cap)
        if carried is None:
            continue
        assert carried == declaration, (
            f"{path} declares an obligation that has drifted from the shared template")


def test_the_declaration_actually_reached_the_generated_corpus():
    """Non-vacuity: the drift gate above passes trivially on a corpus that declares nothing."""
    carrying = [path for path, cap in _shipped_conv_performance_members()
                if LOB.declared_obligation(cap) is not None]
    assert carrying, ("no shipped conv performance member carries the obligation the shared template "
                      "declares, so the declaration reached nothing")


def test_the_declaration_did_not_change_what_the_member_measures():
    """The obligation is a claim precondition, not a change of workload.

    The GSIM equivalence certificate is keyed by operation/shape/semantics. Putting the obligation in
    ``performance`` rather than in ``operation.attributes`` keeps every member's workload identity
    byte-for-byte what it was, so a certificate minted for these members stays valid.
    """
    for path, cap in _shipped_conv_performance_members():
        assert LOB.DECLARATION_KEY not in (cap.get("operation") or {}).get("attributes", {}), path
        assert LOB.DECLARATION_KEY not in cap, path
        if LOB.declared_obligation(cap) is not None:
            assert LOB.DECLARATION_KEY in cap["performance"], path


# ---------------------------------------------------------------------------------------------
# the gate is WIRED: the runner refuses the member, it does not merely record a note
# ---------------------------------------------------------------------------------------------
_LOWERING_PLANE = "lowering_obligation"


def _runner_config():
    """A grading config whose shape, not whose identity, matters here."""
    from merlin.targetgen.runner_config import RunnerConfig

    return RunnerConfig(target="synthetic-endpoint", suite="synthetic-capsule-bench", dtype="i8xi8_i32",
                        fourth_output_name="kernel.S", tier_sim={}, rtl_tiers=frozenset(),
                        oracle_tiers=(), perf_fields=(), trace_gate=None)


def _run_with_buffer(tmp_path, monkeypatch, capsule: dict, cb: dict) -> dict:
    from merlin.targetgen import capsule_runner as CR

    monkeypatch.setattr(CR, "run_entrypoints", lambda *a, **k: (object(), cb, "# kernel.S (stub)\n"))
    return CR.run_capsule(capsule, "unused-package", runs_root=tmp_path, run_id="lowering",
                          config=_runner_config(), oracle_adapters={})


def test_the_runner_fails_a_performance_member_whose_program_omits_the_windowing(tmp_path, monkeypatch):
    res = _run_with_buffer(tmp_path, monkeypatch, _conv_capsule(), _buffer_declaring_a_recipe())
    assert res["status"] == "fail", res.get("failure")
    assert res["failure"]["plane"] == _LOWERING_PLANE, res["failure"]
    assert "IFM_im2col" in res["failure"]["detail"]


def test_the_runner_does_not_raise_that_failure_when_the_program_emits_the_gather(tmp_path, monkeypatch):
    """The other arm of the mutation, at the gate itself.

    The run still ends without a verdict here (no oracle adapter is supplied), which is exactly right --
    what must NOT happen is the lowering plane firing on a program that did emit its own gather.
    """
    res = _run_with_buffer(tmp_path, monkeypatch, _conv_capsule(),
                           _buffer_emitting_an_explicit_gather())
    assert (res.get("failure") or {}).get("plane") != _LOWERING_PLANE, res.get("failure")


def test_the_runner_leaves_a_functional_conv_capsule_alone(tmp_path, monkeypatch):
    """Scope, at the gate: the identical buffer that fails as a performance member passes here."""
    res = _run_with_buffer(tmp_path, monkeypatch, _functional_conv_capsule(),
                           _buffer_declaring_a_recipe())
    assert (res.get("failure") or {}).get("plane") != _LOWERING_PLANE, res.get("failure")
