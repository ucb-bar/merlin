"""A resident-weight program sets its weight up ONCE and reuses it; the check must test that directly.

The point of a resident-weight capsule is that the weight is packed once and then reused by every
activation that consumes it. A backend that re-materializes the pack per command still computes the
right answer -- the repeated transfer writes the bytes that are already there -- so numerics, L0/L1/L2
and the L3 RTL oracle all pass, and only trace conformance can notice. That is exactly how it was
found: ``A6_resident_reuse`` passed every tier while re-loading its weight.

WHAT THIS FILE CORRECTS. The original rule tested a PROXY: ``n_cfg_ex != 1``, under a comment reading
"weights loaded into the resident region once". A config count is not a statement about the resident
region, and nothing counted the loads -- so the defect was reported as one redundant config while the
weight was ALSO being moved in twice, and suppressing that config alone would have turned the capsule
green with the reload intact. Measured on the frozen A6 trace: two CONFIG_EX flagged, while
``MVIN(arg 0 -> on-chip 16368)`` appeared at #5 AND #14 and went unmentioned.

So the rule now has two independent halves, and these tests pin both:

* RESIDENCY -- an on-chip destination rewritten with the same source it already held is a re-load, and
  the region was not resident. Decided from the DECODER's derived ``spad_addr``/``dram`` fields, so no
  operand-field layout is assumed here; a decoder that cannot see the destination yields no finding
  rather than a guess.
* MODE CONFIG -- the execution-mode config is re-issued when it is already active. This is the
  surviving half of the original intent, kept because weight-stationary mode is part of what residency
  means, and scoped to that one config so a residency capsule is not failed for unrelated config
  hoisting.

Plus the two properties that stop either half from degenerating: reuse must still be VISIBLE (>=2
commits), and hoisting must not move a config after the work it configures.

Class names here are the decoder's own derived classes and the expectations come from a capsule's
``expected`` block, so nothing in this file assumes an opcode, a funct value or a target.
"""
from __future__ import annotations

from merlin.targetgen import trace_check as TCK

_EXPECTED = {
    "instruction_classes": ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST",
                            "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"],
    "modes": {"resident_reuse": True},
}

# Every class carries a funct so ``drives_accelerator`` (the one gating signal) is satisfied and the
# only findings under test are the mode findings.
_FUNCT = {"FENCE": None, "FLUSH": 7, "CONFIG_EX": 0, "CONFIG_LD": 0, "CONFIG_ST": 0,
          "MVIN": 2, "MVOUT": 3, "PRELOAD": 6, "COMPUTE_PRELOADED": 4}

# The on-chip destinations A6 actually uses: the weight high in the scratchpad, activations at the base.
_WEIGHT_SPAD, _ACT_SPAD = 16368, 0


def _ins(cls: str, *, decoded: dict | None = None, rs1=None, rs2=None) -> dict:
    return {"class": cls, "funct": _FUNCT[cls], "decoded": decoded or {}, "rs1": rs1, "rs2": rs2}


def _load(arg_index: int, spad: int) -> list[dict]:
    """A CONFIG_LD + MVIN pair moving operand ``arg_index`` to ``spad``, as the decoder reports it."""
    return [_ins("CONFIG_LD"),
            _ins("MVIN", decoded={"dram": {"raw": None, "kind": "argbase",
                                           "arg_index": arg_index, "offset": 0},
                                  "rows": 16, "cols": 16, "addr": spad, "spad_addr": spad})]


def _commit(activation_arg: int, *, config_ex: bool, reload_weight: bool) -> list[dict]:
    out: list[dict] = []
    if config_ex:
        out.append(_ins("CONFIG_EX", rs1=0x3F80000000010004, rs2=0x1000000000000))
    out.append(_ins("CONFIG_ST"))
    if reload_weight:
        out += _load(0, _WEIGHT_SPAD)
    out += _load(activation_arg, _ACT_SPAD)
    out += [_ins("PRELOAD"), _ins("COMPUTE_PRELOADED"), _ins("MVOUT")]
    return out


def _program(*, config_ex_per_commit: bool, reload_weight: bool, commits: int = 2) -> dict:
    ins = [_ins("FENCE"), _ins("FLUSH")]
    if not config_ex_per_commit:
        ins.append(_ins("CONFIG_EX", rs1=0x3F80000000010004, rs2=0x1000000000000))
    # The weight is packed once, up front, when it is genuinely resident.
    if not reload_weight:
        ins += _load(0, _WEIGHT_SPAD)
    for n in range(commits):
        ins += _commit(1 + n, config_ex=config_ex_per_commit, reload_weight=reload_weight)
    ins.append(_ins("FENCE"))
    for i, x in enumerate(ins):
        x["index"] = i
    return {"instructions": ins}


def _mode_violations(result: dict) -> list[str]:
    return [v for v in result["violations"] if v.startswith("mode resident_reuse")]


def test_a_re_materialized_weight_is_rejected():
    """THE DEFECT THE PROXY MISSED: the weight is moved in again for the second matmul."""
    trace = _program(config_ex_per_commit=False, reload_weight=True)
    result = TCK.check(trace, _EXPECTED)
    assert result["status"] == "fail"
    findings = _mode_violations(result)
    assert len(findings) == 1, findings
    assert "redundant load" in findings[0] and "NOT resident" in findings[0]
    assert str(_WEIGHT_SPAD) in findings[0], "the finding must name the destination that was rewritten"


def test_a_genuinely_resident_weight_is_accepted():
    """THE FIX: one pack up front, two activations, two commits."""
    trace = _program(config_ex_per_commit=False, reload_weight=False)
    classes = [i["class"] for i in trace["instructions"]]
    assert classes.count("MVOUT") == 2, "the reuse must still be visible as two output commits"
    assert classes.count("MVIN") == 3, "one weight load plus one activation load per commit"
    result = TCK.check(trace, _EXPECTED)
    assert _mode_violations(result) == []
    assert result["status"] == "pass", result["violations"]


def test_a_re_issued_execution_mode_config_is_rejected():
    """The surviving half of the original intent, now stated as redundancy rather than a count."""
    trace = _program(config_ex_per_commit=True, reload_weight=False)
    assert [i["class"] for i in trace["instructions"]].count("CONFIG_EX") == 2
    findings = _mode_violations(TCK.check(trace, _EXPECTED))
    assert len(findings) == 1, findings
    assert "re-issue the configuration already active" in findings[0]


def test_the_two_halves_are_independent():
    """Both defects at once yields both findings -- neither masks the other.

    This is the property whose absence caused the original miss: one finding was reported, the reader
    fixed the thing it named, and the other defect stayed.
    """
    findings = _mode_violations(
        TCK.check(_program(config_ex_per_commit=True, reload_weight=True), _EXPECTED))
    assert len(findings) == 2, findings
    assert any("redundant load" in f for f in findings)
    assert any("already active" in f for f in findings)


def test_reuse_must_still_be_visible():
    """Neither half is satisfiable by dropping the second activation entirely."""
    findings = _mode_violations(
        TCK.check(_program(config_ex_per_commit=False, reload_weight=False, commits=1), _EXPECTED))
    assert findings == ["mode resident_reuse declared but <2 output commits (no reuse visible)"]


def test_a_reload_after_the_slot_is_reused_is_not_flagged():
    """A destination genuinely repurposed and then re-filled is a REAL load, not a redundant one.

    Tracking live contents per destination is what makes this distinction; a rule that merely looked
    for a repeated payload anywhere in the trace would fail this legitimate program.
    """
    ins = [_ins("FENCE"), _ins("FLUSH"),
           _ins("CONFIG_EX", rs1=0x3F80000000010004, rs2=0x1000000000000)]
    ins += _load(0, _WEIGHT_SPAD)                      # weight in
    ins += _load(1, _WEIGHT_SPAD)                      # slot repurposed for another tensor
    ins += _load(0, _WEIGHT_SPAD)                      # so THIS is a genuine re-load
    for arg in (2, 3):
        ins += [_ins("CONFIG_ST")] + _load(arg, _ACT_SPAD) + [
            _ins("PRELOAD"), _ins("COMPUTE_PRELOADED"), _ins("MVOUT")]
    ins.append(_ins("FENCE"))
    for i, x in enumerate(ins):
        x["index"] = i
    assert _mode_violations(TCK.check({"instructions": ins}, _EXPECTED)) == []


def test_a_decoder_that_cannot_see_the_destination_yields_no_finding():
    """Fail SOFT on the residency half: absent evidence is not a violation, and not a pass either.

    A target whose decoder exposes no on-chip destination gets silence from this half of the rule --
    never an inferred verdict. The mode-config half still applies, since it needs no address.
    """
    ins = [_ins("FENCE"), _ins("FLUSH"),
           _ins("CONFIG_EX", rs1=0x3F80000000010004, rs2=0x1000000000000)]
    for _ in range(2):
        ins += [_ins("CONFIG_ST"), _ins("CONFIG_LD"), _ins("MVIN"),   # decoded={} -> no spad_addr
                _ins("CONFIG_LD"), _ins("MVIN"),
                _ins("PRELOAD"), _ins("COMPUTE_PRELOADED"), _ins("MVOUT")]
    ins.append(_ins("FENCE"))
    for i, x in enumerate(ins):
        x["index"] = i
    assert _mode_violations(TCK.check({"instructions": ins}, _EXPECTED)) == []


def test_the_single_config_still_precedes_the_first_compute():
    """Hoisting must not move the configuration after the work it configures."""
    trace = _program(config_ex_per_commit=False, reload_weight=False)
    ins = [x for x in trace["instructions"] if x["class"] != "CONFIG_EX"]
    first_mvout = next(i for i, x in enumerate(ins) if x["class"] == "MVOUT")
    ins.insert(first_mvout + 1, _ins("CONFIG_EX", rs1=1, rs2=2))
    result = TCK.check({"instructions": ins}, _EXPECTED)
    assert any("CONFIG_EX appears after first PRELOAD/COMPUTE" in v for v in result["violations"])
