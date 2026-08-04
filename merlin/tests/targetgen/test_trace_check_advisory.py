"""trace_check is ADVISORY; the only oracle-independent gate it exposes is the anti-cheese floor.

Contract (see capsule_runner: the trace gate raises ONLY when ``drives_accelerator`` is False):
  * coverage / ordering / UNKNOWN findings are diagnostics (``check`` may report status="fail"), but the
    runner does NOT fail a capsule on them — correctness is the numeric + L2/L3 RTL oracle, and an
    instruction the decoder cannot classify is our limit, not the backend's defect;
  * a kernel that emits >=1 custom-opcode instruction DRIVES the accelerator (gate passes), even if some
    instructions are UNKNOWN; a kernel with only a fence / host ops does not (gate fails) — the one thing
    that cannot be faked away since correctness must come from the device.
The signal is derived (``funct`` present == the target's RTL decoder claimed the instruction) — no
class-name or target literals — so it holds for any target.
"""
from __future__ import annotations

from merlin.targetgen import trace_check as TCK


def _trace(*insts: dict) -> dict:
    return {"instructions": list(insts)}


def test_drives_accelerator_true_when_a_custom_opcode_instruction_is_present():
    # a recognized accelerator op (funct set) — drives the accelerator
    assert TCK.drives_accelerator(_trace({"class": "MVIN", "funct": 2}))
    # UNKNOWN but with a funct (matched the custom opcode, sub-class not recognized) STILL counts
    assert TCK.drives_accelerator(_trace({"class": "FENCE", "funct": None},
                                         {"class": "UNKNOWN", "funct": 9}))


def test_drives_accelerator_false_when_no_custom_opcode_instruction():
    # only a memory fence (funct None) — did not drive the accelerator
    assert not TCK.drives_accelerator(_trace({"class": "FENCE", "funct": None}))
    # a non-custom asm recorded UNKNOWN with no funct — not an accelerator instruction
    assert not TCK.drives_accelerator(_trace({"class": "UNKNOWN", "funct": None, "raw": ".insn r 0x33,..."}))
    assert not TCK.drives_accelerator(_trace())  # empty


def test_dram_provenance_is_address_model_parameterized():
    # A memory-move op whose DRAM operand is a baked const vs one derived from a function arg.
    baked = _trace({"class": "MVIN", "funct": 2, "decoded": {"dram": {"kind": "const", "raw": 0}}})
    argd = _trace({"class": "MVIN", "funct": 2,
                   "decoded": {"dram": {"kind": "argbase", "arg_index": 1, "offset": 0}}})
    # pointer_args: a baked DRAM address is flagged (won't match the runtime buffer); an arg-derived one is fine
    assert len(TCK.dram_address_findings(baked, "pointer_args")) == 1
    assert TCK.dram_address_findings(argd, "pointer_args") == []
    # fixed_preload: a baked const IS the declared base — not flagged (the check would overfit otherwise)
    assert TCK.dram_address_findings(baked, "fixed_preload") == []
    # a non-memory op (no 'dram' operand) is never flagged
    assert TCK.dram_address_findings(_trace({"class": "COMPUTE_PRELOADED", "funct": 4, "decoded": {}}),
                                     "pointer_args") == []
    # threaded through check(): advisory only (does not gate — drives_accelerator is the sole gate)
    res = TCK.check(baked, expected={}, address_model="pointer_args")
    assert any("BAKED DRAM address" in v for v in res["violations"])
    assert TCK.drives_accelerator(baked)


def test_unknown_is_reported_but_is_not_the_gate():
    # An UNKNOWN produces an advisory violation in check(), but it still DRIVES the accelerator (funct set),
    # so the anti-cheese gate the runner uses passes — UNKNOWN never fails a conformant backend by itself.
    tr = _trace({"class": "FENCE", "funct": None},
                {"class": "UNKNOWN", "funct": 9, "decoded": {}},
                {"class": "FENCE", "funct": None})
    res = TCK.check(tr, expected={})
    assert any("UNKNOWN" in v for v in res["violations"])   # reported...
    assert TCK.drives_accelerator(tr)                        # ...but the gate (anti-cheese) still passes


def test_degenerate_scale_flagged_identity_silent():
    # A datapath scale of 0.0 (the np12r bug: mvin loads x0) or non-finite is a universal degeneracy —
    # flagged; identity 1.0 or an unresolved (None) scale is not. Advisory: it lands in violations, no gate.
    zero = _trace({"class": "CONFIG_LD", "funct": 0, "index": 2,
                   "decoded": {"subtype": "LD", "scale": 0.0}})
    assert TCK.degenerate_scale_findings(zero), "a zero datapath scale must be flagged"
    ident = _trace({"class": "CONFIG_LD", "funct": 0, "index": 2,
                    "decoded": {"subtype": "LD", "scale": 1.0}})
    assert TCK.degenerate_scale_findings(ident) == []
    nan = _trace({"class": "CONFIG_ST", "funct": 0, "decoded": {"acc_scale": float("nan")}})
    assert TCK.degenerate_scale_findings(nan)                              # non-finite acc_scale too
    none = _trace({"class": "CONFIG_LD", "funct": 0, "decoded": {"subtype": "LD", "scale": None}})
    assert TCK.degenerate_scale_findings(none) == []                      # unresolved -> fail-open, not flagged
    res = TCK.check(zero, expected={})                                    # surfaces via advisory violations
    assert any("scale" in v for v in res["violations"])
