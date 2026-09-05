"""The encoding merlin derives for a target must agree with that target's own hardware.

A shipped ISA definition is a DOCUMENT. When it is wrong, a backend deriving from it emits a word that
assembles cleanly, disassembles to the intended mnemonic, and executes as a DIFFERENT instruction — the
one failure mode no downstream check can see, because nothing about it looks like an error.

These tests pin the three properties that make the cross-check worth having:
  * it FAILS when a target's hardware contradicts its shipped encoding and nobody wrote that down;
  * it says UNKNOWN, never OK, when it had no evidence — a check that could not run must not pass;
  * it refuses to grade a model against the source that model was BUILT FROM.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import isa_rtl_crosscheck as X


# ======================================================================================================
# Parsers — pure, no target and no checkout needed
# ======================================================================================================
def test_bitpat_dontcare_bits_are_never_compared():
    """``?`` is the decoder saying "this bit is not my identity". Pinning it rejects conformant words —
    which is how a per-instruction field (a channel selector, a RoCC operand-presence bit) silently
    dropped every valid encoding the last time something matched on the whole word."""
    (pat,) = X.parse_bitpats('  def FOO = BitPat("b0000001_00000_?????_???_00000_1111111")').values()
    assert pat.width == 32
    # the five rs1 bits [19:15] and three funct3 bits [14:12] are don't-cares -> absent from the mask
    for bit in list(range(15, 20)) + list(range(12, 15)):
        assert not (pat.mask >> bit) & 1, f"bit {bit} is a don't-care but was pinned"
    assert (pat.value >> 25) & 0x7F == 0b0000001
    assert pat.value & 0x7F == 0b1111111


def test_bitpat_reader_ignores_separators_and_non_literals():
    got = X.parse_bitpats("\n".join([
        'def A = BitPat("b0000000_00000")',
        'def B = BitPat("b" + someRuntimeString)',       # not a literal -> must not be invented
        'val C = BitPat("b1111")',                       # not a `def` -> different construct
        'def D = BitPat("hDEADBEEF")',                   # not binary
    ]))
    assert set(got) == {"A"}


def test_opcode_constants_skip_commented_out_lines():
    """A decoder keeps retired opcodes commented out. Reading one as live invents an encoding."""
    got = X.parse_opcode_constants("\n".join([
        "object Op {",
        '  val LOAD = "b000000011"',
        '  // val LOAD_FP = "b0000111"',
        '  val NAME_WITH_TEXT = "not bits"',
        "}",
    ]))
    assert got == {"LOAD": 3}


def test_projection_fit_refuses_a_table_it_cannot_explain():
    """A table no projection of the model's own fields explains is uninterpretable. Reading it anyway
    would manufacture verdicts in both directions out of a coincidence."""
    class _M:
        by_mnemonic = {"A": {"opcode": 3, "funct7": 0, "fixed_mask": 0x707F, "fixed_value": 3}}
    _name, _fn, cov = X.fit_code_projection(_M(), {90210, 70000, 12345})
    assert cov < X.MIN_PROJECTION_COVERAGE


# ======================================================================================================
# Report semantics
# ======================================================================================================
def _targets():
    from merlin.common.paths import merlin_dir
    base = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    return sorted(p.parent.name for p in base.glob("*/target_experiment.yaml"))


@pytest.mark.parametrize("target", _targets())
def test_no_evidence_is_never_reported_as_agreement(target):
    """The single most important property. A target the check could not verify must land on UNKNOWN, and
    a target reported OK must actually have compared something. `n_fail == 0` over zero comparisons is
    the shape this repo keeps rediscovering the hard way."""
    rep = X.crosscheck(target)
    if rep.status == X.OK:
        assert rep.covered_mnemonics, f"{target} reported OK having compared nothing"
        assert any(s.usable for s in rep.sources), f"{target} reported OK with no usable source"
    if not any(s.usable for s in rep.sources):
        assert rep.status == X.UNKNOWN, f"{target} has no usable evidence but did not report UNKNOWN"


@pytest.mark.parametrize("target", _targets())
def test_uncovered_instructions_are_accounted_for(target):
    """Every mnemonic the check touched is either covered or explicitly not covered — never dropped.
    A class no decode pattern mentions must show up in the not-covered count, because 111 silently
    unmatched classes reading as success is the failure this whole module exists to prevent."""
    rep = X.crosscheck(target)
    assert rep.covered_mnemonics.isdisjoint(rep.uncovered_mnemonics)
    assert (rep.covered_mnemonics | rep.uncovered_mnemonics) == {f.mnemonic for f in rep.findings}


def test_a_model_is_never_graded_against_the_table_it_was_built_from():
    """A target whose model is synthesised from its decode table would otherwise score a perfect,
    permanent, meaningless 100%. Whichever target that is, the source must be marked circular and must
    not count toward the verdict."""
    checked = 0
    for target in _targets():
        rep = X.crosscheck(target)
        if rep.model_provenance != "rtl_decode_table":
            continue
        checked += 1
        src = next(s for s in rep.sources if s.kind == X.RTL_DECODE_TABLE)
        assert src.circular, f"{target}: model built from the decode table but the source is not circular"
        assert not src.usable, f"{target}: a circular source must not count as evidence"
        assert not any(f.source == X.RTL_DECODE_TABLE for f in rep.by_verdict(X.AGREE)), \
            f"{target}: tautological agreements leaked into the verdict"
    assert checked, "no target exercises the circular-provenance path; this test asserted nothing"


# ======================================================================================================
# The gate
# ======================================================================================================
@pytest.mark.parametrize("target", _targets())
def test_declared_encoding_agrees_with_hardware(target):
    """THE GATE. A target's own hardware must not contradict the encoding merlin derives for it, unless a
    human has read the evidence and recorded the disagreement in the errata registry.

    THIS TEST FAILS ON TODAY'S TREE, and that is the point: it is reporting a real, measured defect. To
    make it pass, either fix the encoding or record the erratum with a rationale — never by deleting the
    assertion. Run `build_tools/scripts/check_isa_matches_rtl.py --target <t>` for the evidence and a
    ready-to-paste registry stanza."""
    rep = X.crosscheck(target)
    undeclared = X.undeclared_disagreements(rep)
    assert not undeclared, (
        f"{target}: {len(undeclared)} instruction(s) whose declared encoding this target's own hardware "
        f"contradicts, none recorded in {X.errata_path().name}: "
        + "; ".join(f"{m} spec={r['declared']} vs {sorted(r['evidence'].values())}"
                    for m, r in sorted(undeclared.items())))


def test_the_gate_can_actually_fail():
    """A gate nobody has seen fail is a gate nobody has tested. This drives the same comparison with a
    planted disagreement and asserts it is caught — so a future refactor that quietly turns the check
    into a no-op is detected even on a tree where every real target happens to agree."""
    class _M:
        target = "planted"
        by_mnemonic = {"OP_A": {"opcode": 0x7F, "funct7": 1, "fixed_mask": 0xFE00_007F,
                                "fixed_value": 0x0200_007F}}
    findings = X.compare_bitpats(_M(), {"OP_A": X.BitPattern(mask=0xFE00_007F, value=0x0000_007F, width=32)})
    assert [f.verdict for f in findings] == [X.DISAGREE]
    assert findings[0].declared == "0x0200007f" and findings[0].evidence == "0x0000007f"


# ======================================================================================================
# A correction must not collapse a family onto one of its members
# ======================================================================================================

def test_a_correction_preserves_the_bits_the_pattern_does_not_pin():
    """A BitPat's don't-cares carry the per-instruction field; the correction must keep them.

    `pat.value` alone is not an encoding — its don't-care bits are zero. Atlas ships eight
    `DMA_CONFIG_CH0..7` that differ ONLY in funct3, and the matching BitPat leaves funct3 as `???`, so
    the raw pattern value names one word for all eight. Applying that in the assembler would make every
    channel emit channel 0's instruction: eight distinct DMA targets silently becoming one.
    """
    # funct7 + opcode pinned; funct3 (bits 14:12) and rs1 left as don't-care.
    pat = X.BitPattern(mask=0xFE00_707F & ~0x0000_7000, value=0x0000_007F, width=32)
    for channel in range(8):
        declared = 0x0200_007F | (channel << 12)      # spec: funct7=1 (wrong), funct3=channel
        corrected = X._corrected(declared, pat)
        assert corrected == 0x0000_007F | (channel << 12), (
            f"channel {channel}: correction dropped the channel selector")
    # and the correction really did change the contradicted bit
    assert X._corrected(0x0200_007F, pat) == 0x0000_007F


def test_a_correction_overwrites_every_bit_the_pattern_does_pin():
    """The other half: bits the RTL pins must take the hardware's value, not the spec's."""
    pat = X.BitPattern(mask=0xFFFF_FFFF, value=0x0000_7073, width=32)
    assert X._corrected(0x0000_4073, pat) == 0x0000_7073


# ======================================================================================================
# Applying a reviewed correction — the seam that decides emitted bits
# ======================================================================================================

from merlin.targetgen.isa_model import IsaModel, apply_errata  # noqa: E402


def _model_with(entry: dict) -> IsaModel:
    return IsaModel(target="t", by_mnemonic={"OP": dict(entry)})


_SPEC = {"class": "C", "mnemonic": "OP", "opcode": 0x7F, "funct3": 3, "funct7": 1,
         "fixed_mask": 0xFE00_707F, "fixed_value": 0x0200_307F}


def test_a_reviewed_correction_changes_the_word_the_assembler_emits():
    """isa_asm builds a word as `fixed_value | operands`, so this is the seam that decides bits."""
    m = apply_errata(_model_with(_SPEC), {"OP": {
        "authoritative": "rtl", "declared": "0x0200307f", "hardware": "0x0000307f"}})
    e = m.resolve("OP")
    assert e["fixed_value"] == 0x0000_307F
    assert e["spec_fixed_value"] == 0x0200_307F      # the contradicted word is kept, not erased
    assert e["errata_applied"]["hardware"] == "0x0000307f"


def test_a_correction_updates_the_fields_derived_from_the_old_word():
    """A corrected word makes every field derived from the old one stale.

    Leaving funct7 at the contradicted value would reproduce the same silent-wrong-bits failure one
    level down, in every consumer that reads fields rather than the packed word."""
    m = apply_errata(_model_with(_SPEC), {"OP": {
        "authoritative": "rtl", "declared": "0x0200307f", "hardware": "0x0000307f"}})
    e = m.resolve("OP")
    assert e["funct7"] == 0, "funct7 kept the contradicted value"
    assert e["funct3"] == 3, "funct3 is a don't-care here and carries the channel — must survive"
    assert e["opcode"] == 0x7F
    assert not e.get("errata_dropped_fields")


def test_an_unresolved_entry_changes_no_bits_and_says_so():
    """`unresolved` means nothing may derive from either encoding — not a silent pick of one side."""
    m = apply_errata(_model_with(_SPEC), {"OP": {
        "authoritative": "unresolved", "declared": "25", "hardware": "absent from the decoder",
        "rationale": "one decoder's fan-out is not the whole machine"}})
    e = m.resolve("OP")
    assert e["fixed_value"] == _SPEC["fixed_value"]
    assert "errata_unresolved" in e and e["errata_unresolved"]
    assert "errata_applied" not in e


def test_a_field_whose_layout_cannot_be_confirmed_is_dropped_not_left_stale():
    """Fail closed: an unrecognised field after a correction must not keep the contradicted value."""
    spec = dict(_SPEC, weird_field=0x5)
    m = apply_errata(_model_with(spec), {"OP": {
        "authoritative": "rtl", "declared": "0x0200307f", "hardware": "0x0000307f"}})
    e = m.resolve("OP")
    assert e["weird_field"] is None
    assert "weird_field" in e["errata_dropped_fields"]


def test_a_target_with_no_errata_is_untouched():
    before = _model_with(_SPEC)
    assert apply_errata(before, {}).resolve("OP") == before.resolve("OP")
