"""The DRAM window is a DERIVABLE fact, and half of it was being thrown away.

A target's memory-map green card ships its DRAM region as a row ``| `DRAM` | <start> ~ <end> |``.
`dram_facts` parsed that row with a helper that returned only the FIRST hex token, so the end address
was read and discarded: `dram_base_for` was the only public reading, and the liveness screen's
address-map check -- which bounds every movement transaction against ``[base, base+size)`` -- was never
handed a size at all. Every liveness report on disk therefore carried `dram-window-unknown` and left the
upper bound unchecked, *including* for a target whose own card states the end address.

These tests pin the derivation and the honesty of its refusals: two tokens give a real window, one token
or no card gives UNKNOWN with a reason that says WHOSE gap it is (this tooling's, or the target's data),
and nothing anywhere substitutes a plausible default -- a fabricated window would manufacture false
"unmapped address" faults for exactly the programs the real window contains.
"""
from __future__ import annotations

import textwrap

import pytest

from merlin.liveness import Program, Severity, assess
from merlin.liveness.facts import SiliconFacts
from merlin.liveness.interconnect import simulate
from merlin.targetgen import dram_facts as DF

# The two targets in the live cross-arm roster that bracket the two real outcomes: one ships a
# memory-map card with both addresses, the other ships no card at all. Named here (a test is a
# legitimate edge for a target name; library code stays parameterized).
CARD_TARGET = "atlas"
NO_CARD_TARGET = "gemmini"

_GIB = 1024 ** 3


# --- (A) the derivation -----------------------------------------------------------------------------

def test_a_shipped_card_with_both_addresses_yields_the_real_window():
    base, size, why = DF.dram_window_for(CARD_TARGET)
    assert base == 0x8000_0000
    # end - start, with `hi` exclusive everywhere it is consumed: exactly 32 GiB.
    assert size == 32 * _GIB, "the end address must be parsed, not discarded"
    assert f"{base:#x}" in why and f"{base + size:#x}" in why, \
        f"provenance must name both addresses it derived from, got {why!r}"


def test_a_target_with_no_memory_map_stays_unknown_and_says_so():
    base, size, why = DF.dram_window_for(NO_CARD_TARGET)
    assert base == 0, "no card means the 0-based aperture, not a fabricated base"
    assert size is None, "an underivable window must be UNKNOWN, never a plausible default"
    low = why.lower()
    assert "no memory-map card" in low or "ships no memory map" in low, \
        f"provenance must say the TARGET ships no map, got {why!r}"


def test_a_start_only_row_is_a_different_gap_from_no_card_at_all(tmp_path, monkeypatch):
    """A card that gives a start but no end is the target's data gap; no card is a different one.

    The two must not share a sentence: one is closed by reading a second token the card already has,
    the other only by the target shipping a map at all.
    """
    card = tmp_path / "card.md"
    card.write_text(textwrap.dedent("""
        | Region | Address |
        | --- | --- |
        | `IMEM` | `0x1000` ~ `0x2000` |
        | `DRAM` (cacheable) | `0x8000_0000` |
    """).strip(), encoding="utf-8")
    desc = tmp_path / "target_experiment.yaml"
    desc.write_text(f"target: synthstartonly\nhardware_spec:\n  isa_headers:\n    - {card}\n",
                    encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(desc))
    DF._WINDOW_CACHE.pop("synthstartonly", None)

    base, size, why = DF.dram_window_for("synthstartonly")
    assert base == 0x8000_0000, "the start is still derivable"
    assert size is None, "one token cannot give a size"
    assert "no upper bound" in why.lower(), f"provenance must name the missing end, got {why!r}"
    assert why != DF.dram_window_for(NO_CARD_TARGET)[2], \
        "a card-without-an-end must not read the same as a target with no card"


def test_a_card_that_declares_no_dram_row_is_its_own_third_gap(tmp_path, monkeypatch):
    """A shipped memory map without a DRAM region is neither "no card" nor "no upper bound".

    It must not fabricate a base either: the region the kernel addresses is exactly what the map does
    not say, so 0-based is the model default and the sentence has to name which card was read.
    """
    card = tmp_path / "card.md"
    card.write_text("| Region | Address |\n| `IMEM` | `0x1000` ~ `0x2000` |\n", encoding="utf-8")
    desc = tmp_path / "target_experiment.yaml"
    desc.write_text(f"target: synthnodramrow\nhardware_spec:\n  isa_headers:\n    - {card}\n",
                    encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(desc))
    DF._WINDOW_CACHE.pop("synthnodramrow", None)

    base, size, why = DF.dram_window_for("synthnodramrow")
    assert base == 0, "an absent DRAM row must not borrow another region's address"
    assert size is None
    assert "no DRAM region row" in why, f"provenance must name the missing row, got {why!r}"
    assert why != DF.dram_window_for(NO_CARD_TARGET)[2], \
        "a card without a DRAM row must not read the same as a target with no card"


def test_a_backwards_row_is_rejected_rather_than_believed(tmp_path, monkeypatch):
    """end <= start spans no bytes; a nonsense size would bound every address out of the window."""
    card = tmp_path / "card.md"
    card.write_text("| `DRAM` | `0x8000_0000` ~ `0x1000` |\n", encoding="utf-8")
    desc = tmp_path / "target_experiment.yaml"
    desc.write_text(f"target: synthbackwards\nhardware_spec:\n  isa_headers:\n    - {card}\n",
                    encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(desc))
    DF._WINDOW_CACHE.pop("synthbackwards", None)

    base, size, why = DF.dram_window_for("synthbackwards")
    assert base == 0x8000_0000
    assert size is None
    assert "no bytes" in why.lower() or "unusable" in why.lower()


def test_every_hex_token_in_the_row_is_tokenized_not_just_the_first():
    """The defect in one line: `_first_hex` was the only reader of a two-address cell."""
    assert DF._hex_tokens("`0x8000_0000` ~ `0x8_8000_0000`") == [0x8000_0000, 0x8_8000_0000]
    assert DF._first_hex("`0x8000_0000` ~ `0x8_8000_0000`") == 0x8000_0000
    assert DF._hex_tokens("no addresses here") == []


@pytest.mark.parametrize("target,expected", [(CARD_TARGET, 0x8000_0000), (NO_CARD_TARGET, 0)])
def test_dram_base_for_is_byte_identical_for_existing_callers(target, expected):
    """Six call sites relocate real addresses through this; the window work must not move any of them."""
    assert DF.dram_base_for(target) == expected


# --- (C) the screen's finding names the reason ------------------------------------------------------

def _facts(**kw) -> SiliconFacts:
    base = dict(
        target="synthtest", mesh_rows=16, mesh_cols=16,
        scratchpad_bytes=262144, scratchpad_rows=4096,
        accumulator_bytes=65536, accumulator_rows=1024,
        accumulator_row_bytes=64, acc_ctrl_mask=0xE0000000,
        legal_funct=[2, 3, 4], custom_opcode=0x7B, funct3=3,
        dram_base=0x8000_0000, provenance="synthetic",
    )
    base.update(kw)
    return SiliconFacts(**base)


def _movement_trace():
    return {"source": "synthetic", "abi": {}, "summary": {"class_histogram": {}}, "instructions": [
        {"index": 0, "class": "MVIN", "funct": 2, "rs1": None, "rs2": None,
         "decoded": {"spad_addr": 0, "rows": 16,
                     "dram": {"kind": "argbase", "arg_index": 0, "offset": 0}}},
        {"index": 1, "class": "MVOUT", "funct": 3, "rs1": None, "rs2": None,
         "decoded": {"acc_addr": 0, "dram": {"kind": "argbase", "arg_index": 1, "offset": 0}}},
        {"index": 2, "class": "FENCE", "funct": 1, "rs1": None, "rs2": None, "decoded": {}},
    ]}


def _rules(findings):
    return {f.rule for f in findings}


def test_a_derived_window_suppresses_the_unknown_finding():
    findings, peaks = simulate(_movement_trace(), _facts(), dram_bytes=32 * _GIB,
                              dram_window_why=DF.dram_window_for(CARD_TARGET)[2])
    assert "dram-window-unknown" not in _rules(findings), \
        "a window that WAS derived must not be reported unknown"
    assert peaks["dram_window_bytes"] == 32 * _GIB, "the window must reach resource_peaks"


def test_an_undecidable_window_still_fires_and_names_the_reason():
    why = DF.dram_window_for(NO_CARD_TARGET)[2]
    findings, peaks = simulate(_movement_trace(), _facts(), dram_bytes=None, dram_window_why=why)
    unk = [f for f in findings if f.rule == "dram-window-unknown"]
    assert unk, "an underivable window must still be surfaced (fail closed)"
    assert unk[0].severity == Severity.UNKNOWN
    assert why in unk[0].message, f"the message must carry the reason, got {unk[0].message!r}"
    assert why in (unk[0].derived_from or ""), \
        f"derived_from must carry the reason, got {unk[0].derived_from!r}"
    assert peaks["dram_window_bytes"] is None


def test_a_caller_that_supplies_no_reason_keeps_the_historical_wording():
    """The new parameter defaults to today's behaviour, so no existing caller changes shape."""
    findings, _ = simulate(_movement_trace(), _facts())
    unk = [f for f in findings if f.rule == "dram-window-unknown"]
    assert unk and unk[0].derived_from == "dram_facts (base only)"
    assert unk[0].message.startswith("DRAM window size not supplied")


def test_the_reason_threads_through_the_program_and_assess(monkeypatch):
    """`assess` is the entry point the capsule runner calls; the field must not stop at `simulate`."""
    monkeypatch.setattr("merlin.liveness.oracle.silicon_facts", lambda t: _facts(target=t))
    rep = assess(Program(name="p", trace=_movement_trace(), address_model="fixed_preload",
                         dram_bytes=None, dram_window_why="SENTINEL-REASON"), "synthtest")
    unk = [f for f in rep.findings if f.rule == "dram-window-unknown"]
    assert unk and "SENTINEL-REASON" in unk[0].message


# --- (B) the runner actually supplies the fact ------------------------------------------------------

def test_the_capsule_runner_hands_the_derived_window_to_the_screen():
    """The fact was derivable all along and the ONE call site never passed it.

    Checked structurally over the AST rather than by substring, and checked for *derivation*: a literal
    would be a fabricated window, which is the thing this must never become.
    """
    import ast

    from merlin.common.paths import merlin_dir

    src = (merlin_dir() / "python" / "merlin" / "targetgen" / "capsule_runner.py").read_text()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_LProg"]
    assert calls, "the liveness screen's Program construction site vanished"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "dram_bytes" in kw, "the derived DRAM window size must be supplied"
        assert "dram_window_why" in kw, "so must the provenance of that derivation"
        for name in ("dram_bytes", "dram_window_why"):
            assert not isinstance(kw[name], ast.Constant), \
                f"{name} must be DERIVED, never a literal (a baked window fabricates the fact)"
