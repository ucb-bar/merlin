"""A corpus's ISA coverage is a number with a DERIVED denominator, and the denominator is not curated."""
import struct

import pytest

from merlin.targetgen import isa_corpus_coverage as C
from merlin.targetgen import isa_disasm
from merlin.targetgen.isa_model import IsaModel


def _model():
    """Two ops of ONE class that differ only in the unit they drive, plus an unrelated class.

    This is the shape the check exists for: at class granularity ``UNIT_A``/``UNIT_B`` are one entry, so a
    corpus can exercise only one of them and still look complete.
    """
    return IsaModel(target="t", by_mnemonic={
        "MM_UNIT_A": {"class": "MatMul", "mnemonic": "MM_UNIT_A", "role": "matmul",
                      "fixed_mask": 0xFF, "fixed_value": 0x01, "fields": {}},
        "MM_UNIT_B": {"class": "MatMul", "mnemonic": "MM_UNIT_B", "role": "matmul",
                      "fixed_mask": 0xFF, "fixed_value": 0x02, "fields": {}},
        "MOVE": {"class": "Move", "mnemonic": "MOVE", "role": "memory",
                 "fixed_mask": 0xFF, "fixed_value": 0x03, "fields": {}},
    }, roles={"matmul": ["MatMul"], "memory": ["Move"]})


def _kernel(tmp_path, name, words):
    p = tmp_path / name
    p.write_bytes(struct.pack(f"<{len(words)}I", *words))
    return p


def test_disassembly_reports_both_granularities():
    """The exact mnemonic must survive decode; reporting only the class is what made a unit invisible."""
    recs = isa_disasm.disassemble(_model(), [0x01, 0x02])
    assert [r["class"] for r in recs] == ["MatMul", "MatMul"]
    assert [r["isa_mnemonic"] for r in recs] == ["MM_UNIT_A", "MM_UNIT_B"]
    assert isa_disasm.present_classes(recs) == ["MatMul"]
    assert isa_disasm.present_mnemonics(recs) == ["MM_UNIT_A", "MM_UNIT_B"]


def test_class_granularity_demand_cannot_distinguish_units():
    """A demand written as the CLASS is satisfied by either unit -- the unfalsifiable case, stated."""
    recs = isa_disasm.disassemble(_model(), [0x01])           # unit A only
    assert isa_disasm.coverage(_model(), recs, required=["MatMul"])["missing"] == []


def test_mnemonic_granularity_demand_is_falsifiable():
    """Naming the exact op makes the same kernel FAIL, which is the whole point of the finer axis."""
    recs = isa_disasm.disassemble(_model(), [0x01])           # unit A only
    cov = isa_disasm.coverage(_model(), recs, required=["MM_UNIT_B"])
    assert cov["missing"] == ["MM_UNIT_B"]


def test_uncovered_unit_is_reported(tmp_path):
    m = _model()
    k = {"cap0": _kernel(tmp_path, "a.bin", [0x01, 0x03])}    # unit A + move; unit B never driven
    cov = C.corpus_coverage(m, k)
    assert cov["n_universe"] == 3 and cov["n_covered"] == 2
    assert cov["uncovered"] == ["MM_UNIT_B"]
    assert cov["by_class"]["MatMul"]["n_covered"] == 1 and cov["by_class"]["MatMul"]["n_total"] == 2


def test_denominator_is_not_curated(tmp_path):
    """Every derived op counts. A filtered universe is how a real hole gets excluded from the ratio --
    measured: one target files its DMA ops under the scalar role, so excluding scalar hid 32 of them."""
    m = _model()
    m.by_mnemonic["SCALAR_THING"] = {"class": "Sc", "mnemonic": "SCALAR_THING", "role": "scalar",
                                     "fixed_mask": 0xFF, "fixed_value": 0x04, "fields": {}}
    assert "SCALAR_THING" in C.universe(m)
    cov = C.corpus_coverage(m, {"c": _kernel(tmp_path, "b.bin", [0x01])})
    assert "SCALAR_THING" in cov["uncovered"]


def test_empty_isa_reports_no_ratio_not_full_coverage():
    """A target that ships no ISA must not read as fully covered -- the flattering direction of failure."""
    cov = C.corpus_coverage(IsaModel(target="t"), {})
    assert cov["ratio"] is None and cov["n_universe"] == 0
    assert "(n/a)" in C.render_markdown(cov)


def test_ambiguous_decode_is_surfaced_not_resolved(tmp_path):
    """Coverage credited from a word matching several signatures is not evidence; it must be visible."""
    m = _model()
    m.by_mnemonic["OVERLAP"] = {"class": "MatMul", "mnemonic": "OVERLAP", "role": "matmul",
                                "fixed_mask": 0x0F, "fixed_value": 0x01, "fields": {}}
    cov = C.corpus_coverage(m, {"c": _kernel(tmp_path, "c.bin", [0x01])})
    assert cov["ambiguous_decodes"]["c"]
    assert "Ambiguous decodes" in C.render_markdown(cov)


def test_words_of_drops_trailing_partial_word(tmp_path):
    p = tmp_path / "odd.bin"
    p.write_bytes(struct.pack("<2I", 0x01, 0x02) + b"\xff")
    assert C.words_of(p) == [0x01, 0x02]


def test_words_of_rejects_unsupported_width(tmp_path):
    p = tmp_path / "w.bin"
    p.write_bytes(b"\x00" * 8)
    with pytest.raises(ValueError):
        C.words_of(p, inst_width=24)
