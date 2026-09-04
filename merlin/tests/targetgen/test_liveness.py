"""Tests for the HW-agnostic liveness oracle (merlin.liveness): the static precondition linter (B) and
the dynamic transaction-level model (A). Hermetic — synthetic facts + traces, no RTL toolchain."""
from __future__ import annotations

from merlin.liveness import Program, Severity, assess, silicon_facts
from merlin.liveness.facts import SiliconFacts
from merlin.liveness.interconnect import simulate
from merlin.liveness.preconditions import funct_legality, host_assist, medany_span, vlen_match
from merlin.liveness.report import Finding, LivenessReport


# --- helpers ----------------------------------------------------------------------------------------

def _facts(**kw) -> SiliconFacts:
    base = dict(
        target="synthtest", mesh_rows=16, mesh_cols=16,
        scratchpad_bytes=262144, scratchpad_rows=4096,
        accumulator_bytes=65536, accumulator_rows=1024,
        accumulator_row_bytes=64, acc_ctrl_mask=0xE0000000,
        legal_funct=[2, 3, 4, 8, 9, 10, 14, 15, 16, 18, 19, 23, 24, 25, 27],
        custom_opcode=0x7B, funct3=3, dram_base=0x0, provenance="synthetic",
    )
    base.update(kw)
    return SiliconFacts(**base)


def _ins(idx, cls, funct=None, **decoded):
    return {"index": idx, "class": cls, "funct": funct, "rs1": None, "rs2": None, "decoded": decoded}


def _trace(instructions):
    return {"source": "synthetic", "abi": {}, "instructions": instructions,
            "summary": {"class_histogram": {}}}


# --- (A) interconnect: scratchpad footprint ---------------------------------------------------------

def test_scratchpad_within_capacity_is_clean():
    tr = _trace([
        _ins(0, "FENCE", funct=1),
        _ins(1, "MVIN", funct=2, spad_addr=0, rows=16, dram={"kind": "argbase", "arg_index": 0, "offset": 0}),
        _ins(2, "MVIN", funct=2, spad_addr=16, rows=16, dram={"kind": "argbase", "arg_index": 1, "offset": 0}),
        _ins(3, "MVOUT", funct=3, acc_addr=0, dram={"kind": "argbase", "arg_index": 2, "offset": 0}),
        _ins(4, "FENCE", funct=1),
    ])
    findings, peaks = simulate(tr, _facts())
    assert peaks["scratchpad_max_row"] == 32
    assert not [f for f in findings if f.rule == "scratchpad-overflow"]


def test_scratchpad_overflow_detected():
    # capacity 64 rows; an MVIN reaches row 4096 -> overflow (stall on silicon)
    tr = _trace([
        _ins(0, "FENCE", funct=1),
        _ins(1, "MVIN", funct=2, spad_addr=4090, rows=16, dram={"kind": "argbase", "arg_index": 0, "offset": 0}),
        _ins(2, "FENCE", funct=1),
    ])
    findings, peaks = simulate(tr, _facts(scratchpad_rows=64))
    over = [f for f in findings if f.rule == "scratchpad-overflow"]
    assert over and over[0].severity == Severity.STALL
    assert peaks["scratchpad_max_row"] == 4106


def test_scratchpad_capacity_unknown_is_failclosed():
    tr = _trace([_ins(0, "MVIN", funct=2, spad_addr=10, rows=4, dram={"kind": "argbase"})])
    findings, _ = simulate(tr, _facts(scratchpad_rows=None))
    unk = [f for f in findings if f.rule == "scratchpad-capacity"]
    assert unk and unk[0].severity == Severity.UNKNOWN


# --- (A) interconnect: accumulator footprint --------------------------------------------------------

def test_accumulator_within_capacity_is_clean():
    # acc_addr carries control bits (0x80000000) but a low row index -> masked row small -> no overflow
    tr = _trace([
        _ins(0, "MVOUT", funct=3, acc_addr=0x80000005, dram={"kind": "argbase"}),
        _ins(1, "FENCE", funct=1),
    ])
    findings, peaks = simulate(tr, _facts())
    assert peaks["accumulator_max_row"] == 5
    assert not [f for f in findings if f.rule == "accumulator-overflow"]


def test_accumulator_overflow_detected():
    tr = _trace([
        _ins(0, "MVOUT", funct=3, acc_addr=0x80000000 | 2000, dram={"kind": "argbase"}),  # row 2000 > 1024
        _ins(1, "FENCE", funct=1),
    ])
    findings, _ = simulate(tr, _facts())
    over = [f for f in findings if f.rule == "accumulator-overflow"]
    assert over and over[0].severity == Severity.STALL


def test_accumulator_unknown_when_mask_missing():
    tr = _trace([_ins(0, "MVOUT", funct=3, acc_addr=0x80000005, dram={"kind": "argbase"})])
    findings, _ = simulate(tr, _facts(acc_ctrl_mask=None))
    unk = [f for f in findings if f.rule == "accumulator-capacity"]
    assert unk and unk[0].severity == Severity.UNKNOWN


# --- (A) interconnect: DRAM address-map -------------------------------------------------------------

def test_dram_unmapped_below_base_is_fault():
    tr = _trace([
        _ins(0, "MVIN", funct=2, spad_addr=0, rows=1,
             dram={"kind": "const", "raw": 0x40, "arg_index": None, "offset": None}),
    ])
    findings, _ = simulate(tr, _facts(dram_base=0x80000000), dram_bytes=0x1000)
    bad = [f for f in findings if f.rule == "dram-unmapped"]
    assert bad and bad[0].severity == Severity.FAULT


def test_dram_const_under_pointer_args_is_provenance_fault():
    tr = _trace([
        _ins(0, "MVOUT", funct=3, acc_addr=0,
             dram={"kind": "const", "raw": 0x1000, "arg_index": None, "offset": None}),
    ])
    findings, _ = simulate(tr, _facts(), address_model="pointer_args")
    prov = [f for f in findings if f.rule == "dram-provenance"]
    assert prov and prov[0].severity == Severity.FAULT


def test_dram_argbase_is_clean():
    tr = _trace([
        _ins(0, "MVIN", funct=2, spad_addr=0, rows=1,
             dram={"kind": "argbase", "arg_index": 0, "offset": 0}),
        _ins(1, "FENCE", funct=1),
    ])
    findings, _ = simulate(tr, _facts(), address_model="pointer_args")
    # argbase movement raises no provenance/unmapped fault; the only dram finding allowed is the honest
    # UNKNOWN that the DRAM window size was not supplied (upper bound unchecked).
    assert not [f for f in findings if f.rule in ("dram-provenance", "dram-unmapped", "dram-provenance-unknown")]
    assert all(f.severity == Severity.UNKNOWN for f in findings if f.rule.startswith("dram-"))


def test_dram_unknown_provenance_is_unknown():
    tr = _trace([
        _ins(0, "MVIN", funct=2, spad_addr=0, rows=1,
             dram={"kind": "unknown", "raw": None, "arg_index": None, "offset": None}),
    ])
    findings, _ = simulate(tr, _facts())
    unk = [f for f in findings if f.rule == "dram-provenance-unknown"]
    assert unk and unk[0].severity == Severity.UNKNOWN


# --- (A) interconnect: visibility / drain -----------------------------------------------------------

def test_no_closing_fence_is_warn():
    tr = _trace([
        _ins(0, "FENCE", funct=1),
        _ins(1, "MVIN", funct=2, spad_addr=0, rows=1, dram={"kind": "argbase"}),
        _ins(2, "MVOUT", funct=3, acc_addr=0, dram={"kind": "argbase"}),  # no closing FENCE
    ])
    findings, peaks = simulate(tr, _facts())
    vis = [f for f in findings if f.rule == "visibility-no-drain"]
    assert vis and vis[0].severity == Severity.WARN
    assert peaks["closes_with_fence"] is False


def test_closing_fence_is_clean():
    tr = _trace([
        _ins(0, "MVIN", funct=2, spad_addr=0, rows=1, dram={"kind": "argbase"}),
        _ins(1, "MVOUT", funct=3, acc_addr=0, dram={"kind": "argbase"}),
        _ins(2, "FENCE", funct=1),
    ])
    findings, peaks = simulate(tr, _facts())
    assert not [f for f in findings if f.rule == "visibility-no-drain"]
    assert peaks["closes_with_fence"] is True


# --- (B) preconditions: funct legality --------------------------------------------------------------

def test_funct_legality_pass():
    tr = _trace([_ins(0, "MVIN", funct=2), _ins(1, "MVOUT", funct=3)])
    assert funct_legality(tr, _facts()) == []


def test_funct_legality_illegal_is_fault():
    tr = _trace([_ins(0, "MVIN", funct=2), _ins(1, "UNKNOWN", funct=99)])
    out = funct_legality(tr, _facts())
    assert out and out[0].severity == Severity.FAULT
    assert 99 in out[0].evidence["illegal_functs"]


def test_funct_legality_unknown_when_legalset_missing():
    tr = _trace([_ins(0, "MVIN", funct=2)])
    out = funct_legality(tr, _facts(legal_funct=None))
    assert out and out[0].severity == Severity.UNKNOWN


# --- (B) preconditions: host-assist -----------------------------------------------------------------

def test_host_assist_htif_on_hostless_is_fault():
    out = host_assist(hostless=True, has_htif=True)
    assert out and out[0].severity == Severity.FAULT


def test_host_assist_htif_on_host_is_clean():
    assert host_assist(hostless=False, has_htif=True) == []


def test_host_assist_tohost_symbol_on_hostless_is_fault():
    out = host_assist(hostless=True, has_htif=False, has_tohost=True)
    assert out and out[0].severity == Severity.FAULT


def test_host_assist_unknown_when_unsupplied():
    out = host_assist(hostless=None, has_htif=None)
    assert out and out[0].severity == Severity.UNKNOWN


# --- (B) preconditions: vlen-match ------------------------------------------------------------------

def test_vlen_match_ok():
    assert vlen_match(declared_vlen=256, hw_vlen=256) == []


def test_vlen_mismatch_is_fault():
    out = vlen_match(declared_vlen=128, hw_vlen=256)
    assert out and out[0].severity == Severity.FAULT


def test_vlen_not_applicable_when_hw_none():
    assert vlen_match(declared_vlen=128, hw_vlen=None) == []  # not a vector target


def test_vlen_unknown_when_declared_missing():
    out = vlen_match(declared_vlen=None, hw_vlen=256)
    assert out and out[0].severity == Severity.UNKNOWN


# --- (B) preconditions: medany-span -----------------------------------------------------------------

def test_medany_span_within_window_ok():
    assert medany_span(uses_medany=True, image_span_bytes=256 << 20) == []  # 256 MB, well within


def test_medany_span_exceeds_window_is_fault():
    out = medany_span(uses_medany=True, image_span_bytes=(1 << 31) + 1)  # >2 GB
    assert out and out[0].severity == Severity.FAULT


def test_medany_span_over_half_is_warn():
    out = medany_span(uses_medany=True, image_span_bytes=(1 << 30) + (1 << 29))  # 1.5 GB
    assert out and out[0].severity == Severity.WARN


def test_medany_not_applicable_when_not_medany():
    assert medany_span(uses_medany=False, image_span_bytes=1 << 32) == []
    assert medany_span(uses_medany=None, image_span_bytes=1 << 32) == []


# --- report / verdict -------------------------------------------------------------------------------

def test_verdict_ranking():
    r = LivenessReport(target="t", program="p")
    assert r.verdict == "ok"
    r.add(Finding("a", Severity.WARN, "w"))
    assert r.verdict == "warn"
    r.add(Finding("b", Severity.UNKNOWN, "u"))
    assert r.verdict == "unknown"
    r.add(Finding("c", Severity.STALL, "s"))
    assert r.verdict == "stall"
    r.add(Finding("d", Severity.FAULT, "f"))
    assert r.verdict == "fault"
    d = r.to_dict()
    assert d["verdict"] == "fault"
    assert d["counts"]["fault"] == 1


# --- facts + assess end-to-end (fail-closed) --------------------------------------------------------

def test_silicon_facts_failclosed_on_unknown_target():
    f = silicon_facts("definitely_not_a_real_target_xyz")
    assert "mesh_rows" in f.unknowns() and "legal_funct" in f.unknowns()
    # must not raise; provenance records the degradation
    assert f.dram_base is None or isinstance(f.dram_base, int)


def test_assess_end_to_end_merges_checks():
    tr = _trace([
        _ins(0, "MVIN", funct=99, spad_addr=0, rows=1, dram={"kind": "argbase"}),  # illegal funct
        _ins(1, "MVOUT", funct=3, acc_addr=0, dram={"kind": "argbase"}),           # no closing fence
    ])
    # Inject synthetic facts by monkeypatching is overkill; instead rely on the unknown-target
    # fail-closed path for facts and assert the trace-derived findings still fire where facts allow.
    prog = Program(name="synth", trace=tr, hostless=True, has_htif=True)
    rep = assess(prog, "definitely_not_a_real_target_xyz")
    rules = {f.rule for f in rep.findings}
    # funct legality is UNKNOWN (no legal set derivable), host-assist is FAULT, visibility is STALL
    assert "host-assist" in rules
    assert rep.verdict in ("fault", "stall", "unknown")


def test_a_named_float_datapath_is_not_read_as_a_width_of_its_digits():
    """A registry format name is not a width with letters around it.

    ``_dtype_bits`` used to concatenate a token's digit runs, which is right for the machine spellings
    (``i8``, ``i32``) and catastrophic for a registered one: ``fp8_e4m3`` reads as 843 bits, sizing a
    row 105x too wide so every capacity check reports a fit. Both readings are pinned here, plus the
    rule that a datapath's own measured ``elem_bits`` outranks any reading of its name.
    """
    from merlin.liveness.facts import _datapath_bits, _dtype_bits

    assert _dtype_bits("fp8_e4m3") == 8 and _dtype_bits("bf16") == 16
    assert _dtype_bits("i8") == 8 and _dtype_bits("i32") == 32
    assert _dtype_bits("not-a-dtype") is None
    facts = {"datapaths": [{"name": "input", "dtype": "fp8_e4m3", "elem_bits": 8},
                           {"name": "accumulator", "dtype": "bf16", "elem_bits": 16}]}
    assert _datapath_bits(facts, "input") == 8
    assert _datapath_bits(facts, "accumulator") == 16
    assert _datapath_bits(facts, "absent") is None
