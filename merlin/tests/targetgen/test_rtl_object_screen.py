"""The emitted-object check family: a compiler that emits a lowering must still get screened.

The defect these pin: :mod:`merlin.targetgen.rtl_check_runner` had two check families, one keyed on
``generated/kernel.S`` and one on ``generated/instruction_trace.json``. A target whose codegen endpoint is
an LLVM-dialect MLIR module writes neither — its machine code exists only after a toolchain compiles the
lowering — so ``screen_run`` returned ``None`` for every capsule and the advisory came out ``[]`` beside an
approving note. Measured: 18 consecutive rounds across three repeats, read the whole time as "nothing
wrong".

Every model here is SYNTHETIC and shares no value with any shipped target: the checks must fall out of a
model's own derived facts, so a test model with a different layout, different spaces and different control
ops has to work identically. A check whose derivation is missing must be DROPPED — neither passed nor
failed — which is the property the `checks-that-skip-and-report-success` class is about.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import isa_transcode as TX
from merlin.targetgen import rtl_object_screen as OS
from merlin.targetgen.isa_model import IsaModel

# A synthetic 64-bit fixed-format layout. Deliberately NOT any shipped target's: the opcode field is the
# base ISA's own (that is what makes the standard opcode windows applicable at all), everything else moved.
_LAYOUT = {"opcode": (6, 0), "space": (8, 7), "rd": (16, 9), "f3": (19, 17), "rs1": (27, 20),
           "rs2": (35, 28), "f7": (58, 52), "imm": (59, 36)}
#: The target's OWN status registers vs the architectural ones, told apart by PROVENANCE, never a list.
_CSRS = {"lane_no": 3000, "stream_no": 3001, "mstatus": 768}
_CSR_PROV = {"csr.lane_no": "rtl:src", "csr.stream_no": "rtl:src", "csr.mstatus": "spec:riscv:mstatus"}
#: Standard RISC-V opcode values — LOAD / STORE / OP / OP_FP / MADD / CUSTOM0 / SYSTEM.
_OPCODES = {"LOAD": 0x03, "STORE": 0x23, "OP": 0x33, "OP_FP": 0x53, "MADD": 0x43,
            "CUSTOM0": 0x0B, "SYSTEM": 0x73}
_SFU = {
    "spin": {"opcode": 0x0B, "funct3": 0},        # a control op, in the custom window
    "meet": {"opcode": 0x0B, "funct3": 3},        # another
    "csrrw": {"opcode": 0x73, "funct3": 1},       # a CSR access — NOT warp control, must be excluded
}
_SIMT_FACTS = {"facts": {"simt": {"lanes_per_warp": 4, "warps_per_core": 3, "cores": 5}}}


def _model(**over) -> IsaModel:
    kw = dict(target="probe", inst_width=64, field_layout=dict(_LAYOUT), opcode_table=dict(_OPCODES),
              address_spaces={"near": 0, "far": 1}, address_space_field="space",
              runtime_abi={"base_isa_family": "riscv32", "sfu_ops": dict(_SFU),
                           "special_csrs": dict(_CSRS), "provenance": dict(_CSR_PROV)})
    kw.update(over)
    return IsaModel(**kw)


def _word(opcode: int, *, f3: int = 0, space: int = 0, rd: int = 1, rs1: int = 2, imm: int = 0) -> int:
    w = opcode & 0x7F
    w |= (space & 0x3) << 7
    w |= (rd & 0xFF) << 9
    w |= (f3 & 0x7) << 17
    w |= (rs1 & 0xFF) << 20
    w |= (imm & 0xFFFFFF) << 36
    return w


def _csr_read(name: str) -> int:
    """A CSR access to *name*, packed the way the target's own transcoder packs an immediate."""
    return _word(_SFU["csrrw"]["opcode"], f3=_SFU["csrrw"]["funct3"], imm=_CSRS[name])


# --------------------------------------------------------------------------- base-ISA opcode grouping
def test_opcode_groups_come_from_the_targets_own_table():
    g = TX.base_isa_opcode_groups(_model())
    assert g["memory"] == frozenset({0x03, 0x23})
    assert g["memory_load"] == frozenset({0x03}) and g["memory_store"] == frozenset({0x23})
    assert g["fp_compute"] == frozenset({0x53})
    assert g["fused_multiply_add"] == frozenset({0x43})
    assert g["custom_extension"] == frozenset({0x0B})
    # A form the target's decoder does not define gets no entry at all, so `in` is a real capability test.
    assert "ordering" not in g and "control_flow" not in g


def test_opcode_groups_refuse_a_substrate_the_standard_values_do_not_describe():
    """The standard windows are a fact about RISC-V. Applying them to another substrate would be exactly the
    assumed-encoding the repo forbids, so the grouping returns nothing rather than something plausible."""
    assert TX.base_isa_opcode_groups(_model(runtime_abi={"base_isa_family": "somethingelse"})) == {}
    assert TX.base_isa_opcode_groups(_model(runtime_abi={})) == {}
    assert TX.base_isa_opcode_groups(_model(field_layout={}, opcode_table={})) == {}


def test_opcode_groups_mask_a_table_value_wider_than_the_opcode_field():
    """An opcode-table entry may carry an extension selector above the field; a decoded word never does. So
    the comparison happens at the FIELD's width, the same way the disassembler's reverse map does."""
    g = TX.base_isa_opcode_groups(_model(opcode_table={"EXT": 0x10B}))
    assert g == {"custom_extension": frozenset({0x0B})}


# --------------------------------------------------------------------------- derivations
def test_control_ops_are_the_sfu_ops_inside_the_custom_window():
    sigs = OS.simt_control_signatures(_model())
    assert sigs == {(0x0B, 0): "spin", (0x0B, 3): "meet"}, "a CSR-access sfu_op is not warp control"


def test_control_ops_are_empty_rather_than_guessed_when_the_abi_derives_none():
    assert OS.simt_control_signatures(_model(runtime_abi={"base_isa_family": "riscv32"})) == {}
    # …and with no custom window in the target's own table there is nothing to look in.
    assert OS.simt_control_signatures(_model(opcode_table={"OP": 0x33})) == {}


def test_instruction_streams_is_warps_times_cores_and_none_without_the_geometry():
    assert OS.simt_instruction_streams(_SIMT_FACTS) == 15
    assert OS.simt_instruction_streams({"facts": {}}) is None
    assert OS.simt_instruction_streams({"facts": {"simt": {"warps_per_core": 4}}}) is None


# --------------------------------------------------------------------------- the screen
def test_a_serial_kernel_is_reported_against_the_machines_real_stream_count():
    words = [_word(0x03), _word(0x53), _word(0x23)]
    rep = OS.screen(words, _model(), _SIMT_FACTS)
    ctl = next(c for c in rep["checks"] if c["id"] == "simt_control")
    assert ctl["status"] == "fail" and ctl["severity"] == "warning"
    assert "1 of this machine's 15" in ctl["message"]
    assert "spin" in ctl["message"] and "meet" in ctl["message"]
    assert "identity registers" in ctl["message"], "both kinds of evidence are absent, so say both"
    assert rep["metrics"]["simt_control_count"] == 0
    assert rep["metrics"]["identity_csr_read_count"] == 0
    # The hint must name the TARGET'S OWN registers; an architectural CSR says nothing about which stream
    # this is, and listing one sends the reader to the wrong register.
    assert "lane_no" in ctl["fix_hint"] and "stream_no" in ctl["fix_hint"]
    assert "mstatus" not in ctl["fix_hint"]
    # advisory, never a reject: a single-stream kernel is correct, only serial.
    assert rep["verdict"] == "warn"


def test_a_kernel_that_uses_the_control_ops_passes_that_check():
    words = [_word(0x0B, f3=0), _word(0x03), _word(0x0B, f3=3)]
    rep = OS.screen(words, _model(), _SIMT_FACTS)
    ctl = next(c for c in rep["checks"] if c["id"] == "simt_control")
    assert ctl["status"] == "pass"
    assert rep["metrics"]["simt_control_ops"] == {"meet": 1, "spin": 1}


def test_identity_registers_are_the_ones_provenance_attributes_to_the_target():
    hits, why = OS.identity_csr_reads([_csr_read("lane_no"), _csr_read("lane_no"),
                                       _csr_read("mstatus"), _word(0x03)], _model())
    assert why == ""
    assert hits == {"lane_no": 2}, "an architectural CSR is not an identity register"


def test_reading_the_identity_registers_counts_as_using_the_machine():
    """A kernel can be parallel by reading WHICH lane it is rather than by spawning, so either kind of
    evidence satisfies the check — and the metric is reported separately either way."""
    rep = OS.screen([_csr_read("stream_no"), _word(0x03)], _model(), _SIMT_FACTS)
    assert next(c for c in rep["checks"] if c["id"] == "simt_control")["status"] == "pass"
    assert rep["metrics"]["identity_csr_reads"] == {"stream_no": 1}
    assert "SIMT_IDENTITY_READS 1" in OS.render(rep)


def test_the_identity_count_refuses_a_field_too_narrow_to_hold_the_csr_numbers():
    """The trap this check nearly shipped with. Read from a field that cannot represent the target's own CSR
    numbers and the count is 0 for every kernel — which reads as the finding "nothing reads its warp id"
    when in fact nothing could ever match. Measured on a real target: an 8-bit field, 15 CSRs all >4000."""
    narrow = {k: v for k, v in _LAYOUT.items() if k != "imm"}      # widest remaining field is 8 bits
    hits, why = OS.identity_csr_reads([_csr_read("lane_no")], _model(field_layout=narrow))
    assert hits is None, "a count that cannot match must not be reported as zero"
    assert "too narrow" in why and "0 by construction" in why

    rep = OS.screen([_csr_read("lane_no")], _model(field_layout=narrow), _SIMT_FACTS)
    assert "simt_identity" in rep["dropped"]
    assert "identity_csr_read_count" not in rep["metrics"]
    assert "SIMT_IDENTITY_READS -" in OS.render(rep)
    # …and the surviving control finding must not claim anything about identity reads.
    ctl = next(c for c in rep["checks"] if c["id"] == "simt_control")
    assert "identity" not in (ctl["message"] or "")


def test_identity_reads_are_dropped_with_a_reason_when_no_csr_is_attributed_to_the_target():
    hits, why = OS.identity_csr_reads([_word(0x03)], _model(
        runtime_abi={"base_isa_family": "riscv32", "sfu_ops": dict(_SFU),
                     "special_csrs": {"mstatus": 768}, "provenance": {"csr.mstatus": "spec:riscv:mstatus"}}))
    assert hits is None and "own RTL sources" in why


def test_the_control_check_is_dropped_not_answered_when_the_facts_carry_no_geometry():
    """The whole point of the ledger: a check with no derivation must appear in ``dropped`` with a reason,
    and must NOT appear in ``checks`` at all — not as a pass (which reads as evidence) and not as a fail."""
    rep = OS.screen([_word(0x03)], _model(), {"facts": {}})
    assert "simt_control" not in [c["id"] for c in rep["checks"]]
    assert "simt_control" in rep["dropped"]
    assert "SIMT geometry" in rep["dropped"]["simt_control"]
    assert "simt_control" not in rep["grounded"]


def test_a_kernel_that_never_leaves_the_default_space_is_reported():
    words = [_word(0x03, space=0), _word(0x23, space=0), _word(0x33, space=1)]
    rep = OS.screen(words, _model(), _SIMT_FACTS)
    sp = next(c for c in rep["checks"] if c["id"] == "address_space_use")
    assert sp["status"] == "fail"
    assert "never reaches far" in sp["message"]
    # only MEMORY accesses count — the non-memory word carrying space=1 must not launder the finding.
    assert rep["metrics"]["memory_accesses"] == 2
    assert rep["metrics"]["accesses_by_space"] == {"far": 0, "near": 2}


def test_a_kernel_that_reaches_the_other_space_passes():
    rep = OS.screen([_word(0x03, space=1), _word(0x23, space=0)], _model(), _SIMT_FACTS)
    assert next(c for c in rep["checks"] if c["id"] == "address_space_use")["status"] == "pass"


def test_the_space_check_is_dropped_for_a_single_flat_address_space():
    rep = OS.screen([_word(0x03)], _model(address_spaces={}, address_space_field=""), _SIMT_FACTS)
    assert "address_space_use" not in [c["id"] for c in rep["checks"]]
    assert "address_space" in rep["dropped"]


def test_an_undecodable_word_is_an_error_and_rejects():
    rep = OS.screen([_word(0x03), _word(0x71)], _model(), _SIMT_FACTS)
    legality = next(c for c in rep["checks"] if c["id"] == "decode_legality")
    assert legality["status"] == "fail" and legality["got"] == 1
    assert rep["verdict"] == "reject"


def test_legality_says_when_it_is_an_attestation_rather_than_a_discovery():
    """An emit path that already rejects an undecodable word makes this check unfalsifiable. Reporting 0 as
    if it were a finding is the vacuous-gate failure mode, so the screen records which one it is."""
    plain = OS.screen([_word(0x03)], _model(), _SIMT_FACTS)
    attested = OS.screen([_word(0x03)], _model(), _SIMT_FACTS, lint_enforced=True)
    assert plain["metrics"]["legality_attested_at_emit"] is False
    assert attested["metrics"]["legality_attested_at_emit"] is True
    assert "attestation, not a discovery" in attested["grounded"]["decode_legality"]
    assert "attestation" not in plain["grounded"]["decode_legality"]
    assert "LEGALITY_ATTESTED_AT_EMIT yes" in OS.render(attested)


def test_a_non_fixed_format_model_drops_the_whole_family_instead_of_passing_it():
    rep = OS.screen([1, 2, 3], IsaModel(target="probe"), _SIMT_FACTS)
    assert rep["checks"] == [] and rep["verdict"] == "unknown"
    assert "all" in rep["dropped"]


def test_declared_instruction_classes_are_asserted_when_a_capsule_declares_them():
    caps = {"name": "c", "expected": {"instruction_classes": ["OP_FP", "MADD"]}}
    rep = OS.screen([_word(0x03), _word(0x53)], _model(), _SIMT_FACTS, )
    assert "class_coverage" in rep["dropped"]
    rep2 = OS.screen([_word(0x03), _word(0x53)], _model(), _SIMT_FACTS)
    assert rep2["metrics"]["opcode_histogram"] == {"LOAD": 1, "OP_FP": 1}
    rep3 = OS.screen([_word(0x03), _word(0x53)], _model(), _SIMT_FACTS, capsule=caps)
    cov = next(c for c in rep3["checks"] if c["id"] == "class_coverage")
    assert cov["status"] == "fail" and "MADD" in cov["message"]


def test_a_kernel_that_never_writes_cannot_have_produced_its_output():
    """The family's one correctness claim: no memory write means no result. Severity error, so a prescreen
    can skip the oracle run — justified only because it was checked against the corpus first (every one of
    35 oracle-PASSING runs writes at least once, so this cannot fire on a correct kernel)."""
    caps = {"name": "c"}
    rep = OS.screen([_word(0x03), _word(0x53)], _model(), _SIMT_FACTS, capsule=caps)
    rw = next(c for c in rep["checks"] if c["id"] == "result_write")
    assert rw["status"] == "fail" and rw["severity"] == "error" and rw["got"] == 0
    assert rep["verdict"] == "reject"
    assert "MEMORY_WRITES 0" in OS.render(rep)

    ok = OS.screen([_word(0x03), _word(0x23)], _model(), _SIMT_FACTS, capsule=caps)
    assert next(c for c in ok["checks"] if c["id"] == "result_write")["status"] == "pass"
    assert ok["verdict"] == "warn", "the remaining findings are advisory, so it must not reject"


def test_the_write_check_is_dropped_for_a_target_with_no_derivable_write_opcode():
    rep = OS.screen([_word(0x33)], _model(opcode_table={"OP": 0x33}), _SIMT_FACTS, capsule={"name": "c"})
    assert "result_write" not in [c["id"] for c in rep["checks"]]
    assert "result_write" in rep["dropped"]


# --------------------------------------------------------------------------- render / FileCheck
def test_an_undeterminable_metric_renders_a_dash_not_a_passing_zero():
    txt = OS.render(OS.screen([_word(0x03)], _model(), {"facts": {}}))
    assert "SIMT_CONTROL_COUNT -" in txt and "SIMT_STREAMS -" in txt
    # and the reason travels with it, so a reader of the render knows why.
    assert any(ln.startswith("DROPPED simt_control ") for ln in txt.splitlines())


def test_the_filecheck_file_omits_an_assertion_it_could_not_ground():
    grounded = OS.compile_object_checks({"name": "c"}, OS.screen([_word(0x03)], _model(), _SIMT_FACTS))
    assert "DECODE_ILLEGAL 0" in grounded
    bare = OS.compile_object_checks({"name": "c"},
                                    OS.screen([1], IsaModel(target="probe"), _SIMT_FACTS))
    assert bare is None or "DECODE_ILLEGAL" not in bare


def test_declared_classes_become_filecheck_assertions_over_the_render():
    caps = {"name": "c", "expected": {"instruction_classes": ["OP_FP"]}}
    rep = OS.screen([_word(0x53)], _model(), _SIMT_FACTS, capsule=caps)
    checks = OS.compile_object_checks(caps, rep)
    assert "OPCODE_PRESENT OP_FP" in checks
    assert "OPCODE_PRESENT OP_FP" in OS.render(rep)


# --------------------------------------------------------------------------- the artifact
def test_the_words_artifact_round_trips(tmp_path):
    OS.write_words(tmp_path, [1, 2, 3], inst_width=64, source="obj", symbol="k", lint_enforced=True)
    doc = OS.load_words(tmp_path)
    assert doc["words"] == [1, 2, 3] and doc["inst_width"] == 64
    assert doc["source"] == "obj" and doc["symbol"] == "k" and doc["lint_enforced"] is True
    assert doc["schema"] == OS.WORDS_SCHEMA


@pytest.mark.parametrize("body", [None, "not json", '{"inst_width": 64}', '[]'])
def test_a_missing_or_malformed_artifact_reads_as_absent_never_as_empty_success(tmp_path, body):
    if body is not None:
        (tmp_path / OS.WORDS_ARTIFACT).write_text(body)
    assert OS.load_words(tmp_path) is None


# --------------------------------------------------------------------------- routing
def test_screen_run_routes_a_lowering_only_run_to_the_object_family(tmp_path, monkeypatch):
    """The regression itself: a run dir with no kernel.S and no trace, only recorded words, must produce a
    result. Before this family it produced ``None``, which the advisory rendered as an empty findings list."""
    from merlin.targetgen import rtl_check_runner as RUN

    run = tmp_path / "CAP0"
    gen = run / "generated"
    gen.mkdir(parents=True)
    OS.write_words(gen, [_word(0x03), _word(0x53)], inst_width=64, source="obj")
    monkeypatch.setattr(RUN, "compiled_checks", lambda *a, **k: {"kernel": "// checks", "trace": None})
    monkeypatch.setattr("merlin.targetgen.isa_model.isa_model_for_target", lambda t: _model())

    res = RUN.screen_run(run, _SIMT_FACTS, {}, None, write=True, target="probe")
    assert res is not None, "a run whose compiler emitted a lowering must be screened, not skipped"
    assert res["capsule"] == "CAP0"
    assert [c["id"] for c in res["screen"]["checks"]]
    assert res["screen"]["grounded"]
    assert json.loads((run / "rtl_checks.json").read_text())["object_decode"]


def test_a_screen_that_grounded_nothing_reports_no_passing_filecheck(tmp_path, monkeypatch):
    """A render whose every value is ``-`` would satisfy the assertion file trivially, so FileCheck must not
    run at all: an ``ok`` beside a verdict of ``unknown`` is the reassuring-noise this whole family exists
    to stop producing."""
    from merlin.targetgen import rtl_check_runner as RUN

    run = tmp_path / "CAP2"
    gen = run / "generated"
    gen.mkdir(parents=True)
    OS.write_words(gen, [1, 2], inst_width=32, source="obj")
    monkeypatch.setattr(RUN, "compiled_checks", lambda *a, **k: {"kernel": "// checks", "trace": None})
    monkeypatch.setattr("merlin.targetgen.isa_model.isa_model_for_target",
                        lambda t: IsaModel(target="probe"))          # not fixed-format -> grounds nothing
    monkeypatch.setattr(RUN, "find_filecheck", lambda: "/bin/true")

    res = RUN.screen_run(run, _SIMT_FACTS, {}, "/bin/true", write=False, target="probe")
    assert res["verdict"] == "unknown"
    assert res["filecheck"] == {}, "no assertion may be reported as passing when nothing was grounded"
    assert res["screen"]["dropped"]


def test_screen_run_still_returns_none_when_nothing_was_emitted(tmp_path, monkeypatch):
    """The complement: with no artifact from any family the answer stays None, so the coverage census counts
    it as unscreenable rather than clean."""
    from merlin.targetgen import rtl_check_runner as RUN

    run = tmp_path / "CAP1"
    (run / "generated").mkdir(parents=True)
    monkeypatch.setattr(RUN, "compiled_checks", lambda *a, **k: {"kernel": "// checks", "trace": None})
    assert RUN.screen_run(run, _SIMT_FACTS, {}, None, write=False, target="probe") is None


def test_run_dir_discovery_covers_every_familys_entry_artifact(tmp_path):
    from merlin.targetgen import rtl_check_runner as RUN

    for name, artifact in (("a", "instruction_trace.json"), ("b", "kernel.S"), ("c", OS.WORDS_ARTIFACT)):
        d = tmp_path / name / "generated"
        d.mkdir(parents=True)
        (d / artifact).write_text("{}")
    found = {p.name for p in RUN.iter_run_dirs(tmp_path)}
    assert found == {"a", "b", "c"}
