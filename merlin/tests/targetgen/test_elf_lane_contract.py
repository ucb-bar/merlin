"""A NEGATIVE lane contract, proved on the operator path from the linked ELF.

Ten host-lane capsules declare ``lanes.forbid: [<accelerator lane>]`` -- "this family is not admitted
by the target's manifest, so the compiler must leave it on the host". Only the whole-model path builds
the dynamic dispatch ledger a lane verdict was read from, so on the operator path the assertion was
recorded as ``LANE_CONTRACT_NOT_EVALUATED`` and the capsules could not pass however clean their
numerics were -- and, worse, a submission that DID accelerate the inadmissible family could not be
caught by the capsule written to catch exactly that.

The refusal was right and stays. What changes is that the operator path now has evidence it can
actually prove. The assertion is a negative, and absence is what a complete instruction stream CAN
establish: the linked ELF holds every instruction the program can execute, whatever spelling the
backend used to emit it. That is strictly MORE than the IR-level decoder sees (which reads
``llvm.inline_asm`` ops and is blind to a raw ``.word`` datum), so this is a strengthening of the
existing gate, not a way around it.

The asymmetry is the whole design and is pinned below in both directions:

* an ELF that CARRIES the accelerator opcode while the capsule forbids it -> **fail** (disproved);
* an ELF that genuinely carries none -> **pass** (the negative is settled);
* an ELF that cannot be read, or a target whose opcode is not derivable -> **unmeasured**, and the
  capsule stays ``incomplete``. Never a pass.

A REQUIRED lane is never credited here: an instruction present in a binary is not one that executed.
"""
from __future__ import annotations

import struct

import pytest

from merlin.targetgen import capsule_runner as R
from merlin.targetgen import elf_lanes as EL
from merlin.targetgen.capsule_common import make_run_paths
from merlin.targetgen.target_registry import all_targets

# --- a synthetic linked ELF, built structurally so a test needs no cross toolchain --------------
_EM_RISCV = 243


def build_elf(sections, *, machine: int = _EM_RISCV, ei_class: int = 2, ei_data: int = 1,
              drop_section_headers: bool = False) -> bytes:
    """A minimal ELF64 with the given ``(name, flags, bytes)`` sections. ``flags`` is the raw
    ``sh_flags`` so a test can place bytes in an executable section or a merely-allocated one."""
    names = [""] + [n for n, _f, _d in sections] + [".shstrtab"]
    shstr, offsets = b"", {}
    for n in names:
        offsets[n] = len(shstr)
        shstr += n.encode() + b"\0"
    body, blobs = b"", []
    cursor = 64
    for _n, _f, data in sections:
        blobs.append((cursor, len(data)))
        body += data
        cursor += len(data)
    shstr_off = cursor
    body += shstr
    cursor += len(shstr)
    shoff = cursor
    hdr = bytearray(64)
    hdr[0:4] = b"\x7fELF"
    hdr[4], hdr[5], hdr[6] = ei_class, ei_data, 1
    struct.pack_into("<HHI", hdr, 0x10, 2, machine, 1)
    struct.pack_into("<QQQ", hdr, 0x18, 0x80000000, 0, 0 if drop_section_headers else shoff)
    struct.pack_into("<IHHHHH", hdr, 0x30, 0, 64, 0, 0, 64,
                     0 if drop_section_headers else len(sections) + 2)
    struct.pack_into("<H", hdr, 0x3E, 0 if drop_section_headers else len(sections) + 1)
    if drop_section_headers:
        return bytes(hdr) + body
    shdrs = bytearray()

    def _sh(name_off, typ, flags, addr, off, size):
        return struct.pack("<IIQQQQIIQQ", name_off, typ, flags, addr, off, size, 0, 0, 4, 0)

    shdrs += _sh(offsets[""], 0, 0, 0, 0, 0)
    for (n, f, _d), (off, size) in zip(sections, blobs):
        shdrs += _sh(offsets[n], 1, f, 0x80000000 + off, off, size)
    shdrs += _sh(offsets[".shstrtab"], 3, 0, 0, shstr_off, len(shstr))
    return bytes(hdr) + body + bytes(shdrs)


_EXEC = 0x2 | 0x4          # SHF_ALLOC | SHF_EXECINSTR
_ALLOC = 0x2               # SHF_ALLOC only -- data, not instructions

_NOP = struct.pack("<I", 0x00000013)      # addi x0, x0, 0
_C_NOP = struct.pack("<H", 0x0001)        # a 16-bit instruction: shifts everything after it off 4-align


def _accel_word(opcode: int) -> bytes:
    """A 32-bit word in the target's custom major-opcode space. The opcode is DERIVED and passed in;
    the surrounding bits are arbitrary, because the scan's claim is about the opcode field alone."""
    return struct.pack("<I", (0x2A << 25) | (0x03 << 12) | opcode)


@pytest.fixture(scope="module")
def derivable():
    """A target whose accelerator major opcode is derivable, plus that opcode.

    Chosen by DERIVATION over the registry rather than named here, so this test does not pin a target;
    it asserts (never skips) that at least one exists -- a suite where none did would report success
    while proving nothing."""
    found = [(t, EL.accelerator_opcode(t)) for t in all_targets()]
    have = [(t, op) for t, (op, _src) in found if op is not None]
    assert have, ("no registered target has a derivable accelerator opcode; this test would otherwise "
                  f"pass while measuring nothing (looked at {[t for t, _ in found]})")
    return have[0]


def test_the_scanned_file_is_the_one_the_compile_step_links():
    """The scan is worthless if it looks at a path the build never writes, and "no linked executable"
    is an UNMEASURED verdict -- which is quiet. Pin the name against the compile step that produces it."""
    from merlin.common.paths import merlin_dir
    src = (merlin_dir() / "python/merlin/targetgen/contract/compile.py").read_text(encoding="utf-8")
    assert f'"{EL.PACKAGE_ELF_NAME}"' in src, (
        f"the compile step no longer links {EL.PACKAGE_ELF_NAME}; the lane scan would silently find "
        f"nothing to read and report every forbidden lane as unmeasured")


# --- the derivation ----------------------------------------------------------------------------
def test_the_opcode_is_derived_and_says_where_from(derivable):
    target, opcode = derivable
    got, source = EL.accelerator_opcode(target)
    assert got == opcode
    assert "funct_decode_table" in source or "capability_manifest" in source, source
    assert 0 <= opcode <= 0x7F, "a RISC-V major opcode is a 7-bit field"


def test_an_underivable_target_is_unknown_not_a_default(tmp_path):
    """Fail closed: a target whose facts and manifest ground no custom opcode yields UNKNOWN, and the
    scan measures nothing rather than substituting a value from a target that does have one."""
    blind = [t for t in all_targets() if EL.accelerator_opcode(t)[0] is None]
    assert blind, "expected at least one target with no host-issued custom opcode (SIMT / self-hosted)"
    elf = tmp_path / "k.elf"
    elf.write_bytes(build_elf([(".text", _EXEC, _NOP * 4)]))
    scan = EL.scan_elf_for_accelerator(elf, blind[0])
    assert scan.status == "unmeasured" and scan.opcode == EL.UNKNOWN
    assert "UNDERIVABLE" in scan.opcode_source


# --- the scan ----------------------------------------------------------------------------------
def test_a_clean_elf_measures_zero(derivable, tmp_path):
    target, _op = derivable
    elf = tmp_path / "k.elf"
    elf.write_bytes(build_elf([(".text", _EXEC, _NOP * 16), (".rodata", _ALLOC, b"\x00" * 32)]))
    scan = EL.scan_elf_for_accelerator(elf, target)
    assert scan.status == "measured" and scan.n_hits == 0
    assert scan.n_instruction_words == 16, "every word in the executable section was walked"


def test_an_accelerator_word_is_found_even_with_no_ir_spelling(derivable, tmp_path):
    """THE `.word` HOLE. The IR decoder reads ``llvm.inline_asm`` ops, so an accelerator instruction
    emitted as a raw datum decodes as silence. Bytes have no spelling: this ELF has no IR at all and
    the instruction is still seen."""
    target, opcode = derivable
    elf = tmp_path / "k.elf"
    elf.write_bytes(build_elf([(".text", _EXEC, _NOP + _accel_word(opcode) + _NOP)]))
    scan = EL.scan_elf_for_accelerator(elf, target)
    assert scan.status == "measured" and scan.n_hits == 1
    assert scan.hits[0]["section"] == ".text"


def test_a_compressed_instruction_does_not_hide_the_next_one(derivable, tmp_path):
    """A 16-bit instruction shifts everything after it to a 2-mod-4 address. A 4-byte stride would walk
    straight past the accelerator word -- measured on real kernels, whose accelerator instructions sit
    at 2-mod-4 addresses. The walk follows the ISA's length encoding instead."""
    target, opcode = derivable
    elf = tmp_path / "k.elf"
    elf.write_bytes(build_elf([(".text", _EXEC, _C_NOP + _accel_word(opcode) + _C_NOP)]))
    scan = EL.scan_elf_for_accelerator(elf, target)
    assert scan.n_hits == 1, "the length-encoded walk must survive a compressed instruction"
    assert scan.hits[0]["addr"] % 4 == 2


def test_data_sections_are_not_read_as_instructions(derivable, tmp_path):
    """Widening the scan to every ALLOCATED section was measured on the ten host-lane capsules and
    manufactured 1-6 phantom hits on EVERY one of them: constant-pool bytes whose low 7 bits happen to
    carry the opcode. A phantom hit fails a conformant submission, so the scan stays inside sections the
    linker marked executable."""
    target, opcode = derivable
    elf = tmp_path / "k.elf"
    elf.write_bytes(build_elf([(".text", _EXEC, _NOP * 4),
                               (".rodata", _ALLOC, _accel_word(opcode) * 4)]))
    scan = EL.scan_elf_for_accelerator(elf, target)
    assert scan.status == "measured" and scan.n_hits == 0
    assert ".rodata" not in scan.sections


@pytest.mark.parametrize("make", [
    pytest.param(lambda p: None, id="absent"),
    pytest.param(lambda p: p.write_bytes(b"not an elf at all"), id="not_an_elf"),
    pytest.param(lambda p: p.write_bytes(build_elf([(".text", _EXEC, _NOP)],
                                                   drop_section_headers=True)), id="stripped"),
    pytest.param(lambda p: p.write_bytes(build_elf([(".text", _EXEC, _NOP)], machine=62)),
                 id="wrong_machine"),
    pytest.param(lambda p: p.write_bytes(build_elf([(".text", _EXEC, _NOP)], ei_data=2)),
                 id="big_endian"),
    pytest.param(lambda p: p.write_bytes(build_elf([(".rodata", _ALLOC, _NOP)])), id="no_exec_sections"),
])
def test_an_unreadable_elf_is_unmeasured_never_clean(derivable, tmp_path, make):
    target, _op = derivable
    elf = tmp_path / "k.elf"
    make(elf)
    scan = EL.scan_elf_for_accelerator(elf, target)
    assert scan.status == "unmeasured", scan.detail
    assert scan.n_hits == 0 and scan.detail


# --- the lane report ---------------------------------------------------------------------------
def _forbidding():
    return {"name": "c", "lanes": {"forbid": [R._ACCELERATOR_LANE]}}


def test_no_declared_lanes_produces_no_report(derivable, tmp_path):
    target, _op = derivable
    assert EL.lane_report_from_elf({}, tmp_path / "k.elf", target=target) is None
    assert EL.lane_report_from_elf({"lanes": {}}, tmp_path / "k.elf", target=target) is None


def test_a_forbidden_lane_with_no_accelerator_instruction_is_settled(derivable, tmp_path):
    target, _op = derivable
    elf = tmp_path / "k.elf"
    elf.write_bytes(build_elf([(".text", _EXEC, _NOP * 8)]))
    rep = EL.lane_report_from_elf(_forbidding(), elf, target=target)
    assert rep["violated"] == [] and "unmeasured_forbidden" not in rep
    assert rep["evidence"][R._ACCELERATOR_LANE] == EL.LINKED_ELF_EVIDENCE
    assert EL.unjudged_lanes(rep, _forbidding()["lanes"]) == []


def test_a_forbidden_lane_the_elf_carries_is_violated(derivable, tmp_path):
    target, opcode = derivable
    elf = tmp_path / "k.elf"
    elf.write_bytes(build_elf([(".text", _EXEC, _NOP + _accel_word(opcode))]))
    rep = EL.lane_report_from_elf(_forbidding(), elf, target=target)
    assert rep["violated"] == [R._ACCELERATOR_LANE]
    assert rep["elf_scan"]["n_hits"] == 1


def test_a_required_lane_is_never_credited_by_a_static_scan(derivable, tmp_path):
    """An instruction present in a binary is not one that executed, so this evidence cannot satisfy a
    positive obligation. The lane stays unmeasured and its capsule cannot pass on it."""
    target, _op = derivable
    elf = tmp_path / "k.elf"
    elf.write_bytes(build_elf([(".text", _EXEC, _NOP * 8)]))
    lanes = {"require": ["scalar_rvv_lane"], "forbid": [R._ACCELERATOR_LANE]}
    rep = EL.lane_report_from_elf({"lanes": lanes}, elf, target=target)
    assert rep["observed"] == [] and rep["unexercised"] == ["scalar_rvv_lane"]
    assert rep["evidence"]["scalar_rvv_lane"] == EL.NO_EVIDENCE
    assert EL.unjudged_lanes(rep, lanes) == ["scalar_rvv_lane"]


def test_the_evidence_rung_is_not_folded_into_the_executed_vocabulary():
    """``EXECUTED_LANE_EVIDENCE`` means "something RAN". A static scan did not run anything, so folding
    this rung in would let a required lane be credited by mere presence -- at BOTH ends, since
    ``capsule_grade`` reads that same tuple. It is admissible for the negative direction only."""
    assert EL.LINKED_ELF_EVIDENCE not in R.EXECUTED_LANE_EVIDENCE
    assert EL.LINKED_ELF_EVIDENCE in EL.negative_lane_evidence()
    assert set(R.EXECUTED_LANE_EVIDENCE) < set(EL.negative_lane_evidence())


def test_unjudged_lanes_refuses_a_missing_or_malformed_report():
    lanes = {"forbid": [R._ACCELERATOR_LANE], "require": ["scalar_rvv_lane"]}
    assert EL.unjudged_lanes(None, lanes) == sorted([R._ACCELERATOR_LANE, "scalar_rvv_lane"])
    assert EL.unjudged_lanes({"evidence": "ledger"}, lanes) == sorted(
        [R._ACCELERATOR_LANE, "scalar_rvv_lane"])
    assert EL.unjudged_lanes({"evidence": {}}, {}) == []


def test_a_lane_both_required_and_forbidden_is_refused(derivable, tmp_path):
    target, _op = derivable
    with pytest.raises(ValueError):
        EL.lane_report_from_elf({"lanes": {"require": ["x"], "forbid": ["x"]}},
                                tmp_path / "k.elf", target=target)


# --- the verdict: the finalizer, in all three directions ---------------------------------------
def _finalize(paths, capsule, *, target, status="pass"):
    return R._finalize_capsule_result(
        name="cap", capsule=capsule, status=status, failure=None,
        tiers={"L2": R.TierResult("L2", "pass", True)},
        trace_check_res={"status": "skipped", "violations": []},
        numeric={"status": "pass"}, required={"L2"}, no_oracle=False, eff_target=target,
        paths=paths, run_id="cap", cfg=R._config_for_target(target, "t", "fp32"), contract=None)


@pytest.fixture()
def paths(tmp_path, derivable):
    target, _op = derivable
    return make_run_paths(tmp_path / "runs", "cap", suite="t", target=target,
                          dtype="fp32", benchmark="cap")


def _write_elf(paths, blob: bytes):
    paths.generated.mkdir(parents=True, exist_ok=True)
    (paths.generated / EL.PACKAGE_ELF_NAME).write_bytes(blob)


def test_mutation_clean_elf_reaches_pass(derivable, paths):
    """Direction 1: forbids the accelerator, and the linked ELF genuinely carries none of its
    instructions. The contract is SETTLED, so the capsule passes -- this is the row the ten host-lane
    capsules could not reach."""
    target, _op = derivable
    _write_elf(paths, build_elf([(".text", _EXEC, _NOP * 8)]))
    row = _finalize(paths, {"name": "cap", "kind": "model_slice", "label": "public",
                            "lanes": {"forbid": [R._ACCELERATOR_LANE]}}, target=target)
    assert row["status"] == "pass", row.get("failure")
    assert row["lane_report"]["evidence"][R._ACCELERATOR_LANE] == EL.LINKED_ELF_EVIDENCE
    assert row["lane_report"]["elf_scan"]["opcode_source"]


def test_mutation_an_accelerated_elf_fails(derivable, paths):
    """Direction 2: the same capsule, the same clean numerics, one accelerator instruction added to the
    binary. It must FAIL -- and the row must name the lane, not merely go quiet."""
    target, opcode = derivable
    _write_elf(paths, build_elf([(".text", _EXEC, _NOP * 4 + _accel_word(opcode))]))
    row = _finalize(paths, {"name": "cap", "kind": "model_slice", "label": "public",
                            "lanes": {"forbid": [R._ACCELERATOR_LANE]}}, target=target)
    assert row["status"] == "fail"
    assert row["failure"]["category"] == "ACCELERATED_A_FORBIDDEN_LANE"
    assert row["lane_report"]["violated"] == [R._ACCELERATOR_LANE]


def test_mutation_an_unreadable_elf_stays_unmeasured(derivable, paths):
    """Direction 3: nothing to read. Not a pass and not a fail -- ``incomplete``, because nothing about
    the submission was disproved and nothing was proved either."""
    target, _op = derivable
    _write_elf(paths, b"truncated")
    row = _finalize(paths, {"name": "cap", "kind": "model_slice", "label": "public",
                            "lanes": {"forbid": [R._ACCELERATOR_LANE]}}, target=target)
    assert row["status"] == "incomplete"
    assert row["failure"]["category"] == "LANE_CONTRACT_NOT_EVALUATED"
    assert R._ACCELERATOR_LANE in row["failure"]["detail"]
    assert row["status"] not in ("pass",)


def test_a_missing_elf_stays_unmeasured(derivable, paths):
    """The compile never produced an executable: same verdict, and the capsule is still not passed."""
    target, _op = derivable
    row = _finalize(paths, {"name": "cap", "kind": "model_slice", "label": "public",
                            "lanes": {"forbid": [R._ACCELERATOR_LANE]}}, target=target)
    assert row["status"] == "incomplete"
    assert row["status"] in R.NOT_MEASURED_STATUSES or row["status"] != "pass"


def test_a_required_lane_keeps_the_capsule_incomplete(derivable, paths):
    """A capsule that also REQUIRES a lane cannot be finished by an ELF scan, and must not be quietly
    passed on the half that could be settled."""
    target, _op = derivable
    _write_elf(paths, build_elf([(".text", _EXEC, _NOP * 8)]))
    row = _finalize(paths, {"name": "cap", "kind": "model_slice", "label": "public",
                            "lanes": {"require": ["scalar_rvv_lane"],
                                      "forbid": [R._ACCELERATOR_LANE]}}, target=target)
    assert row["status"] == "incomplete"
    assert "scalar_rvv_lane" in row["failure"]["detail"]


def test_a_capsule_with_no_lane_contract_is_untouched(derivable, paths):
    """Non-regression for every other capsule in the corpus: no declared lanes, no lane report, no
    change to the verdict."""
    target, _op = derivable
    row = _finalize(paths, {"name": "cap", "kind": "isa", "label": "public"}, target=target)
    assert row["status"] == "pass" and "lane_report" not in row


def test_a_caller_supplied_lane_report_still_wins(derivable, paths):
    """The whole-model path attaches its own ledger-evidenced report. It must be used as-is -- the ELF
    scan is the operator path's fallback, never an override of real execution evidence."""
    target, _op = derivable
    _write_elf(paths, build_elf([(".text", _EXEC, _NOP * 8)]))
    supplied = {"required": [], "observed": [], "unexercised": [], "forbidden": [R._ACCELERATOR_LANE],
                "violated": [], "evidence": {R._ACCELERATOR_LANE: "dynamic_dispatch_ledger"}}
    row = R._finalize_capsule_result(
        name="cap", capsule={"name": "cap", "kind": "model_slice", "label": "public",
                             "lanes": {"forbid": [R._ACCELERATOR_LANE]}},
        status="pass", failure=None, tiers={"L2": R.TierResult("L2", "pass", True)},
        trace_check_res={"status": "skipped", "violations": []}, numeric={"status": "pass"},
        required={"L2"}, no_oracle=False, eff_target=target, paths=paths, run_id="cap",
        cfg=R._config_for_target(target, "t", "fp32"), contract=None,
        extra={"lane_report": supplied})
    assert row["status"] == "pass"
    assert row["lane_report"] is supplied or row["lane_report"] == supplied
    assert "elf_scan" not in row["lane_report"]


def test_a_caller_supplied_violation_still_fails(derivable, paths):
    """The other half of deferring to the caller: a report that DID disprove the contract is acted on,
    whoever produced it."""
    target, _op = derivable
    _write_elf(paths, build_elf([(".text", _EXEC, _NOP * 8)]))
    supplied = {"required": [], "observed": [R._ACCELERATOR_LANE], "unexercised": [],
                "forbidden": [R._ACCELERATOR_LANE], "violated": [R._ACCELERATOR_LANE],
                "evidence": {R._ACCELERATOR_LANE: "dynamic_dispatch_ledger"}}
    row = R._finalize_capsule_result(
        name="cap", capsule={"name": "cap", "kind": "model_slice", "label": "public",
                             "lanes": {"forbid": [R._ACCELERATOR_LANE]}},
        status="pass", failure=None, tiers={"L2": R.TierResult("L2", "pass", True)},
        trace_check_res={"status": "skipped", "violations": []}, numeric={"status": "pass"},
        required={"L2"}, no_oracle=False, eff_target=target, paths=paths, run_id="cap",
        cfg=R._config_for_target(target, "t", "fp32"), contract=None,
        extra={"lane_report": supplied})
    assert row["status"] == "fail"
    assert row["failure"]["category"] == "ACCELERATED_A_FORBIDDEN_LANE"


# --- non-regression: the whole-model ladder is untouched ---------------------------------------
def test_the_whole_model_ladder_is_unchanged():
    """The dispatch-ledger path builds its own report and does not consult the ELF at all. Pinned here
    because a change that silently re-routed it would replace execution evidence with a static scan."""
    rep = R.lane_report({"lanes": {"require": ["on_mesh"], "forbid": ["scalar_rvv_lane"]}},
                        {"on_mesh": {"matmul": 3}},
                        {"dispatch_ledger": [{"status": "pass", "lane": "on_mesh"}]})
    assert rep["evidence"]["on_mesh"] == "dynamic_dispatch_ledger"
    assert rep["observed"] == ["on_mesh"] and rep["violated"] == []
    assert "elf_scan" not in rep
    assert EL.unjudged_lanes(rep, {"require": ["on_mesh"], "forbid": ["scalar_rvv_lane"]}) == []
