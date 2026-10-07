"""Phase 1: every graded capsule's linked program is held to the experiment's prohibited roles.

The sealed instruction policy says ``applies_to: phase1_capsule_elfs``, and nothing scanned a capsule's
ELF: the rule reached the whole-model gate and Phase 2, while the capsule grades it was meant to gate
passed programs no one had read. The finalizer every op capsule goes through now scans every ELF the
grade linked (:func:`merlin.targetgen.capsule_runner.prohibited_instruction_report`) and, in the three
directions this test pins:

* a prohibited instruction anywhere in the binary -> **fail** (``PROHIBITED_INSTRUCTION``);
* every ELF measured and none carries one -> the capsule's own verdict stands;
* nothing to scan, or roles that name no instruction of the target -> **incomplete**, never a pass.

The target, its opcode and its role table are synthetic facts patched in; nothing here is a real
encoding, and the ELFs are built structurally (no cross toolchain).
"""

from __future__ import annotations

import struct

import pytest
from test_elf_lane_contract import _EXEC, _NOP, build_elf

from merlin.perf import task_instruction_evidence as TIE
from merlin.perf.whole_model_gate import ROLES_ENV
from merlin.targetgen import capsule_runner as R
from merlin.targetgen import elf_lanes as EL
from merlin.targetgen.capsule_common import make_run_paths

#: A made-up target: no manifest, so the runner's conventional config; its facts are patched in below.
TARGET = "toy-accelerator"
#: The synthetic opcode and role table (made up; the scan derives both from these facts).
OPCODE = 0x2B
FACTS = {
    "isa": {"CUSTOM_OPCODE": OPCODE},
    "instruction_names": {"1": "MOVE", "5": "STEP_A", "9": "FENCE_LIKE"},
    "roles_by_selector": {"1": ["dma"], "5": ["loop_descriptor"], "9": ["sync"]},
}


def _word(selector: int) -> bytes:
    return struct.pack("<I", (selector << 25) | OPCODE)


@pytest.fixture()
def synthetic(monkeypatch):
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: FACTS)
    monkeypatch.setattr(EL, "accelerator_opcode", lambda _target: (OPCODE, "synthetic facts"))
    monkeypatch.setenv(ROLES_ENV, "loop_descriptor")


@pytest.fixture()
def paths(tmp_path):
    return make_run_paths(tmp_path / "runs", "cap", suite="t", target=TARGET, dtype="fp32", benchmark="cap")


def _finalize(paths, target=TARGET, capsule=None, *, status="pass", no_oracle=False):
    return R._finalize_capsule_result(
        name="cap",
        capsule=capsule or {"name": "cap", "kind": "isa", "label": "public"},
        status=status,
        failure=None,
        tiers={"L2": R.TierResult("L2", "pass", True)},
        trace_check_res={"status": "skipped", "violations": []},
        numeric={"status": "pass"},
        required={"L2"},
        no_oracle=no_oracle,
        eff_target=target,
        paths=paths,
        run_id="cap",
        cfg=R._config_for_target(target, "t", "fp32"),
        contract=None,
    )


def _link(paths, *words: bytes, name: str = EL.PACKAGE_ELF_NAME):
    paths.generated.mkdir(parents=True, exist_ok=True)
    (paths.generated / name).write_bytes(build_elf([(".text", _EXEC, _NOP * 2 + b"".join(words))]))


def test_the_roles_come_from_the_run_and_from_the_capsules_own_sealed_arm(monkeypatch):
    monkeypatch.delenv(ROLES_ENV, raising=False)
    assert R.graded_prohibited_roles({"name": "cap"}) == ()
    monkeypatch.setenv(ROLES_ENV, "loop_descriptor, sync")
    assert R.graded_prohibited_roles({"name": "cap"}) == ("loop_descriptor", "sync")
    monkeypatch.delenv(ROLES_ENV)
    sealed = {
        "performance": {"arms": {"candidate": {"instruction_policy": {"prohibited_instruction_roles": ["sync"]}}}}
    }
    assert R.graded_prohibited_roles(sealed) == ("sync",)


def test_mutation_a_capsule_elf_carrying_a_prohibited_instruction_fails(synthetic, paths):
    _link(paths, _word(1), _word(5))
    row = _finalize(paths)
    assert row["status"] == "fail" and row["failure"]["category"] == "PROHIBITED_INSTRUCTION"
    assert row["isa_prohibition"]["summary"] == {"STEP_A": 1}
    assert row["failure"]["prohibited_instructions"] == {"5": "STEP_A"}


def test_every_linked_elf_is_scanned_not_only_the_kernel(synthetic, paths):
    """A grade can link more than one executable (a harness beside the kernel); the rule is about all
    of them, so a prohibited instruction in the second one fails the capsule too."""
    _link(paths, _word(1))
    _link(paths, _word(5), name="harness.elf")
    row = _finalize(paths)
    assert row["status"] == "fail" and len(row["isa_prohibition"]["scans"]) == 2


def test_a_clean_capsule_elf_keeps_its_verdict(synthetic, paths):
    _link(paths, _word(1), _word(9))
    row = _finalize(paths)
    assert row["status"] == "pass", row.get("failure")
    assert row["isa_prohibition"]["clean"] is True and row["isa_prohibition"]["status"] == "measured"


def test_mutation_an_empty_role_table_is_never_clean(synthetic, paths, monkeypatch):
    """The vacuous case: the program is full of the prohibited instruction, and the role table that
    should name it names nothing. Unmeasured -> incomplete; never a pass."""
    unroled = {**FACTS, "roles_by_selector": {k: [] for k in FACTS["roles_by_selector"]}}
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _t: unroled)
    _link(paths, _word(5), _word(5))
    row = _finalize(paths)
    assert row["status"] == "incomplete" and row["failure"]["category"] == "PROHIBITION_NOT_MEASURED"
    assert row["isa_prohibition"]["clean"] is None


def test_a_grade_that_linked_no_elf_is_not_a_pass(synthetic, paths):
    row = _finalize(paths)
    assert row["status"] == "incomplete" and row["failure"]["category"] == "PROHIBITION_NOT_MEASURED"
    smoke = _finalize(paths, no_oracle=True)
    assert smoke["status"] == "not_gradeable_no_oracle"  # a structure-only smoke: never a pass, no phantom fix


def test_a_capsule_held_to_no_role_is_not_scanned(paths, monkeypatch):
    monkeypatch.delenv(ROLES_ENV, raising=False)
    row = _finalize(paths)
    assert "isa_prohibition" not in row
