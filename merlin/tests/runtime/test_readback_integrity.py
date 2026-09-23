"""A readback with a periodic hole must be REFUSED, not compared against a tolerance.

The shape under test was measured on an elaborated-RTL cert run: every 4-byte word at
an odd index came back exactly 0.0 while every other word was bit-exact.  A magnitude-
scaled numeric gate cannot see that -- on one 7232-element capsule only 988 of 5172
zeroed elements were reported as mismatches, and the remaining 4184 were ACCEPTED
because their goldens were smaller than the absolute tolerance.  So the transport could
certify a capsule at the mandatory tier with half its buffer never read back.

These tests pin the structural refusal, and pin that it does NOT fire on the two
readbacks that were genuine complete certifications on the same engine.
"""

from __future__ import annotations

import struct

import pytest

from merlin.common import readback_integrity as RI
from merlin.runtime.backends.base import get_backend

MO = get_backend("muon").muon_oracles
MU = get_backend("muon").muon


def _words(values: list[int]) -> bytes:
    return b"".join(struct.pack("<I", v) for v in values)


def test_every_other_word_zeroed_is_refused():
    """The measured defect: alternating bit-exact word / exact zero."""
    values = [0x458DB70E + i if i % 2 == 0 else 0 for i in range(690)]
    defect = RI.residue_class_defect(values)
    assert defect is not None
    assert defect.startswith("readback_residue_class_zeroed:")
    assert "index % 2 == 1" in defect
    assert "345 of 345" in defect


def test_quarter_and_eighth_word_holes_are_refused():
    """A transport that loses one word in four, or one in eight, is the same defect."""
    for stride in (4, 8, 16):
        for residue in range(stride):
            values = [0 if i % stride == residue else 0x40000000 + i for i in range(512)]
            defect = RI.residue_class_defect(values)
            assert defect is not None, (stride, residue)
            assert f"index % {stride} == {residue}" in defect or "index % 2" in defect


def test_a_complete_readback_is_accepted():
    values = [0x3F800000 + i for i in range(690)]
    assert RI.residue_class_defect(values) is None


def test_scattered_zeros_are_not_a_residue_class():
    """Real outputs contain zeros; only a PERIODIC hole is a transport defect."""
    values = [0x40000000 + i for i in range(690)]
    for i in (3, 5, 11, 200, 201, 400, 689):
        values[i] = 0
    assert RI.residue_class_defect(values) is None


def test_a_wholly_zero_buffer_does_not_trip_this_rule():
    """No complement means no contrast; an all-zero buffer is any value gate's job."""
    assert RI.residue_class_defect([0] * 690) is None


def test_short_buffers_are_not_evidence():
    assert RI.residue_class_defect([1, 0, 2, 0, 3, 0, 4, 0]) is None


def test_bit_pattern_zero_not_float_zero():
    """-0.0 is a live value that was written; it must not read as a hole."""
    neg_zero = struct.unpack("<I", struct.pack("<f", -0.0))[0]
    values = [0x40000000 + i if i % 2 == 0 else neg_zero for i in range(690)]
    assert RI.residue_class_defect(values) is None


def test_require_intact_names_the_transport():
    raw = _words([0x458DB70E if i % 2 == 0 else 0 for i in range(690)])
    with pytest.raises(RI.ReadbackIntegrityError) as exc:
        RI.require_intact(raw, transport="some_transport")
    assert "some_transport:" in str(exc.value)


def test_ragged_byte_count_fails_closed():
    with pytest.raises(RI.ReadbackIntegrityError):
        RI.require_intact(b"\x00\x01\x02", transport="t")


def test_oracle_decode_refuses_the_broken_readback(tmp_path):
    """The wiring: a holed dump must never reach the value comparison."""
    raw = _words([0x458DB70E if i % 2 == 0 else 0 for i in range(690)])
    dump = tmp_path / "gsim.output.bin"
    dump.write_bytes(raw)
    manifest = {
        "output": {
            "name": "Y0",
            "dtype": "f32",
            "rows": 345,
            "cols": 2,
            "elements": 690,
            "byte_length": 2760,
            "dump_file": "gsim.output.bin",
        }
    }
    console = (
        "[gsim-emu] FINISHED: cycles=403408 wall=80.88s (4988 cyc/s) done=0 "
        "model_finished=1 exit_code=0\n"
        "[gsim-emu] BINARY_DUMP complete bytes=2760\n"
    )
    with pytest.raises(MU.MuonError) as exc:
        MO._gsim_host_dump_outputs(console, dump, manifest, workdir=tmp_path)
    assert "readback_residue_class_zeroed" in str(exc.value)
    assert "gsim_evaluator_owned_gmem_dump" in str(exc.value)


def test_oracle_decode_accepts_a_complete_readback(tmp_path):
    raw = _words([0x3F800000 + i for i in range(690)])
    dump = tmp_path / "gsim.output.bin"
    dump.write_bytes(raw)
    manifest = {
        "output": {
            "name": "Y0",
            "dtype": "f32",
            "rows": 345,
            "cols": 2,
            "elements": 690,
            "byte_length": 2760,
            "dump_file": "gsim.output.bin",
        }
    }
    console = (
        "[gsim-emu] FINISHED: cycles=403408 wall=80.88s (4988 cyc/s) done=0 "
        "model_finished=1 exit_code=0\n"
        "[gsim-emu] BINARY_DUMP complete bytes=2760\n"
    )
    out = MO._gsim_host_dump_outputs(console, dump, manifest, workdir=tmp_path)
    assert len(out["Y0"]) == 345


def test_the_false_pass_hazard_itself():
    """Half a buffer zeroed, yet a magnitude-scaled gate accepts most of it."""
    atol = 0.03125
    golden = [0.5 if i % 2 == 0 else 0.001 for i in range(7232)]
    read = [g if i % 2 == 0 else 0.0 for i, g in enumerate(golden)]
    accepted = sum(1 for g, r in zip(golden, read) if abs(g - r) <= atol)
    assert accepted > len(golden) // 2  # the gate would pass most of the hole
    words = [struct.unpack("<I", struct.pack("<f", v))[0] for v in read]
    assert RI.residue_class_defect(words) is not None  # the structural check will not


def test_cyclotron_decode_refuses_the_broken_readback(tmp_path):
    """The same refusal guards the timing-model transport, not only the RTL one."""
    raw = _words([0x458DB70E if i % 2 == 0 else 0 for i in range(690)])
    dump = tmp_path / "cyclotron.output.bin"
    dump.write_bytes(raw)
    manifest = {
        "output": {
            "name": "Y0",
            "dtype": "f32",
            "rows": 345,
            "cols": 2,
            "elements": 690,
            "byte_length": 2760,
            "dump_file": "cyclotron.output.bin",
        }
    }
    with pytest.raises(MU.MuonError) as exc:
        MO._cyclotron_host_dump_outputs("DONE\n", dump, manifest)
    assert "readback_residue_class_zeroed" in str(exc.value)
    assert "cyclotron_host_gmem_dump" in str(exc.value)


def test_cyclotron_decode_accepts_a_complete_readback(tmp_path):
    raw = _words([0x3F800000 + i for i in range(690)])
    dump = tmp_path / "cyclotron.output.bin"
    dump.write_bytes(raw)
    manifest = {
        "output": {
            "name": "Y0",
            "dtype": "f32",
            "rows": 345,
            "cols": 2,
            "elements": 690,
            "byte_length": 2760,
            "dump_file": "cyclotron.output.bin",
        }
    }
    out = MO._cyclotron_host_dump_outputs("DONE\n", dump, manifest)
    assert len(out["Y0"]) == 345
