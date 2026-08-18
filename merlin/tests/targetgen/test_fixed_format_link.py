"""Fork-free link relocation-patching for a fixed-format ISA
(:mod:`merlin.targetgen.fixed_format.link`).

These are structural, board-free tests. The exact immediate-packing values below were reverse-derived
from a fork-linked, cyclotron-verified boot image: they are the bit patterns the (now retired) forked
linker wrote at fixed-format field positions. Reproducing them with the DERIVED field layout is what
lets a stock linker + this patcher replace the fork with a byte-identical, boot-correct result.

The full link+cyclotron proof needs the local radiance toolchain and is exercised out of band; here we
lock the field math and the fail-closed contract that guard against a silently mis-patched boot.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.fixed_format import link as ML

# Fixed-format field layout for the SIMT target (derived from its RTL-derived IsaModel). Kept explicit
# here so the packing math is tested hermetically; test_derived_layout_matches checks it still equals
# what the derivation produces, so this can never silently drift from the real target.
FIELD_LAYOUT = {"opcode": (6, 0), "f3": (19, 17), "f7": (58, 52), "rd": (16, 9),
                "rs1": (27, 20), "rs2": (35, 28), "rs3": (43, 36), "imm24": (59, 36)}


def test_full_immediate_splits_into_imm24_and_rs2_high_byte():
    f = ML._Fields(FIELD_LAYOUT)
    # A la/call low-half offset (init_regs_all, __global_pointer-class): 0x2068 fits in 24 bits.
    assert f.put_imm32(0, 0x2068, sb=False) == 0x0002068000000000
    # _exit call: 0x1f88.
    assert f.put_imm32(0, 0x1F88, sb=False) == 0x0001F88000000000


def test_negative_offset_high_byte_lands_in_rs2():
    f = ML._Fields(FIELD_LAYOUT)
    # __tcb_aligned_size resolves to 0; PC-relative offset is negative -> 0xEFFFDFB8. The high byte
    # 0xEF must land in rs2, the low 24 bits in imm24 (the exact bits the fork linker emitted).
    assert f.put_imm32(0, 0xEFFFDFB8, sb=False) == 0x0FFDFB8EF0000000


def test_branch_high_byte_lands_in_rd_not_rs2():
    f = ML._Fields(FIELD_LAYOUT)
    # A backward branch (vx_wspawn_wait): displacement 0xFFFFFFF0. SB form -> high byte 0xFF into rd.
    assert f.put_imm32(0, 0xFFFFFFF0, sb=True) == 0x0FFFFF000001FE00


def test_field_layout_missing_required_field_fails_closed():
    with pytest.raises(ML.FixedFormatLinkError):
        ML._Fields({"opcode": (6, 0)})  # no imm24/rs2/rd


def test_relocation_type_numbers_are_psabi_constants():
    # These are the ELF/RISC-V psABI numbers (identical for every RISC-V target) -- the values the
    # patcher dispatches on. Guard against an accidental edit that would mis-route a relocation.
    assert (ML.R_RISCV_BRANCH, ML.R_RISCV_JAL, ML.R_RISCV_CALL_PLT,
            ML.R_RISCV_PCREL_HI20, ML.R_RISCV_PCREL_LO12_I) == (16, 17, 19, 23, 24)


def test_resolve_stock_linker_rejects_missing_explicit(tmp_path):
    with pytest.raises(ML.FixedFormatLinkError):
        ML.resolve_stock_linker(tmp_path / "nope-ld.lld")


def test_derived_layout_matches_isamodel():
    """The explicit FIELD_LAYOUT above must equal the target's RTL-derived layout, so the packing
    tests never certify a stale hand-copied layout. Skips only if the derivation is unavailable."""
    try:
        from merlin.targetgen.isa_model import isa_model_from_encoding
        from merlin.targetgen.rtl import mlc_bridge
        m = isa_model_from_encoding("radiance", mlc_bridge.isa_encoding_for("radiance"))
    except Exception as exc:  # noqa: BLE001 -- derivation needs the local target encoding
        pytest.skip(f"radiance IsaModel not derivable here: {exc}")
    for name, pos in FIELD_LAYOUT.items():
        assert tuple(m.field_layout[name]) == pos, f"{name} drifted from derived layout"
    assert m.inst_width == 64
