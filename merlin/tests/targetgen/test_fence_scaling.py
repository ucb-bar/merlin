"""Synchronisation must not scale with output tiles.

A schedule that fences after every output tile is numerically correct and catastrophically slow, so
no numeric oracle can notice it. Measured on a 1024x1024 QK slice: 4,096 fences for 4,096 tiles, and
with twelve batch slices across repeated attention layers the FPGA run appeared stalled. Batching the
synchronisation -- one fence before the kernel, the reservation station ordering the intervening
scratchpad and accumulator hazards, one load-bearing fence at the end so output DMA completes before
the CPU reads -- took the same kernel to two fences with identical arithmetic, tiling, addresses and
commands.

The invariant under test is SCALING, not a fixed budget. A schedule may legitimately carry a small
constant number of fences; what can never be right is one per tile.
"""
from __future__ import annotations

from merlin.targetgen import trace_check as TCK

_EXPECTED = {
    "instruction_classes": ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST",
                            "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"],
    "modes": {},
}

#: Every class carries a funct so ``drives_accelerator`` is satisfied and the only findings under
#: test are the tile/synchronisation ones.
_FUNCT = {"FLUSH": 7, "CONFIG_EX": 0, "CONFIG_LD": 0, "CONFIG_ST": 0,
          "MVIN": 2, "MVOUT": 3, "PRELOAD": 6, "COMPUTE_PRELOADED": 4}


def _ins(cls: str) -> dict:
    return {"class": cls, "funct": _FUNCT[cls], "decoded": {}, "rs1": None, "rs2": None}


def _command_buffer(m: int, n: int) -> dict:
    """One resident matmul over an m x k activation and a k x n weight."""
    return {
        "tensors": {"A0": {"shape": [m, 16], "dtype": "i8"},
                    "W": {"shape": [16, n], "dtype": "i8"},
                    "W_res": {"shape": [16, n], "dtype": "i8"},
                    "Y0": {"shape": [m, n], "dtype": "i32"}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
             "attributes": {"epilogue": [], "output_dtype": "i32"}},
        ],
    }


def _program(*, tiles: int, flushes: int) -> dict:
    """A kernel issuing ``tiles`` stores and ``flushes`` fences."""
    ins = [_ins("FLUSH") for _ in range(1)]
    ins += [_ins("CONFIG_EX"), _ins("CONFIG_LD"), _ins("MVIN"), _ins("CONFIG_ST")]
    for index in range(tiles):
        ins += [_ins("PRELOAD"), _ins("COMPUTE_PRELOADED"), _ins("MVOUT")]
        # A per-tile fence lands between stores; the batched form has none here.
        if flushes > 1 and index < flushes - 1:
            ins.append(_ins("FLUSH"))
    return {"instructions": ins}


def _fence_findings(violations: list[str]) -> list[str]:
    return [v for v in violations if "FLUSH count" in v and "scales with" in v]


def test_batched_synchronisation_is_accepted() -> None:
    """CONTROL. Two fences around a four-tile kernel is the shape the fix produces; without this
    passing, the refusal below would only prove the check fires on everything."""
    cb = _command_buffer(32, 32)                      # Mt=2, Nt=2 -> 4 tiles
    out = TCK.check(_program(tiles=4, flushes=2), _EXPECTED, cb)
    assert _fence_findings(out["violations"]) == [], out["violations"]


def test_per_tile_synchronisation_is_refused() -> None:
    """The defect: one fence per output tile."""
    cb = _command_buffer(32, 32)
    out = TCK.check(_program(tiles=4, flushes=4), _EXPECTED, cb)
    found = _fence_findings(out["violations"])
    assert found, out["violations"]
    assert "4 output tile(s)" in found[0]
    assert "Mt=2" in found[0] and "Nt=2" in found[0]


def test_a_single_tile_kernel_is_never_flagged() -> None:
    """A one-tile kernel cannot distinguish per-tile from batched fencing, so it must not be
    accused. Flagging it would make the check fire on the smallest capsules in the corpus, which is
    how an advisory diagnostic gets ignored."""
    cb = _command_buffer(16, 16)                      # Mt=1, Nt=1 -> 1 tile
    for flushes in (1, 2, 3):
        out = TCK.check(_program(tiles=1, flushes=flushes), _EXPECTED, cb)
        assert _fence_findings(out["violations"]) == [], (flushes, out["violations"])


def test_the_check_measures_scaling_not_a_fixed_budget() -> None:
    """A small constant number of fences is legitimate at any tile count -- the pathology is one per
    tile. Nine tiles with three fences must pass; nine tiles with nine must not."""
    cb = _command_buffer(48, 48)                      # Mt=3, Nt=3 -> 9 tiles
    ok = TCK.check(_program(tiles=9, flushes=3), _EXPECTED, cb)
    assert _fence_findings(ok["violations"]) == [], ok["violations"]
    bad = TCK.check(_program(tiles=9, flushes=9), _EXPECTED, cb)
    assert _fence_findings(bad["violations"]), bad["violations"]
