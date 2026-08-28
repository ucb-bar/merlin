"""The vector length reaches codegen through three independent pins, and they must not drift.

A model object, the Zephyr/spike cflags, and the simulator's ISA string each append `_zvl<N>b` in
their own function. That is not cosmetic duplication: `-march=...zvl<N>b` makes the compiler emit
scalable-vector spill slots whose addresses are computed from `vlenb` READ AT RUN TIME, so if the
simulator reports a different vlenb than the build assumed, every such slot lands at the wrong offset
and writes over whatever sits nearby. Build and run disagreeing is not a slowdown; it is corruption.

These pin two things. The three sites must produce the SAME suffix for the same number -- drift
between them is exactly the failure above. And the number itself now has a single typed home on
`Host.vlen`, derived from the board, rather than each caller rediscovering it.
"""
from __future__ import annotations

import pytest

_VLENS = (128, 256, 512)


def _suffix(text: str) -> str | None:
    """The `zvl<N>b` token in a march string, or None."""
    for tok in str(text).replace(",", " ").replace("-march=", " ").split():
        for part in tok.split("_"):
            if part.startswith("zvl") and part.endswith("b"):
                return part
    return None


@pytest.mark.parametrize("vlen", _VLENS)
def test_every_site_pins_the_same_vector_length(vlen):
    from merlin.mining.k1 import codegen_march
    from merlin.runtime.backends.zephyr_model import march_with_vlen, spike_isa

    got = {
        "board object": _suffix(codegen_march(vlen=vlen)),
        "cflags": _suffix(" ".join(march_with_vlen(["-march=rv64gcv"], vlen))),
        "simulator": _suffix(spike_isa(vlen)),
    }
    pinned = {k: v for k, v in got.items() if v}
    assert len(set(pinned.values())) == 1, (
        f"the vector length is pinned differently across sites at VLEN={vlen}: {pinned}. "
        f"A build and a run that disagree on vlenb corrupt every scalable spill slot.")
    assert next(iter(pinned.values())) == f"zvl{vlen}b"


def test_the_board_is_where_the_number_comes_from():
    """`Host.vlen` is derived from the board descriptor, which is the one place that knows it. A
    caller that re-reads an env var instead can pin a different length than the image was built for."""
    from merlin.system.derive import host_from_board

    h = host_from_board("chipyard_kodiak")
    assert h.vlen, "this board declares a vector length; the Host must carry it"
    assert isinstance(h.vlen, int)


def test_an_undeclared_vector_length_stays_unknown():
    """None means unknown, and codegen must then not pin a zvl at all -- pinning a guessed length is
    how an image runs at half its declared width, or corrupts a spill slot."""
    from merlin.system.derive import host_from_board

    h = host_from_board("spike_riscv64")
    assert h.vlen is None
    assert h.vector_capable() is None, "unknown width and unknown capability are both real answers"
