"""MX datapath certification — merlin's mx_oracle vs a LIVE Chisel-RTL run.

This locks the numerical chain for the microscaling (MX) datapath against real hardware RTL:

    merlin mx_oracle  ==  mx golden (mlc.validate.mx_ref, itself validated vs the radiance-kernels
                          C++ RTL-mirror golden)  ==  real Chisel RTL output

The reference vector under ``merlin/tests/data/mx_rtl_cert/`` was captured by executing the
``gemm_mxgemmini`` fp8 single-tile kernel (M=N=K=64, block group=32) on the real MX datapath in the
``RadianceTapeoutSimConfig`` Verilator simulator (chipyard, gemmini-mx sources: MxPE / MxFPMul /
MxRequantizer). It stores the exact device operand codes + E8M0 block scales fed to the RTL, the
kernel's bf16 golden, and the bf16 output reconstructed from the simulator's SIMT store trace.

Two independent legs, asserted honestly:

  * ``test_mx_oracle_reproduces_golden_bitexact`` — merlin's mx_oracle, run on the SAME operand codes
    that drove the RTL, reproduces the golden BIT-EXACT (4096/4096, err 0). This is the merlin
    numerical-tier certification. Skipped only when mlc is unavailable.

  * ``test_live_rtl_reproduces_golden`` — pure numpy, runs everywhere: the values the real MX RTL
    wrote to memory agree with the golden on the overwhelming majority of cells. The (documented)
    residual is a kernel move-out memory-consistency hazard (mesh-write -> SIMT-read in the
    radiance-kernels moveout), NOT an mx datapath error — see ``meta.json`` ``rtl_run``. The bound is
    deliberately loose so the test certifies "the real datapath produces the golden values" without
    over-claiming a clean 100% capture.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import mx_oracle

_FIXTURE = repo_root() / "merlin" / "tests" / "data" / "mx_rtl_cert"

pytestmark = pytest.mark.skipif(
    not (_FIXTURE / "meta.json").is_file(),
    reason="mx RTL certification reference vector not present",
)


def _load():
    meta = json.loads((_FIXTURE / "meta.json").read_text())
    arr = {name: np.load(_FIXTURE / f"{name}.npy") for name in
           ("A_in", "B_in", "SA", "SB", "golden_bf16", "rtl_bf16")}
    return meta, arr


@pytest.mark.skipif(not mx_oracle.mx_datapath_available(),
                    reason="mlc MX reference (mlc.validate.mx_ref) not importable")
def test_mx_oracle_reproduces_golden_bitexact():
    """merlin's mx_oracle, on the exact fp8 codes + E8M0 scales fed to the real RTL, reproduces the
    mx golden bit-exact. Certifies merlin's MX numerical tier against the RTL-mirror reference."""
    meta, arr = _load()
    got = mx_oracle.mx_matmul(
        arr["A_in"].astype(np.uint8), arr["B_in"].astype(np.uint8),
        arr["SA"].astype(np.int32), arr["SB"].astype(np.int32),
        meta["M"], meta["N"], meta["K"], fmt=meta["fmt"], g=meta["G"],
    )
    assert got is not None, "mx_oracle failed closed on the certified operands"
    got_bits = (got.view(np.uint32) >> 16).astype(np.uint16)
    golden = arr["golden_bf16"].astype(np.uint16)
    assert got_bits.shape == golden.shape == (meta["M"], meta["N"])
    assert np.array_equal(got_bits, golden), "mx_oracle diverges from the mx golden"


def test_live_rtl_reproduces_golden():
    """The bf16 output the real MX Chisel RTL wrote to memory agrees with the golden on the vast
    majority of cells. The residual is the documented kernel move-out readout hazard, not a datapath
    error; the recorded per-cell result is pinned in meta.json['rtl_run']."""
    meta, arr = _load()
    golden = arr["golden_bf16"].astype(np.uint16)
    rtl = arr["rtl_bf16"].astype(np.uint16)
    assert rtl.shape == golden.shape == (meta["M"], meta["N"])
    n_exact = int(np.sum(rtl == golden))
    recorded = int(meta["rtl_run"]["cells_bit_exact"])
    # The committed vector must reproduce the recorded live-RTL result exactly (it is a fixed capture).
    assert n_exact == recorded, f"committed RTL vector drifted: {n_exact} exact vs recorded {recorded}"
    # And that recorded agreement must be the overwhelming majority (datapath is faithful; the rest is
    # the documented moveout hazard). Loose bound — this is corroboration, not a clean-capture claim.
    assert n_exact >= int(0.9 * golden.size), (
        f"real RTL agrees with golden on only {n_exact}/{golden.size} cells — below the datapath-"
        f"faithfulness bound; investigate beyond the known moveout readout hazard"
    )
