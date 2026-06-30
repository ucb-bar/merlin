"""Ingest adapters parse op/dtype/target from real-shaped fixtures."""
import os
from merlin.common.paths import merlin_dir

from merlin.kernels.ingest.generic import ingest_generic

DATA = str(merlin_dir() / "tests" / "data" / "kernels")


def _one(path, **kw):
    return list(ingest_generic(path, **kw))[0]


def test_generic_reads_file():
    nk = _one(os.path.join(DATA, "xnnpack_qs8_gemm_rvv.c"), source="xnnpack", target="rvv")
    assert nk.source == "xnnpack" and nk.target == "rvv"
    assert "__riscv_vsetvl" in nk.raw_text


def test_xnnpack_signature_parse():
    from merlin.kernels.ingest.xnnpack import _record_from_file
    from pathlib import Path
    p = Path(DATA) / "xnnpack_qs8_gemm_rvv.c"
    nk = _record_from_file(p, Path(DATA), "rvv")
    assert nk.op == "gemm"
    assert nk.dtype == "i8"
    assert nk.shape.get("MR") == 4


def test_autocomp_signature_parse():
    from merlin.kernels.ingest.autocomp import _parse_signature
    text = (Path_open(os.path.join(DATA, "autocomp_gemmini_matmul.c")))
    op, dtype, shape = _parse_signature(text)
    assert op == "matmul"
    assert dtype == "i8"
    assert shape == {"M": 512, "K": 512, "N": 512}


def Path_open(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()
