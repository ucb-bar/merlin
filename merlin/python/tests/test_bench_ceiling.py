"""Kernel-performance ceiling harness (S4.2): measure a curated kernel's spike cycle
count and join it 1:1 to a compiler-measured cycle count via the compare.py key.

Unit tests (no toolchain) cover the jsonl round-trip, attainment math, shape_regime
stability, the compare.py key contract, and graceful None paths. The real spike-run
tests are skipped when spike/riscv-gcc/saturn-corpus is unavailable.
"""
import json

import pytest

from merlin.kernels import bench_ceiling as bc
from merlin.kernels.compare import RvvFingerprint

_HAVE_SPIKE = bc.saturn_available()
_needs_spike = pytest.mark.skipif(
    not _HAVE_SPIKE,
    reason="spike/riscv-gcc/saturn-corpus/encoding.h unavailable (set MERLIN_CHIPYARD)")


# ---------------------------------------------------------------- toolchain-independent
def test_fingerprint_key_matches_compare():
    """The ceiling join key must be exactly compare.py's RvvFingerprint key shape."""
    key = bc.fingerprint_key("matmul", "i64", "square_small")
    assert key == {"op": "matmul", "dtype": "i64", "shape_regime": "square_small"}
    # and it is consumable as a compare.py key (same {op,dtype,shape_regime} dict)
    fp = RvvFingerprint.from_objdump("", key, "run")
    assert fp.key == key


def test_shape_regime_is_deterministic_string():
    assert bc.shape_regime("matmul", 8, 8, 8) == "square_small"
    assert bc.shape_regime("matmul", 32, 32, 32) == "square_medium"
    assert bc.shape_regime("matmul", 512, 512, 512) == "square_large"
    assert bc.shape_regime("dotprod", 512, 1, 1) == "vector"
    assert bc.shape_regime("matmul", 8, 256, 256) == "skinny"
    assert bc.shape_regime("matmul", 64, 128, 64) == "rectangular"
    # stable: same inputs -> same string
    assert bc.shape_regime("matmul", 32, 32, 32) == bc.shape_regime("matmul", 32, 32, 32)


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "ceiling.jsonl"
    row = {
        "op": "matmul", "dtype": "i64", "M": 32, "N": 32, "K": 32,
        "shape_regime": "square_medium", "source": "saturn", "target": "spike",
        "bench": "vec-igemm", "kernel_ref": "imatmul", "cycles": 26055,
        "isa": bc.DEFAULT_ISA,
        "fingerprint_key": bc.fingerprint_key("matmul", "i64", "square_medium"),
    }
    bc.append_ceiling(row, path)
    bc.append_ceiling({**row, "M": 64, "N": 64, "K": 64, "cycles": 200087}, path)
    rows = bc.load_ceiling(path)
    assert len(rows) == 2
    assert rows[0] == row
    assert rows[1]["cycles"] == 200087
    # each line is independently valid JSON
    for line in path.read_text().splitlines():
        json.loads(line)


def test_load_ceiling_missing_file(tmp_path):
    assert bc.load_ceiling(tmp_path / "absent.jsonl") == []


def test_attainment_math():
    row = {"cycles": 26055}
    # compiler matches the ceiling exactly -> 1.0
    assert bc.attainment(row, 26055) == pytest.approx(1.0)
    # compiler slower than expert -> < 1.0 (leaving performance on the table)
    assert bc.attainment(row, 30000) == pytest.approx(26055 / 30000)
    # compiler faster than expert -> > 1.0
    assert bc.attainment(row, 20000) > 1.0


def test_attainment_guards():
    assert bc.attainment({"cycles": 100}, 0) is None
    assert bc.attainment({"cycles": 100}, -5) is None
    assert bc.attainment({}, 100) is None
    assert bc.attainment({"cycles": 100}, None) is None


def test_find_ceiling_by_key():
    rows = [
        {"fingerprint_key": bc.fingerprint_key("matmul", "i64", "square_small"),
         "cycles": 573},
        {"fingerprint_key": bc.fingerprint_key("dotprod", "i64", "vector"),
         "cycles": 401},
    ]
    hit = bc.find_ceiling(rows, "dotprod", "i64", "vector")
    assert hit is not None and hit["cycles"] == 401
    assert bc.find_ceiling(rows, "matmul", "f32", "square_small") is None


def test_ceiling_joins_to_compiler_fingerprint():
    """End-to-end key contract: a ceiling row and a generated RvvFingerprint built for the
    SAME op-shape share the exact join key, so attainment pairs them 1:1."""
    regime = bc.shape_regime("matmul", 32, 32, 32)
    ceiling_row = {"cycles": 26055,
                   "fingerprint_key": bc.fingerprint_key("matmul", "i64", regime)}
    gen = RvvFingerprint.from_objdump("", bc.fingerprint_key("matmul", "i64", regime), "ours")
    assert ceiling_row["fingerprint_key"] == gen.key
    # pretend our compiler took 30k cycles for the same shape
    assert bc.attainment(ceiling_row, 30000) == pytest.approx(26055 / 30000)


def test_vopacc_excluded_and_bad_target():
    assert bc.run_kernel_ceiling("saturn", "vec-VOPACC-x", "matmul", "i64", (8, 8, 8)) is None
    assert bc.run_kernel_ceiling("saturn", "vec-igemm", "", "", (8, 8, 8),
                                 target="board") is None
    assert bc.run_kernel_ceiling("xnnpack", "vec-igemm", "", "", (8, 8, 8)) is None
    assert bc.run_kernel_ceiling("saturn", "no-such-bench", "", "", (8, 8, 8)) is None


# --------------------------------------------------------------------------- spike-gated
@_needs_spike
def test_igemm_ceiling_real_spike():
    row = bc.run_kernel_ceiling("saturn", "vec-igemm", "", "", (32, 32, 32))
    assert row is not None, "vec-igemm 32x32x32 should build + run on spike"
    assert row["op"] == "matmul" and row["target"] == "spike"
    assert row["cycles"] > 0
    assert row["fingerprint_key"] == bc.fingerprint_key("matmul", row["dtype"],
                                                        row["shape_regime"])
    # demonstrate attainment against a dummy compiler cycle count
    att = bc.attainment(row, row["cycles"] * 1.2)
    assert att == pytest.approx(1 / 1.2)


@_needs_spike
def test_igemm_unmeasured_size_returns_none():
    # vec-igemm only sweeps power-of-two squares up to M(=64); 7x7x7 is not measured.
    assert bc.run_kernel_ceiling("saturn", "vec-igemm", "", "", (7, 7, 7)) is None


@_needs_spike
def test_dotprod_ceiling_real_spike():
    row = bc.run_kernel_ceiling("saturn", "vec-dotprod", "", "", (512, 1, 1))
    assert row is not None
    assert row["op"] == "dotprod" and row["shape_regime"] == "vector"
    assert row["cycles"] > 0 and row.get("instructions", 0) > 0
