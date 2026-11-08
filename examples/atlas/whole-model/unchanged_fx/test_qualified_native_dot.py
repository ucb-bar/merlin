"""Exercise raw arithmetic storage, qualification binding, and proxy refusal."""
import json
from pathlib import Path

import numpy as np
import pytest

import qualified_native_dot as native


def unit_product(a, b):
    return ((a ^ b) >> 7, 13 if a & 127 and b & 127 else 0, 0)


class OriginalMath:
    @staticmethod
    def inner_product_matmul(a, b, c):
        raise AssertionError("the selected dot must execute the native candidate")

    @staticmethod
    def other_operation(value):
        return ("original", value)


@pytest.fixture
def candidate(tmp_path):
    return native.candidate_for_qualification(unit_product, tmp_path)


def test_exhaustive_table_indexing_and_readonly_metadata(candidate):
    a = np.repeat(np.arange(256, dtype=np.uint8), 256)
    b = np.tile(np.arange(256, dtype=np.uint8), 256)
    expected = np.array([(s << 16) | (e << 8) | m for s, e, m in
                         (unit_product(int(left), int(right)) for left, right in zip(a, b, strict=True))], np.uint32)
    np.testing.assert_array_equal(candidate.lookup(a, b), expected)
    assert not candidate.table.flags.writeable


def test_non_model_shapes_and_defined_signed_wrap(candidate):
    a = np.ones((3, 32), np.uint8)
    b = np.broadcast_to(np.array([1, 129, 0], np.uint8), (32, 3)).copy()
    seed = np.zeros((3, 3), np.uint16)
    frozen = [value.tobytes() for value in (a, b, seed)]
    expected = np.broadcast_to(np.array([0x4200, 0xc200, 0], np.uint16), (3, 3))
    np.testing.assert_array_equal(candidate.matmul_codes(a, b, seed), expected)
    assert frozen == [value.tobytes() for value in (a, b, seed)]
    wrapped = candidate.matmul_codes(np.ones((1, 128), np.uint8), np.ones((128, 1), np.uint8), np.zeros((1, 1), np.uint16))
    assert int(wrapped[0, 0]) == 0xc300  # +128 unit terms wrap the signed anchor integer to -128.
    flushed = candidate.matmul_codes(np.zeros((1, 32), np.uint8), np.zeros((32, 1), np.uint8), np.array([[1]], np.uint16))
    assert int(flushed[0, 0]) == 0


def test_missing_qualification_refuses_before_build(tmp_path):
    with pytest.raises(ValueError, match="matching passing qualification"):
        native.load_qualified_math(None, None, {}, None, tmp_path)


def test_product_field_contract_rejects_invalid_callback():
    with pytest.raises(ValueError, match="format"):
        native.derive_product_table(lambda a, b: (2, 0, 0))


def test_qualification_binds_sources_library_table_and_proxy(candidate, tmp_path):
    torch = pytest.importorskip("torch")
    original = OriginalMath()
    sources = {"math": Path(__file__)}
    identity = native.backend_identity(candidate, original, unit_product, sources)
    record = {"schema": "merlin.qualified_native_dot.v1", "status": "PASS", "mismatches": 0,
              "original_python_comparison": True, "backend_identity": identity,
              "case_groups": {key: {"status": "PASS", "mismatches": 0} for key in
                              ("product_table", "all_seed_codes", "directed", "random_matrices", "continuation", "non_model_shapes")}}
    receipt = tmp_path / "qualification.json"
    receipt.write_text(json.dumps(record))
    proxy = native.load_qualified_math(original, unit_product, sources, receipt, tmp_path)
    assert proxy.backend_name == "qualified_native"
    assert proxy.other_operation(7) == ("original", 7)
    a, b = torch.ones((1, 32), dtype=torch.uint8), torch.ones((32, 1), dtype=torch.uint8)
    seed = torch.zeros((1, 1), dtype=torch.bfloat16)
    assert int(proxy.inner_product_matmul(a, b, seed).view(torch.uint16)[0, 0]) == 0x4200
    assert proxy.calls == 1
    assert proxy.execution_metadata()["original_arithmetic_implementation_unchanged"] is False
    with pytest.raises(ValueError, match="K=32"):
        proxy.inner_product_matmul(a[:, :16], b[:16], seed)
    with pytest.raises(ValueError, match="byte-code"):
        proxy.inner_product_matmul(a.to(torch.bfloat16), b, seed)
    record["backend_identity"] = {**identity, "product_table_sha256": "0" * 64}
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not match"):
        native.load_qualified_math(original, unit_product, sources, receipt, tmp_path)
