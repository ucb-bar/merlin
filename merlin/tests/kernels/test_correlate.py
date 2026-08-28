"""Which instructions came from which model region - and an honest refusal when none can.

The provenance chain survives model op -> dispatch -> kernel symbol -> runtime program -> measured
ticks. It breaks at symbol -> instruction range: with no MLIR Location threading and no debug info,
a monolithic image has one compute symbol and every instruction in the model belongs to it.

Returning that single span with attributable=True would be literally correct and would read as "we
can attribute instructions to regions". This pins the refusal.
"""
from __future__ import annotations

from merlin.kernels.correlate import attribute, symbol_of
from merlin.kernels.decode.objdump import RawInsn


def _insns(section: str, n: int, base: int = 0):
    return [RawInsn(addr=base + i * 4, mnemonic="add", operands=[], hexcode="00000013",
                    section=section) for i in range(n)]


class TestSymbolOf:
    def test_reads_the_angle_bracketed_name(self):
        assert symbol_of("10002000 <compute_kernel>") == "compute_kernel"

    def test_a_plain_section_header_is_not_a_symbol(self):
        """NEGATIVE CASE: `.text` names a section, not a symbol."""
        assert symbol_of(".text") is None
        assert symbol_of("") is None

    def test_an_unterminated_bracket_is_not_a_symbol(self):
        assert symbol_of("10002000 <oops") is None


class TestAttribution:
    def test_a_monolithic_image_is_not_attributable(self):
        a = attribute(_insns("0 <forward>", 5))
        assert a.attributable is False
        assert a.by_region() == {} and a.region_of_index(0) is None
        assert "monolithic" in a.reason

    def test_region_tagged_symbols_are_attributable(self):
        a = attribute(_insns("0 <forward$kernel_0__rmatmul_3>", 4))
        assert a.attributable and list(a.by_region()) == ["matmul_3"]
        assert a.region_of_index(3) == "matmul_3"

    def test_spans_are_split_per_symbol_with_their_address_range(self):
        raws = (_insns("0 <forward$kernel_0__ra>", 2, base=0)
                + _insns("8 <forward$kernel_1__rb>", 3, base=8))
        a = attribute(raws)
        by = a.by_region()
        assert set(by) == {"a", "b"}
        assert by["a"].n_insns == 2 and by["a"].low_addr == 0 and by["a"].high_addr == 4
        assert by["b"].n_insns == 3 and by["b"].low_addr == 8

    def test_an_untagged_symbol_beside_tagged_ones_does_not_claim_a_region(self):
        raws = _insns("0 <memcpy>", 2) + _insns("8 <forward$kernel_0__rm>", 2, base=8)
        a = attribute(raws)
        assert a.attributable and list(a.by_region()) == ["m"]
        assert a.region_of_index(0) is None       # inside memcpy: real, and not a model region

    def test_an_empty_stream_is_a_refusal_not_an_empty_success(self):
        a = attribute([])
        assert a.attributable is False and "nothing to attribute" in a.reason


class TestAgainstARealBinary:
    def test_a_real_expert_object_reports_its_symbols_and_refuses_attribution(self):
        """An expert kernel carries no merlin region ids, so it must refuse - loudly, with symbols."""
        import os
        import pytest

        from merlin.kernels.decode import rvv as rvv_decode

        obj = os.environ.get("MERLIN_TEST_EXPERT_OBJ")
        if not obj or not os.path.exists(obj):
            pytest.skip("set MERLIN_TEST_EXPERT_OBJ to a built expert object")
        raws = [i.raw for i in rvv_decode.decode(obj).insns]
        a = attribute(raws)
        assert a.spans, "a real object must yield at least one symbol span"
        assert a.attributable is False, "an expert kernel has no merlin region ids to attribute to"
