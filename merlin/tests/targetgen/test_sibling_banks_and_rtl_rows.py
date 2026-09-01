"""A store's ROW is derivable from the memory itself, not only from a compute array.

Two coupled facts, both measured 2026-09-01:

1. Sibling banks are a STRUCTURAL relation, not a spelling. The previous rule required the bank index to
   parse as `<base>_<int>` in some segment of the representative's path. The SIMT target's discovered
   operand store is `shared_mem/mem/radiance_smem_bank_ext` -- bank 0 carries NO index -- so no segment
   parsed, the group was empty, and a store of 64 banks x 512 rows x 4 bytes read as no store at all.

2. With the bank list in hand the store's own RTL SRAM word width is available, and that is the row width
   a backend addresses. Deriving the row from the compute-array edge x datapath element width is a
   systolic-only framing: a machine with no array had no row, hence no addressable row count, hence a
   memory-regime axis that required nothing of any corpus and could never fail.
"""
from __future__ import annotations

from merlin.targetgen import address_space as AS


class TestSiblingBankRelation:
    """`_are_siblings` is exercised through the public capacity derivation's own helper shape."""

    @staticmethod
    def _siblings(a: str, b: str) -> bool:
        from merlin.targetgen.rtl import mlc_bridge as MB
        import inspect
        # The relation is a closure inside discovered_capacities; re-derive it the same way rather than
        # duplicating the rule here, so the test cannot drift from the implementation.
        src = inspect.getsource(MB.discovered_capacities)
        ns: dict = {}
        body = src[src.index("    def _debank"):src.index("    def _group")]
        exec("\n".join(l[4:] for l in body.splitlines()), ns)          # noqa: S102 -- test-local
        return ns["_are_siblings"](a, b)

    def test_an_unindexed_bank_zero_groups_with_its_indexed_siblings(self):
        """The shape that reported a 64-bank store as no store."""
        assert self._siblings("shared_mem/mem/radiance_smem_bank_ext",
                              "shared_mem/mem_1/radiance_smem_bank_ext")
        assert self._siblings("shared_mem/mem/radiance_smem_bank_ext",
                              "shared_mem/mem_63/radiance_smem_bank_ext")

    def test_the_index_may_sit_in_any_segment(self):
        assert self._siblings("spad/spad_mems_0/mem_ext", "spad/spad_mems_1/mem_ext")
        assert self._siblings("mreg/banks_30_ext/mem_ext", "mreg/banks_31_ext/mem_ext")

    def test_two_different_stores_are_not_siblings(self):
        assert not self._siblings("spad/spad_mems_0/mem_ext", "spad/acc_mems_0/mem/mem_ext")
        assert not self._siblings("a/x_0/m", "a/y_1/m"), "differing bases are different stores"

    def test_paths_differing_in_two_segments_are_not_siblings(self):
        assert not self._siblings("a/x_0/m_0", "a/x_1/m_1")


class TestRtlRowWidthIsAFirstClassSource:
    """A memory that declares its own word width yields a row with no array in the facts at all."""

    @staticmethod
    def _space(mem: dict, **extra):
        return AS.derive_address_space("t", facts={"memories": [mem], **extra})

    def test_a_row_derives_with_no_array_and_no_datapath(self):
        sp = self._space({"name": "smem", "bytes": 131072, "depth": 512, "row_bytes": 4})
        (st,) = sp.stores
        assert st.row_bytes == 4, "the memory's own declared word width is the row"
        assert st.total_rows == 32768, f"131072/4 rows, got {st.total_rows}"
        assert "row_bytes" in st.sources and "RTL SRAM word" in st.sources["row_bytes"]

    def test_no_row_elems_unknown_is_raised_when_the_rtl_row_is_present(self):
        """Absent an array the old code reported row_elems UNKNOWN; with an RTL row that is not a gap."""
        sp = self._space({"name": "smem", "bytes": 4096, "row_bytes": 4})
        qs = {u.quantity for u in sp.unknowns}
        assert "row_elems" not in qs, f"array geometry is not needed here: {qs}"
        assert "element_bits" not in qs, f"a row is measured directly: {qs}"

    def test_the_array_route_still_works_when_no_memory_declares_a_word(self):
        sp = AS.derive_address_space("t", facts={
            "memories": [{"name": "spad", "bytes": 262144, "depth": 4096}],
            "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
            "datapaths": [{"name": "spad", "dtype": "i8", "elem_bits": 8}]})
        (st,) = sp.stores
        assert st.row_bytes == 16 and st.total_rows == 16384

    def test_a_disagreement_between_the_two_routes_is_stated_not_hidden(self):
        """Both routes present and contradicting: the row is still produced, and the conflict is named."""
        sp = AS.derive_address_space("t", facts={
            "memories": [{"name": "spad", "bytes": 262144, "depth": 4096, "row_bytes": 32}],
            "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
            "datapaths": [{"name": "spad", "dtype": "i8", "elem_bits": 8}]})
        (st,) = sp.stores
        assert st.row_bytes == 32, "the memory's own word width is the measurement"
        conflict = [u for u in sp.unknowns if u.quantity == "row_bytes"]
        assert conflict, "a 32-vs-16 contradiction must be recorded"
        assert "disagreement is unresolved" in conflict[0].reason
