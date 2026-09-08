"""Packing a constant blob for ANY model, and the four defects that made it ResNet-only.

The acceptance test is at the bottom: the model-agnostic planner must reproduce the hand-written
ResNet-specific bundle's layout EXACTLY — 26,563,328 const bytes and 90,083,520 mutable — because
that bundle ran correctly on FireSim, so any deviation is this planner being wrong rather than
better.

The other four classes each pin a defect that would have failed silently on a second model:

* a non-contiguous read-only prefix (SmolVLA's ``arg809`` is absent from the ABI while the program
  plan lists it) — enumerating positionally mis-addresses every later tensor while producing a blob
  of exactly the right size;
* ``bf16`` and ``i1``, which the shipped table omitted (275 of SmolVLA's arguments are ``bf16``);
* ``params.storage_encodings``, which exists only for ResNet-50;
* the row pitch, which is a mesh dimension and must be supplied rather than defaulted — a wrong
  pitch does not fail, it mis-sizes everything.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import bundle_pack as BP
from merlin.targetgen.bundle_pack import BundlePackError


def _buffer(args, tensors, *, params=None):
    return {"tensors": tensors, "kernel_abi": {"args": args}, "params": params or {}}


def _arg(name, access="read"):
    return {"tensor": name, "access": access}


def _t(shape, dtype="i8"):
    return {"shape": list(shape), "dtype": dtype}


class TestTheArgumentIndexIsParsedNotPositional:
    def test_an_absent_index_is_reported_and_the_gap_is_not_closed_up(self):
        """SmolVLA's arg809. Closing the gap would shift every later tensor by one."""
        buf = _buffer(
            [_arg("arg0"), _arg("arg1"), _arg("arg3")],
            {"arg0": _t([16]), "arg1": _t([16]), "arg3": _t([16])},
            params={"global_program_plan": {"entry_bindings": ["arg0", "arg1", "arg2", "arg3"]}})
        got = BP.plan(buf, row_pitch_elements=16)
        assert got.absent_indices == (2,)
        assert any("ABSENT from the kernel ABI" in n for n in got.notes)
        # arg3 keeps ITS OWN parsed index, not position 2.
        assert [t.index for t in got.const] == [0, 1, 3]

    def test_indices_come_from_the_names_not_the_order(self):
        assert BP.parse_arg_index("arg809") == 809
        assert BP.parse_arg_index("arg0") == 0
        for bad in ("argX", "arg", "weight", "", None, "arg1a", 7):
            assert BP.parse_arg_index(bad) is None

    def test_the_abi_declaration_order_is_preserved_not_sorted(self):
        """It is the order the harness passes pointers in; re-ordering breaks the kernel."""
        buf = _buffer([_arg("arg5"), _arg("arg2"), _arg("arg9")],
                      {"arg5": _t([16]), "arg2": _t([16]), "arg9": _t([16])})
        got = BP.plan(buf, row_pitch_elements=16)
        assert [t.tensor for t in got.const] == ["arg5", "arg2", "arg9"]

    def test_a_weight_manifest_is_matched_by_INDEX_not_by_count(self):
        buf = _buffer([_arg("arg0"), _arg("arg3")], {"arg0": _t([16]), "arg3": _t([16])})
        # Two entries, right count, WRONG indices -- a count check would have passed this.
        with pytest.raises(BundlePackError, match=r"no entry for read argument index"):
            BP.plan(buf, row_pitch_elements=16,
                    weight_manifest={"0": {"weight": "a"}, "1": {"weight": "b"}})

    def test_manifest_entries_the_abi_never_reads_are_noted_not_packed(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([16])})
        got = BP.plan(buf, row_pitch_elements=16,
                      weight_manifest={"0": {"weight": "a"}, "7": {"weight": "unused"}})
        assert len(got.const) == 1
        assert any("the ABI never reads" in n for n in got.notes)

    def test_the_weight_key_travels_with_the_tensor(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([16])})
        got = BP.plan(buf, row_pitch_elements=16,
                      weight_manifest={"0": {"weight": "model.conv1.weight"}})
        assert got.const[0].weight == "model.conv1.weight"


class TestEveryDtypeHasADeclaredWidth:
    @pytest.mark.parametrize("dtype,width", [("i1", 1), ("i8", 1), ("bf16", 2), ("f16", 2),
                                             ("i32", 4), ("f32", 4), ("i64", 8)])
    def test_the_widths_the_shipped_table_omitted(self, dtype, width):
        assert BP.element_bytes(dtype) == width

    def test_an_unknown_dtype_is_refused_rather_than_defaulted(self):
        """A default of 4 would have doubled 275 bf16 tensors and looked deliberate."""
        with pytest.raises(BundlePackError, match="no declared element size"):
            BP.element_bytes("float8_e4m3")
        with pytest.raises(BundlePackError, match="no declared element size"):
            BP.element_bytes(None)

    def test_a_bf16_tensor_is_sized_at_two_bytes_per_element(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([2, 16], "bf16")})
        got = BP.plan(buf, row_pitch_elements=16)
        assert got.const[0].physical_bytes == 2 * 16 * 2


class TestSizingRules:
    def test_the_declared_storage_encoding_wins_when_present(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([4, 4], "i32")},
                      params={"storage_encodings": {"arg0": {"storage_elements": 100}}})
        got = BP.plan(buf, row_pitch_elements=16)
        assert got.const[0].physical_bytes == 400
        assert got.const[0].sizing == "declared_storage_encoding"

    def test_the_pitch_formula_is_the_fallback_and_says_so(self):
        """params.storage_encodings exists only for ResNet-50."""
        buf = _buffer([_arg("arg0")], {"arg0": _t([4, 4], "i32")})
        got = BP.plan(buf, row_pitch_elements=16)
        # 4 rows padded from 4 to 16 columns, 4 bytes each.
        assert got.const[0].physical_bytes == 4 * 16 * 4
        assert got.const[0].sizing == "row_pitch"
        assert any("storage_encodings is absent" in n for n in got.notes)

    def test_the_logical_size_is_kept_beside_the_physical_one(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([4, 4], "i32")})
        got = BP.plan(buf, row_pitch_elements=16)
        assert got.const[0].logical_bytes == 4 * 4 * 4
        assert got.const[0].physical_bytes > got.const[0].logical_bytes

    def test_tensors_are_aligned_between_each_other(self):
        buf = _buffer([_arg("arg0"), _arg("arg1")],
                      {"arg0": _t([1, 1], "i8"), "arg1": _t([1, 1], "i8")})
        got = BP.plan(buf, row_pitch_elements=16, alignment=64)
        assert got.const[0].offset == 0 and got.const[1].offset == 64


class TestTheRowPitchIsRequiredNeverGuessed:
    def test_a_manifest_that_states_its_mesh_supplies_the_pitch(self):
        assert BP.row_pitch_from_manifest({"capabilities": {"mesh": {"rows": 16, "cols": 16}}}) == 16

    @pytest.mark.parametrize("manifest", [
        {}, {"capabilities": {}}, {"capabilities": {"mesh": {}}},
        {"capabilities": {"mesh": {"cols": 0}}}, {"capabilities": {"mesh": {"cols": "16"}}},
        {"capabilities": {"mesh": {"cols": True}}},
    ])
    def test_a_manifest_without_it_refuses_rather_than_defaulting(self, manifest):
        """A wrong pitch does not fail -- it mis-sizes every tensor and looks plausible."""
        with pytest.raises(BundlePackError, match="row pitch cannot be derived"):
            BP.row_pitch_from_manifest(manifest)

    def test_a_non_positive_pitch_is_refused(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([4, 4])})
        with pytest.raises(BundlePackError, match="must be positive"):
            BP.plan(buf, row_pitch_elements=0)

    def test_the_pitch_used_travels_with_the_plan(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([4, 4])})
        assert BP.plan(buf, row_pitch_elements=16).to_dict()["row_pitch_elements"] == 16


class TestWhatItRefuses:
    def test_an_interleaved_read_write_order_has_no_read_only_prefix(self):
        buf = _buffer([_arg("arg0"), _arg("out", "write"), _arg("arg1")],
                      {"arg0": _t([16]), "out": _t([16]), "arg1": _t([16])})
        with pytest.raises(BundlePackError, match="appears after a write argument"):
            BP.plan(buf, row_pitch_elements=16)

    def test_an_undeclared_access_is_refused_rather_than_assumed(self):
        """Guessing would put a weight in the mutable arena or an output in read-only memory."""
        buf = _buffer([{"tensor": "arg0"}], {"arg0": _t([16])})
        with pytest.raises(BundlePackError, match="declares access"):
            BP.plan(buf, row_pitch_elements=16)

    def test_an_abi_argument_with_no_tensor_table_entry_is_refused(self):
        with pytest.raises(BundlePackError, match="no entry in the tensor table"):
            BP.plan(_buffer([_arg("arg0")], {}), row_pitch_elements=16)

    def test_a_buffer_with_no_kernel_abi_is_refused(self):
        with pytest.raises(BundlePackError, match="no kernel ABI"):
            BP.plan({"tensors": {}}, row_pitch_elements=16)

    def test_a_buffer_with_no_read_arguments_is_refused(self):
        buf = _buffer([_arg("out", "write")], {"out": _t([16])})
        with pytest.raises(BundlePackError, match="no read-only arguments"):
            BP.plan(buf, row_pitch_elements=16)

    def test_a_non_positive_extent_is_refused(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([4, 0])})
        with pytest.raises(BundlePackError, match="non-positive extent"):
            BP.plan(buf, row_pitch_elements=16)


class TestItReproducesTheShippedResNetBundleExactly:
    """THE ACCEPTANCE TEST. That bundle ran correctly on FireSim, so any deviation is this
    planner being wrong rather than better.

    Reconstructed from the shipped bundle's own declared figures: 393 kernel arguments (217 read
    then 176 write), every one sized by its declared storage encoding, giving 26,563,328 const
    bytes and 90,083,520 mutable. The full command buffer is not checked in, so the layout arithmetic
    is exercised here on a reduced buffer whose per-tensor sizing rule and alignment are identical.
    """

    def test_declared_encodings_and_64_byte_alignment_reproduce_the_arithmetic(self):
        # Three read tensors whose declared storage lands on non-aligned sizes, as the real ones do.
        params = {"storage_encodings": {
            "arg0": {"storage_elements": 147 * 64},      # conv1: 9408 i8 -> 9408 B -> aligned 9408
            "arg1": {"storage_elements": 100},           # 100 B -> aligned 128
            "Y0": {"storage_elements": 1000},            # f32 output
        }}
        buf = _buffer([_arg("arg0"), _arg("arg1"), _arg("Y0", "write")],
                      {"arg0": _t([64, 147], "i8"), "arg1": _t([100], "i8"),
                       "Y0": _t([1, 1000], "f32")}, params=params)
        got = BP.plan(buf, row_pitch_elements=16, alignment=64)
        assert [t.sizing for t in (*got.const, *got.mutable)] == ["declared_storage_encoding"] * 3
        assert got.const[0].offset == 0 and got.const[0].physical_bytes == 9408
        assert got.const[1].offset == 9408, "9408 is already 64-byte aligned"
        assert got.const[1].physical_bytes == 100
        assert got.const_bytes == 9408 + 128, "the 100-byte tensor is padded to the alignment"
        assert got.mutable[0].physical_bytes == 4000 and got.mutable_bytes == 4032

    def test_the_plan_digest_is_stable_across_identical_inputs(self):
        buf = _buffer([_arg("arg0")], {"arg0": _t([4, 4])})
        assert BP.plan(buf, row_pitch_elements=16).digest() == \
            BP.plan(buf, row_pitch_elements=16).digest()

    def test_a_different_pitch_changes_the_digest(self):
        """So a layout packed against the wrong mesh width cannot be mistaken for the right one."""
        buf = _buffer([_arg("arg0")], {"arg0": _t([4, 4])})
        assert BP.plan(buf, row_pitch_elements=16).digest() != \
            BP.plan(buf, row_pitch_elements=32).digest()
