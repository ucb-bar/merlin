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


class TestWritingTheBytes:
    """Planning a layout and producing one are different claims; this is the second.

    THE ACCEPTANCE TEST is byte-for-byte agreement with the shipped ResNet-50 blob: 217/217 tensors,
    all 26,563,328 bytes, packed from the capture that bundle pinned. That bundle ran correctly on
    FireSim, so a difference is this writer being wrong. Getting there needed the declared weight
    prepack -- without it 216 of 217 matched and the dense layer was transposed, which is correct
    arithmetic on the wrong bytes for one layer out of 54 and is invisible to any size check.
    """

    def _plan(self, shape=(2, 16), dtype="i8", extra=None):
        tensors = {"arg0": _t(shape, dtype), "Y0": _t((1, 4), "i32")}
        args = [_arg("arg0"), _arg("Y0", "write")]
        return BP.plan(_buffer(args, tensors, params=extra or {}), row_pitch_elements=16)

    def _source(self, payload):
        def source(key):
            if key not in payload:
                raise KeyError(key)
            return payload[key]
        return source

    def test_it_writes_exactly_the_planned_size(self, tmp_path):
        plan = self._plan()
        receipt = BP.write_const_blob(plan, self._source({"arg0": bytes(32)}),
                                      tmp_path / "const.bin")
        assert receipt["bytes"] == plan.const_bytes
        assert (tmp_path / "const.bin").stat().st_size == plan.const_bytes

    def test_the_receipt_digests_the_bytes_actually_written(self, tmp_path):
        import hashlib
        plan = self._plan()
        receipt = BP.write_const_blob(plan, self._source({"arg0": bytes(range(32))}),
                                      tmp_path / "const.bin")
        on_disk = hashlib.sha256((tmp_path / "const.bin").read_bytes()).hexdigest()
        assert receipt["sha256"] == on_disk

    def test_a_missing_weight_is_refused_never_packed_as_zeros(self, tmp_path):
        """A zero weight is a real number the device would happily compute with."""
        with pytest.raises(BundlePackError, match="cannot be packed as zeros"):
            BP.write_const_blob(self._plan(), self._source({}), tmp_path / "const.bin")

    def test_a_short_or_long_source_is_refused(self, tmp_path):
        for wrong in (bytes(31), bytes(33)):
            with pytest.raises(BundlePackError, match="byte\\(s\\) and the source supplied"):
                BP.write_const_blob(self._plan(), self._source({"arg0": wrong}),
                                    tmp_path / "const.bin")

    def test_row_padding_is_written_explicitly_as_zeros(self, tmp_path):
        """A reader cannot tell an intentional zero from an uninitialised one."""
        plan = self._plan(shape=(2, 4), dtype="i8")          # 4 cols padded to a 16-element pitch
        BP.write_const_blob(plan, self._source({"arg0": bytes([7] * 8)}), tmp_path / "const.bin")
        written = (tmp_path / "const.bin").read_bytes()
        assert written[:4] == bytes([7] * 4), "the row's real bytes"
        assert written[4:16] == bytes(12), "its pad, explicitly zero"

    def test_the_manifest_names_the_source_under_either_spelling(self, tmp_path):
        """A capture spells a parameter's source `weight` and a graph INPUT's `name`."""
        plan = self._plan()
        for field in ("weight", "name"):
            receipt = BP.write_const_blob(
                plan, self._source({"the_source": bytes(32)}), tmp_path / f"c_{field}.bin",
                weight_manifest={"0": {field: "the_source"}})
            assert receipt["tensors"][0]["weight"] == "the_source"


class TestThePrepackIsAppliedFromTheDeclaredRecipe:
    def test_a_reshape_transition_leaves_the_bytes_alone(self):
        raw = bytes(range(24))
        recipe = {"tensor": "w", "source_shape": [2, 3, 2, 2], "packed_shape": [2, 12],
                  "source_layout": "OIHW", "packed_layout": "CoK_dim_padded"}
        assert BP.prepack_bytes(raw, recipe, dtype="i8") == raw

    def test_a_transpose_transition_moves_the_elements(self):
        """The dense layer: NK [1000,2048] -> KN_dim_padded [2048,1000]."""
        raw = bytes([0, 1, 2, 3, 4, 5])                       # 2x3, row-major
        recipe = {"tensor": "w", "source_shape": [2, 3], "packed_shape": [3, 2],
                  "source_layout": "NK", "packed_layout": "KN_dim_padded"}
        assert BP.prepack_bytes(raw, recipe, dtype="i8") == bytes([0, 3, 1, 4, 2, 5])

    def test_a_transpose_carries_multi_byte_elements_intact(self):
        """Byte-wise, so a dtype numpy cannot represent (bf16) permutes losslessly."""
        raw = bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22])   # 2x2 of 2-byte elements
        recipe = {"tensor": "w", "source_shape": [2, 2], "packed_shape": [2, 2],
                  "source_layout": "NK", "packed_layout": "KN_dim_padded"}
        got = BP.prepack_bytes(raw, recipe, dtype="bf16")
        assert got == bytes([0xAA, 0xBB, 0xEE, 0xFF, 0xCC, 0xDD, 0x11, 0x22])

    def test_an_undeclared_transition_is_refused_not_assumed_to_be_a_reshape(self):
        """A permutation assumed away is invisible: the element count always matches."""
        recipe = {"tensor": "w", "source_shape": [2, 3], "packed_shape": [3, 2],
                  "source_layout": "HWIO", "packed_layout": "something_new"}
        with pytest.raises(BundlePackError, match="does not describe"):
            BP.prepack_bytes(bytes(6), recipe, dtype="i8")

    def test_a_source_shape_of_the_wrong_rank_is_refused(self):
        recipe = {"tensor": "w", "source_shape": [6], "packed_shape": [3, 2],
                  "source_layout": "NK", "packed_layout": "KN_dim_padded"}
        with pytest.raises(BundlePackError, match="permutes 2 axes"):
            BP.prepack_bytes(bytes(6), recipe, dtype="i8")

    def test_a_source_of_the_wrong_size_is_refused(self):
        recipe = {"tensor": "w", "source_shape": [2, 3], "packed_shape": [3, 2],
                  "source_layout": "NK", "packed_layout": "KN_dim_padded"}
        with pytest.raises(BundlePackError, match="needs 6 source byte"):
            BP.prepack_bytes(bytes(5), recipe, dtype="i8")

    def test_the_receipt_records_which_transition_was_applied(self, tmp_path):
        tensors = {"arg0": _t((3, 2), "i8"), "Y0": _t((1, 4), "i32")}
        plan = BP.plan(_buffer([_arg("arg0"), _arg("Y0", "write")], tensors),
                       row_pitch_elements=16)
        receipt = BP.write_const_blob(
            plan, lambda k: bytes(range(6)), tmp_path / "c.bin",
            prepack_recipes=[{"tensor": "arg0", "source_shape": [2, 3], "packed_shape": [3, 2],
                              "source_layout": "NK", "packed_layout": "KN_dim_padded"}])
        assert receipt["tensors"][0]["prepack"] == "NK->KN_dim_padded"

    def test_a_tensor_with_no_recipe_is_left_alone(self, tmp_path):
        plan = BP.plan(_buffer([_arg("arg0"), _arg("Y0", "write")],
                               {"arg0": _t((1, 16), "i8"), "Y0": _t((1, 4), "i32")}),
                       row_pitch_elements=16)
        receipt = BP.write_const_blob(plan, lambda k: bytes(range(16)), tmp_path / "c.bin",
                                      prepack_recipes=[{"tensor": "somethingelse"}])
        assert receipt["tensors"][0]["prepack"] is None


class TestARecurrentSessionsCarriedStateCannotLiveInReadOnlyMemory:
    """The fifth defect, and the one the command buffer cannot possibly report.

    Within ONE invocation a carried state's input genuinely is a read, so the emitted ABI marks it
    ``read`` and the planner puts it in the const blob. Only the capture's session contract knows
    the loop writes the output back into it every step. Measured on SmolVLA's flow_denoise: all
    three carries (``prefix_kv_cache`` 2,314,240 B, ``flow_state`` 6,400 B, ``timestep`` 64 B) land
    in the read-only blob from the command buffer alone.

    So the placement is driven by the declaration, and each carried input keeps a ``seed`` row in
    the const blob while its ABI pointer targets a mutable working copy.
    """

    def _buffer(self):
        tensors = {"arg0": _t((4, 4), "i8"), "arg1": _t((1, 8), "f32"),
                   "Y0": _t((1, 8), "f32"), "Y1": _t((1, 4), "i32")}
        return _buffer([_arg("arg0"), _arg("arg1"),
                        _arg("Y0", "write"), _arg("Y1", "write")], tensors)

    def _states(self):
        return (BP.SessionState(name="hidden", input_arg=1, output_index=0),)

    def test_without_the_contract_the_carried_input_is_READ_ONLY(self):
        """The state the defect lives in, pinned so the fix cannot be mistaken for a no-op."""
        plan = BP.plan(self._buffer(), row_pitch_elements=16)
        assert [t.tensor for t in plan.const] == ["arg0", "arg1"]
        assert all(t.role == "argument" for t in plan.const)

    def test_with_the_contract_it_gains_a_mutable_working_copy_and_a_const_seed(self):
        plan = BP.plan(self._buffer(), row_pitch_elements=16, session_states=self._states())
        seeds = [t for t in plan.const if t.role == "seed"]
        assert [t.tensor for t in seeds] == ["arg1"]
        working = [t for t in plan.mutable if t.tensor == "arg1"]
        assert len(working) == 1 and working[0].role == "argument"
        assert working[0].physical_bytes == seeds[0].physical_bytes

    def test_the_carry_row_names_all_three_offsets(self):
        plan = BP.plan(self._buffer(), row_pitch_elements=16, session_states=self._states())
        assert len(plan.carried) == 1
        row = plan.carried[0]
        assert row["state"] == "hidden" and row["input_arg"] == 1
        assert row["output_tensor"] == "Y0"
        for key in ("seed_offset", "working_offset", "output_offset", "bytes"):
            assert key in row
        assert row["bytes"] > 0

    def test_the_ABI_ORDER_is_recorded_and_const_plus_mutable_is_NOT_it(self):
        """The latent bug a second model surfaced: the harness built its pointer list as
        ``const + mutable``, which equals the ABI order only while every read precedes every write.
        """
        plan = BP.plan(self._buffer(), row_pitch_elements=16, session_states=self._states())
        assert plan.abi_order == ("arg0", "arg1", "Y0", "Y1")
        assert [t.tensor for t in plan.arguments] == list(plan.abi_order)
        concatenated = [t.tensor for t in (*plan.const, *plan.mutable)]
        assert concatenated != list(plan.abi_order), \
            "if these agree the test proves nothing; the whole point is that they diverge"

    def test_a_feed_forward_plan_is_BYTE_FOR_BYTE_unchanged(self):
        """No carries declared must mean nothing moves -- the ResNet-50 acceptance test still holds."""
        buffer = self._buffer()
        assert BP.plan(buffer, row_pitch_elements=16).digest() == \
            BP.plan(buffer, row_pitch_elements=16, session_states=()).digest()

    def test_the_const_blob_size_is_unchanged_by_the_reclassification(self):
        """The seed keeps the bytes, so a packed const blob is identical; only mutable grows."""
        plain = BP.plan(self._buffer(), row_pitch_elements=16)
        session = BP.plan(self._buffer(), row_pitch_elements=16, session_states=self._states())
        assert session.const_bytes == plain.const_bytes
        assert session.mutable_bytes > plain.mutable_bytes

    def test_a_state_naming_an_input_the_ABI_does_not_READ_is_REFUSED(self):
        with pytest.raises(BundlePackError, match="does not describe this program"):
            BP.plan(self._buffer(), row_pitch_elements=16,
                    session_states=(BP.SessionState("bogus", input_arg=97, output_index=0),))

    def test_an_out_of_range_output_index_is_REFUSED(self):
        with pytest.raises(BundlePackError, match="out-of-range carry"):
            BP.plan(self._buffer(), row_pitch_elements=16,
                    session_states=(BP.SessionState("hidden", input_arg=1, output_index=9),))

    def test_a_carry_between_MISMATCHED_layouts_is_REFUSED(self):
        """The failure no size check can see: Y1 is i32[1,4] = 16 B and arg1 is f32[1,8] = 32 B, but
        a carry between two tensors of the SAME byte count and different shapes copies the right
        number of bytes into the wrong elements."""
        tensors = {"arg0": _t((4, 4), "i8"), "arg1": _t((1, 8), "f32"),
                   "Y0": _t((2, 4), "f32"), "Y1": _t((1, 4), "i32")}
        buffer = _buffer([_arg("arg0"), _arg("arg1"),
                          _arg("Y0", "write"), _arg("Y1", "write")], tensors)
        with pytest.raises(BundlePackError, match="copies the right byte count into the wrong"):
            BP.plan(buffer, row_pitch_elements=16,
                    session_states=(BP.SessionState("hidden", input_arg=1, output_index=0),))

    def test_a_state_whose_input_is_ALREADY_a_write_argument_needs_no_move(self):
        """A backend that emits the carried state as a write argument has already done the work."""
        tensors = {"arg0": _t((4, 4), "i8"), "arg5": _t((1, 8), "f32")}
        buffer = _buffer([_arg("arg0"), _arg("arg5", "write")], tensors)
        plan = BP.plan(buffer, row_pitch_elements=16,
                       session_states=(BP.SessionState("s", input_arg=5, output_index=0),))
        assert plan.carried == ()
        assert [t.tensor for t in plan.mutable] == ["arg5"]
        assert all(t.role == "argument" for t in (*plan.const, *plan.mutable))


class TestTheContractsStateListIsParsedStructurally:
    def test_it_reads_the_declared_states(self):
        states = BP.session_states_from_contract(
            {"states": [{"name": "kv", "input_arg": 809, "output_index": 1},
                        {"name": "flow", "input_arg": 810, "output_index": 0}]})
        assert [s.name for s in states] == ["kv", "flow"]
        assert [s.input_arg for s in states] == [809, 810]

    def test_a_feed_forward_contract_yields_no_states(self):
        assert BP.session_states_from_contract({"version": 1}) == ()

    def test_a_state_missing_an_endpoint_is_REFUSED(self):
        for row in ({"name": "kv", "output_index": 1}, {"name": "kv", "input_arg": 809}):
            with pytest.raises(BundlePackError, match="write fault at step 1"):
                BP.session_states_from_contract({"states": [row]})

    def test_two_states_sharing_an_endpoint_are_REFUSED(self):
        with pytest.raises(BundlePackError, match="overwrite the other's carry"):
            BP.session_states_from_contract(
                {"states": [{"name": "a", "input_arg": 1, "output_index": 0},
                            {"name": "b", "input_arg": 1, "output_index": 1}]})
        with pytest.raises(BundlePackError, match="overwrite the other's carry"):
            BP.session_states_from_contract(
                {"states": [{"name": "a", "input_arg": 1, "output_index": 0},
                            {"name": "b", "input_arg": 2, "output_index": 0}]})

    def test_a_non_list_states_field_is_REFUSED(self):
        with pytest.raises(BundlePackError, match="not a list"):
            BP.session_states_from_contract({"states": "kv"})


class TestTheRealSmolVLAContractPlansAgainstTheRealCommandBuffer:
    """The acceptance test for the session path: the shipped contract against the emitted buffer."""

    def _inputs(self):
        import glob
        import json

        from merlin.common.paths import artifacts_dir
        from merlin.common.yaml import load_yaml
        capture = (artifacts_dir() / "recaptures" / "smolvla_int8_w8a8_consistent"
                   / "stages" / "flow_denoise")
        if not capture.is_dir():
            pytest.skip("no smolvla recapture in this tree")
        found = glob.glob(str(artifacts_dir() / "perf-bench" / "gemmini"
                              / "_global_phase2_baseline_emission_cache_v1" / "*"
                              / "command_buffer.json"))
        buffers = []
        for path in sorted(found):
            with open(path, encoding="utf-8") as handle:
                buffer = json.load(handle)
            abi = (buffer.get("kernel_abi") or {}).get("args") or []
            if len(abi) > 1000:
                buffers.append(buffer)
        if not buffers:
            pytest.skip("no emitted smolvla-sized command buffer in this tree")
        return buffers[0], load_yaml(capture / "session_contract.yaml")

    def test_all_three_declared_carries_resolve_and_move_out_of_read_only(self):
        buffer, contract = self._inputs()
        states = BP.session_states_from_contract(contract)
        assert len(states) == 3
        plan = BP.plan(buffer, row_pitch_elements=16, session_states=states)
        assert {row["state"] for row in plan.carried} == {"prefix_kv_cache", "flow_state",
                                                          "timestep"}
        assert len([t for t in plan.const if t.role == "seed"]) == 3

    def test_the_concatenation_the_harness_used_to_build_DISAGREES_with_the_ABI(self):
        buffer, contract = self._inputs()
        plan = BP.plan(buffer, row_pitch_elements=16,
                       session_states=BP.session_states_from_contract(contract))
        assert [t.tensor for t in plan.arguments] == list(plan.abi_order)
        assert [t.tensor for t in (*plan.const, *plan.mutable)] != list(plan.abi_order)


class TestAnImageTooLargeForTheCodeModelIsCaughtBEFOREItIsLinked:
    """Under a PC-relative code model an oversized image links and MIS-ADDRESSES.

    It does not fail. So the span is projected from the plan and checked before a build spends the
    minutes to produce a binary nobody should run. Measured: SmolVLA's linked image is 0.689 GiB and
    clears the medany window; tiny_llama's plan projects 2.237 GiB and does not.
    """

    def test_the_projection_is_const_plus_mutable_plus_what_the_plan_cannot_know(self):
        plan = BP.plan(_buffer([_arg("arg0"), _arg("Y0", "write")],
                               {"arg0": _t((1, 16), "i8"), "Y0": _t((1, 4), "i32")}),
                       row_pitch_elements=16)
        assert plan.projected_image_bytes() == plan.const_bytes + plan.mutable_bytes
        assert plan.projected_image_bytes(additional_bytes=1024) == \
            plan.const_bytes + plan.mutable_bytes + 1024

    def test_a_negative_allowance_is_refused(self):
        plan = BP.plan(_buffer([_arg("arg0"), _arg("Y0", "write")],
                               {"arg0": _t((1, 16), "i8"), "Y0": _t((1, 4), "i32")}),
                       row_pitch_elements=16)
        with pytest.raises(BundlePackError):
            plan.projected_image_bytes(additional_bytes=-1)

    def _plan_for(self, read_args):
        """A synthetic plan of a declared const size, so the boundary is exercised without a model."""
        tensors, args = {}, []
        for index, count in enumerate(read_args):
            tensors[f"arg{index}"] = _t((count, 16), "i8")
            args.append(_arg(f"arg{index}"))
        tensors["Y0"] = _t((1, 16), "i32")
        args.append(_arg("Y0", "write"))
        return BP.plan(_buffer(args, tensors), row_pitch_elements=16)

    def test_an_image_inside_the_window_yields_NO_finding(self):
        from merlin.liveness.preconditions import medany_span
        plan = self._plan_for([1024])
        assert medany_span(uses_medany=True,
                           image_span_bytes=plan.projected_image_bytes()) == []

    def test_an_image_PAST_the_window_is_a_FAULT_naming_the_fix(self):
        from merlin.liveness.preconditions import medany_span
        # One 2.5 GiB read argument: past the ±2 GiB PC-relative reach.
        plan = self._plan_for([(5 * (1 << 30)) // (2 * 16)])
        findings = medany_span(uses_medany=True,
                               image_span_bytes=plan.projected_image_bytes())
        assert len(findings) == 1
        assert findings[0].severity.name == "FAULT"
        assert "mis-address" in findings[0].message
        assert "linker script" in (findings[0].fix_hint or "")

    def test_a_target_that_is_not_medany_is_not_judged_by_this_rule(self):
        from merlin.liveness.preconditions import medany_span
        plan = self._plan_for([(5 * (1 << 30)) // (2 * 16)])
        assert medany_span(uses_medany=False,
                           image_span_bytes=plan.projected_image_bytes()) == []


class TestTheRealTinyLlamaPlanIsPastTheMedanyWindow:
    """The finding, held so it cannot be forgotten and then rediscovered as a wrong answer."""

    def _plan(self):
        import glob
        import json

        from merlin.common.paths import artifacts_dir
        found = glob.glob(str(artifacts_dir() / "perf-bench" / "gemmini"
                              / "_global_phase2_baseline_emission_cache_v1" / "*"
                              / "command_buffer.json"))
        for path in sorted(found):
            with open(path, encoding="utf-8") as handle:
                buffer = json.load(handle)
            abi = (buffer.get("kernel_abi") or {}).get("args") or []
            if len(abi) == 825:                      # tiny_llama's argument count
                return BP.plan(buffer, row_pitch_elements=16)
        pytest.skip("no emitted tiny_llama command buffer in this tree")

    def test_its_projected_image_exceeds_the_two_gigabyte_reach(self):
        from merlin.liveness.preconditions import medany_span
        plan = self._plan()
        span = plan.projected_image_bytes(additional_bytes=48_976_384 + 2 * 1024 * 1024)
        assert span > (1 << 31), f"{span} bytes; the finding this pins is no longer present"
        findings = medany_span(uses_medany=True, image_span_bytes=span)
        assert [f.severity.name for f in findings] == ["FAULT"]

    def test_the_const_blob_alone_is_over_a_gibibyte(self):
        assert self._plan().const_bytes > (1 << 30)
