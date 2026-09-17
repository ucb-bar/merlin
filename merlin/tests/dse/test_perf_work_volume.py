from __future__ import annotations

from merlin.perf.work_volume import work_from_command_buffer


def test_matmul_work_is_recovered_from_tensor_lineage_not_a_benchmark_field():
    cb = {
        "tensors": {"left": {"shape": [3, 5]}, "weight": {"shape": [5, 7]}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "weight", "dst": "opaque_handle"}},
            {"opcode": "MATMUL_RESIDENT",
             "operands": {"lhs": "left", "rhs": "opaque_handle", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "out"}},
        ],
    }
    got = work_from_command_buffer(cb)
    assert got.exact_macs == 3 * 5 * 7
    assert got.basis == "compiler_command_buffer" and got.unit == "macs"
    assert got.to_dict()["commands"][0]["provenance"].startswith("command_buffer.commands")


def test_attention_qk_counts_the_transposed_rhs_semantics():
    cb = {"tensors": {"q": {"shape": [11, 13]}, "k": {"shape": [17, 13]}},
          "commands": [{"opcode": "ATTENTION_QK",
                        "operands": {"q": "q", "k": "k", "dst": "out"}}]}
    assert work_from_command_buffer(cb).exact_macs == 11 * 13 * 17


def _conv_buffer(*, ifm_shape, weight_shape, layout="nhwc"):
    attributes = {"kernel": [3, 2, 3, 5], "stride": [2, 1],
                  "padding": [1, 0, 1, 0], "dilation": [1, 2]}
    if layout is not None:
        attributes["layout"] = layout
    return {
        "tensors": {"x": {"shape": list(ifm_shape)}, "w": {"shape": list(weight_shape)}},
        "commands": [{"opcode": "CONV2D", "operands": {"ifm": "x", "weight": "w", "dst": "out"},
                      "attributes": attributes}],
    }


#: output 5x8; each of 2*5*8*5 outputs performs 3*2*3 MACs.
_CONV_MACS = 2 * 5 * 8 * 3 * 2 * 3 * 5


def test_convolution_work_uses_explicit_ir_geometry():
    cb = _conv_buffer(ifm_shape=[2, 9, 10, 3], weight_shape=[18, 5])
    assert work_from_command_buffer(cb).exact_macs == _CONV_MACS


def test_convolution_reads_the_input_axis_order_from_the_declared_layout():
    """The same convolution, extents permuted, prices the same work under the matching layout."""
    nhwc = _conv_buffer(ifm_shape=[2, 9, 10, 3], weight_shape=[18, 5], layout="nhwc")
    nchw = _conv_buffer(ifm_shape=[2, 3, 9, 10], weight_shape=[18, 5], layout="nchw")
    assert work_from_command_buffer(nhwc).exact_macs == _CONV_MACS
    assert work_from_command_buffer(nchw).exact_macs == _CONV_MACS
    # A scheduling suffix on the spelling names the same axis order, and must not defeat the read.
    suffixed = _conv_buffer(ifm_shape=[2, 3, 9, 10], weight_shape=[18, 5],
                            layout="nchw_streamed_row_im2col")
    assert work_from_command_buffer(suffixed).exact_macs == _CONV_MACS


def test_convolution_with_no_declared_layout_refuses_rather_than_assuming_one():
    """A SILENCE-FAILURE GUARD, not a preference.

    Reading NCHW extents in NHWC positions does not fail -- it prices a different convolution and
    returns a number. That is how 53 of 54 ResNet-50 commands came to be refused while the total that
    survived still looked like a result. An undeclared axis order is UNKNOWN, and UNKNOWN must be
    louder than a plausible default.
    """
    got = work_from_command_buffer(
        _conv_buffer(ifm_shape=[2, 9, 10, 3], weight_shape=[18, 5], layout=None))
    assert got.exact_macs is None and got.is_lower_bound
    assert got.refusals and "convolution geometry" in got.refusals[0]


def test_convolution_accepts_either_declared_weight_orientation():
    """A prepacking backend emits (co, kh*kw*ci); a row-major im2col one emits (kh*kw*ci, co)."""
    rowmajor = _conv_buffer(ifm_shape=[2, 9, 10, 3], weight_shape=[18, 5])
    prepacked = _conv_buffer(ifm_shape=[2, 9, 10, 3], weight_shape=[5, 18])
    assert work_from_command_buffer(rowmajor).exact_macs == _CONV_MACS
    assert work_from_command_buffer(prepacked).exact_macs == _CONV_MACS
    # A weight matching NEITHER orientation is still refused.
    wrong = _conv_buffer(ifm_shape=[2, 9, 10, 3], weight_shape=[19, 5])
    assert work_from_command_buffer(wrong).exact_macs is None


def test_residency_evicts_the_handle_the_pack_declared_not_just_its_destination():
    """Both ABI spellings of a resident pack must price, because both are executed.

    ``RES_PACK {src, dst}`` names one thing; ``RES_PACK {src, dst, handle}`` names the packed buffer
    and the residency separately. The reference engines register only ``dst``, which is harmless
    there because EVICT has no numerical effect -- but here an unresolvable eviction is a refusal,
    and one refusal makes the whole program uncounted. That is what left every plain-MATMUL program
    on this tree unpriced while its arithmetic was perfectly well described.
    """
    def buffer(pack_operands):
        return {"tensors": {"W": {"shape": [4, 8]}, "A": {"shape": [2, 4]},
                            "Wp": {"shape": [4, 8]}},
                "commands": [{"opcode": "RES_PACK", "operands": pack_operands},
                             {"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "Wp",
                                                               "dst": "acc"}},
                             {"opcode": "EVICT", "operands": {"handle": "W_res"}}]}

    two_names = buffer({"src": "W", "dst": "Wp", "handle": "W_res"})
    assert work_from_command_buffer(two_names).exact_macs == 2 * 4 * 8
    # The legacy EVICT spelling reference.py also accepts.
    legacy = buffer({"src": "W", "dst": "W_res", "handle": "W_res"})
    legacy["commands"][1]["operands"]["rhs"] = "W"
    legacy["commands"][2]["operands"] = {"src": "W_res"}
    assert work_from_command_buffer(legacy).exact_macs == 2 * 4 * 8
    # An eviction naming nothing the program ever made resident is still refused.
    stray = buffer({"src": "W", "dst": "Wp"})
    stray["commands"][2]["operands"] = {"handle": "never_packed"}
    assert work_from_command_buffer(stray).exact_macs is None


def test_unknown_compute_work_is_a_lower_bound_never_an_exact_zero():
    got = work_from_command_buffer({"tensors": {}, "commands": [{"opcode": "FUTURE_ENGINE"}]})
    assert got.known_macs == 0 and got.exact_macs is None and got.is_lower_bound
    assert "UNKNOWN" in got.refusals[0]


def test_incompatible_shapes_refuse_the_whole_program_but_keep_other_known_work():
    cb = {
        "tensors": {"a": {"shape": [2, 3]}, "b": {"shape": [3, 4]},
                    "bad": {"shape": [9, 8]}},
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "b"}},
            {"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "bad"}},
        ],
    }
    got = work_from_command_buffer(cb)
    assert got.known_macs == 24 and got.exact_macs is None
    assert got.to_dict()["is_lower_bound"] is True


def test_convolution_requires_a_real_compatible_weight_tensor():
    cb = {
        "tensors": {"x": {"shape": [1, 8, 8, 3]}},
        "commands": [{"opcode": "CONV2D",
                      "operands": {"ifm": "x", "weight": "missing", "dst": "out"},
                      "attributes": {"kernel": [3, 3, 3, 4], "stride": [1, 1],
                                     "padding": [0, 0, 0, 0], "dilation": [1, 1]}}],
    }
    got = work_from_command_buffer(cb)
    assert got.exact_macs is None and got.is_lower_bound


def test_resident_handle_lifetime_is_tracked_through_evict():
    cb = {
        "tensors": {"a": {"shape": [2, 3]}, "w": {"shape": [3, 4]}},
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "w", "dst": "resident"}},
            {"opcode": "EVICT", "operands": {"handle": "resident"}},
            {"opcode": "MATMUL_RESIDENT",
             "operands": {"lhs": "a", "rhs": "resident", "dst": "out"}},
        ],
    }
    got = work_from_command_buffer(cb)
    assert got.exact_macs is None and got.known_macs == 0


def test_work_receipt_is_the_hash_of_compiler_ir_not_a_corpus_field():
    cb = {"tensors": {"a": {"shape": [2, 3]}, "b": {"shape": [3, 4]}},
          "commands": [{"opcode": "MATMUL", "operands": {"lhs": "a", "rhs": "b"}}],
          "macs": 999999}
    first = work_from_command_buffer(cb)
    changed = work_from_command_buffer({**cb, "macs": 1})

    assert first.exact_macs == changed.exact_macs == 24
    assert len(first.artifact_sha256) == 64
    assert first.artifact_sha256 != changed.artifact_sha256
