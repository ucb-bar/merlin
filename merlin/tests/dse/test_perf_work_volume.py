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


def test_convolution_work_uses_explicit_ir_geometry():
    cb = {
        "tensors": {"x": {"shape": [2, 9, 10, 3]}, "w": {"shape": [18, 5]}},
        "commands": [{"opcode": "CONV2D", "operands": {"ifm": "x", "weight": "w",
                                                          "dst": "out"},
                      "attributes": {"kernel": [3, 2, 3, 5], "stride": [2, 1],
                                     "padding": [1, 0, 1, 0], "dilation": [1, 2]}}],
    }
    # output 5x8; each of 2*5*8*5 outputs performs 3*2*3 MACs.
    assert work_from_command_buffer(cb).exact_macs == 2 * 5 * 8 * 3 * 2 * 3 * 5


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
