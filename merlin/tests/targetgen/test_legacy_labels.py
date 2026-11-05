"""Labels that named a lowering are written in their neutral spelling and still READ in the old one."""

from __future__ import annotations

from merlin_experiments.phase0 import hidden_disjointness as HD

from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import legacy_labels as LL
from merlin.targetgen.semantic_families import from_op


def test_old_spellings_read_as_the_neutral_ones_and_new_ones_pass_unchanged():
    assert LL.semantic("conv2d_im2col") == LL.semantic("conv2d") == "conv2d"
    assert LL.is_gathered_conv_path("im2col_matmul") and LL.is_gathered_conv_path(LL.CONV_PATH_GATHERED)
    assert not LL.is_gathered_conv_path("direct_contraction")
    assert LL.is_gathered_conv_op("convolution_im2col_matmul") and LL.is_gathered_conv_op(LL.CONV_OP_GATHERED)
    assert LL.operation_attributes({"semantic": "conv2d_im2col", "kh": 3}) == {"semantic": "conv2d", "kh": 3}
    assert from_op("convolution_im2col_matmul") == from_op(LL.CONV_OP_GATHERED) == "contraction"
    # Nothing writes an old spelling.
    assert not set(LL.SEMANTIC_ALIASES) & set(LL.SEMANTIC_ALIASES.values())


def test_a_new_conv2d_capsule_is_labelled_conv2d_and_names_no_lowering(monkeypatch):
    import inspect

    source = inspect.getsource(CS.build_conv2d)
    assert '"semantic": "conv2d"' in source and "im2col" not in source.lower()


def _capsule(semantic: str) -> dict:
    return {
        "inputs": [
            {"name": "W", "role": "weight", "shape": [36, 8], "dtype": "i8"},
            {"name": "IFM", "role": "input", "shape": [1, 8, 8, 4], "dtype": "i8"},
        ],
        "operation": {
            "op": "conv2d",
            "attributes": {"ifm": "IFM", "weight": "W", "out": "Y0", "kh": 3, "semantic": semantic},
        },
    }


def test_a_retained_capsule_with_the_old_label_is_the_same_program_point():
    assert HD.capsule_point(_capsule("conv2d_im2col")) == HD.capsule_point(_capsule("conv2d"))
