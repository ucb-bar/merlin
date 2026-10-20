"""Binding a submission's own kernel objects into the whole-model program.

The failure this guards against runs. A kernel handed an operand in the wrong order, or a
convolution handed a raw feature map where its lowering demands a gathered one, still returns and
still reports a cycle count -- for numbers nobody should quote.

Which buffer each argument IS is decided upstream, once: the whole-program statement records the
binding it spliced, and `merlin.perf.whole_model_build.bind_groups` hands it here in the contract's
argument order. What this module still owns is finding the C buffer that holds each program tensor
and refusing when the two disagree -- by name, by size, or by content.
"""

from __future__ import annotations

import numpy as np
import pytest


def _module():
    from selected_driver import load

    return load("gemmini", "group_model_submission_kernels.py")


def _model(**arrays):
    return {
        "steps": [
            {
                "group": 3,
                "kind": "matmul",
                "m": 64,
                "k": 32,
                "n": 16,
                "in": "B_g2",
                "weight": "W_g3",
                "bias": "BIAS_g3",
                "out": "B_g3",
            },
        ],
        "buffers": [{"name": "B_g2", "elements": 64 * 32}, {"name": "B_g3", "elements": 64 * 16}],
        "arrays": {
            "W_g3": arrays.get("W_g3", np.zeros((32, 16), dtype=np.int8)),
            "BIAS_g3": np.zeros(16, dtype=np.int32),
        },
    }


_OPERANDS = {"lhs": "B_g2", "rhs": "layer3.weight", "bias": "layer3.bias", "dst": "B_g3"}


def _row(**over):
    """A group the package answered, in `bind_groups`' words: its arguments in ABI order, each naming
    the PROGRAM tensor the splice bound it to."""
    row = {
        "group": 3,
        "on": "package",
        "shape": "resident_matmul",
        "symbol": "kernel_g3",
        "object": "/dev/null",
        "operands": dict(_OPERANDS),
        "args": [
            {"tensor": "W", "role": "weight", "program": "layer3.weight", "declared": [32, 16]},
            {"tensor": "A0", "role": "input", "program": "B_g2", "declared": [64, 32]},
            {"tensor": "Y0", "role": "output", "program": "B_g3", "declared": [64, 16]},
            {"tensor": "B", "role": "bias", "program": "layer3.bias", "declared": [16]},
        ],
    }
    row.update(over)
    return row


def test_a_group_is_called_with_the_buffers_its_bound_arguments_are():
    m = _module()
    out = m.render_kernels(_model(), [_row()], row_padding=_contract_row_padding())
    assert out["calls"][3] == "kernel_g3((void *)W_g3, (void *)B_g2, (void *)B_g3, (void *)BIAS_g3);"
    assert out["census"][0]["on"] == "submission"
    assert out["census"][0]["arguments"] == ["W_g3", "B_g2", "B_g3", "BIAS_g3"]


def test_an_argument_of_another_size_goes_to_the_vendor_rather_than_being_called():
    m = _module()
    row = _row()
    row["args"][1] = dict(row["args"][1], declared=[64, 64])
    out = m.render_kernels(_model(), [row], row_padding=_contract_row_padding())
    assert out["census"][0]["on"] == "vendor" and "'A0'" in out["census"][0]["why"]
    assert 3 not in out["calls"] and out["objects"] == []


def test_two_statements_naming_different_committed_buffers_is_a_refusal():
    """Both the statement and the driver name a group's committed buffers, and they must name the
    same one. A sum whose operands arrive swapped is exactly this: the counts agree, the scales do not."""
    m = _module()
    model = {
        "steps": [{"group": 6, "kind": "sum", "rows": 4, "cols": 16, "lhs": "B_g4", "rhs": "B_g5", "out": "B_g6"}],
        "buffers": [{"name": n, "elements": 64} for n in ("B_g4", "B_g5", "B_g6")],
        "arrays": {},
    }
    row = {
        "group": 6,
        "on": "package",
        "shape": "whole_program",
        "symbol": "kernel_g6",
        "object": "/dev/null",
        "operands": {"lhs": "B_g4", "rhs": "B_g5", "dst": "B_g6"},
        "args": [
            {"tensor": "X0", "role": "input", "program": "B_g4", "declared": [4, 16]},
            {"tensor": "X1", "role": "input", "program": "B_g5", "declared": [4, 16]},
            {"tensor": "Y0", "role": "output", "program": "B_g6", "declared": [4, 16]},
        ],
    }
    out = m.render_kernels(model, [row], row_padding=_contract_row_padding())
    assert out["calls"][6] == "kernel_g6((void *)B_g4, (void *)B_g5, (void *)B_g6);"
    swapped = dict(row, operands={"lhs": "B_g4", "rhs": "B_g5", "dst": "B_g6"})
    model["steps"][0]["lhs"], model["steps"][0]["rhs"] = "B_g5", "B_g4"
    out = m.render_kernels(model, [swapped], row_padding=_contract_row_padding())
    assert out["census"][0]["on"] == "vendor" and "B_g4" in out["census"][0]["why"]


def test_a_weight_whose_bytes_are_not_the_laid_out_ones_is_refused():
    """A weight is compared BY CONTENT: the statement carries the prepack's digest of the bytes it
    laid out, and a driver buffer holding other bytes of the same size is a mis-bound pointer."""
    import hashlib

    m = _module()
    weight = np.arange(32 * 16, dtype=np.int64).astype(np.int8).reshape(32, 16)
    row = _row()
    row["args"][0] = dict(row["args"][0], sha256=hashlib.sha256(weight.tobytes()).hexdigest())
    assert (
        m.render_kernels(_model(W_g3=weight), [row], row_padding=_contract_row_padding())["census"][0]["on"]
        == "submission"
    )
    out = m.render_kernels(
        _model(W_g3=np.ascontiguousarray(weight.T).reshape(32, 16)), [row], row_padding=_contract_row_padding()
    )
    assert out["census"][0]["on"] == "vendor" and "bytes" in out["census"][0]["why"]


def test_a_convolution_gathers_its_operand_and_the_gather_is_timed_separately():
    m = _module()
    model = {
        "steps": [
            {
                "group": 3,
                "kind": "conv2d",
                "in_dim": 8,
                "ci": 4,
                "n": 16,
                "out_dim": 8,
                "stride": 1,
                "padding": 1,
                "kernel": 3,
                "pool": {},
                "in": "B_g2",
                "weight": "W_g3",
                "bias": "BIAS_g3",
                "out": "B_g3",
            }
        ],
        "buffers": [{"name": "B_g2", "elements": 64 * 4}, {"name": "B_g3", "elements": 64 * 16}],
        "arrays": {"W_g3": np.zeros((3, 3, 4, 16), dtype=np.int8), "BIAS_g3": np.zeros(16, dtype=np.int32)},
    }
    # The package declares HOW its im2col is built; the gather is generated from that recipe, never
    # re-derived from the driver's step, so the two cannot drift.
    recipe = {
        "target": "IFM_im2col",
        "source": "IFM",
        "kh": 3,
        "kw": 3,
        "ci": 4,
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
        "dilation": [1, 1],
        "layout": "nhwc",
    }
    row = _row(
        args=[
            {"tensor": "W", "role": "weight", "program": "layer3.weight", "declared": [36, 16]},
            {
                "tensor": "IFM_im2col",
                "role": "input",
                "declared": [64, 36],
                "gather": {"recipe": recipe, "source": "B_g2", "source_declared": [1, 8, 8, 4]},
            },
            {"tensor": "Y0", "role": "output", "program": "B_g3", "declared": [64, 16]},
            {"tensor": "B", "role": "bias", "program": "layer3.bias", "declared": [16]},
        ]
    )
    out = m.render_kernels(model, [row], row_padding=_contract_row_padding())
    call = out["calls"][3]
    assert "im2col_g3((const elem_t *)B_g2, IM2COL)" in call
    # The gather is timed on its own INSIDE the group's bracket: both halves stay in the measured
    # window, because both must happen for the model to compute its answer, but one is this
    # harness's code and one is the submission's and a single number cannot tell them apart.
    assert "fm_g = read_cycles() - i0" in call, "the gather must be attributable on its own"
    assert "fm_im2col += fm_g" in call
    assert "(void *)IM2COL" in call and "(void *)B_g2" not in call.partition("kernel_g3")[2]
    # 36 columns per row, written at the contract's padded row pitch (the tile edge rounds 36 up).
    pad = _contract_row_padding()
    assert out["im2col_scratch_elements"] == 64 * (-(-36 // pad) * pad)
    # The column order is the weight's row order, which the CONV2D contract states as
    # "column order kh, kw, ci": the row segment is kr * (Kw*Ci) and the tap within it is kc * Ci,
    # which is (kr*Kw + kc)*Ci factored so a whole kernel row can move as one run.
    assert "kr * 12" in out["definitions"] and "kc * 4" in out["definitions"]
    # A kernel row is Kw*Ci adjacent elements in the source, so the interior moves in one copy.
    assert "12 * sizeof(elem_t)" in out["definitions"]
    assert "__builtin_memcpy(col," in out["definitions"]
    assert "__builtin_memset(col, 0," in out["definitions"]

    # A declared im2col its own recipe does not produce is refused, never gathered into.
    row["args"][1] = dict(row["args"][1], declared=[64, 27])
    out = m.render_kernels(model, [row], row_padding=_contract_row_padding())
    assert out["census"][0]["on"] == "vendor" and "does not produce" in out["census"][0]["why"]


def test_a_group_the_package_never_answered_keeps_its_library_call():
    m = _module()
    out = m.render_kernels(
        _model(),
        [{"group": 3, "on": "vendor", "why": "declined", "cause": "package_declined"}],
        row_padding=_contract_row_padding(),
    )
    assert out["calls"] == {} and out["census"][0]["on"] == "vendor"
    assert out["census"][0]["cause"] == "package_declined"


def _numpy_im2col(src, kh, kw, ci, stride, pad, out_dim, in_dim):
    """im2col in the CONV2D contract's own words: column order kh, kw, ci; out-of-bounds reads 0."""
    import numpy as np

    out = np.zeros((out_dim * out_dim, kh * kw * ci), dtype=np.int8)
    for oh in range(out_dim):
        for ow in range(out_dim):
            for kr in range(kh):
                ih = oh * stride[0] - pad[0] + kr
                for kc in range(kw):
                    iw = ow * stride[1] - pad[1] + kc
                    if 0 <= ih < in_dim and 0 <= iw < in_dim:
                        lo = (kr * kw + kc) * ci
                        out[oh * out_dim + ow, lo : lo + ci] = src[ih, iw]
    return out


@pytest.mark.parametrize(
    ("kh", "kw", "ci", "stride", "pad", "in_dim"),
    [
        (3, 3, 4, [1, 1], [1, 1, 1, 1], 8),  # padded 3x3 -- the common ResNet case
        (3, 3, 8, [2, 2], [1, 1, 1, 1], 8),  # strided: taps stay contiguous, positions do not
        (1, 1, 16, [1, 1], [0, 0, 0, 0], 6),  # 1x1, no padding, no edge path
        (7, 7, 3, [3, 3, 3, 3][:2], [3, 3, 3, 3], 16),  # the stem's geometry
    ],
)
def test_the_generated_gather_builds_the_matrix_numpy_does(kh, kw, ci, stride, pad, in_dim, tmp_path):
    """COMPILED AND RUN, not inspected. A structural assertion cannot catch a wrong column order or
    a mishandled edge: both produce a matrix of the right shape that multiplies to the wrong answer.
    This builds the actual C the arm links and compares every element against numpy.
    """
    import shutil
    import subprocess

    import numpy as np

    if shutil.which("cc") is None:
        pytest.skip("no host C compiler")
    out_dim = (in_dim + pad[0] + pad[2] - kh) // stride[0] + 1
    recipe = {"kh": kh, "kw": kw, "ci": ci, "stride": stride, "padding": pad, "dilation": [1, 1], "layout": "nhwc"}
    body = _module()._gather(recipe, out_dim, in_dim, "gather_under_test")
    src = np.random.default_rng(0).integers(-128, 127, size=(in_dim, in_dim, ci), dtype=np.int8)
    source = tmp_path / "g.c"
    source.write_text(
        f"#include <stdio.h>\n#include <stdint.h>\n#include <string.h>\ntypedef int8_t elem_t;\n{body}\n"
        f"int main(void) {{ static elem_t s[{in_dim * in_dim * ci}], d[{out_dim * out_dim * kh * kw * ci}];\n"
        f"  if (fread(s, 1, sizeof s, stdin) != sizeof s) return 2;\n"
        f"  gather_under_test(s, d); fwrite(d, 1, sizeof d, stdout); return 0; }}\n",
        encoding="utf-8",
    )
    subprocess.run(["cc", "-O2", "-o", str(tmp_path / "g"), str(source)], check=True, capture_output=True)
    got = subprocess.run([str(tmp_path / "g")], input=src.tobytes(), capture_output=True, check=True).stdout
    want = _numpy_im2col(src, kh, kw, ci, stride, pad, out_dim, in_dim)
    assert np.array_equal(np.frombuffer(got, dtype=np.int8).reshape(want.shape), want)


# ------------------------------------------------------------------ the contract's pointee layout
def _contract_row_padding() -> int:
    """The row padding the version-1 resident kernel ABI declares, resolved for this target (never a
    literal). These kernels implement that ABI, which a support selects explicitly."""
    from merlin.llvmlower import device_shim
    from merlin.targetgen.contract.schemas import render_legacy_kernel_abi

    edge = device_shim.tile_edge_for("gemmini")
    padding = {"layout": render_legacy_kernel_abi("gemmini").get("pointee_layout"), "multiple": edge}
    layout = str(padding["layout"])
    assert "derived geometry, padding" in layout and "distinct layouts" in layout, (
        "the contract no longer distinguishes logical buffers from target-derived device padding"
    )
    assert padding["multiple"], "the target's tile edge is not derivable"
    return int(padding["multiple"])


def _stem_model(channels, dim=4):
    """A first group reading the model input: a ``dim`` x ``dim`` image of ``channels`` channels."""
    image = np.arange(dim * dim * channels, dtype=np.int64).astype(np.int8).reshape(1, dim, dim, channels)
    return {
        "steps": [
            {
                "group": 1,
                "kind": "conv2d",
                "in": "IMAGE",
                "weight": "W_g1",
                "bias": "BIAS_g1",
                "out": "B_g1",
                "in_dim": dim,
                "ci": channels,
                "n": 16,
                "out_dim": dim,
                "stride": 1,
                "padding": 1,
                "kernel": 3,
                "pool": {"size": 0, "stride": 0, "padding": 0},
            },
        ],  # fmt: skip
        "buffers": [{"name": "B_g1", "elements": dim * dim * 16}],
        "arrays": {
            "IMAGE_DATA": image,
            "W_g1": np.zeros((3 * 3 * channels, 16), dtype=np.int8),
            "BIAS_g1": np.zeros(16, dtype=np.int32),
        },
    }


def _stem_row(channels, dim=4):
    return {
        "group": 1,
        "on": "package",
        "shape": "whole_program",
        "symbol": "kernel_g1",
        "object": "/dev/null",
        "operands": {"lhs": "input", "rhs": "conv1.weight", "bias": "conv1.bias", "dst": "B_g1"},
        "args": [
            {"tensor": "IFM", "role": "input", "program": "input", "declared": [1, dim, dim, channels]},
            {"tensor": "W", "role": "weight", "program": "conv1.weight", "declared": [3 * 3 * channels, 16]},
            {"tensor": "Y0", "role": "output", "program": "B_g1", "declared": [1, dim, dim, 16]},
        ],
    }


def test_a_first_group_whose_channels_miss_the_padding_reads_the_declared_layout():
    """The model input with 3 channels is handed to a package kernel as the contract lays a pointee
    out -- each pixel's row zero-padded to the row padding -- not densely, which a package reads at
    the padded pitch and gets plausible, wrong numbers from (every lowering of ResNet-50's g1 did)."""
    m, pad = _module(), _contract_row_padding()
    model = _stem_model(channels=3)
    out = m.render_kernels(model, [_stem_row(channels=3)], entry="input", row_padding=pad)
    assert out["calls"][1] == f"kernel_g1((void *){m.PADDED_IMAGE}, (void *)W_g1, (void *)B_g1);"
    padded = model["arrays"][m.PADDED_IMAGE].reshape(16, pad)
    dense = model["arrays"]["IMAGE_DATA"].reshape(16, 3)
    assert (padded[:, :3] == dense).all() and (padded[:, 3:] == 0).all()
    assert out["census"][0]["row_padded"]["IFM"]["row_padding"] == pad
    # The dense image is unchanged: the library call and every host-side reference read it.
    assert model["arrays"]["IMAGE_DATA"].size == 4 * 4 * 3


def test_an_input_already_on_the_padding_is_handed_over_as_it_is():
    m, pad = _module(), _contract_row_padding()
    model = _stem_model(channels=pad)
    out = m.render_kernels(model, [_stem_row(channels=pad)], entry="input", row_padding=pad)
    assert out["calls"][1].startswith("kernel_g1((void *)IMAGE,") and m.PADDED_IMAGE not in model["arrays"]


def test_an_unknown_padding_is_a_refusal_never_a_dense_hand_over():
    m = _module()
    out = m.render_kernels(_stem_model(channels=3), [_stem_row(channels=3)], entry="input", row_padding=None)
    assert out["census"][0]["on"] == "vendor" and "not derivable" in out["census"][0]["why"]


def test_a_gathered_matrix_whose_rows_miss_the_padding_is_written_at_the_padded_pitch(tmp_path):
    """COMPILED AND RUN. The stem's im2col rows (7*7*3 = 147 columns) are written at the contract's
    padded pitch with zero tails -- including over a scratch another group left dirty."""
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no host C compiler")
    pad = _contract_row_padding()
    kh, ci, in_dim, stride, padding = 7, 3, 16, [2, 2], [3, 3, 3, 3]
    out_dim = (in_dim + padding[0] + padding[2] - kh) // stride[0] + 1
    width = kh * kh * ci
    pitch = -(-width // pad) * pad
    recipe = {"kh": kh, "kw": kh, "ci": ci, "stride": stride, "padding": padding, "dilation": [1, 1], "layout": "nhwc"}
    body = _module()._gather(recipe, out_dim, in_dim, "gather_under_test", pitch=pitch)
    src = np.random.default_rng(1).integers(-128, 127, size=(in_dim, in_dim, ci), dtype=np.int8)
    source = tmp_path / "g.c"
    source.write_text(
        f"#include <stdio.h>\n#include <stdint.h>\n#include <string.h>\ntypedef int8_t elem_t;\n{body}\n"
        f"int main(void) {{ static elem_t s[{in_dim * in_dim * ci}], d[{out_dim * out_dim * pitch}];\n"
        f"  memset(d, 0x5a, sizeof d);\n"
        f"  if (fread(s, 1, sizeof s, stdin) != sizeof s) return 2;\n"
        f"  gather_under_test(s, d); fwrite(d, 1, sizeof d, stdout); return 0; }}\n",
        encoding="utf-8",
    )
    subprocess.run(["cc", "-O2", "-o", str(tmp_path / "g"), str(source)], check=True, capture_output=True)
    got = subprocess.run([str(tmp_path / "g")], input=src.tobytes(), capture_output=True, check=True).stdout
    rows = np.frombuffer(got, dtype=np.int8).reshape(out_dim * out_dim, pitch)
    want = _numpy_im2col(src, kh, kh, ci, stride, padding, out_dim, in_dim)
    assert np.array_equal(rows[:, :width], want) and not rows[:, width:].any()
