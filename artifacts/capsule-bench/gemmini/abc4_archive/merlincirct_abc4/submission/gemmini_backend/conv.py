"""conv2d (im2col + weight-stationary matmul) device lowering.

A conv2d is lowered to a *derived* im2col activation plus a standard resident
matmul.  The im2col matrix ``[N*Ho*Wo, Kh*Kw*Ci]`` is gathered from the NHWC
activation by the runner (declared via ``params.im2col_recipes`` in the command
buffer), so the device program is just the WS-matmul tile schedule over that
matrix with the packed weight ``[Kh*Kw*Ci, Co]`` resident.  This mirrors how a
real Gemmini conv is compiler-lowered (bareMetalC/conv.c): the spatial gather is
host/DRAM-side and the systolic array sees a plain matmul.
"""
from __future__ import annotations


def _conv_out_dims(H, W, kh, kw, stride, padding, dilation):
    sh, sw = stride
    pt, pl, pb, pr = padding
    dh, dw = dilation
    ho = (H + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    wo = (W + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    return ho, wo


def conv_geometry(rec: dict) -> dict:
    """Derive the im2col matmul geometry for a conv2d program record.

    Returns the derived activation name, the matmul extents ``M`` (=N*Ho*Wo),
    ``K`` (=Kh*Kw*Ci), ``N`` (=Co), and the recipe fields the command buffer
    needs to declare the im2col gather.
    """
    n, h, w, c = rec["ifm_shape"]
    kh, kw, ci, co = rec["kernel"]
    ho, wo = _conv_out_dims(h, w, kh, kw, rec["stride"], rec["padding"], rec["dilation"])
    m = n * ho * wo
    k = kh * kw * ci
    return {
        "im2col": rec["ifm"] + "_im2col",
        "M": m, "K": k, "N": co,
        "kh": kh, "kw": kw, "ci": ci, "ho": ho, "wo": wo,
        "ifm_dtype": "i8",
        "recipe": {
            "target": rec["ifm"] + "_im2col", "source": rec["ifm"],
            "kh": kh, "kw": kw, "ci": ci,
            "stride": list(rec["stride"]), "padding": list(rec["padding"]),
            "dilation": list(rec["dilation"]), "layout": rec["layout"],
        },
    }
