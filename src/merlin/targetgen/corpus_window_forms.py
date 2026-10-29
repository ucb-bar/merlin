"""The shape-general FORMS of the standalone exact window sum (``window_sum_capsule``), one per declared form.

``reduction_standalone`` declares one SUM reduction row; the window sum (``merlin_iface.window_sum``) is the
op that writes it, and its ``mode`` is the form: a forward window pool, an adaptive pool, their transposes
(the pool backwards), a fold (``col2im``) and a plain reduce (``mean``). A form is a geometry, not a tile
probe: it fixes its own extents, and carries the frontend operators it stands for so a coverage reader can
tell which captured operators each capsule exercises. Geometries follow the captured windows (kernel 2x2,
stride 2x2 on a 4x4 map in the core-ATen coverage captures) and add the cases the captures' operators
define: a window whose count is not a power of two (a 3x3 window, a clipped adaptive window) and the
1-d / 3-d ranks.

CONFORMANCE RISK (documented, not a spec change): the exact definition divides by the window count as one
exact rational and rounds once (``integer_ops._finish``). The RTL facts give a SUM and a float readout
scale, not an exact-rational divide, so for a count that is not a power of two (``pool2d_odd_count``, the
clipped ``adaptive*`` windows, ``mean_*`` over 6x6) a scale of ``1/count`` in f32 can differ from the
golden by one LSB near a tie. Admission is unaffected (the reduction_standalone row claims no readout
rounding); the forms stay in the corpus so a disagreement is measured, not hidden.

The forms are DECLARED by the ``reduction_standalone`` spec row (``forms:``) and registered in
``corpus_form_table.FORM_TABLE`` (family ``reduction``, op ``window_sum``); a geometry is fixed, so each is
written once, at the aligned cell.
"""

from __future__ import annotations

from typing import Any

def _window(mode: str, src: list[int], out: list[int], ops: tuple[str, ...], **geometry: Any) -> dict[str, Any]:
    return {
        "entry": {"mode": mode, "shape": src, "out_shape": out, **geometry},
        "frontend_ops": ops,
    }


#: form name -> {"entry": BUILDERS entry fields, "frontend_ops": operators the form stands for}.
WINDOW_FORMS: dict[str, dict[str, Any]] = {
    "pool1d": _window(
        "window", [1, 4, 16], [1, 4, 4], ("avg_pool1d",),
        kernel=[4], stride=[4], padding=[0], count_include_pad=True,
    ),
    "pool2d": _window(
        "window", [1, 4, 8, 8], [1, 4, 4, 4], ("avg_pool2d",),
        kernel=[2, 2], stride=[2, 2], padding=[0, 0], count_include_pad=True,
    ),
    "pool2d_odd_count": _window(
        "window", [1, 4, 8, 8], [1, 4, 4, 4], ("avg_pool2d",),
        kernel=[3, 3], stride=[2, 2], padding=[1, 1], count_include_pad=False,
    ),
    "pool3d": _window(
        "window", [1, 2, 4, 4, 4], [1, 2, 2, 2, 2], ("avg_pool3d",),
        kernel=[2, 2, 2], stride=[2, 2, 2], padding=[0, 0, 0], count_include_pad=True,
    ),
    "adaptive1d": _window("adaptive", [1, 4, 10], [1, 4, 3], ("adaptive_avg_pool1d",)),
    "adaptive2d": _window("adaptive", [1, 4, 7, 7], [1, 4, 3, 3], ("_adaptive_avg_pool2d",)),
    "adaptive3d": _window("adaptive", [1, 2, 5, 5, 5], [1, 2, 2, 2, 2], ("_adaptive_avg_pool3d",)),
    "pool2d_backward": _window(
        "window_transpose", [1, 4, 4, 4], [1, 4, 8, 8], ("avg_pool2d_backward",),
        kernel=[2, 2], stride=[2, 2], padding=[0, 0], count_include_pad=True,
    ),
    "adaptive2d_backward": _window(
        "adaptive_transpose", [1, 4, 3, 3], [1, 4, 7, 7], ("_adaptive_avg_pool2d_backward",)
    ),
    "fold": _window(
        "fold", [1, 16, 9], [1, 4, 4, 4], ("col2im",),
        kernel=[2, 2], stride=[1, 1], padding=[0, 0], dilation=[1, 1], output_dtype="i32",
    ),
    # Second geometries per operator (all with a power-of-two window count, so the f32 readout 1/count is exact):
    # a different kernel / stride / rank than the first form, so no operator rests on one capsule.
    "pool1d_k2": _window(
        "window", [1, 4, 16], [1, 4, 8], ("avg_pool1d",),
        kernel=[2], stride=[2], padding=[0], count_include_pad=True,
    ),
    "pool2d_k4": _window(
        "window", [1, 4, 8, 8], [1, 4, 2, 2], ("avg_pool2d",),
        kernel=[4, 4], stride=[4, 4], padding=[0, 0], count_include_pad=True,
    ),
    "pool3d_overlap": _window(
        "window", [1, 2, 4, 4, 4], [1, 2, 3, 3, 3], ("avg_pool3d",),
        kernel=[2, 2, 2], stride=[1, 1, 1], padding=[0, 0, 0], count_include_pad=True,
    ),
    "adaptive1d_even": _window("adaptive", [1, 4, 8], [1, 4, 4], ("adaptive_avg_pool1d",)),
    "adaptive2d_even": _window("adaptive", [1, 4, 8, 8], [1, 4, 2, 2], ("_adaptive_avg_pool2d",)),
    "adaptive3d_even": _window("adaptive", [1, 2, 4, 4, 4], [1, 2, 2, 2, 2], ("_adaptive_avg_pool3d",)),
    "pool2d_backward_overlap": _window(
        "window_transpose", [1, 4, 3, 3], [1, 4, 4, 4], ("avg_pool2d_backward",),
        kernel=[2, 2], stride=[1, 1], padding=[0, 0], count_include_pad=True,
    ),
    "adaptive2d_backward_even": _window(
        "adaptive_transpose", [1, 4, 2, 2], [1, 4, 4, 4], ("_adaptive_avg_pool2d_backward",)
    ),
    "mean_dim_h": _window("reduce", [1, 4, 4, 4], [1, 4, 1, 4], ("mean.dim",), dims=[2], average=True),
    "mean_all_small": _window("reduce", [1, 1, 2, 4], [1, 1, 1, 1], ("mean.default",), dims=[0, 1, 2, 3], average=True),
    "mean_dim": _window("reduce", [1, 4, 6, 6], [1, 4, 1, 1], ("mean.dim",), dims=[2, 3], average=True),
    "mean_all": _window("reduce", [1, 4, 6, 6], [1, 1, 1, 1], ("mean.default",), dims=[0, 1, 2, 3], average=True),
}
