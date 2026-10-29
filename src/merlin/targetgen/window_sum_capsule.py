"""The standalone exact window-sum capsule (``op == window_sum``): mean, average pool, adaptive average pool.

Written as ``merlin_iface.window_sum`` -- the operation the frozen author interface already defines
(``interface_dialect_contract.yaml`` / ``command_buffer_abi.yaml`` ``WINDOW_SUM``). The reference is the
exact integer definition in ``runtime.integer_ops.window_sum`` (called by ``capsule_golden``): exact int32
window sums, then ONE rounding per output,

    dst[o] = clamp(round_half_even(acc[o] * readout_scale / count[o]) + zero_point, -128, 127).

``readout_scale`` is 1.0 here, so the result is the mean of the window rounded half to even and saturated
to i8; ``count`` is the in-bounds window size, the padded size when ``count_include_pad``, or the
override. This module only states the program and what the capsule owes the target; no arithmetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding

_WINDOW_KEYS = (
    "kernel", "stride", "padding", "count_include_pad", "ceil_mode", "divisor_override", "dims", "average", "dilation",
)  # fmt: skip
_MODES = {"window", "adaptive", "reduce", "window_transpose", "adaptive_transpose", "fold"}


def _list(values: list[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def build_window_sum(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """``X0[shape] -> Y0[out_shape]``: the operand dtype in, the operand dtype out (one readout per output)."""
    from .corpus_spec import _iface_prelude, _numeric_policy

    mode = entry.get("mode")
    shape, out_shape = entry.get("shape"), entry.get("out_shape")
    if mode not in _MODES:
        raise ValueError(f"{entry.get('name')}: window_sum mode must be one of {sorted(_MODES)}, got {mode!r}")
    for what, extent in (("shape", shape), ("out_shape", out_shape)):
        if not isinstance(extent, list) or not extent or any(type(d) is not int or d < 1 for d in extent):
            raise ValueError(f"{entry.get('name')}: window_sum needs a resolved positive {what}")
    src, out = entry.get("src", "X0"), entry.get("out", "Y0")
    idt, midt = binding.cap_dtype(binding.operand_dtype), binding.mlir_dtype(binding.operand_dtype)
    # the result is the operand dtype (one readout per output) unless the entry asks for the raw i32 sum
    # (readout_scale 1.0 and no divisor only: fold / col2im)
    wide = entry.get("output_dtype") in ("i32", "int32")
    odt = binding.cap_dtype(binding.accum_dtype) if wide else idt
    modt = binding.mlir_dtype(binding.accum_dtype) if wide else midt
    attrs: dict[str, Any] = {
        "src": src,
        "out": out,
        "shape": list(out_shape),
        "mode": mode,
        "readout_scale": 1.0,
        "output_dtype": odt,
        "rounding": "half_to_even",
        "layout": entry.get("layout", "nchw"),
    }
    attrs.update({key: entry[key] for key in _WINDOW_KEYS if key in entry})
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": src, "role": "input", "shape": list(shape), "dtype": idt}],
        "operation": {"op": "window_sum", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype if wide else binding.operand_dtype, None),
        "expected": {
            # Standalone reduction: the operand is moved and the accumulate-on-load unit sums it. The
            # family must be known to semantic_families, or the list silently comes out empty.
            "instruction_classes": binding.classes_for(op="window_sum", output_dtype=odt),
            "modes": {},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    xs = "x".join(str(d) for d in shape)
    ys = "x".join(str(d) for d in out_shape)
    fields = [f'name = "{out}"', f'mode = "{mode}"']
    for key in ("kernel", "stride", "padding", "dims", "dilation"):
        if key in entry:
            fields.append(f"{key} = {_list(entry[key])}")
    for key in ("count_include_pad", "ceil_mode", "average"):
        if key in entry:
            fields.append(f"{key} = {'true' if entry[key] else 'false'}")
    if "divisor_override" in entry:
        fields.append(f"divisor_override = {int(entry['divisor_override'])}")
    fields += [f'layout = "{attrs["layout"]}"', "readout_scale = 1.0 : f32", f'output_dtype = "{odt}"']
    lines = _iface_prelude(binding.target, entry.get("comment", ""))
    lines += [
        f'  %{src} = merlin_iface.tensor {{name = "{src}", role = "input"}} : tensor<{xs}x{midt}>',
        f"  %{out} = merlin_iface.window_sum %{src} {{{', '.join(fields)}}} : (tensor<{xs}x{midt}>) -> tensor<{ys}x{modt}>",
        "}",
    ]
    return cap, "\n".join(lines) + "\n"
