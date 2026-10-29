"""The pointwise convolution-backward capsule: the three gradients of a 1x1, stride-1, unpadded convolution.

``Y[M, N] = X[M, K] @ W[K, N] + b`` (``M`` = batch x spatial positions, ``K`` = input channels, ``N`` = output
channels) has the gradients of ``aten.convolution_backward``:

* ``grad_input  = dY @ W^T``  (``[M, K]``, the transposed convolution: one contraction with the transposed weight)
* ``grad_weight = X^T @ dY``  (``[K, N]``, the correlation of input and output gradient: one contraction)
* ``grad_bias   = sum_m dY``  (``[1, N]``, the standalone integer sum down the outer axis)

Each is a form the spec already declares (the contraction row's rank-2 matmul and the standalone sum reduction row);
the capsule composes them as ONE program so the three gradients are demanded together, the way the op is captured.
A spatial kernel needs the fold of the per-tap grad_input back onto the image (a scatter-add, the col2im form); that is
not claimed here. Written as a generic component program (``component_program.render``); the golden is the independent
exact integer evaluator ``component_numerics``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .component_program import SCHEMA_V2

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding


def conv_backward_program(entry: dict, binding: CorpusBinding) -> dict:
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    if any(type(extent) is not int or extent < 1 for extent in (M, K, N)):
        raise ValueError(f"{entry.get('name')}: convolution backward needs resolved positive M, K and N")
    return {
        "schema": SCHEMA_V2,
        "inputs": [
            {"name": "X", "role": "input", "shape": [M, K], "dtype": "operand"},
            {"name": "dY", "role": "input", "shape": [M, N], "dtype": "operand"},
            {"name": "W", "role": "weight", "shape": [K, N], "dtype": "operand"},
        ],
        "nodes": [
            {"name": "xt", "op": "transpose", "inputs": ["X"]},
            {"name": "wt", "op": "transpose", "inputs": ["W"]},
            {"name": "gw", "op": "matmul", "inputs": ["xt", "dY"]},
            {"name": "gx", "op": "matmul", "inputs": ["dY", "wt"]},
            {"name": "gb", "op": "reduce_sum", "inputs": ["dY"], "axis": 0, "dtype": "accumulator"},
        ],
        "outputs": [
            {"name": "grad_input", "value": "gx"},
            {"name": "grad_weight", "value": "gw"},
            {"name": "grad_bias", "value": "gb"},
        ],
    }


def build_conv_backward(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    """A pointwise convolution backward (op == conv_backward), written as a component program."""
    from .corpus_spec import build_component_program

    cap, source = build_component_program({**entry, "program": conv_backward_program(entry, binding)}, binding)
    # The three computing regions are the one captured operation: each carries its source operator.
    source = re.sub(
        r'(prov\.region_id = "component_[2-4]", prov\.op = "(?:matmul|reduce_sum)",)',
        r'\1 prov.aten = "aten.convolution_backward.default",',
        source,
    )
    cap["expected"]["instruction_classes"] = binding.classes_for(
        op="conv_backward", output_dtype=binding.cap_dtype(binding.accum_dtype)
    )
    return cap, source
