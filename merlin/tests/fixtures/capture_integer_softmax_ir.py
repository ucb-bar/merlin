"""Capture the integer-softmax IR fixtures of ``merlin/tests/data/int_softmax_table`` (capture venv).

usage: capture_integer_softmax_ir.py <targetgen dir> <output dir>

Run with the capture interpreter and the PINNED model2MLIR on ``PYTHONPATH`` (``software_pins.yaml``,
``model2mlir_capture``): the integer exp's shifts and floor division lower exactly only there. Each
fixture is a module the way the capture worker makes it -- ``_activation_contractions.install`` then
``_integer_nonlinear.install`` on the model, ``level="linalg-on-tensors"`` -- at shapes small enough to
execute in a test:

* ``softmax_wide.mlir``: ``integer_softmax`` alone over rows of 11200, wide enough for one row to reach
  every grid index the clamp allows;
* ``attention_bf16_masked.mlir``: bf16 attention with an additive -inf mask, both contractions int8;
* ``attention_long_f32.mlir``: f32 attention over 1024 keys, where a flat row's probabilities are small
  enough that the next contraction's quantization scale clamps to its eps.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, sys.argv[1])
import _activation_contractions as AC  # noqa: E402
import _integer_nonlinear as NL  # noqa: E402
import m2m  # noqa: E402

OUT = Path(sys.argv[2])


class Softmax(torch.nn.Module):
    def forward(self, x):
        return NL.integer_softmax(x, -1)


class Attention(torch.nn.Module):
    def forward(self, q, k, v, mask=None):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def capture(model, args, name, *, modes=True):
    model = model.eval()
    if modes:
        AC.install(model)
        NL.install(model)
    result = m2m.convert(model, args, backend="fx_importer", level="linalg-on-tensors")
    (OUT / f"{name}.mlir").write_text(result.mlir_text, encoding="utf-8")
    print("ok", name)


torch.manual_seed(0)
capture(Softmax(), (torch.randn(3, 11200),), "softmax_wide", modes=False)
q, k, v = torch.randn(1, 2, 20, 32), torch.randn(1, 2, 24, 32), torch.randn(1, 2, 24, 32)
mask = torch.zeros(1, 1, 20, 24)
mask[..., 5::7] = float("-inf")
capture(Attention(), tuple(t.bfloat16() for t in (q, k, v, mask)), "attention_bf16_masked")
capture(
    Attention(),
    (torch.randn(1, 1, 6, 32), torch.randn(1, 1, 1024, 32), torch.randn(1, 1, 1024, 32)),
    "attention_long_f32",
)
