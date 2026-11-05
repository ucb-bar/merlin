"""A quantized capture's INT8 reference evaluation equals PyTorch eager on the same integerized program.

WHY THIS EXISTS. A recipe-quantized capture is checked by evaluating its MLIR with ``linalg_numpy``, and its
quality is screened by running the same quantized model in PyTorch. When the two disagreed (TinyLlama
prefill logits 17% apart), the question was whether the lowering changes the program. Localized: on an f32
program the two are bit-identical on every int8 contraction until a float difference upstream of a static
quantize moves one value across a rounding boundary. The first such difference was PyTorch's FUSED
attention kernel against the math decomposition the capture lowers; with that decomposition on both sides
the contractions stay identical, and a single flipped int8 element is what then grows into the logit gap
(an ill-conditioned per-tensor program amplifies it; the lowering does not create it).

What this pins, on a small two-layer transformer quantized exactly as the capture worker quantizes
(``apply_recipe`` + ``integerize_pt2e``, f32, per-channel int8 weights, static per-tensor activations):

* the capture lowers as many int8 contractions as eager executes, and ``linalg_numpy`` reproduces every
  one of them bit-exactly when eager evaluates attention by the capture's own decomposition;
* the program output agrees with eager within float rounding.

Known limitation, not pinned: a bf16 program differs by more, because ``linalg_numpy`` rounds every
operation to bf16 while PyTorch computes a bf16 composite (norm, softmax, GELU) in f32 and rounds once.

The capture half runs in the capture venv (``MERLIN_M2M_PYTHON`` with ``MERLIN_M2M_DIR``); skipped without it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import env
from merlin.targetgen import quant_recipe as QR

_TARGETGEN = Path(QR.__file__).parent

_CAPTURE = r"""
import json
import sys
from pathlib import Path

targetgen, m2m_dir, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
for path in (str(targetgen), str(targetgen.parents[1]), m2m_dir):
    sys.path.insert(0, path)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import _recipe_quantizer as RQ
from m2m import convert
from m2m.capture.pt2e_integerize import integerize_pt2e
from m2m.capture.torchao_pipeline import QuantizationConfig
from m2m.ir.torchmlir_decomps import _sdpa
from torch.ao.quantization import allow_exported_model_train_eval


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.qkv, self.o = nn.Linear(64, 192), nn.Linear(64, 64)
        self.up, self.down = nn.Linear(64, 128), nn.Linear(128, 64)

    def forward(self, x):
        q, k, v = self.qkv(self.norm1(x)).reshape(1, 24, 3, 2, 32).permute(2, 0, 3, 1, 4)
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(context.transpose(1, 2).reshape(1, 24, 64))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


torch.manual_seed(0)
torch.set_num_threads(1)
model = Block().eval()
recipe = json.loads(sys.argv[4])
calibration = [(torch.randn(1, 24, 64),) for _ in range(4)]
args = calibration[0]
quantized = RQ.apply_recipe(model, recipe, example_inputs=args, calibration_inputs=calibration)
quantized, receipt = integerize_pt2e(quantized, args)

SDPA = torch.ops.aten.scaled_dot_product_attention.default
INT_MM = torch.ops.aten._int_mm.default
records = []


class Eager(torch.fx.Interpreter):
    # Attention by the decomposition the capture lowers; every int8 contraction recorded in order.
    def call_function(self, target, args, kwargs):
        if target == SDPA:
            return _sdpa(*args, **kwargs)
        value = super().call_function(target, args, kwargs)
        if target == INT_MM:
            records.append(value.numpy().copy())
        return value


with torch.no_grad():
    result = Eager(quantized).run(*args)
allow_exported_model_train_eval(quantized)
lowered = convert(
    quantized, args, backend="fx_importer", level="linalg-on-tensors",
    quantization=QuantizationConfig(scheme="int8_static_act_int8_weight"), quantization_preapplied=True,
)
program = lowered.exported_program
state = {**program.state_dict, **getattr(program, "constants", {})}
user = iter(args)
bound = [next(user) if s.kind.name == "USER_INPUT" else state[s.target] for s in program.graph_signature.input_specs]
(out / "model.mlir").write_text(lowered.mlir_text)
np.savez(out / "args.npz", **{f"a{i:04d}": t.detach().contiguous().numpy() for i, t in enumerate(bound)})
np.savez(out / "int_mm.npz", **{f"m{i:04d}": r for i, r in enumerate(records)})
np.save(out / "output.npy", (result[0] if isinstance(result, (tuple, list)) else result).numpy())
emitted = receipt["integer_mm_emitted"]
print(json.dumps({"ok": bool(lowered.ok), "integer_mm_emitted": emitted, "recorded": len(records)}))
"""


def _recipe() -> dict:
    body = {
        "schema": QR.SCHEMA,
        "target": "t",
        "unit": "u",
        "status": QR.DERIVED,
        "families": [QR.CONTRACTION],
        "weight": {"dtype": "int8", "granularity": "channel", "symmetric": True, "quant_min": -127, "quant_max": 127},
        "activation": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -128,
            "quant_max": 127,
            "mode": "static",
            "observer": "minmax",
        },
        "bias_domain": None,
        "unquantized": {},
        "why": {},
        "underivable": {},
    }
    body["recipe_sha256"] = QR.digest(body)
    return body


def _capture(out: Path) -> dict:
    python, m2m_dir = env("MERLIN_M2M_PYTHON"), env("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m_dir:
        pytest.skip("no capture interpreter and model2MLIR checkout are configured")
    completed = subprocess.run(
        [python, "-c", _CAPTURE, str(_TARGETGEN), m2m_dir, str(out), json.dumps(_recipe())],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_int8_reference_reproduces_every_eager_contraction_of_a_quantized_capture(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.runtime import linalg_numpy as LN
    from merlin.runtime.backends.zephyr_model import _prepare_model_mlir

    receipt = _capture(tmp_path)
    assert receipt["ok"] and receipt["integer_mm_emitted"] == receipt["recorded"] == 8
    recorded = np.load(tmp_path / "int_mm.npz")
    expected = [recorded[k] for k in sorted(recorded.files)]
    with np.load(tmp_path / "args.npz") as data:
        args = [np.asarray(data[k]) for k in sorted(data.files)]
    (tmp_path / "prep").mkdir()
    prepared = _prepare_model_mlir(
        tmp_path / "model.mlir", tmp_path / "prep", int8_compute=True, respect_captured_quantization_scope=True
    )
    seen: list[np.ndarray] = []

    def observe(op, index, results, operands):
        provenance = getattr(op.attributes.get("prov.aten"), "data", None)
        if op.name == "linalg.generic" and provenance == "aten._int_mm.default":
            seen.append(np.asarray(results[0]))

    outputs = LN.evaluate(parse_mlir_file(prepared), args, observe=observe)
    assert len(seen) == len(expected), "the lowered program runs every int8 contraction eager runs"
    for index, (got, want) in enumerate(zip(seen, expected, strict=True)):
        assert got.dtype == np.int32 and got.shape == want.shape, index
        assert np.array_equal(got, want), f"int8 contraction {index} differs in {(got != want).sum()} elements"
    got = np.asarray(outputs[0], dtype=np.float64)
    want = np.load(tmp_path / "output.npy").astype(np.float64)
    assert float(np.abs(got - want).max() / np.abs(want).max()) < 1e-5
