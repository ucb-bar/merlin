"""A recipe may opt into quantizing contractions of two run-time activations (attention), and only then.

WHY THIS EXISTS. A static per-tensor recipe annotates a contraction only when its second operand is a
stored weight, so attention's ``Q @ K^T`` and ``P @ V`` stay in floating point and run on the host
(measured on the r53b captures: 44 batched matmuls per TinyLlama program, 19.7 GMAC of SmolVLA's
SigLIP attention). ``activation_contraction`` in a recipe's ``families`` is the explicit switch:

* :func:`quant_recipe.with_activation_contractions` adds it to a derived static per-tensor integer
  recipe (and refuses every other form, saying why); the content digest changes with it, so a capture
  made under the variant is never served for one made without it;
* the quantizer then spells a fused ``scaled_dot_product_attention`` as its two matmuls around the
  softmax, and annotates each activation x activation matmul with the ACTIVATION spec on both
  operands (one calibrated static scale per tensor each, probabilities included);
* a product whose reduction extent is 16 or less (a rotary outer product) is never quantized, and a
  fused call outside SDPA's plain math definition (causal flag, boolean mask, dropout, GQA) stays fused;
* without the family nothing changes: the fused call is left alone and a run-time matmul is refused.

The quantizer runs in the capture venv (``MERLIN_M2M_PYTHON``); its half is skipped without it.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import quant_recipe as QR

_QUANTIZER = merlin_dir() / "python/merlin/targetgen/_recipe_quantizer.py"
_DYNAMIC = merlin_dir() / "python/merlin/targetgen/_activation_contractions.py"


def _recipe(**activation) -> dict:
    body = {
        "schema": QR.SCHEMA,
        "target": "t",
        "unit": "u",
        "status": QR.DERIVED,
        "families": [QR.CONTRACTION, QR.WINDOW_MEAN],
        "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True, "quant_min": -127, "quant_max": 127},
        "activation": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -128,
            "quant_max": 127,
            "mode": "static",
            **activation,
        },
        "bias_domain": None,
        "unquantized": {},
        "why": {},
        "underivable": {},
    }
    body["recipe_sha256"] = QR.digest(body)
    return body


# --- the recipe switch -----------------------------------------------------------------------------


def test_the_variant_adds_only_the_family_and_changes_the_digest():
    base = _recipe()
    variant = QR.with_activation_contractions(base)
    assert variant["families"] == [*base["families"], QR.ACTIVATION_CONTRACTION]
    assert variant["recipe_sha256"] == QR.digest(variant) != base["recipe_sha256"]
    assert QR.ACTIVATION_CONTRACTION in variant["why"]
    for key in ("weight", "activation", "unquantized", "underivable", "target"):
        assert variant[key] == base[key], key
    assert base["families"] == [QR.CONTRACTION, QR.WINDOW_MEAN], "the input recipe is not mutated"
    assert QR.with_activation_contractions(variant) == variant, "applying the switch twice is the switch"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"mode": "dynamic"}, "not static"),
        ({"granularity": "token"}, "not one per tensor"),
        ({"symmetric": False}, "symmetric integer"),
        ({"dtype": "fp8_e4m3"}, "symmetric integer"),
    ],
)
def test_only_a_static_per_tensor_integer_recipe_is_extended(change, reason):
    with pytest.raises(ValueError, match=reason):
        QR.with_activation_contractions(_recipe(**change))


def test_a_recipe_without_contractions_or_not_derived_is_refused():
    no_contraction = {**_recipe(), "families": [QR.WINDOW_MEAN]}
    with pytest.raises(ValueError, match="no contraction"):
        QR.with_activation_contractions(no_contraction)
    with pytest.raises(ValueError, match="derived"):
        QR.with_activation_contractions({**_recipe(), "status": QR.UNDERIVABLE})


def test_the_switch_has_a_command_line(tmp_path):
    source = tmp_path / "recipe.json"
    source.write_text(json.dumps(_recipe()))
    out = tmp_path / "variant" / "recipe.json"
    assert QR._main(["activation-contractions", str(source), str(out)]) == 0  # noqa: SLF001
    assert json.loads(out.read_text()) == QR.with_activation_contractions(_recipe())


def test_the_family_is_spelled_the_same_on_both_sides_of_the_capture_boundary():
    """The quantizer is read in the capture venv without merlin, so it repeats the constants."""
    values = {}
    for path, names in (
        (_QUANTIZER, ("ACTIVATION_CONTRACTION", "MIN_ACTIVATION_REDUCTION")),
        (_DYNAMIC, ("MIN_REDUCTION",)),
    ):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        values[target.id] = node.value.value
    assert values["ACTIVATION_CONTRACTION"] == QR.ACTIVATION_CONTRACTION
    assert values["MIN_ACTIVATION_REDUCTION"] == values["MIN_REDUCTION"]


# --- the quantizer (capture venv) ------------------------------------------------------------------

_SCRIPT = r"""
import importlib.util
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source.parent))
sys.path.insert(0, str(source.parents[2]))
import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location("recipe_quantizer", source)
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)
recipe = json.loads(sys.argv[2])


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(64, 192)

    def forward(self, x, mask, keep):
        # 32 tokens, 2 heads of 32: both reductions (head dim, keys) exceed the outer-product bound.
        q, k, v = self.qkv(x).reshape(1, 32, 3, 2, 32).permute(2, 0, 3, 1, 4)
        inv = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32, 1)
        table = torch.matmul(inv, torch.ones(1, 1, 32))  # an outer product (K = 1): never quantized
        fused = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.25)
        # Half precision with a boolean mask: computed in f32, excluded keys filled with -inf.
        half = F.scaled_dot_product_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), attn_mask=keep)
        causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return fused + half.float() + causal + table.sum()


torch.manual_seed(0)
model = Attention().eval()
keep = torch.ones(1, 1, 32, 32, dtype=torch.bool)
keep[..., 0, 3:] = False
inputs = (torch.randn(1, 32, 64), torch.zeros(1, 1, 32, 32), keep)
reference = model(*inputs)
graph = torch.export.export(model, inputs).module()
listed = quant.ACTIVATION_CONTRACTION in recipe["families"]
rows = quant.spell_attention(graph) if listed else []
sdpa = torch.ops.aten.scaled_dot_product_attention.default
q = quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, model))
q.annotate(graph)
pairs, others = [], []
for node in graph.graph.nodes:
    note = node.meta.get("quantization_annotation")
    if note is None:
        continue
    specs = list(note.input_qspec_map.values())
    row = {"target": str(node.target), "activation_specs": sum(s is q.activation for s in specs),
           "weight_specs": sum(s is q.weight for s in specs)}
    (pairs if str(node.target) == "aten.matmul.default" else others).append(row)
result = {"rows": rows, "pairs": pairs, "others": others, "count": q.annotated_pairs,
          "fused_left": sum(n.target == sdpa for n in graph.graph.nodes),
          "refused": [r["operator"] for r in q.refused_by_placement]}
if listed:
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    fresh = torch.export.export(model, inputs).module()
    quant.spell_attention(fresh)
    prepared = prepare_pt2e(fresh, quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, model)))
    with torch.no_grad():
        for _ in range(4):
            prepared(torch.randn(1, 32, 64), *inputs[1:])
        prepared(*inputs)
    spelled = torch.export.export(model, inputs).module()
    quant.spell_attention(spelled)
    with torch.no_grad():
        result["spelled_max_error"] = float((spelled(*inputs) - reference).abs().max())
    converted = convert_pt2e(prepared, fold_quantize=True)
    quantize = torch.ops.quantized_decomposed.dequantize_per_tensor.default
    matmuls = [n for n in converted.graph.nodes if n.target == torch.ops.aten.matmul.default]
    result["dequantized_inputs"] = sorted(
        sum(isinstance(a, torch.fx.Node) and a.target == quantize for a in n.args[:2]) for n in matmuls)
    with torch.no_grad():
        out = converted(*inputs)
    result["cosine"] = float(F.cosine_similarity(out.flatten(), reference.flatten(), dim=0))
print(json.dumps(result))
"""


def _run(recipe: dict) -> dict:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    completed = subprocess.run(
        [python, "-c", _SCRIPT, str(_QUANTIZER), json.dumps(recipe)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_with_the_family_attention_is_spelled_and_both_operands_take_the_activation_spec():
    data = _run(QR.with_activation_contractions(_recipe()))
    assert [row["status"] for row in data["rows"]] == ["spelled", "spelled", "kept"], data["rows"]
    assert data["rows"][2]["why"] == "causal flag"
    assert data["fused_left"] == 1, "only the causal call stays fused"
    # The spelled form is the operator's own math: equal to the fused calls up to f32 rounding (bf16
    # for the half-precision call's result).
    assert data["spelled_max_error"] < 2e-2, data["spelled_max_error"]
    # Q @ K^T and P @ V of both spelled calls; the K = 1 outer product is not among them.
    assert data["count"] == 4
    assert data["pairs"] == [{"target": "aten.matmul.default", "activation_specs": 2, "weight_specs": 0}] * 4
    assert {"target": "aten.linear.default", "activation_specs": 1, "weight_specs": 1} in data["others"]
    assert data["dequantized_inputs"] == [0, 2, 2, 2, 2], "every quantized matmul reads two dequantized operands"
    assert data["cosine"] > 0.99, data["cosine"]


def test_without_the_family_nothing_changes():
    data = _run(_recipe())
    assert data["rows"] == [] and data["fused_left"] == 3
    assert data["count"] == 0 and data["pairs"] == []
    assert "aten.matmul.default" in data["refused"], "a run-time matmul is still refused, as before"
