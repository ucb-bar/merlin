"""Each rewritten dynamic site carries its own check against the float contraction of its operands.

WHY THIS EXISTS. A recipe's dynamic block rewrites contractions (attention under static per-head scales)
into int8 device calls that integerization never sees, so the capture's golden agreement does not test
them. The sealed issuer only admits such a capture when every site records its own agreement with the
float contraction of the very operands the rewritten graph gives it, on the recipe agreement's samples,
within a bar the recipe declares (``dynamic.site_agreement``). This pins:

* ``quant_recipe.with_site_agreement`` states the bar, binds it into the digest and refuses a malformed
  one; ``site_agreement_of`` validates it;
* the worker's check (capture venv) reproduces an independently computed float reference: on a two-head
  attention, the relu-read site's recorded cosine and max error equal a hand-computed comparison;
* a planted wrong per-head output scale lowers the recorded cosine, so the issuer's bar would refuse it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import quant_recipe as QR

_QUANTIZER = merlin_dir() / "python/merlin/targetgen/_recipe_quantizer.py"


def _recipe(scales: str = QR.STATIC_HEADS) -> dict:
    body = {
        "schema": QR.SCHEMA,
        "target": "t",
        "unit": "u",
        "status": QR.DERIVED,
        "families": [QR.CONTRACTION],
        "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True, "quant_min": -127, "quant_max": 127},
        "activation": {
            "dtype": "int8",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": -128,
            "quant_max": 127,
            "mode": "static",
        },
        "bias_domain": None,
        "unquantized": {},
        "why": {},
        "underivable": {},
    }
    body["recipe_sha256"] = QR.digest(body)
    return QR.with_dynamic_contractions(
        body, planes={QR.SOFTMAX_OUTPUT: 2}, smooth=True, static_families=[], scales=scales
    )


def test_the_bar_is_declared_bound_and_validated(tmp_path):
    body = QR.with_site_agreement(_recipe(), cosine_min=0.999, rel_rmse_max=0.05, basis="measured, 2x margin")
    assert QR.site_agreement_of(body) == {"schema": QR.SITE_AGREEMENT_SCHEMA, "cosine_min": 0.999, "rel_rmse_max": 0.05}
    assert body["recipe_sha256"] == QR.digest(body) != _recipe()["recipe_sha256"]
    assert body["why"]["dynamic.site_agreement"] == "measured, 2x margin"
    assert QR.site_agreement_of(_recipe()) is None
    for bad in ({"cosine_min": 1.5}, {"cosine_min": 0.0}, {"rel_rmse_max": 0.0}, {"rel_rmse_max": float("inf")}):
        changed = json.loads(json.dumps(body))
        changed["dynamic"]["site_agreement"].update(bad)
        with pytest.raises(ValueError, match="site_agreement"):
            QR.site_agreement_of(changed)
    with pytest.raises(ValueError, match="basis"):
        QR.with_site_agreement(_recipe(), cosine_min=0.9, rel_rmse_max=0.1, basis=" ")
    plain = {k: v for k, v in _recipe().items() if k != "dynamic"}
    with pytest.raises(ValueError, match="dynamic block"):
        QR.with_site_agreement(plain, cosine_min=0.9, rel_rmse_max=0.1, basis="x")
    source, out = tmp_path / "in.json", tmp_path / "out.json"
    source.write_text(json.dumps(_recipe()))
    argv = ["site-agreement", str(source), str(out), "--cosine-min", "0.99", "--rel-rmse-max", "0.1", "--basis", "b"]
    assert QR._main(argv) == 0
    assert QR.site_agreement_of(json.loads(out.read_text()))["cosine_min"] == 0.99


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

spec = importlib.util.spec_from_file_location("recipe_quantizer", source)
quant = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant)
recipe = json.loads(sys.argv[2])
PLANT = float(sys.argv[3])


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 64)
        self.kv = nn.Linear(64, 128)

    def heads(self, x):
        return x.view(1, 32, 2, 32).transpose(1, 2)

    def forward(self, x):
        q, k, v = self.heads(self.q(x)), self.heads(self.k(x)), self.heads(self.v(x))
        probs = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.5, dim=-1)
        gate = torch.relu(torch.matmul(v, q.transpose(-2, -1)))  # no softmax reads it: no shift
        # Keys and values unpacked from one projection: the key shift stays a host subtraction.
        k2, v2 = self.kv(x).view(1, 32, 2, 2, 32).permute(2, 0, 3, 1, 4)
        eager = torch.matmul(torch.softmax(torch.matmul(q, k2.transpose(-2, -1)) * 0.25, dim=-1), v2)
        return torch.matmul(probs, v).sum() + gate.sum() + eager.sum()


torch.manual_seed(0)
model = Block().eval()
calibration = [(torch.randn(1, 32, 64),) for _ in range(4)]
example = (torch.randn(1, 32, 64),)
graph = torch.export.export(model, example).module()
for index, node in enumerate(graph.graph.nodes):  # what the capture's frontend trace stamps
    node.meta["custom"] = {"m2m_node_id": f"g:original:root:n{index}"}
q = quant.build_quantizer(recipe, layer_plan=quant._plan_layers(recipe, model))
rows = q.rewrite_dynamic(graph, calibration, 100)
gate = next(r for r in rows if not r["smooth"] and r["planes"] == [1, 1])
if PLANT != 1.0:
    wanted = f"_static_scale_{gate['node']}_out"
    name = next(n.target for n in graph.graph.nodes if n.op == "get_attr" and wanted in str(n.target))
    scale = getattr(graph, name)
    scale[:, 0] *= PLANT  # a wrong scale on the first head only
quant._verify_dynamic_sites(graph, rows, example, calibration)
# The independent statement: the gate site's emitted value against v @ q^T, on the capture input and then
# the calibration stream (the recipe agreement's samples), compared exactly as the record defines it.
seen = {}

def tap(name):
    def hook(_module, _inputs, output):
        seen[name] = output
    return hook

cos, worst = [], 0.0
for sample in [example, *calibration]:
    with torch.no_grad():
        vh, qh = model.heads(model.v(sample[0])), model.heads(model.q(sample[0]))
        want = (vh.double() @ qh.double().transpose(-2, -1)).float().double().reshape(-1)
        node = next(n for n in graph.graph.nodes if n.name.startswith("relu"))
        captured = {}
        def keep(value):
            captured["v"] = value
            return value
        with graph.graph.inserting_before(node):
            probe = graph.graph.call_function(keep, (node.args[0],))
        node.args = (probe, *node.args[1:])
        graph.recompile()
        graph(*sample)
        node.args = (probe.args[0], *node.args[1:])
        graph.graph.erase_node(probe)
        graph.recompile()
        got = captured["v"].double().reshape(-1)
    cos.append(float((got * want).sum() / (got.norm() * want.norm())))
    worst = max(worst, float((got - want).abs().max()))
print(json.dumps({"rows": rows, "gate": gate["node"], "cosine_min": min(cos), "max_abs": worst}, default=str))
"""


def _run(plant: float, scales: str = QR.STATIC_HEADS) -> dict:
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("the TorchAO capture interpreter is not configured")
    completed = subprocess.run(
        [python, "-c", _SCRIPT, str(_QUANTIZER), json.dumps(_recipe(scales)), str(plant)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_each_site_records_its_agreement_with_the_float_contraction_of_its_operands():
    data = _run(1.0)
    rows = {row["node"]: row for row in data["rows"]}
    assert all(row["status"] == "rewritten" and "_verify" not in row for row in rows.values())
    for row in rows.values():
        check = row["verification"]
        assert check["schema"] == "m2m.dynamic_site_verification.v1" and check["samples"] == 5 and check["finite"]
        assert 0.99 < check["cosine_min"] <= check["cosine_mean"] <= 1.0 + 1e-12
        assert row["lineage"], "the issuer ties each site's device calls to its lineage"
    gate = rows[data["gate"]]["verification"]
    assert gate["reference"] == "float64_matmul_of_site_operands_rounded_f32"
    assert abs(gate["cosine_min"] - data["cosine_min"]) < 1e-9
    assert abs(gate["max_abs_error"] - data["max_abs"]) < 1e-6
    # The key shift is folded into the key projection's bias here, so the site's own second operand is
    # already the shifted keys: its reference needs no further subtraction, and still agrees.
    # An unfolded shift is subtracted from the reference's operand, as the rewrite does; both agree.
    smoothed = {row["shift_folded"]: row["verification"]["reference"] for row in rows.values() if row["smooth"]}
    assert smoothed == {
        True: "float64_matmul_of_site_operands_rounded_f32",
        False: "float64_matmul_of_site_operands_rounded_f32_shift_adjusted",
    }


def test_a_planted_wrong_scale_lowers_the_recorded_cosine():
    good = {row["node"]: row["verification"] for row in _run(1.0)["rows"]}
    data = _run(5.0)
    bad = {row["node"]: row["verification"] for row in data["rows"]}
    assert bad[data["gate"]]["cosine_min"] < good[data["gate"]]["cosine_min"] - 0.05
    assert bad[data["gate"]]["rel_rmse_max"] > good[data["gate"]]["rel_rmse_max"] * 10


def test_run_time_scale_sites_are_checked_the_same_way():
    data = _run(1.0, scales=QR.NON_REDUCTION)
    rows = {row["node"]: row for row in data["rows"]}
    assert rows and all(row["verification"]["finite"] and row["verification"]["samples"] == 5 for row in rows.values())
    assert all(row["verification"]["cosine_min"] > 0.99 for row in rows.values())
    gate = rows[data["gate"]]["verification"]
    assert abs(gate["cosine_min"] - data["cosine_min"]) < 1e-9 and abs(gate["max_abs_error"] - data["max_abs"]) < 1e-6
    # A run-time smooth shift subtracts the operand's own mean; the reference does the same.
    assert {row["verification"]["reference"] for row in rows.values() if row["smooth"]} == {
        "float64_matmul_of_site_operands_rounded_f32_shift_adjusted"
    }
