"""A model's quantization is derived from what the target's readout holds, or is refused with why."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common import quant_formats as qf
from merlin.targetgen import quant_recipe as QR
from merlin.targetgen import readout_facet as RF

_ABI = {
    "schema": RF.SCALAR_ABI_SCHEMA,
    "accumulator_dtype": "i32",
    "output_dtype": "i8",
    "scale_dtype": "f32",
    "clamp_min": -128,
    "clamp_max": 127,
    "provenance": {"scope": "test"},
}
_READOUTS = [{"selector": "i8", "applies": ["acc_scale", "relu", "bias_add"]}]


def _facts(scale_field: str = "out_scale") -> dict:
    return {
        "facts": {
            "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
            "memories": [{"name": "accumulator", "bytes": 4096}],
            "interfaces": [
                {
                    "name": "register_bundle_layouts",
                    "unresolved": {},
                    "bundles": {
                        "StoreConfig": {
                            "width": 64,
                            "fields": {
                                scale_field: {
                                    "offset": 32,
                                    "width": None,
                                    "width_param": "scale_bits",
                                    "slot_width": 32,
                                }
                            },
                        }
                    },
                }
            ],
        }
    }


def _facet(**kwargs) -> RF.ReadoutFacet:
    return RF.derive(
        "t", facts=_facts(), unit={"name": "u", "dtypes": ["int8"]}, scalar_abi=_ABI, readouts=_READOUTS, **kwargs
    )


def test_a_readout_holding_one_scale_per_command_derives_per_tensor_symmetric_int8() -> None:
    recipe = QR.derive(_facet()).to_dict()
    assert recipe["status"] == QR.DERIVED
    assert recipe["weight"] == {
        "dtype": "int8",
        "granularity": "tensor",
        "symmetric": True,
        "quant_min": -127,
        "quant_max": 127,
        "block": None,
        "mode": "static",
    }
    assert (recipe["activation"]["granularity"], recipe["activation"]["mode"]) == ("tensor", "static")
    assert (recipe["activation"]["quant_min"], recipe["activation"]["quant_max"]) == (-128, 127)
    assert recipe["bias_domain"] == "accumulator"
    assert "per tensor" in recipe["why"]["weight.granularity"]


_SUM = {
    "schema": RF.OPERAND_SUM_SCHEMA,
    "operands": 2,
    "operand_dtype": "i8",
    "scale_dtype": "f32",
    "operand_rounding": "half_even",
    "operand_saturates": True,
}


def test_an_add_is_quantized_only_where_a_unit_can_sum_operands() -> None:
    plain, summing = QR.derive(_facet()), QR.derive(_facet(operand_sum=_SUM))
    assert QR.OPERAND_SUM not in plain.families and "load-scale" in plain.unquantized[QR.OPERAND_SUM]
    assert QR.OPERAND_SUM in summing.families and not summing.unquantized
    assert "1 output step" in summing.why[QR.OPERAND_SUM]
    # Two recipes are two captures: the family is content, so the digest moves with it.
    assert plain.to_dict()["recipe_sha256"] != summing.to_dict()["recipe_sha256"]


def test_a_window_mean_is_quantized_where_one_static_scale_can_carry_the_reciprocal() -> None:
    assert QR.WINDOW_MEAN in QR.derive(_facet()).families
    # Per-token dynamic activations have no single compile-time scale to fold 1/count into.
    dynamic = QR.derive(
        RF.derive(
            "t",
            facts=_facts(),
            unit={"name": "u", "dtypes": ["int8"]},
            scalar_abi=_ABI,
            readouts=_READOUTS,
        ),
        prefer_activation_mode="dynamic",
    )
    assert (QR.WINDOW_MEAN in dynamic.families) == (dynamic.activation.mode == "static")
    name = next(n for n, f in sorted(qf.registry().items()) if f.kind == "mx_block")
    block = QR.derive(RF.derive("t", unit={"name": "u", "dtypes": [name]}))
    assert block.status == QR.DERIVED
    assert QR.WINDOW_MEAN not in block.families and "per block" in block.unquantized[QR.WINDOW_MEAN]


def test_a_finer_readout_derives_a_finer_weight_scale() -> None:
    facet = _facet()
    facet.scale_granularities = ("tensor", "column")
    assert QR.derive(facet).weight.granularity == "channel"
    facet.scale_granularities = ("tensor", "row", "rank1")
    dynamic = QR.derive(facet, prefer_activation_mode="dynamic")
    assert (dynamic.activation.granularity, dynamic.activation.mode) == ("token", "dynamic")


def test_a_dynamic_request_the_readout_cannot_absorb_is_recorded_not_honoured() -> None:
    recipe = QR.derive(_facet(), prefer_activation_mode="dynamic")
    assert recipe.activation.mode == "static"
    assert "host reduction" in recipe.why["activation.mode"]


def test_an_underived_facet_yields_no_recipe_and_never_a_default() -> None:
    # The mutation: without the scale field the granularity is unknown. The recipe must not fall
    # back to the common per-channel default, which is the defect this whole path exists to end.
    facet = RF.derive("t", facts=_facts("out_gain"), unit={"name": "u", "dtypes": ["int8"]}, scalar_abi=_ABI)
    recipe = QR.derive(facet)
    assert (recipe.status, recipe.weight, recipe.activation) == (QR.UNDERIVABLE, None, None)
    assert "scale role" in recipe.underivable["granularity"]


def test_a_block_scaled_unit_derives_its_recipe_from_the_format_alone() -> None:
    name, fmt = next((n, f) for n, f in sorted(qf.registry().items()) if f.kind == "mx_block")
    recipe = QR.derive(RF.derive("t", facts={}, unit={"name": "u", "dtypes": [name]}))
    assert recipe.status == QR.DERIVED
    assert (recipe.weight.dtype, recipe.weight.granularity, recipe.weight.block) == (name, "block", fmt.scale.block)
    assert recipe.activation.mode == "dynamic"


def test_the_digest_is_of_the_content_and_not_of_the_explanation() -> None:
    body = QR.derive(_facet()).to_dict()
    assert body["recipe_sha256"] == QR.digest(body)
    reworded = {**body, "why": {"weight.granularity": "said differently"}}
    assert QR.digest(reworded) == body["recipe_sha256"]
    changed = json.loads(json.dumps(body))
    changed["weight"]["granularity"] = "channel"
    assert QR.digest(changed) != body["recipe_sha256"]


def test_every_declared_target_gets_a_recipe_or_its_reasons() -> None:
    from merlin.common.paths import merlin_dir
    from merlin.targetgen.target_experiment import load_capability_manifest

    seen = 0
    for contract in sorted((merlin_dir() / "targets").glob("*/contracts/target_contract.yaml")):
        target = contract.parent.parent.name
        try:
            load_capability_manifest(target)
        except Exception:  # noqa: BLE001 -- a manifest that does not load is another test's
            continue
        for recipe in QR.for_target(target):
            body = recipe.to_dict()
            assert (body["status"] == QR.DERIVED) == (body["weight"] is not None)
            assert body["status"] == QR.DERIVED or body["underivable"]
            seen += 1
    assert seen >= 3


def test_a_capture_is_keyed_by_its_recipe_and_refuses_an_underived_one(tmp_path: Path) -> None:
    from merlin.targetgen import capsule_source as CS

    source = CS.PytorchRefSource.__new__(CS.PytorchRefSource)
    source.m2m_dir, source.python, source.timeout = tmp_path, tmp_path / "python", 5
    plain = source._cache_slot("model", "int8", "src", None)
    keyed = source._cache_slot("model", "int8", "src", None, recipe_sha256="abc")
    other = source._cache_slot("model", "int8", "src", None, recipe_sha256="abd")
    assert len({str(plain), str(keyed), str(other)}) == 3
    underived = QR.derive(RF.derive("t", facts={})).to_dict()
    with pytest.raises(CS.M2MUnavailable, match="not derived"):
        source._run(tmp_path / "loader.py", "model", "int8", workdir=tmp_path, src="", recipe=underived)


_TINY_LOADER = """
import torch
from torch import nn

def get_model_and_inputs():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1))
    return model.eval(), (torch.randn(1, 3, 8, 8),)
"""


_RESIDUAL_LOADER = """
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem, self.body = nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        skip = torch.relu(self.stem(x))
        return self.head(torch.relu(self.body(skip) + skip))

def get_model_and_inputs():
    torch.manual_seed(0)
    return Block().eval(), (torch.randn(1, 3, 8, 8),)
"""


_NORMED_LOADER = """
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem, self.body = nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        skip = torch.relu(self.stem(x))
        summed = self.body(skip) + skip
        scale = torch.rsqrt(summed.pow(2).mean(dim=1, keepdim=True) + 1e-6)
        return self.head(summed * scale)

def get_model_and_inputs():
    torch.manual_seed(0)
    return Block().eval(), (torch.randn(1, 3, 8, 8),)
"""


@pytest.mark.slow
def test_a_sum_a_normalization_reads_is_left_in_floating_point(tmp_path: Path) -> None:
    """The family being listed is not enough: a sum nobody requantizes cannot close, so quantizing
    its operands would cost accuracy and place nothing (measured on a two-layer decoder)."""
    from merlin.common.paths import env

    python = env("MERLIN_M2M_PYTHON") or (
        str(Path(env("MERLIN_M2M_VENV") or "") / "bin/python") if env("MERLIN_M2M_VENV") else ""
    )
    m2m_dir = env("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m_dir:
        pytest.skip("no capture interpreter is configured")
    worker = Path(QR.__file__).with_name("_m2m_capture_worker.py")
    (tmp_path / "loader.py").write_text(_NORMED_LOADER, encoding="utf-8")
    recipe = QR.derive(_facet(operand_sum=_SUM))
    assert {QR.OPERAND_SUM, QR.WINDOW_MEAN} <= set(recipe.families)
    (tmp_path / "recipe.json").write_text(json.dumps(recipe.to_dict()), encoding="utf-8")
    result = subprocess.run(
        [python, str(worker), "--loader", str(tmp_path / "loader.py"), "--dtype", "int8", "--out", str(tmp_path / "o")]
        + ["--m2m-dir", m2m_dir, "--recipe", str(tmp_path / "recipe.json")],
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    stats = json.loads((tmp_path / "o" / "meta.json").read_text())["quantization_stats"]
    assert (stats["annotated_sums"], stats["annotated_means"]) == (0, 0)
    assert stats["sums_and_means_left_in_float"] >= 2 and stats["annotated_contractions"] == 3


@pytest.mark.slow
def test_a_residual_is_captured_as_an_integer_sum_only_under_a_recipe_that_lists_it(tmp_path: Path) -> None:
    """The recipe decides whether an add is quantized, and group formation reads the difference."""
    from fake_quant_layer import Oracle

    from merlin.common import mlir_query as mq
    from merlin.common.paths import env
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    python = env("MERLIN_M2M_PYTHON") or (
        str(Path(env("MERLIN_M2M_VENV") or "") / "bin/python") if env("MERLIN_M2M_VENV") else ""
    )
    m2m_dir = env("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m_dir:
        pytest.skip("no capture interpreter is configured")
    worker = Path(QR.__file__).with_name("_m2m_capture_worker.py")
    (tmp_path / "loader.py").write_text(_RESIDUAL_LOADER, encoding="utf-8")
    facet = _facet(operand_sum=_SUM)
    oracle = Oracle(readout=RF.TargetReadout((facet,)))
    placed = {}
    for label, recipe in (("float_skip", QR.derive(_facet())), ("integer_sum", QR.derive(facet))):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")
        out = tmp_path / label
        result = subprocess.run(
            [python, str(worker), "--loader", str(tmp_path / "loader.py"), "--dtype", "int8", "--out", str(out)]
            + ["--m2m-dir", m2m_dir, "--recipe", str(path)],
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, "TMPDIR": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr[-2000:]
        stats = json.loads((out / "meta.json").read_text())["quantization_stats"]
        assert stats["annotated_sums"] == (1 if label == "integer_sum" else 0)
        assert stats["annotated_means"] == 0  # the model has none, and none is invented
        groups = CG.form_groups(mq.parse((out / "linalg.mlir").read_text()), "synthetic", oracle=oracle)
        placed[label] = [g for g in groups if g.operand_sum is not None]
    assert not placed["float_skip"]
    (summed,) = placed["integer_sum"]
    assert summed.placement == "u" and summed.operand_sum["relu"] is True


@pytest.mark.slow
def test_the_generic_quantizer_realises_the_recipe_in_the_capture_venv(tmp_path: Path) -> None:
    """End to end in the interpreter that has torch: recipe JSON in, the recipe's arithmetic out."""
    from merlin.common.paths import env

    python = env("MERLIN_M2M_PYTHON") or (
        str(Path(env("MERLIN_M2M_VENV") or "") / "bin/python") if env("MERLIN_M2M_VENV") else ""
    )
    m2m_dir = env("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m_dir:
        pytest.skip("no capture interpreter is configured")
    worker = Path(QR.__file__).with_name("_m2m_capture_worker.py")
    (tmp_path / "loader.py").write_text(_TINY_LOADER, encoding="utf-8")
    seen = {}
    for granularity in ("tensor", "channel"):
        body = QR.derive(_facet()).to_dict()
        body["weight"]["granularity"] = granularity
        body["recipe_sha256"] = QR.digest(body)
        recipe = tmp_path / f"{granularity}.json"
        recipe.write_text(json.dumps(body), encoding="utf-8")
        out = tmp_path / granularity
        result = subprocess.run(
            [
                python,
                str(worker),
                "--loader",
                str(tmp_path / "loader.py"),
                "--dtype",
                "int8",
                "--out",
                str(out),
                "--m2m-dir",
                m2m_dir,
                "--recipe",
                str(recipe),
            ],
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, "TMPDIR": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr[-2000:]
        meta = json.loads((out / "meta.json").read_text())
        assert meta["recipe_sha256"] == body["recipe_sha256"] and meta["scheme"] is None
        assert meta["quantization_stats"]["annotated_contractions"] == 2
        assert meta["recipe_agreement"]["samples"] >= 1
        seen[granularity] = (out / "linalg.mlir").read_text()
    # The recipe decides the program: per-tensor weights dequantize per tensor, per-channel per axis.
    assert "dequantize_per_channel" not in seen["tensor"]
    assert "dequantize_per_channel" in seen["channel"]


def test_a_capture_takes_the_targets_recipe_only_when_it_is_derived_for_that_format(monkeypatch):
    from merlin.targetgen import capsule_source as CS

    monkeypatch.setattr(QR, "for_target", lambda target, **_: [QR.derive(_facet())])
    assert CS.derived_recipe("t", "int8")["weight"]["granularity"] == "tensor"
    assert CS.derived_recipe("t", "i8")["weight"]["dtype"] == "int8"  # alias of the same format
    assert CS.derived_recipe("t", "fp8_e4m3") is None  # another format: not ours
    monkeypatch.setattr(QR, "for_target", lambda target, **_: [QR.derive(RF.derive("t", facts={}))])
    assert CS.derived_recipe("t", "int8") is None  # underivable: scheme stands
    assert CS.derived_recipe(None, "int8") is None
