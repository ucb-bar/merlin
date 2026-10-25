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
    assert "0 output step" in summing.why[QR.OPERAND_SUM]
    # Two recipes are two captures: the family is content, so the digest moves with it.
    assert plain.to_dict()["recipe_sha256"] != summing.to_dict()["recipe_sha256"]
    # Exact is the default; the historical bounded policy remains an explicit compatibility choice.
    exact = QR.derive(_facet(operand_sum=_SUM), integer_sum_policy="exact")
    bounded = QR.derive(_facet(operand_sum=_SUM), integer_sum_policy="bounded")
    assert "0 output step" in exact.why[QR.OPERAND_SUM]
    assert summing.integer_sum_policy == exact.integer_sum_policy == "exact"
    assert "integer_sum_policy" not in summing.to_dict()
    assert exact.to_dict()["integer_sum_policy"] == "exact"
    assert bounded.to_dict()["integer_sum_policy"] == "bounded"
    assert len({r.to_dict()["recipe_sha256"] for r in (exact, bounded, summing)}) == 3
    with pytest.raises(ValueError, match="explicit per-operand requantization capture"):
        QR.derive(_facet(operand_sum=_SUM), integer_sum_policy="per_operand")


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
    # Without owned upstream files caching deliberately returns None; exercise
    # request identities with a complete direct-owner fixture, not an empty path.
    for relative in (
        "m2m/api.py",
        "m2m/ir/import_fx.py",
        "m2m/capture/torchao_pipeline.py",
        "m2m/capture/torchao_schemes.py",
        "m2m/capture/torch_export.py",
        "m2m/capture/torch_mlir_bridge.py",
    ):
        owner = tmp_path / relative
        owner.parent.mkdir(parents=True, exist_ok=True)
        owner.write_text("# owned test implementation\n")
    plain = source._cache_slot("model", "int8", "src", None)
    keyed = source._cache_slot("model", "int8", "src", None, recipe_sha256="abc")
    other = source._cache_slot("model", "int8", "src", None, recipe_sha256="abd")
    assert all(slot is not None for slot in (plain, keyed, other))
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


_FP8_LOADER = """
import torch
from torch import nn

def get_model_and_inputs():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)).eval(), (torch.randn(1, 4),)
"""


@pytest.mark.slow
def test_static_fp8_recipe_records_projected_storage_as_diagnostic(tmp_path: Path) -> None:
    """A valid FP8 recipe reaches capture but cannot claim native FP8 from projected f32 storage."""
    from merlin.common.paths import env

    python = env("MERLIN_M2M_PYTHON") or (
        str(Path(env("MERLIN_M2M_VENV") or "") / "bin/python") if env("MERLIN_M2M_VENV") else ""
    )
    m2m_dir = env("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m_dir:
        pytest.skip("no capture interpreter is configured")
    worker = Path(QR.__file__).with_name("_m2m_capture_worker.py")
    (tmp_path / "loader.py").write_text(_FP8_LOADER, encoding="utf-8")
    recipe = {
        "schema": "quant_recipe_v1",
        "target": "synthetic",
        "unit": "tensor_core",
        "status": "derived",
        "families": ["contraction"],
        "weight": {
            "dtype": "fp8_e4m3",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": None,
            "quant_max": None,
            "block": None,
            "mode": "static",
        },
        "activation": {
            "dtype": "fp8_e4m3",
            "granularity": "tensor",
            "symmetric": True,
            "quant_min": None,
            "quant_max": None,
            "block": None,
            "mode": "static",
        },
    }
    recipe["recipe_sha256"] = QR.digest(recipe)
    (tmp_path / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    result = subprocess.run(
        [
            python,
            str(worker),
            "--loader",
            str(tmp_path / "loader.py"),
            "--dtype",
            "fp8_e4m3",
            "--out",
            str(tmp_path / "capture"),
            "--m2m-dir",
            m2m_dir,
            "--recipe",
            str(tmp_path / "recipe.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "TMPDIR": str(tmp_path)},
    )
    assert result.returncode == 3, result.stderr[-3000:]
    assert (tmp_path / "capture" / "linalg.mlir").is_file()
    meta = json.loads((tmp_path / "capture" / "meta.json").read_text())
    assert meta["ok"] is False
    assert meta["precision_realization"]["status"] == "projected"
    assert meta["scheme"] == "fp8_e4m3_static_act_weight"
    assert meta["quantization_stats"]["framework_capture_policy"]["activation_observer"] == "minmax"


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
    # The exact-rational oracle cannot qualify this graph: the float sum rounds an already rounded
    # result, and rsqrt has no exact model. The per-graph selection must say so visibly and hand the
    # golden to the framework-float integer reference rather than claim exact arithmetic.
    from merlin.targetgen.pt2e_reference_selection import FRAMEWORK_ENGINE

    assert result.returncode == 0, result.stderr[-2000:]
    meta = json.loads((tmp_path / "o" / "meta.json").read_text())
    stats = meta["quantization_stats"]
    assert (stats["annotated_sums"], stats["annotated_means"]) == (0, 0), stats
    assert stats["sums_and_means_left_in_float"] >= 2 and stats["annotated_contractions"] == 3, stats
    receipt = meta["integerization_receipt"]
    selection = receipt["reference_selection"]
    assert selection["engine"] == FRAMEWORK_ENGINE and selection["closed_rational"] is False, selection
    assert selection["reason"], selection
    agreement = receipt["golden_agreement"]
    assert agreement["status"] == "passed", agreement
    assert agreement["arithmetic"] == "integer_contractions_framework_float", agreement
    inspect = subprocess.run(
        [
            python,
            "-c",
            """
import importlib.util,json,sys
from merlin.targetgen._recipe_quantizer import apply_recipe
spec=importlib.util.spec_from_file_location('loader',sys.argv[1])
loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
model,inputs=loader.get_model_and_inputs()
quantized=apply_recipe(model,json.load(open(sys.argv[2])),example_inputs=inputs)
s=quantized._recipe_quantization_stats
assert (s['annotated_sums'],s['annotated_means'])==(0,0),s
assert s['sums_and_means_left_in_float']>=2 and s['annotated_contractions']==3,s
""",
            str(tmp_path / "loader.py"),
            str(tmp_path / "recipe.json"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(worker.parent.parent.parent)},
    )
    assert inspect.returncode == 0, inspect.stdout + inspect.stderr


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
    # Legacy grouping has no exact-product proof for these two load scales.
    # It must leave the sum on the host; the region backend's exact full-i32
    # implementation is exercised separately by the cross-repo device gate.
    assert not placed["integer_sum"]


@pytest.mark.slow
def test_the_generic_quantizer_realises_tensor_and_channel_integer_capture(tmp_path: Path) -> None:
    """A realized TorchAO format is not automatically a supported integer lowering."""
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
        body["software_numerical_engine"] = "integer_reference"
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
        meta = json.loads((out / "meta.json").read_text())
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
        assert meta["integerization_receipt"]["quantized_contractions_remaining"] == 0
        assert meta["integerization_receipt"]["golden_agreement"]["status"] == "passed"
        assert meta["integerization_receipt"]["golden_agreement"]["reference"] == "pt2e_integer"
        assert meta["integerization_receipt"]["portable_agreement"]["informational"] is True
        assert meta["integerization_receipt"]["exported_integer_mm_count"] == 2
        assert meta["determinism"]["deterministic_algorithms"] == "required"
        assert meta["recipe_sha256"] == body["recipe_sha256"]
        assert meta["scheme"] == "int8_static_act_int8_weight"
        assert meta["quantization_stats"]["annotated_contractions"] == 2
        assert meta["recipe_agreement"]["samples"] >= 1
        seen[granularity] = (out / "linalg.mlir").read_text()
    # Both paths perform integer contractions; channel scales remain explicit
    # output-side reconstruction, never floating-point weight dequantization.
    assert "dequantize_per_channel" not in seen["tensor"]
    assert "dequantize_per_channel" not in seen["channel"]


def test_a_capture_takes_the_targets_recipe_only_when_it_is_derived_for_that_format(monkeypatch):
    from merlin.targetgen import capsule_source as CS

    monkeypatch.setattr(QR, "for_target", lambda target, **_: [QR.derive(_facet())])
    assert CS.derived_recipe("t", "int8")["weight"]["granularity"] == "tensor"
    assert CS.derived_recipe("t", "i8")["weight"]["dtype"] == "int8"  # alias of the same format
    assert CS.derived_recipe("t", "fp8_e4m3") is None  # another format: not ours
    monkeypatch.setattr(QR, "for_target", lambda target, **_: [QR.derive(RF.derive("t", facts={}))])
    assert CS.derived_recipe("t", "int8") is None  # underivable: scheme stands
    assert CS.derived_recipe(None, "int8") is None


def test_capture_cpu_policy_keeps_reference_bytes_when_inherited_threads_change():
    from merlin.common.paths import env

    python = env("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("no capture interpreter is configured")
    worker = Path(QR.__file__).with_name("_m2m_capture_worker.py")
    program = """
import importlib.util, json, sys, torch
spec = importlib.util.spec_from_file_location("capture_worker", sys.argv[1])
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
observations = []
for inherited_threads in (2, 4):
    torch.set_num_threads(inherited_threads)
    policy = worker._seed_capture(0, torch)
    assert policy["cpu_threads"] == torch.get_num_threads() == 1
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16)).eval()
    with torch.no_grad():
        output = model(torch.linspace(-1, 1, 272).reshape(17, 16))
    observations.append(output.contiguous().view(torch.uint8).numpy().tobytes())
assert observations[0] == observations[1]
print(json.dumps({"cpu_threads": policy["cpu_threads"], "same_bytes": True}))
"""
    result = subprocess.run(
        [python, "-c", program, str(worker)], capture_output=True, text=True, timeout=60, check=True
    )
    assert json.loads(result.stdout) == {"cpu_threads": 1, "same_bytes": True}


def _scheme_facet(columns):
    from merlin.targetgen.output_path import from_record

    path = from_record({"scale": {"status": "derived", "rounding": "half_even", "columns_per_scale": columns}})
    return RF.derive(
        "t",
        facts=_facts(),
        unit={"name": "u", "dtypes": ["int8"], "ops": ["matmul"]},
        scalar_abi=_ABI,
        readouts=_READOUTS,
        output_path=path,
    )


def _scheme(granularity="tensor", value="power_of_two"):
    return dict(weight_scale_granularity=granularity, scale_value=value, readout_strategy="f32_product_rne")


@pytest.mark.parametrize("columns", [3, 7, 24])
def test_selected_scheme_chooses_weight_granularity_from_output_path(columns):
    facet = _scheme_facet(columns)
    tensor = QR.derive(facet, scheme=_scheme()).to_dict()
    blocked = QR.derive(facet, scheme=_scheme("column_block")).to_dict()
    column = QR.derive(facet, scheme=_scheme("column")).to_dict()
    assert tensor["weight"]["granularity"] == "tensor"
    assert blocked["weight"]["granularity"] == column["weight"]["granularity"] == "channel"
    assert blocked["weight"]["channels_per_scale"] == columns
    assert "channels_per_scale" not in column["weight"]
    assert blocked["activation"]["granularity"] == "tensor"
    assert "integer_sum_policy" not in blocked
    assert len({r["recipe_sha256"] for r in (tensor, blocked, column)}) == 3


def test_scheme_calibration_policy_is_explicit_and_does_not_mutate_declaration():
    domain = dict(
        schema="merlin.declared_observer_domain.v1",
        minimum=-0.7,
        maximum=1.3,
        evaluation_inputs="never_read",
        scale_policy="enclosing_power_of_two",
    )
    recipe = QR.derive(
        _scheme_facet(5), scheme=_scheme("column_block", "arbitrary_f32"), calibration_domain=domain
    ).to_dict()
    assert recipe["calibration_domain"]["scale_policy"] == "declared_observer"
    assert domain["scale_policy"] == "enclosing_power_of_two"
    assert (
        recipe["recipe_sha256"]
        != QR.derive(_scheme_facet(5), scheme=_scheme("column_block"), calibration_domain=domain).to_dict()[
            "recipe_sha256"
        ]
    )
    assert QR.derive(_scheme_facet(5), scheme=_scheme(value="arbitrary_f32")).status == QR.UNDERIVABLE


def test_unresolved_or_unproved_schemes_never_choose_a_default():
    assert QR.derive(_facet(), scheme=_scheme("column_block")).status == QR.UNDERIVABLE
    mismatch = dict(_scheme("column_block"), columns_per_scale=9)
    assert QR.derive(_scheme_facet(5), scheme=mismatch).status == QR.UNDERIVABLE
    with pytest.raises(ValueError, match="resolved"):
        QR.derive(_facet(), scheme=dict(_scheme(), status="unresolved"))
    with pytest.raises(ValueError, match="f32_product_rne"):
        QR.derive(_facet(), scheme=dict(_scheme(value="arbitrary_f32"), readout_strategy="exact_rational_host"))
    with pytest.raises(ValueError, match="explicit per-operand"):
        QR.derive(_facet(), scheme=dict(_scheme(), operand_sum_rounding="per_operand"))


@pytest.mark.parametrize(
    "fixture,expected",
    [
        ("tensor", "4391f1817eeab83d01891f4cd8554c5b56c70d6928122c20db7fd43bcb71bd0a"),
        ("sum", "010d6881ef1bde1fc78254a56c5d670a81977c68a62a14fcca49bf6cf4a75258"),
        ("bounded", "08289e2b34468c4c02c4d98bac69dff186a9d163454d05a7420302b451498d0c"),
        ("column", "25fd93afd60a7ecb92b1cdf2b53c8da3031306cedb1537c891b46cc69abf0784"),
    ],
)
def test_explicit_v1_recipe_fixture_digests(fixture, expected):
    facet = _facet(operand_sum=_SUM) if fixture in ("sum", "bounded") else _facet()
    if fixture == "column":
        facet.scale_granularities = ("tensor", "column")
    # Explicit bounded now carries its declaration; implicit bytes are pinned below.
    recipe = QR.derive(facet, integer_sum_policy="bounded" if fixture == "bounded" else "exact").to_dict()
    assert recipe["recipe_sha256"] == expected
    if fixture == "tensor":
        selected = QR.derive(facet, scheme=_scheme(), integer_sum_policy="exact").to_dict()
        partitionable = QR.derive(_scheme_facet(5), integer_sum_policy="exact").to_dict()
        # A newly derived partition capability must not silently change a v1 capture.
        assert json.dumps(selected, sort_keys=True) == json.dumps(recipe, sort_keys=True)
        assert json.dumps(partitionable, sort_keys=True) == json.dumps(recipe, sort_keys=True)


def test_arbitrary_scales_use_capture_instead_of_the_power_of_two_native_proof():
    from merlin.llvmlower.quantized_native import parameters

    spec = dict(
        numerical_semantics=dict(
            quantized_native_fallback=dict(admission="after_native_exact_failure"),
            calibration_domain=dict(
                minimum=-1, maximum=1, evaluation_inputs="never_read", scale_policy="declared_observer"
            ),
        )
    )
    with pytest.raises(ValueError, match="capture-time PT2E"):
        parameters(spec)


def test_implicit_exact_policy_preserves_merge_base_default_digest():
    recipe = QR.derive(_facet())
    assert recipe.integer_sum_policy == "exact"
    # Computed by loading git show 377f1b8a4:src/merlin/targetgen/quant_recipe.py
    # in a separate module and evaluating its derive(_facet()).to_dict().
    assert recipe.to_dict()["recipe_sha256"] == "e6db69a15415b903d7ae1a91be9d624911389a9ff4d1ced45043b8183ddcbd4a"
    assert "integer_sum_policy" not in recipe.to_dict()


def test_only_explicit_integer_sum_policies_are_serialized():
    implicit = QR.derive(_facet()).to_dict()
    exact = QR.derive(_facet(), integer_sum_policy="exact").to_dict()
    bounded = QR.derive(_facet(), integer_sum_policy="bounded").to_dict()
    assert "integer_sum_policy" not in implicit
    assert exact["integer_sum_policy"] == "exact"
    assert bounded["integer_sum_policy"] == "bounded"
    assert len({r["recipe_sha256"] for r in (implicit, exact, bounded)}) == 3


def test_recipe_constructor_preserves_implicit_and_explicit_policy_provenance():
    implicit = QR.QuantRecipe(target="t", unit="u")
    assert implicit.integer_sum_policy == "exact"
    assert "integer_sum_policy" not in implicit.to_dict()
    for policy in ("exact", "bounded"):
        explicit = QR.QuantRecipe(target="t", unit="u", integer_sum_policy=policy)
        assert explicit.to_dict()["integer_sum_policy"] == policy
