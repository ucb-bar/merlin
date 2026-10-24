"""Reviewer regressions: exact readout, contract drift and reviewed declarations."""

import copy
import inspect
from fractions import Fraction

import numpy as np
import pytest


def test_window_override_uses_the_actual_sum_without_a_divisor_domain_assumption():
    from merlin.targetgen import recipe_numerics as rn

    sa, so = 0.03933464363217354, 0.049954403191804886
    scale = rn.window_mean_readout(sa, 3, so)
    sums = np.arange(-128 * 4, 127 * 4 + 1)
    expected = [min(127, max(-128, round(Fraction(int(v)) * Fraction(sa) / (3 * Fraction(so))))) for v in sums]
    assert rn.readout(sums, scale).tolist() == expected
    assert expected[sums.tolist().index(421)] == 111


def test_window_reference_quantizes_once_after_exact_division():
    from merlin.runtime.integer_ops import window_sum
    from merlin.runtime.tensor import Tensor

    ratio = Fraction(0.03933464363217354) / Fraction(0.049954403191804886)
    attrs = dict(
        mode="window",
        kernel=[2, 2],
        stride=[1, 1],
        padding=[0, 0],
        count_include_pad=True,
        divisor_override=3,
        layout="nchw",
        output_dtype="i8",
        scale_numerator=str(ratio.numerator),
        scale_denominator=str(ratio.denominator),
    )
    assert window_sum(Tensor((1, 1, 2, 2), [128, 128, 128, 37], "i32"), attrs, [1, 1, 1, 1]).data == [111]


def test_integer_readout_defaults_cover_wrapper_and_contraction_free_reference():
    from merlin.capture.integerization import integerize_static_pt2e
    from merlin.targetgen.core_aten_w8a8_reference import selected_pt2e_reference

    for function in (integerize_static_pt2e, selected_pt2e_reference):
        assert inspect.signature(function).parameters["numerics"].default == "integer_readout"


@pytest.mark.parametrize("recipe", [None, {}, {"readout_numerics": "integer_readout"}])
def test_accelerator_capture_default_is_integer_readout(recipe):
    from merlin.capture.integerization import accelerator_readout_numerics

    assert accelerator_readout_numerics(recipe) == "integer_readout"


def test_explicit_float_readout_cannot_feed_an_accelerator_capture():
    from merlin.capture.integerization import accelerator_readout_numerics

    with pytest.raises(ValueError, match="host-only"):
        accelerator_readout_numerics({"readout_numerics": "float_readout"})


@pytest.mark.parametrize("selection", ["explicit", "implicit", "observed"])
def test_provider_drift_is_never_swallowed_by_fence_enforcement(tmp_path, monkeypatch, selection):
    from merlin.common.yaml import write_yaml
    from merlin.targetgen import target_registry as registry
    from merlin.targetgen.capsule_common import unfenced_memory_reads
    from merlin.targetgen.providers import ProviderError

    provider, reference = tmp_path / "support", tmp_path / "reference"
    selected = dict(
        name="synthetic",
        memory_model=dict(hazard_resolution="explicit"),
        compiler_obligations=["must_fence_dram_read_after_write"],
    )
    stale = copy.deepcopy(selected)
    stale["memory_model"]["hazard_resolution"] = "interlocked"
    write_yaml(
        provider / "provider.yaml",
        dict(
            schema="merlin.provider.v1",
            id="stale",
            target="synthetic",
            role="support",
            contract="contracts/target_contract.yaml",
        ),
    )
    write_yaml(provider / "contracts/target_contract.yaml", stale)
    path = reference / "contracts/target_contract.yaml"
    write_yaml(path, selected)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    monkeypatch.setattr(registry, "reference_targets", lambda: {"synthetic": reference})
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    if selection == "explicit":
        monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(path))
    cb = dict(target="synthetic", commands=[], tensors={})
    from contextlib import nullcontext

    context = registry.observed_contract("synthetic", selected) if selection == "observed" else nullcontext()
    with context:
        with pytest.raises(ProviderError, match="drifts"):
            unfenced_memory_reads(cb)
        with pytest.raises(ProviderError, match="drifts"):
            registry.load_contract("synthetic")


@pytest.mark.parametrize("status", [None, "unreviewed", "reviewed"])
def test_unchecked_paths_require_explicit_operation_review(status):
    from merlin.targetgen import capability_roles as cr
    from merlin.targetgen import compute_units as cu

    cap = cu._sem_cap(
        dict(
            family="elementwise_map",
            dtypes=["int8"],
            standalone_evidence=[dict(fact="accumulate", roles=["operand_load"])],
        ),
        "unit",
    )
    result = cr.standalone_admission(cap, review_status=status)
    assert result["standalone"] is (status == "reviewed")
    assert result["basis"] == "declared_unchecked"
    assert all(path["fact_status"] == "not_checked" for path in result["paths"])


def test_recipe_quantizer_keeps_storage_scheme_and_scale_precision():
    import os
    import subprocess

    from merlin.common.paths import repo_root

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required")
    script = r"""
import torch
import torch.ao.quantization.fx._decomposed
from merlin.targetgen._recipe_integer import _per_tensor_scale
DQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
# Centered int32 operands and exact correction planes retain signed/asymmetric
# and unsigned byte schemes. Device admission still requires the provider proof.
for lo, hi, zp, dtype, scale, allowed in [(-128,127,0,torch.int8,.5,True),
    (-127,127,0,torch.int8,.5,True),(-128,127,-3,torch.int8,.5,True),
    (0,255,123,torch.uint8,.5,True),(-128,127,0,torch.int8,.03,False)]:
 class Model(torch.nn.Module):
  def forward(self,x):return DQ(x,scale,zp,lo,hi,dtype)
 graph = torch.export.export(Model(), (torch.ones(2,3,dtype=dtype),)).module().graph
 node = next(n for n in graph.nodes if n.op=='call_function')
 actual = _per_tensor_scale(node,'quantized_decomposed.dequantize_per_tensor')
 assert (actual is not None)==allowed,(lo,hi,zp,dtype,scale,actual)
"""
    result = subprocess.run(
        [python, "-c", script],
        env={**os.environ, "PYTHONPATH": str(repo_root() / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_scaled_load_and_mean_paths_cannot_round_before_quantization():
    from merlin.targetgen.readout_facet import ReadoutFacet
    from merlin.xdsl_dialects.lowering.compute_groups import Admission, _window_mean_refusal

    facet = ReadoutFacet(
        target="synthetic",
        unit="u",
        accumulator_kind="addressable",
        operand_sum={"operands": 2, "operand_dtype": "int8"},
    )
    assert "round before" in facet.operand_sum_refusal((0.5, 0.25))
    assert facet.operand_sum_refusal((1, 1)) is None
    assert all(p["status"] == "proven" for p in facet.operand_sum_product_proofs((1, 1)))

    class Oracle:
        def ask(self, **kwargs):
            return Admission(admitted=True, units=("u",))

    mean = dict(window=4, exactness="float_reassociation")
    assert "exact rational host" in _window_mean_refusal(mean, Oracle(), "i8")
