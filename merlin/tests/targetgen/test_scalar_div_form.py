"""div.Scalar by a power-of-two divisor: the one-element window sum read out at scale 1/d (form ``scalar_div``).

The row ``reduction_scalar_div`` declares the form; ``FORM_TABLE`` says ``window_sum`` writes it; the golden is the
exact rational ``round_half_even(x / d)`` of ``runtime.integer_ops.window_sum``. The claim is limited to power-of-two
divisors, where the f32 readout scale is exact (``prove_fp32_products``)."""

from fractions import Fraction

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.runtime import exact_readout, integer_ops
from merlin.runtime.tensor import Tensor
from merlin.targetgen import corpus_form_table as ft
from merlin.targetgen import corpus_spec as SPEC
from merlin.targetgen.corpus_cell_forms import cell_forms
from merlin.targetgen.corpus_synth import available_ops

pytestmark = pytest.mark.target("gemmini")


def _binding():
    return SPEC.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip


def _golden(values, divisor):
    attrs = {
        "mode": "window", "kernel": [1, 1], "stride": [1, 1], "padding": [0, 0], "count_include_pad": True,
        "divisor_override": divisor, "readout_scale": 1.0, "output_dtype": "i8", "layout": "nchw",
        "rounding": "half_to_even",
    }  # fmt: skip
    shape = (1, 1, 1, len(values))
    return integer_ops.window_sum(Tensor(shape, list(values), "i8"), attrs, list(shape)).data


def test_exact_rational_golden_is_round_half_even_of_x_over_d():
    xs = list(range(-128, 128))
    for d in (1, 2, 4, 8, 16, 64, 128, 256):
        assert [int(v) for v in _golden(xs, d)] == [round(Fraction(x, d)) for x in xs]  # python round() is half-even
    assert [int(v) for v in _golden([-3, -2, -1, 0, 1, 2, 3, 5], 2)] == [-2, -1, 0, 0, 0, 1, 2, 2]


@pytest.mark.parametrize("d", [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024])
def test_f32_readout_equals_the_exact_golden_for_every_int8_for_a_power_of_two_divisor(d):
    xs = np.arange(-128, 128, dtype=np.int64)
    proof = exact_readout.prove_fp32_products(Fraction(1, d), -128, 127)
    assert proof["status"] == "proven" and proof["equals_policy"] == "exact_rational_host"
    device = exact_readout.f32_readout(xs, Fraction(1, d), "i8")
    assert [int(v) for v in device] == [int(v) for v in _golden([int(x) for x in xs], d)]


@pytest.mark.parametrize("d", [3, 5, 6, 7, 10, 12])
def test_a_divisor_that_is_not_a_power_of_two_is_not_declared_and_has_no_equality_proof(d):
    with pytest.raises(ValueError):
        exact_readout.prove_fp32_products(Fraction(1, d), -128, 127)
    with pytest.raises(ValueError):
        ft._scalar_div(d)


def test_the_reviewed_row_declares_the_form_and_states_the_claim():
    from merlin.targetgen.software_spec import validate_software_spec

    spec = load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml")
    row = spec["operations"]["reduction_scalar_div"]
    assert row["status"] == "reviewed" and row["forms"] == ["scalar_div"]
    assert "approved in operator chat 2026-10-29" in row["evidence"]["reviewed_by"]
    assert "power of two" in row["evidence"]["requires_hardware"] and "NOT claimed" in row["evidence"]["requires_hardware"]
    kept, refused = ft.declared_forms(validate_software_spec(spec, "gemmini"), {})
    assert kept.get("reduction") == ["scalar_div"] and not [r for r in refused if r["family"] == "reduction"]


def test_the_form_is_one_window_sum_cellform():
    forms = cell_forms("reduction", "i8", None, available_ops(), ("scalar_div",))
    assert {f.form: f.op for f in forms if f.form} == {"scalar_div": "window_sum"}


@pytest.mark.parametrize("m,n", [(16, 16), (16, 15), (4, 4)])
def test_the_form_builds_a_one_element_window_sum_with_the_divisor(m, n):
    entry = {
        "name": "T_div", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test", "op": "window_sum",
        "label": "public", **ft.entry_fields("reduction", "scalar_div", m, n),
    }  # fmt: skip
    cap, text = SPEC.build(entry, _binding())
    attrs = cap["operation"]["attributes"]
    assert cap["operation"]["op"] == "window_sum" and "merlin_iface.window_sum" in text
    assert attrs["kernel"] == [1, 1] and attrs["divisor_override"] == 2 and attrs["readout_scale"] == 1.0
    assert cap["inputs"][0]["shape"] == [1, 1, m, n]


def _fixture_module():
    """The phase-0 admission test module's contract and spec fixtures (it is not an importable package)."""
    import importlib.util

    path = repo_root() / "packages/merlin-experiments/tests/test_phase0_program_admission.py"
    spec = importlib.util.spec_from_file_location("_phase0_admission_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_one_element_window_command_is_observed_and_admitted_by_the_standalone_sum_row():
    from copy import deepcopy

    from merlin.targetgen.interface_observations import command_rows
    from merlin.targetgen.operation_accounting import admit_operation_row
    from merlin.targetgen.software_spec import validate_software_spec

    fx = _fixture_module()
    entry = ft.entry_fields("reduction", "scalar_div", 16, 16)
    cb = fx._window_sum_cb(
        {k: entry[k] for k in ("mode", "kernel", "stride", "padding", "count_include_pad", "divisor_override")},
        entry["shape"], entry["out_shape"],
    )  # fmt: skip
    contract = deepcopy(fx._CONTRACT)
    contract["compute_units"][0]["semantic_capabilities"].append(
        {"family": "reduction", "dtypes": ["int8"], "kinds": ["sum"]}
    )
    row_spec = load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml")["operations"]["reduction_scalar_div"]
    spec = dict(fx._SPEC)
    spec["operations"] = {
        **fx._SPEC["operations"],
        "reduction_scalar_div": {
            k: row_spec[k] for k in ("families", "arithmetic", "kinds", "ranks", "layouts", "tails", "aliasing")
        } | {"placement": "accelerator", "dtypes": ["int8"]},
    }
    spec = validate_software_spec(spec, "fixture")
    [row] = [r for r in command_rows(cb) if r["operation"] == "window_sum"]
    assert row["semantic_family"] == "reduction" and row["reduction_kind"] == "sum"
    assert row["tails"] == "zero_pad_valid_window"
    admitted = admit_operation_row(row, software_spec=spec, capability_contract=contract, host_capabilities=None)
    assert admitted["accelerator_admission"]["status"] == "admitted", admitted["accelerator_admission"]
