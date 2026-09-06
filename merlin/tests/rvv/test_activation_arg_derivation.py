"""The ACTIVATION / WEIGHT_INVARIANT split must come from the manifest, never from argument position.

WHY THIS TEST EXISTS. `build_tools/scripts/model_math_census.py` originally derived the activation
set as ``range(len(input_order.json))`` -- "the first n `@forward` arguments are the inputs, the rest
are weights". That is false for every capture measured in this repo, and it inverts the census:
resnet50_v1_5's only activation is argument 320 (0-160 are params, 161-319 buffers) and lstmnetvit's
five are 98-102, so the positional rule labelled a CONVOLUTION WEIGHT "the activation" and the image
"weight-invariant". The aggregate percentages barely moved (the two operands of a contraction are
similar in size), so the defect was invisible in the headline number and only showed up in WHICH ops
were named hoistable -- which was the whole point of the census.

`input_order.json` maps an input NAME to its index in the inputs npz. It carries no argument
positions and must never be read as if it did.
"""
from __future__ import annotations

import json

import importlib.util

import pytest

from merlin.common.paths import repo_root


def _census_module():
    """Load the census SCRIPT by path -- it lives in `build_tools/scripts`, which is not a package.
    Loading the real file (rather than re-stating its rule here) is what makes this a regression
    test of the shipped derivation instead of a test of a copy of it."""
    path = repo_root() / "build_tools" / "scripts" / "model_math_census.py"
    spec = importlib.util.spec_from_file_location("model_math_census", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bundles():
    root = repo_root() / "out" / "artifacts" / "recaptures"
    for name in ("resnet50_v1_5_int8_w8a8_consistent", "lstmnetvit_int8_w8a8_consistent"):
        b = root / name
        if (b / "weights.safetensors.manifest.json").is_file() and (b / "input_order.json").is_file():
            yield b


def test_manifest_declares_which_arguments_are_inputs():
    """Every bundle names its activations explicitly, so nothing has to be inferred."""
    activation_arg_indices = _census_module().activation_arg_indices

    seen = 0
    for bundle in _bundles():
        manifest = json.loads((bundle / "weights.safetensors.manifest.json").read_text())
        idx = activation_arg_indices(manifest)
        assert idx, f"{bundle.name}: manifest declares no kind='input' argument"
        for i in idx:
            assert manifest[str(i)]["kind"] == "input"
        seen += 1
    if not seen:
        pytest.skip("no int8 recapture bundles present in this checkout")


def test_positional_rule_disagrees_with_the_manifest():
    """THE REGRESSION GUARD. The old rule and the correct one must not be confusable.

    Asserting mere inequality would pass on an off-by-one; these captures disagree far more strongly
    than that, so the assertion is DISJOINTNESS -- the positional guess and the truth share no
    argument at all. A future change that reintroduces ``range(len(input_order.json))`` cannot make
    this pass by accident.
    """
    activation_arg_indices = _census_module().activation_arg_indices

    seen = 0
    for bundle in _bundles():
        manifest = json.loads((bundle / "weights.safetensors.manifest.json").read_text())
        truth = activation_arg_indices(manifest)
        positional = set(range(len(json.loads((bundle / "input_order.json").read_text()))))
        assert not (truth & positional), (
            f"{bundle.name}: the positional rule {sorted(positional)} overlaps the manifest-derived "
            f"activation set {sorted(truth)}; this test can no longer detect the old defect")
        assert len(positional) == len(truth), (
            f"{bundle.name}: the two rules should differ in POSITION, not in count")
        seen += 1
    if not seen:
        pytest.skip("no int8 recapture bundles present in this checkout")


def test_activation_set_must_be_declared_not_guessed():
    """An empty activation set is refused: it would classify the whole module weight-invariant."""
    activation_arg_indices = _census_module().activation_arg_indices

    with pytest.raises(SystemExit):
        activation_arg_indices({"0": {"kind": "param"}, "1": {"kind": "buffer"}})
