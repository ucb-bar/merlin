"""A derived capsule must ASSERT the modes its epilogue implies, not an empty dict.

``corpus_spec`` treats the PRESENCE of a ``modes`` key as "the author declared these verbatim" and
skips its own derivation. The synthesizer emitted ``modes: {}`` on every entry -- which reads like "no
modes yet" and means "assert nothing" -- so a derived capsule declaring ``epilogue: [relu]`` shipped no
relu mode. Mode coverage over the derived corpus therefore measured nothing, and every mode-asserting
hand-authored capsule was structurally unmatchable by anything the synthesizer could produce, which is
what blocked retiring the hand corpus.
"""
from __future__ import annotations

import inspect

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import corpus_synth as CSY
from merlin.targetgen.target_experiment import load_target_experiment


def _binding():
    desc = merlin_dir() / "experiments" / "capsule_bench" / "targets" / "gemmini" / "target_experiment.yaml"
    prof = yaml.safe_load((merlin_dir() / "contract" / "capsules" / "profiles" / "gemmini.yaml").read_text())
    if not desc.is_file():
        pytest.skip("this target's descriptor is not present in this checkout")
    return CS.derive_binding(load_target_experiment(desc), prof["datapath"])


def _modes(entry):
    built = CS.build(entry, _binding())
    cap = built[0] if isinstance(built, tuple) else built
    return (cap.get("expected") or {}).get("modes")


_BASE = {"cat": "isa", "kind": "isa", "op": "matmul", "M": 16, "K": 16, "N": 16,
         "lhs": "A", "weight": "W", "out": "Y", "label": "public",
         "source_role": "derived_sweep", "source_reference": "probe"}


def test_the_synthesizer_emits_no_empty_modes_override():
    """THE REGRESSION. An empty dict is not 'no modes' -- it suppresses the builder's derivation."""
    src = inspect.getsource(CSY)
    assert '"modes": {}' not in src, (
        "an empty modes override suppresses corpus_spec's own derivation, so every derived capsule "
        "asserts no modes however its epilogue is declared"
    )


def test_an_epilogue_implies_its_mode_when_the_entry_declares_none():
    m = _modes({**_BASE, "name": "P1", "epilogue": ["relu"]})
    assert m.get("relu") is True, m


def test_a_different_epilogue_derives_a_different_mode():
    """MUTATION CONTROL: if modes were not derived from the epilogue these two would be identical."""
    relu = _modes({**_BASE, "name": "P1", "epilogue": ["relu"]})
    scale = _modes({**_BASE, "name": "P2", "epilogue": ["acc_scale"]})
    assert relu != scale
    assert relu.get("relu") is True and scale.get("acc_scale") is True
    assert scale.get("relu") is False and relu.get("acc_scale") is False


def test_an_explicit_modes_declaration_is_still_honoured_verbatim():
    """An author who DOES declare modes keeps them -- the fix removes the empty override, not the hook."""
    m = _modes({**_BASE, "name": "P3", "epilogue": ["relu"], "modes": {"k_accumulate": True}})
    assert m == {"k_accumulate": True}
