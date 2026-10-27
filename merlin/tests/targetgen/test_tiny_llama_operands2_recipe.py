"""The reviewed score-operand variant keeps the refusal and its original bars."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import quant_methodology as M
from merlin.targetgen import quant_recipe as Q
from merlin.targetgen import quant_recipe_policy as P

_ROOT = repo_root() / "examples/gemmini"
_RECIPES = _ROOT / "protocol/recipes"
_OLD = "static-heads-softmax2-bounded-partner2"
_NEW = _OLD + "-operands2"
_BAR = {"schema": "dynamic_site_agreement_v1", "cosine_min": 0.947588, "rel_rmse_max": 0.457351}
_DIGESTS = {
    "tiny_llama.json": "4f14c71f4195ee4481de9814fee6ed578953a14d21adc0aab20e7b4f261b886d",
    "tiny_llama-native-speed.json": "0f20ddfe1939ea961d55b5fe825d0f73dd4d130b201e3b07350dd439d4476566",
    "tiny_llama-native-scalar-speed.json": "1ddf5660d25d2cf6504ca31cb009bee864583c21c2d49bc3e97e7fc98dfa3faa",
    "tiny_llama-speed-first.json": "32da5632e6d12ff2b0582fe266b4f56a7b9fdb8a49495c8c07a186356af53b28",
}


def _recipe(filename="tiny_llama.json"):
    return json.loads((_RECIPES / filename).read_bytes())


def _method():
    return yaml.safe_load((_ROOT / "target/software-spec.yaml").read_bytes())["quantization"]["methodology"]


def test_review_records_the_evidence_without_relabeling_the_refusal():
    method = M.validate(_method())
    rows = M.variants_for(method, "tiny_llama")
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == _NEW and row["status"] == "reviewed"
    assert row["changes"]["dynamic"] == _recipe()["dynamic"]
    basis = row["measured_accuracy"]["basis"]
    for evidence in ("0.93311", "0.94296", "0.94711", "0.947588", "0.457351", "/tmp/claude-2629/tl_admit.py"):
        assert evidence in basis
    assert f"The prior variant {_OLD} and its decode refusal remain in the history" in basis
    assert "the prior refusal is not relabeled a pass" in basis
    assert "unmeasured" in row["measured_accuracy"]["new_variant"]
    history = _ROOT / "phase0/history/software-spec-seals1-3.yaml"
    assert hashlib.sha256(history.read_bytes()).hexdigest() == "1722caeff44673a0803d5e671b22b4311743b68fda9ffb93584c8eb2c8f4a281"
    old_rows = yaml.safe_load(history.read_bytes())["quantization"]["methodology"]["model_variants"]
    old = next(r for r in old_rows if r["model"] == "tiny_llama")
    assert old["name"] == _OLD
    assert old["changes"]["dynamic"]["site_agreement"] == _BAR
    assert Q.RESIDUAL_OPERAND not in old["changes"]["dynamic"]["planes"]
    assert "preserve any refusal without relabeling it a pass" in (_ROOT / "protocol/native-speed.yaml").read_text()


@pytest.mark.parametrize("filename", list(_DIGESTS))
def test_recipe_schema_identity_and_explicit_plane_policy(filename):
    recipe = _recipe(filename)
    assert recipe["schema"] == Q.SCHEMA and recipe["status"] == Q.DERIVED
    assert recipe["recipe_sha256"] == Q.digest(recipe) == _DIGESTS[filename]
    assert recipe["methodology"]["variants"][0]["name"] == _NEW
    block = recipe["dynamic"]
    assert block["schema"] == Q.DYNAMIC_SCHEMA
    assert block["scales"] == Q.STATIC_HEADS and block["smooth"] is None
    assert block["families"] == [Q.ACTIVATION_CONTRACTION]
    # The speed-first profile is an existing, separate one-plane selection. Its
    # provenance must follow the new source without claiming operands2 admission.
    speed_first = filename == "tiny_llama-speed-first.json"
    planes = 1 if speed_first else 2
    assert block["planes"] == {Q.SOFTMAX_OUTPUT: planes, Q.SOFTMAX_PARTNER: planes, Q.RESIDUAL_OPERAND: planes}
    rebuilt = Q.with_dynamic_contractions(
        recipe, families=block["families"], scales=block["scales"], planes=block["planes"]
    )
    assert rebuilt["dynamic"] == {k: v for k, v in block.items() if k != Q.SITE_AGREEMENT}
    assert Q.site_agreement_of(recipe) == (None if speed_first else _BAR)
    if not speed_first:
        assert recipe["why"]["dynamic.site_agreement"] == M.variants_for(_method(), "tiny_llama")[0]["measured_accuracy"]["basis"]


def test_every_protocol_pin_and_source_lineage_uses_the_new_identity():
    recipe_path = _RECIPES / "tiny_llama.json"
    sha = _DIGESTS[recipe_path.name]
    file_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    experiment = yaml.safe_load((_ROOT / "protocol/experiment.yaml").read_bytes())
    selected = experiment["models"]["tiny_llama"]["quantization"]
    assert selected["recipe"] == "recipes/tiny_llama.json"
    assert selected["recipe_sha256"] == sha and selected["file_sha256"] == file_sha
    assert selected["variant"] == _NEW
    assert selected["attention"]["ordinary_qk_planes"] == 2
    manifest = yaml.safe_load((_ROOT / "protocol/run-manifest.template.yaml").read_bytes())
    pinned = manifest["models"]["tiny_llama"]
    assert pinned["recipe_sha256"] == sha and pinned["recipe_file_sha256"] == file_sha
    derivative = _recipe("tiny_llama-speed-first.json")
    assert derivative["numerical_selection"]["source_recipe_sha256"] == _DIGESTS["tiny_llama-native-speed.json"]
    # Sweep descriptors, numerics (including MANIFEST.json), protocol and docs:
    # no active declaration may retain any of the four old recipe identities.
    stale = (
        "4300d9d2f46160cab679672d384bdda664fd310862137f354b36c2845f45ed5f",
        "1259c1a80ad0ea8bdc74e7b31a80477fcc4b84cd7b1bfc8a9b6e8c59e9119dec",
        "5be01d39a4bded36889b74d1d41ca2e3ebffc99d79a409916a59259e93b8dcdf",
        "f447e97ad23bc571eac80f7fdab345f33492e5b48df7b8a1825880d9b16ffb3f",
        "f113dad1ac0f52d0cbff5f9398afae17282c9ab71c3a6107e98bdf8a096af9ad",
    )
    for path in _ROOT.rglob("*"):
        if path.suffix not in (".json", ".yaml", ".md") or "history" in path.parts:
            continue
        text = path.read_text()
        assert not any(old in text for old in stale), path


def test_one_plane_derivative_can_still_be_derived_from_the_reviewed_source():
    source = _recipe("tiny_llama-native-speed.json")
    epilogue = source["epilogue"]
    # Bind an explicit synthetic spelling of the same physical output path,
    # as in the native/scalar recipe tests; this is not hardware qualification.
    path = {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [
            {"dtype": "i32", "kind": "preload", "range": [-2147483648, 2147483647], "rounding": "none"},
            {"dtype": "i32", "kind": "accumulate", "operand_scale": "none"},
        ],
        "readout": [{"kind": k, **epilogue[k]} for k in epilogue["order"]],
        "exits": epilogue["exits"],
    }
    source["methodology"]["output_path_sha256"] = M.digest(path)
    source["recipe_sha256"] = Q.digest(source)
    derivative = _recipe("tiny_llama-speed-first.json")
    actual = P.select(source, _method(), path, policy=derivative["numerical_selection"]["policy"], model="tiny_llama")
    expected = copy.deepcopy(derivative)
    expected["methodology"]["output_path_sha256"] = M.digest(path)
    expected["numerical_selection"]["source_output_path_sha256"] = M.digest(path)
    expected["numerical_selection"]["source_recipe_sha256"] = source["recipe_sha256"]
    expected["recipe_sha256"] = Q.digest(expected)
    assert actual == expected


# File digests from bd41d066e, before the TinyLlama-only review. These guard
# against regenerating unrelated model choices while repinning the new recipe.
_OTHER_RECIPES = {
    "resnet50-native-speed.json": "e9a4a97db87a665a2a2fb23d55e1e05cd37bdb9ef3df8936d71d765c058880f7",
    "resnet50-shared-grid-sums.json": "59d220417544e018901ca39399ba1a3d368df63e9eaffc26c8761b7a5bc1bc29",
    "resnet50-speed-first.json": "7040d3607627e4e22adeb67085db0b66c42a7440740cecf10058d4ab41d923c2",
    "resnet50.json": "b828e83887c53fd440668c664685989949a66edaf78ff79c7fe2c5127032f3f2",
    "smolvla-native-scalar-speed.json": "99773f018d919cb7bf06c221cdb082dc1f737ff80c734ff07902e57d89452841",
    "smolvla-native-speed.json": "8dd1066d24562c2d021b999a57864867962332271a44d90c22e2ad7d5bf7c525",
    "smolvla-speed-first.json": "ac4a4bb75a5164b05f8206740f21700f5aa15d46a6a2332eb00792e9b23900cd",
    "smolvla.json": "ce0027045998257aa6d2bb91afd401ce6be4b8d06c5937bd469bce5ceb48628c",
}


@pytest.mark.parametrize("filename,sha256", _OTHER_RECIPES.items())
def test_other_models_recipe_files_are_byte_identical(filename, sha256):
    assert hashlib.sha256((_RECIPES / filename).read_bytes()).hexdigest() == sha256
