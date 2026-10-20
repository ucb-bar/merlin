"""Explicit successor recipes reopen ordinary indexed inputs; they grant no admission."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_model_inputs as I
from merlin_experiments.phase1.feedback import private_numerical_recipe as N

from merlin.common.paths import repo_root
from merlin.targetgen import quant_methodology as M
from merlin.targetgen import quant_recipe as Q
from merlin.targetgen import quant_recipe_policy as P


def _raw(body):
    return (json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fixture(model="tiny_llama", *, prepare=True):
    root = repo_root() / "examples/gemmini"
    original = json.loads((root / f"protocol/recipes/{model}.json").read_bytes())
    native = json.loads((root / f"protocol/recipes/{model}-native-speed.json").read_bytes())
    method = yaml.safe_load((root / "target/software-spec.yaml").read_bytes())["quantization"]["methodology"]
    epilogue = original["epilogue"]
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
    # The synthetic unit fixture explicitly rebinds its own path; real metadata
    # reproduction is retained separately and uses the original exported contract.
    original["methodology"]["output_path_sha256"] = M.digest(path)
    original["recipe_sha256"] = Q.digest(original)
    policy = {
        "schema": P.SCHEMA,
        "weight_granularity": "tensor",
        "float_quality": "report_only",
        "native_outputs": {k: v for k, v in native["epilogue"]["output_quantization"].items() if k != "schema"},
        "dynamic_activation_planes": 1 if "dynamic" in original else None,
        "integer_extensions": {
            k: native[k] for k in ("accumulator_residual", "integer_mean", "projection_input_reuse") if k in native
        },
    }
    seed = original
    if prepare:
        seed = Q.with_native_output_quantization(seed, **policy["native_outputs"])
        for key, value in policy["integer_extensions"].items():
            seed = Q.with_integer_extension(seed, key=key, policy=value)
    selected = P.select(seed, method, path, policy=policy, model=model)
    raw = _raw(original)
    member = f"software/quantization-recipes/{_sha(raw)}.json"
    return _raw(selected), {
        "model": model,
        "target": original["target"],
        "methodology": method,
        "output_path": path,
        "index": {
            "recipes": [],
            "model_variants": [
                {"model": model, "path": member, "sha256": _sha(raw), "recipe_sha256": original["recipe_sha256"]}
            ],
        },
        "artifacts": {member: raw},
    }


@pytest.mark.parametrize("model", ["resnet50", "tiny_llama", "smolvla"])
@pytest.mark.parametrize("prepare", [False, True])
def test_exact_recomputation_from_original_index_without_modifying_common_recipes(model, prepare):
    selected, inputs = _fixture(model, prepare=prepare)
    before = copy.deepcopy(inputs)
    proof = N.recompute(selected, **inputs)
    assert inputs == before and inputs["index"]["recipes"] == []
    assert proof["selected_file_sha256"] == _sha(selected)
    assert proof["selected_recipe_sha256"] == json.loads(selected)["recipe_sha256"]
    assert proof["preparation"] == ("native_output_extensions_v1" if prepare else "indexed_recipe")
    assert "no admission" in proof["scope"]


@pytest.mark.parametrize("field", ["source_recipe_sha256", "source_methodology_sha256", "source_output_path_sha256"])
def test_each_preselected_source_digest_is_required(field):
    selected, inputs = _fixture()
    body = json.loads(selected)
    body["numerical_selection"][field] = "0" * 64
    body["recipe_sha256"] = Q.digest(body)
    with pytest.raises(ValueError):
        N.recompute(_raw(body), **inputs)


@pytest.mark.parametrize("mutation", ["extra", "policy", "body", "target", "format"])
def test_closed_policy_entire_body_and_original_serialization_are_bound(mutation):
    selected, inputs = _fixture()
    body = json.loads(selected)
    if mutation == "extra":
        body["numerical_selection"]["unreviewed"] = True
    elif mutation == "policy":
        body["numerical_selection"]["policy"]["weight_granularity"] = "channel"
    elif mutation == "body":
        body["activation"]["symmetric"] = False
    elif mutation == "target":
        body["target"] = "another_target"
    body["recipe_sha256"] = Q.digest(body)
    raw = json.dumps(body).encode() if mutation == "format" else _raw(body)
    with pytest.raises(ValueError):
        N.recompute(raw, **inputs)


@pytest.mark.parametrize("mutation", ["bytes", "semantic", "path", "missing", "duplicate", "other_model"])
def test_original_indexed_variant_is_reopened_and_unique(mutation):
    selected, inputs = _fixture()
    row = inputs["index"]["model_variants"][0]
    if mutation == "bytes":
        inputs["artifacts"][row["path"]] += b" "
    elif mutation == "semantic":
        row["recipe_sha256"] = "0" * 64
    elif mutation == "path":
        row["path"] = "software/quantization-recipes/../foreign.json"
    elif mutation == "missing":
        inputs["index"]["model_variants"] = []
    elif mutation == "duplicate":
        inputs["index"]["model_variants"].append(copy.deepcopy(row))
    else:
        row["model"] = "another_model"
    with pytest.raises(ValueError):
        N.recompute(selected, **inputs)


def _intake(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import evidence

    selected, inputs = _fixture()
    root = tmp_path / "export"
    root.mkdir()
    manifest = root / "evidence-manifest.json"
    manifest.write_bytes(b"verified-export-fixture")
    recipe = tmp_path / "selected.json"
    recipe.write_bytes(selected)
    roles = {"software-spec": "1" * 64, "target-contract": "2" * 64, "rtl-facts": "3" * 64}
    host_sha = "4" * 64
    software = {"quantization": {"methodology": inputs["methodology"]}}
    index = {
        "schema": "merlin.phase0.capture_recipes.v1",
        "target": inputs["target"],
        "software_review": "reviewed",
        **inputs["index"],
    }
    quantization = {"methodology": {"declaration": inputs["methodology"], "output_path": inputs["output_path"]}}
    exported = SimpleNamespace(
        target=inputs["target"],
        source_snapshots=[
            *(SimpleNamespace(role=k, sha256=v) for k, v in roles.items()),
            SimpleNamespace(role="host-capability-spec:selected", sha256=host_sha),
        ],
        software_spec=software,
        contract={"output_epilogue": {"output_path": inputs["output_path"]}},
        host_capabilities={"selected": {"status": "reviewed"}},
        archived_artifacts=[
            *inputs["artifacts"].items(),
            ("software/quantization-recipes.json", _raw(index)),
            ("software/quantization-contract.json", _raw(quantization)),
        ],
    )
    monkeypatch.setattr(evidence, "load_exported_evidence", lambda _root: exported)

    def pin(_spec, path, sha):
        result = Path(path)
        if _sha(result.read_bytes()) != sha:
            raise ValueError("pinned file changed")
        return result

    kwargs = dict(
        target=inputs["target"],
        software_sha256=roles["software-spec"],
        capability_sha256=roles["target-contract"],
        facts_sha256=roles["rtl-facts"],
        host_capabilities_sha256=host_sha,
        pinned_file=pin,
        file_digest=lambda path: _sha(path.read_bytes()),
    )
    row = {
        "id": inputs["model"],
        "recipe_derivation_root": str(root),
        "recipe_derivation_manifest_sha256": _sha(manifest.read_bytes()),
    }
    plan = {
        "recipe": {
            "path": str(recipe),
            "sha256": _sha(selected),
            "recipe_sha256": json.loads(selected)["recipe_sha256"],
        }
    }
    return lambda: I._recipe_derivation(tmp_path / "spec.yaml", row, plan, **kwargs), recipe, exported, kwargs, plan


def test_private_intake_uses_current_export_and_exact_capture_preselection(tmp_path, monkeypatch):
    call, recipe, exported, _, _ = _intake(tmp_path, monkeypatch)
    proof, software, hosts = call()
    assert proof["recipe_sha256"] == _sha(recipe.read_bytes())
    assert proof["numerical_selection"]["schema"] == "merlin.phase1.numerical_recipe_derivation.v1"
    assert software is exported.software_spec and hosts == exported.host_capabilities


@pytest.mark.parametrize("mutation", ["facts", "method_join", "path_join", "capture_semantic", "selected_bytes"])
def test_private_intake_preserves_original_outer_joins(tmp_path, monkeypatch, mutation):
    call, recipe, exported, kwargs, plan = _intake(tmp_path, monkeypatch)
    if mutation == "facts":
        kwargs["facts_sha256"] = "0" * 64
    elif mutation in ("method_join", "path_join"):
        artifacts = dict(exported.archived_artifacts)
        contract = json.loads(artifacts["software/quantization-contract.json"])
        contract["methodology"]["declaration" if mutation == "method_join" else "output_path"] = {}
        artifacts["software/quantization-contract.json"] = _raw(contract)
        exported.archived_artifacts = list(artifacts.items())
    elif mutation == "capture_semantic":
        plan["recipe"]["recipe_sha256"] = "0" * 64
    else:
        recipe.write_bytes(recipe.read_bytes() + b" ")
    with pytest.raises(ValueError):
        call()


def test_private_intake_rejects_mutation_during_reconstruction(tmp_path, monkeypatch):
    call, recipe, _, _, _ = _intake(tmp_path, monkeypatch)
    actual = N.recompute

    def mutate(*args, **kwargs):
        proof = actual(*args, **kwargs)
        recipe.write_bytes(recipe.read_bytes() + b" ")
        return proof

    monkeypatch.setattr(N, "recompute", mutate)
    with pytest.raises(ValueError, match="changed during"):
        call()


def test_new_reconstruction_and_core_policy_are_bound_to_current_source_epoch(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import source_inputs
    from merlin_experiments.spec import SpecError

    selected = tmp_path / "policy.py"
    selected.write_bytes(Path(P.__file__).read_bytes())
    original = source_inputs._source
    monkeypatch.setattr(source_inputs, "_source", lambda name: selected if name == P.__name__ else original(name))
    arguments = {"repo": tmp_path, "entrypoint": tmp_path / "external.py"}
    record = source_inputs.record(**arguments)
    assert record["inputs"]["phase1:source:feedback/private_numerical_recipe.py"]["path"] == N.__file__
    assert record["inputs"]["phase1:startup:numerical_recipe_policy"]["path"] == str(selected)
    selected.write_bytes(selected.read_bytes() + b"\n# changed policy source\n")
    with pytest.raises(SpecError, match="source identity changed"):
        source_inputs.verify(record, **arguments)
