"""Study fixtures keep their original numerical regions across shape changes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir


def _load(name):
    directory = repo_root() / "examples/gemmini/phase1/studies/kernel_abstractions"
    # The operator recipes are plain executable files, not another package.
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("study_" + name, directory / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(directory))


def _dense():
    commit = (
        '  %Y = merlin_iface.commit %acc {name = "Y", epilogue = ["acc_scale"], '
        'output_dtype = "i8", acc_scale = 0.0078125 : f32} '
        ": (!merlin_iface.acc<i32>) -> tensor<16x16xi8>"
    )
    return parse_interface_mlir(
        """module {
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<32x16xi8>
  %A = merlin_iface.tensor {name = "A", role = "input"} : tensor<16x32xi8>
  %R = merlin_iface.resident_pack %W {layout = "packed_rhs"} : (tensor<32x16xi8>) -> !merlin_iface.resident
  %acc = merlin_iface.matmul %A, %R : (tensor<16x32xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
COMMIT_LINE
  merlin_iface.evict %R : (!merlin_iface.resident) -> ()
}""".replace("COMMIT_LINE", commit)
    )


def test_dense_tails_keep_selected_scale_and_no_duplicate_output_declaration():
    recipe = _load("inputs")
    original = _dense()
    changed = recipe.resize_dense(original, m=17, k=19, ns=[31])
    text = emit_interface_mlir(changed)
    again = parse_interface_mlir(text)
    contract = recipe.source_contract(again)
    assert contract["products"][0]["shape_mkn"] == [17, 19, 31]
    assert contract["tensors"]["Y"] == {"shape": [17, 31], "dtype": "i8", "role": "output"}
    assert contract["numerics_and_fusion"][0]["attributes"]["acc_scale"] == 1 / 128
    assert 'merlin_iface.tensor {name = "Y"' not in text
    assert original["tensors"]["A"]["shape"] == [16, 32]


def test_attention_resize_keeps_shared_rhs_and_all_outputs():
    recipe = _load("inputs")
    cb = _dense()
    cb["tensors"]["W"]["shape"] = [16, 64]
    cb["tensors"]["A"]["shape"] = [16, 16]
    cb["tensors"]["A1"] = {"shape": [16, 16], "dtype": "i8", "role": "input"}
    cb["commands"][3:3] = [
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A1", "rhs": "R", "dst": "acc1"}},
        {
            "opcode": "COMMIT",
            "operands": {"src": "acc1", "dst": "Y1"},
            "attributes": {"epilogue": ["acc_scale"], "acc_scale": 1 / 128, "output_dtype": "i8"},
        },
    ]
    resized = recipe.resize_heads(cb, query=17, context=19, pv=True)
    contract = recipe.source_contract(parse_interface_mlir(emit_interface_mlir(resized)))
    assert len(contract["products"]) == 2
    assert {tuple(p["shape_mkn"]) for p in contract["products"]} == {(17, 19, 64)}
    assert {p["rhs"] for p in contract["products"]} == {"R"}
    assert contract["tensors"]["Y1"]["shape"] == [17, 64]


def test_component_head_tails_have_exact_two_product_output_geometry():
    recipe = _load("inputs")
    program = {
        "schema": "merlin.component_program.v2",
        "logical_io": "merlin.logical_io.v1",
        "inputs": [
            {"name": name, "role": "input", "dtype": "i8", "shape": shape}
            for name, shape in [("KT0", [64, 163]), ("V0", [163, 64]), ("Q0", [50, 64]), ("P0", [50, 163])]
        ],
        "nodes": [
            {"name": "S0", "op": "matmul", "inputs": ["Q0", "KT0"]},
            {"name": "O0", "op": "matmul", "inputs": ["P0", "V0"]},
        ],
        "outputs": [{"name": "Y0", "value": "S0"}, {"name": "Y1", "value": "O0"}],
    }
    normalized, source = recipe.resize_component(program, query=51, context=165)
    assert [o["shape"] for o in normalized["outputs"]] == [[51, 165], [51, 64]]
    assert "tensor<51x165xi32>" in source and "tensor<51x64xi32>" in source
    assert "softmax" not in source


def test_e0_reopens_unique_pins_and_never_ranks_changed_or_unmeasured_evidence(tmp_path):
    recipe = _load("lift_checks")
    leaf = tmp_path / "receipt.json"
    leaf.write_text("original")
    actual = recipe.pin(leaf)
    row = {
        "id": "attention_pv",
        "reference": "Jack",
        "shape": "bounded fixture",
        "output": "i8",
        "evidence": "missing_matched_physical_pair",
        "outputs_exact": None,
        "receipts": [actual],
    }
    study = tmp_path / "study.json"
    study.write_text(json.dumps({"source_audits": [actual], "rows": [row]}))
    before = recipe.audit_oracle(study)
    assert before["counts"] == {"unique_pins": 1, "matched": 1}
    assert before["rows"][0]["within_8pct"] is None
    leaf.write_text("changed")
    assert recipe.audit_oracle(study)["pins"][0]["status"] == "changed"


def test_expert_target_projection_omits_candidate_scores_and_selected_schedules(tmp_path):
    recipe = _load("inputs")
    study = tmp_path / "study.json"
    study.write_text(
        json.dumps(
            {
                "hardware": {"runtime_config": "selected-runtime"},
                "rows": [
                    {
                        "id": "measured",
                        "reference": "Jack",
                        "family": "matmul",
                        "shape": "M1xN3xK2",
                        "output": "rawi32",
                        "evidence": "retained_FireSim_completed_call",
                        "outputs_exact": True,
                        "ordered_samples": {"ours": [1, 2], "reference": [99, 101]},
                        "schedule": {"winning": "PRIVATE"},
                    },
                    {
                        "id": "pending",
                        "reference": "Jack",
                        "family": "matmul",
                        "shape": "M2xN3xK2",
                        "output": "rawi32",
                        "evidence": "missing_matched_physical_pair",
                        "outputs_exact": None,
                    },
                ],
            }
        )
    )
    result = recipe.expert_targets(study)
    assert result["rows"][0]["ordered_cycles"] == [99, 101]
    assert result["rows"][0]["mean_cycles"] == 100
    assert result["rows"][1]["mean_cycles"] is None
    assert "ours" not in json.dumps(result) and "PRIVATE" not in json.dumps(result)


def _projection_fixture(recipe, tmp_path, *, raw=True):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    source = corpus / recipe.TINY[2] / "input.mlir"
    source.parent.mkdir()
    cb = recipe.resize_dense(_dense(), m=16, k=2048, ns=[2048])
    if raw:
        commit = next(c for c in cb["commands"] if c["opcode"] == "COMMIT")
        commit["attributes"] = {"epilogue": [], "output_dtype": "i32"}
    source.write_text(emit_interface_mlir(cb))
    oracle, review = tmp_path / "oracle.json", tmp_path / "review.json"
    oracle.write_text(json.dumps({"hardware": {}, "rows": []}))
    review.write_text("{}")
    return dict(
        corpus=corpus,
        historical_kernels=corpus,
        reference_manifest=review,
        compiler_root=tmp_path / "unused-compiler",
        oracle_study=oracle,
    )


def test_projection_core_prepares_only_anchor_and_three_single_axis_hidden_cases(tmp_path, monkeypatch):
    from merlin_experiments.corpus import public_references

    recipe = _load("inputs")
    kwargs = _projection_fixture(recipe, tmp_path)
    staged = []
    monkeypatch.setattr(public_references, "stage", lambda review, output: staged.append(review) or {})
    monkeypatch.setattr(recipe, "capacity_boundary", lambda *_: pytest.fail("unrelated capacity sweep"))
    monkeypatch.setattr(recipe, "prepare_reference_review", lambda *_args, **_kw: pytest.fail("bank changed"))
    out = tmp_path / "projection"
    doc = recipe.prepare_inputs(out, profile="projection-core", **kwargs)
    assert doc["counts"] == {"train": 1, "heldout": 3}
    assert staged == [kwargs["reference_manifest"]]
    expected = [(16, 2048, 2048), (32, 2048, 2048), (16, 1024, 2048), (16, 2048, 1024)]
    assert [tuple(row["contract"]["products"][0]["shape_mkn"]) for row in doc["cases"]] == expected
    for row in doc["cases"]:
        cb = parse_interface_mlir(Path(row["source"]["path"]).read_text())
        contract = recipe.source_contract(cb)
        assert all(t["dtype"] == "i8" for t in contract["tensors"].values() if t["role"] != "output")
        assert [t["dtype"] for t in contract["tensors"].values() if t["role"] == "output"] == ["i32"]
        assert contract["numerics_and_fusion"][0]["attributes"] == {"epilogue": [], "output_dtype": "i32"}
    panel = json.loads((out / "small-panel.json").read_text())
    assert panel["inputs_manifest"] == recipe.pin(out / "manifest.json")
    assert panel["selection"] == {
        "training_case_ids": [recipe.TINY[2]],
        "heldout_case_ids": [f"holdout_projection_m{m}k{k}n{n}" for m, k, n in expected[1:]],
    }
    assert not list((out / "training").glob("holdout*"))
    assert json.loads((out / "training/selection.json").read_text())["cases"] == [recipe.TINY[2]]
    assert doc["capacity_boundary"] is None


def test_projection_core_refuses_quantized_anchor_instead_of_relabeling_it(tmp_path):
    recipe = _load("inputs")
    kwargs = _projection_fixture(recipe, tmp_path, raw=False)
    with pytest.raises(ValueError, match="raw-i32 projection anchor"):
        recipe.prepare_inputs(tmp_path / "projection", profile="projection-core", **kwargs)


def test_projection_core_reuses_reviewed_bank_and_rejects_additional_containers(tmp_path):
    recipe = _load("inputs")
    with pytest.raises(ValueError, match="complete pinned reference review"):
        recipe.prepare_inputs(
            tmp_path / "out",
            profile="projection-core",
            raw_containers=["new.zip"],
            corpus=tmp_path,
            historical_kernels=tmp_path,
            reference_manifest=tmp_path,
            compiler_root=tmp_path,
            oracle_study=tmp_path,
        )


def _conv_fixture(recipe, tmp_path):
    corpus = tmp_path / "corpus"
    source = corpus / recipe.CONVS[1] / "input.mlir"
    source.parent.mkdir(parents=True)
    convolution = (
        '  %Y = merlin_iface.conv2d %IFM, %R {name = "Y", kernel = [3, 3, 128, 128], stride = [1, 1], '
        'padding = [0, 0, 0, 0], dilation = [1, 1], layout = "nhwc", output_dtype = "i8", '
        'epilogue = ["bias_add"], bias = "B"} '
        ': (tensor<4x30x30x128xi8>, !merlin_iface.resident) -> tensor<3136x128xi8>'
    )
    source.write_text('''module {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<4x30x30x128xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<1152x128xi8>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<128xi32>
  %R = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<1152x128xi8>) -> !merlin_iface.resident
CONVOLUTION
  merlin_iface.evict %R : (!merlin_iface.resident) -> ()
}'''.replace("CONVOLUTION", convolution))
    oracle, review = tmp_path / "oracle.json", tmp_path / "review.json"
    oracle.write_text(json.dumps({"hardware": {}, "rows": []}))
    review.write_text("{}")
    return dict(corpus=corpus, historical_kernels=corpus, reference_manifest=review,
        compiler_root=tmp_path / "unused-compiler", oracle_study=oracle)


def test_conv_composition_keeps_contract_and_single_axis_hidden_changes(tmp_path, monkeypatch):
    import yaml
    from merlin_experiments.corpus import public_references

    recipe = _load("inputs")
    kwargs = _conv_fixture(recipe, tmp_path)
    monkeypatch.setattr(public_references, "stage", lambda *_args: {})
    monkeypatch.setattr(recipe, "capacity_boundary", lambda *_: pytest.fail("unrelated capacity sweep"))
    monkeypatch.setattr(recipe, "prepare_reference_review", lambda *_args, **_kw: pytest.fail("bank changed"))
    out = tmp_path / "conv"
    doc = recipe.prepare_inputs(out, profile="conv-composition", **kwargs)
    assert doc["counts"] == {"train": 1, "heldout": 3}
    expected = [(4, 28, 128), (8, 28, 128), (4, 32, 128), (4, 28, 64)]
    for row, (batch, output, channels) in zip(doc["cases"], expected, strict=True):
        cb = parse_interface_mlir(Path(row["source"]["path"]).read_text())
        conv = next(c for c in cb["commands"] if c["opcode"] == "CONV2D")
        attrs = conv["attributes"]
        assert cb["tensors"]["IFM"]["shape"] == [batch, output + 2, output + 2, 128]
        assert cb["tensors"]["W"]["shape"] == [1152, channels]
        assert cb["tensors"]["B"] == {"shape": [channels], "dtype": "i32", "role": "bias"}
        assert cb["tensors"]["Y"]["shape"] == [batch * output * output, channels]
        assert attrs["kernel"] == [3, 3, 128, channels]
        assert attrs["epilogue"] == ["bias_add"] and attrs["output_dtype"] == "i8"
        cap = yaml.safe_load(Path(row["capsule"]["path"]).read_text())
        assert cap["operation"]["attributes"]["kernel"] == attrs["kernel"]
        assert cap["operation"]["attributes"]["weight"] == "W"
        assert cap["numeric_policy"]["dtype"] == "i8"
    assert doc["early_slot"]["case"] == recipe.CONVS[1]
    assert doc["capacity_boundary"] is None
    panel = json.loads((out / "small-panel.json").read_text())
    assert panel["inputs_manifest"] == recipe.pin(out / "manifest.json")
    assert panel["selection"]["training_case_ids"] == [recipe.CONVS[1]]
    assert len(panel["selection"]["heldout_case_ids"]) == 3
    assert not list((out / "training").glob("holdout*"))


def test_conv_resize_preserves_scale_subset_and_rejects_invented_coefficients(tmp_path):
    recipe = _load("inputs")
    kwargs = _conv_fixture(recipe, tmp_path)
    cb = parse_interface_mlir((kwargs["corpus"] / recipe.CONVS[1] / "input.mlir").read_text())
    conv = next(c for c in cb["commands"] if c["opcode"] == "CONV2D")
    conv["attributes"]["acc_scale_by_column"] = [1 / (i + 1) for i in range(128)]
    changed = recipe.resize_conv(cb, batch=4, output=28, output_channels=64)
    after = next(c for c in changed["commands"] if c["opcode"] == "CONV2D")
    assert after["attributes"]["acc_scale_by_column"] == conv["attributes"]["acc_scale_by_column"][:64]
    assert len(conv["attributes"]["acc_scale_by_column"]) == 128
    with pytest.raises(ValueError, match="preserve original readout coefficients"):
        recipe.resize_conv(cb, batch=4, output=28, output_channels=256)


def test_conv_composition_refuses_changed_numerical_contract(tmp_path):
    recipe = _load("inputs")
    kwargs = _conv_fixture(recipe, tmp_path)
    source = kwargs["corpus"] / recipe.CONVS[1] / "input.mlir"
    source.write_text(source.read_text().replace('epilogue = ["bias_add"]', 'epilogue = ["bias_add", "relu"]'))
    with pytest.raises(ValueError, match="original valid K3 NHWC bias INT8 anchor"):
        recipe.prepare_inputs(tmp_path / "conv", profile="conv-composition", **kwargs)


def test_same_conv_family_uses_source_padding_and_one_axis_frozen_sweeps(tmp_path, monkeypatch):
    import yaml
    from merlin_experiments.corpus import public_references

    recipe = _load("inputs")
    kwargs = _conv_fixture(recipe, tmp_path)
    monkeypatch.setattr(public_references, "stage", lambda *_args: {})
    monkeypatch.setattr(recipe, "capacity_boundary", lambda *_: pytest.fail("unrelated capacity sweep"))
    out = tmp_path / "same-family"
    doc = recipe.prepare_inputs(out, profile="conv-family-v2", **kwargs)
    assert doc["counts"] == {"train": 3, "heldout": 15}
    expected_train = {(4, 28, 128, 128), (2, 28, 128, 128), (4, 21, 128, 128)}
    expected_hidden = (
        {(b, 28, 128, 128) for b in (1, 6, 8)}
        | {(4, h, 128, 128) for h in (14, 32, 35, 42)}
        | {(4, 28, ci, 128) for ci in (64, 96, 192, 256)}
        | {(4, 28, 128, co) for co in (64, 96, 192, 256)}
    )
    shapes = {"train": set(), "heldout": set()}
    for row in doc["cases"]:
        cb = parse_interface_mlir(Path(row["source"]["path"]).read_text())
        conv = next(c for c in cb["commands"] if c["opcode"] == "CONV2D")
        attrs = conv["attributes"]
        batch, height, width, cin = cb["tensors"]["IFM"]["shape"]
        cout = attrs["kernel"][-1]
        shape = (batch, height, cin, cout)
        shapes[row["split"]].add(shape)
        assert width == height and attrs["padding"] == [1, 1, 1, 1]
        assert attrs["stride"] == attrs["dilation"] == [1, 1]
        assert cb["tensors"]["W"]["shape"] == [9 * cin, cout]
        assert cb["tensors"]["B"]["shape"] == [cout]
        assert cb["tensors"]["Y"]["shape"] == [batch * height * height, cout]
        assert attrs["epilogue"] == ["bias_add"] and attrs["output_dtype"] == "i8"
        assert "acc_scale" not in attrs and "acc_scale_by_column" not in attrs
        cap = yaml.safe_load(Path(row["capsule"]["path"]).read_text())
        assert cap["operation"]["attributes"]["padding"] == [1, 1, 1, 1]
        assert cap["operation"]["attributes"]["ci"] == cin
        assert not row["original_source_bytes"]
        if row["split"] == "heldout":
            assert sum(a != b for a, b in zip(shape, (4, 28, 128, 128), strict=True)) == 1
    assert shapes == {"train": expected_train, "heldout": expected_hidden}
    assert shapes["train"].isdisjoint(shapes["heldout"])
    panel = json.loads((out / "small-panel.json").read_text())
    assert len(panel["selection"]["training_case_ids"]) == 3
    assert len(panel["selection"]["heldout_case_ids"]) == 15
    assert all(name.startswith("CV2_") for names in panel["selection"].values() for name in names)
    original = parse_interface_mlir((kwargs["corpus"] / recipe.CONVS[1] / "input.mlir").read_text())
    assert next(c for c in original["commands"] if c["opcode"] == "CONV2D")["attributes"]["padding"] == [0] * 4


def test_same_conv_family_refuses_inventing_varying_readout_coefficients(tmp_path):
    recipe = _load("inputs")
    kwargs = _conv_fixture(recipe, tmp_path)
    cb = parse_interface_mlir((kwargs["corpus"] / recipe.CONVS[1] / "input.mlir").read_text())
    next(c for c in cb["commands"] if c["opcode"] == "CONV2D")["attributes"]["acc_scale_by_column"] = [0.5] * 128
    with pytest.raises(ValueError, match="original uniform bias/i8 identity readout"):
        recipe.same_conv_case(cb, batch=4, spatial=28, input_channels=256, output_channels=256)


def test_projection_family_has_public_variety_and_disjoint_single_axis_sweeps(tmp_path, monkeypatch):
    from merlin_experiments.corpus import public_references

    recipe = _load("inputs")
    kwargs = _projection_fixture(recipe, tmp_path)
    monkeypatch.setattr(public_references, "stage", lambda *_args: {})
    monkeypatch.setattr(recipe, "capacity_boundary", lambda *_: pytest.fail("unrelated capacity sweep"))
    doc = recipe.prepare_inputs(tmp_path / "projection-family", profile="projection-family-v2", **kwargs)
    assert doc["counts"] == {"train": 3, "heldout": 14}
    shapes = {split: {tuple(row["contract"]["products"][0]["shape_mkn"])
        for row in doc["cases"] if row["split"] == split} for split in ("train", "heldout")}
    assert shapes["train"] == {(16, 2048, 2048), (8, 2048, 2048), (16, 1024, 2048)}
    assert shapes["train"].isdisjoint(shapes["heldout"])
    assert len(shapes["heldout"]) == 14
    assert all(sum(a != b for a, b in zip(shape, (16, 2048, 2048), strict=True)) == 1
        for shape in shapes["heldout"])
    assert {m for m, k, n in shapes["heldout"] if (k, n) == (2048, 2048)} == {4, 24, 32, 48, 64}
    assert {k for m, k, n in shapes["heldout"] if (m, n) == (16, 2048)} == {512, 1536, 3072, 4096}
    assert {n for m, k, n in shapes["heldout"] if (m, k) == (16, 2048)} == {512, 1024, 1536, 3072, 4096}
