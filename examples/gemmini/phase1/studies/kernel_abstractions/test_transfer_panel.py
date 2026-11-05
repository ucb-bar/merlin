"""Source, bounds and visibility checks; no simulator or author jobs are launched."""

import copy
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml


def module():
    path = Path(__file__).with_name("transfer_panel.py")
    spec = importlib.util.spec_from_file_location("test_transfer_panel_api", path)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    return api


@pytest.fixture
def anchors(tmp_path):
    from merlin.targetgen.component_program import LOGICAL_IO_V1, SCHEMA_V2, render
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir

    sem = dict(model={"engine": "integer_reference"}, operand_dtype="int8", accumulator_dtype="i32",
               readout_dtype="i32", overflow="wrap_internal_mac", internal_arithmetic=dict(
                   signed_operand_bits=8, mac_result_bits=20, mac_result_overflow="wrap_to_20_bits",
                   full_operation_overflow_policy="bounded_exact_requires_each_partial_sum"))
    cb = dict(target="selected", abi_version="0.1", tensors={
        "W": dict(shape=[2048, 2048], dtype="i8", role="weight"),
        "A0": dict(shape=[16, 2048], dtype="i8", role="input")}, commands=[
            dict(opcode="RES_PACK", operands=dict(src="W", dst="W_res"), attributes=dict(layout="packed_rhs")),
            dict(opcode="MATMUL_RESIDENT", operands=dict(lhs="A0", rhs="W_res", dst="acc0")),
            dict(opcode="COMMIT", operands=dict(src="acc0", dst="Y0"), attributes=dict(epilogue=[], output_dtype="i32")),
            dict(opcode="EVICT", operands=dict(handle="W_res"))])
    gemm = dict(name="public_anchor", kind="layer", label="public", source_role="model_derived",
                interface_mlir="capsule.interface.mlir", stimulus_range=[-4, 3],
                inputs=[dict(name=k, **v) for k, v in cb["tensors"].items()],
                operation=dict(op="matmul", attributes=dict(lhs="A0", weight="W", out="Y0", epilogue=[], output_dtype="i32")),
                numeric_policy=dict(compare="exact_int", dtype="i32"), software_screen={"status": "stale"},
                integer_partial_sum_bound={"status": "stale"})
    program = dict(schema=SCHEMA_V2, logical_io=LOGICAL_IO_V1,
        inputs=[dict(name=f"KT{g}", role="input", shape=[64, 16], dtype="operand") for g in range(4)] +
               [dict(name=f"Q{h}", role="input", shape=[16, 64], dtype="operand") for h in range(32)],
        nodes=[dict(name=f"s{h}", op="matmul", inputs=[f"Q{h}", f"KT{h // 8}"]) for h in range(32)],
        outputs=[dict(name=f"S{h}", value=f"s{h}") for h in range(32)])
    typed, source = render(program, operand_dtype="i8", accumulator_dtype="i32")
    qk = dict(name="public_qk", kind="model_slice", label="public", interface_mlir="capsule.interface.mlir",
        linalg_mlir="capsule.interface.mlir", stimulus_range=[-4, 3], inputs=typed["inputs"],
        operation=dict(op="component_program", attributes=dict(program=program, out="S0", outs=[r["name"] for r in typed["outputs"]],
            arg_order=[r["name"] for r in typed["inputs"] + typed["outputs"]])),
        component_program=typed, numerical_semantics=sem, numeric_policy=dict(compare="exact_int", dtype="i32"),
        software_screen={"status": "stale"}, integer_partial_sum_bound={"status": "stale"})
    dirs = []
    for name, cap, text in (("gemm", gemm, emit_interface_mlir(cb)), ("qk", qk, source)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
        (directory / "capsule.interface.mlir").write_text(text)
        dirs.append(directory)
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(dict(facts=dict(compute_datapaths=[dict(name="input", elem_bits=8),
                                                                dict(name="accumulator", elem_bits=20)]))))
    return dict(gemm=gemm, cb=cb, qk=qk, program=program, sem=sem, gemm_dir=dirs[0], qk_dir=dirs[1], facts=facts)


def test_roster_has_34_unique_ofat_cells_and_explicit_exposure():
    api = module()
    rows = api.specifications()
    assert len(rows) == len({row["case"] for row in rows}) == 34
    assert Counter(row["family"] for row in rows) == {"gemm": 21, "qk": 13}
    assert Counter(row["stratum"] for row in rows) == {"anchor": 2, "scaling": 22, "stress": 10}
    for row in rows:
        anchor = api.GEMM_ANCHOR if row["family"] == "gemm" else api.QK_ANCHOR
        differences = [key for key in anchor if row["geometry"][key] != anchor[key]]
        assert differences == ([] if row["stratum"] == "anchor" else [row["axis"]])
        assert row["split"] == "heldout"
    g = [row for row in rows if row["family"] == "gemm"]
    assert {r["geometry"]["M"] for r in g if r["axis"] == "M" and r["stratum"] == "scaling"} == {8, 32, 48, 64}
    assert {r["geometry"]["K"] for r in g if r["axis"] == "K" and r["stratum"] == "scaling"} == {1024, 1536, 2560, 3072, 4096}
    assert {r["geometry"]["N"] for r in g if r["axis"] == "N" and r["stratum"] == "scaling"} == {1024, 1536, 2560, 3072, 4096}
    assert sum("fresh6_public_exact_contract" in r["exposure"] for r in rows) == 1
    assert sum("public_xs_geometry_different_i8_readout" in r["exposure"] for r in rows) == 1
    assert sum("public_xs_exact_contract" in r["exposure"] for r in rows) == 2


def test_all_sources_preserve_bounds_and_gqa_sharing(anchors):
    from merlin_experiments.phase0.component_integer_bounds import preflight_capsule

    from merlin.targetgen.component_program import render
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir

    api, a = module(), anchors
    original = copy.deepcopy(a)
    for row in api.specifications():
        cap, text, contract = api.make_case(row, a["gemm"], a["cb"], a["qk"], a["program"], a["sem"])
        assert "software_screen" not in cap
        assert cap["integer_partial_sum_bound"]["status"] == "proven_safe"
        assert cap["stimulus_range"] == [-4, 3]
        assert contract["epilogue"] == [] and contract["output_dtype"] == "i32"
        g = row["geometry"]
        if row["family"] == "gemm":
            cb = parse_interface_mlir(text)
            assert cb["tensors"]["A0"]["shape"] == [g["M"], g["K"]]
            assert cb["tensors"]["W"]["shape"] == [g["K"], g["N"]]
            assert cap["integer_partial_sum_bound"]["bound"] == 16 * g["K"]
            assert cap["integer_partial_sum_bound"]["signed_positive_limit"] == 524287
            assert contract["outputs"] == [dict(name="Y0", shape=[g["M"], g["N"]], dtype="i32")]
        else:
            typed, emitted = render(cap["operation"]["attributes"]["program"], operand_dtype="i8", accumulator_dtype="i32")
            assert emitted == text and typed == cap["component_program"]
            assert len(typed["inputs"]) == 36 and len(typed["nodes"]) == len(typed["outputs"]) == 32
            assert all(r["role"] == "input" for r in typed["inputs"])
            assert Counter(r["actual_inputs"][1] for r in typed["nodes"]) == {f"KT{i}": 8 for i in range(4)}
            assert all(r["shape"] == [g["query"], g["key"]] and r["dtype"] == "i32" for r in typed["outputs"])
            assert preflight_capsule(cap)["status"] == "proven_safe"
    assert a == original


def test_anchor_roundtrips_and_bad_source_or_widths_refuse(anchors):
    api, a = module(), anchors
    cap, text, _ = api._template(a["gemm_dir"])
    assert api._gemm_template(cap, text) == a["cb"]
    cap, text, _ = api._template(a["qk_dir"])
    assert api._qk_template(cap, text) == a["program"]
    bad = copy.deepcopy(cap)
    bad["operation"]["attributes"]["program"]["nodes"][8]["inputs"][1] = "KT0"
    with pytest.raises(ValueError, match="sharing"):
        api._qk_template(bad, text)
    bad = copy.deepcopy(cap)
    bad["operation"]["attributes"]["program"]["outputs"].pop()
    with pytest.raises(ValueError):
        api._qk_template(bad, text)
    facts = json.loads(a["facts"].read_text())
    assert api._semantics(a["qk"], facts) == a["sem"]
    facts["facts"]["compute_datapaths"][1]["elem_bits"] = 19
    with pytest.raises(ValueError, match="widths"):
        api._semantics(a["qk"], facts)
    with pytest.raises(ValueError, match="round-trip"):
        api._gemm_template(a["gemm"], (a["gemm_dir"] / "capsule.interface.mlir").read_text() + "\n")


def test_fixture_bytes_are_canonical_independent_and_deterministic(anchors, tmp_path):
    from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
    from merlin.targetgen.golden_store import load_golden

    api, a = module(), anchors
    # Small geometry tests the actual ordinary oracle/fixture writer, not a mock.
    row = dict(case="tiny", family="gemm", geometry=dict(M=2, K=3, N=4))
    cap, _, _ = api.make_case(row, a["gemm"], a["cb"], a["qk"], a["program"], a["sem"])
    paths = [tmp_path / "first", tmp_path / "second"]
    pins = []
    for path in paths:
        path.mkdir()
        pins.append(api.write_fixtures(cap, path))
    for key in ("inputs", "outputs", "golden", "golden_arrays"):
        assert pins[0][key]["sha256"] == pins[1][key]["sha256"]
    with np.load(paths[0] / "inputs.npz", allow_pickle=False) as ins, np.load(paths[0] / "outputs.npz", allow_pickle=False) as outs:
        assert set(ins.files) == {"W", "A0"} and set(outs.files) == {"Y0"}
        leaves = materialize_capsule_leaves(cap)
        for name in ins.files:
            assert np.array_equal(ins[name], np.asarray(leaves[name].data).reshape(leaves[name].shape))
        assert np.array_equal(outs["Y0"], ins["A0"].astype(np.int64) @ ins["W"].astype(np.int64))
        assert outs["Y0"].dtype == np.dtype("int32")
    assert load_golden(paths[0])["outputs"]["Y0"] == outs_value(paths[0])


def outs_value(path):
    with np.load(path / "outputs.npz", allow_pickle=False) as data:
        return data["Y0"].tolist()


def test_same_conv_fixtures_preserve_bias_and_narrow_output_contract(tmp_path):
    from merlin.targetgen.capsule_golden import golden

    api = module()
    tensors = {
        "IFM": dict(shape=[1, 3, 3, 2], dtype="i8", role="input"),
        "W": dict(shape=[18, 2], dtype="i8", role="weight"),
        "B": dict(shape=[2], dtype="i32", role="bias"),
        "Y0": dict(shape=[9, 2], dtype="i8", role="output"),
    }
    cap = dict(name="same_conv", kind="layer", label="public", stimulus_range=[-4, 3],
        inputs=[dict(name=name, **value) for name, value in tensors.items() if name != "Y0"],
        operation=dict(op="conv2d", attributes=dict(ifm="IFM", weight="W", bias="B", out="Y0",
            ci=2, kh=3, kw=3, kernel=[3, 3, 2, 2], layout="nhwc", stride=[1, 1],
            padding=[1, 1, 1, 1], dilation=[1, 1], epilogue=["bias_add"], output_dtype="i8")),
        numeric_policy=dict(compare="exact_int", dtype="i8"))
    expected = golden(cap)
    directory = tmp_path / "conv"
    directory.mkdir()
    api.write_fixtures(cap, directory, tensor_contract=tensors)
    with (np.load(directory / "inputs.npz", allow_pickle=False) as inputs,
          np.load(directory / "outputs.npz", allow_pickle=False) as outputs):
        assert inputs["IFM"].dtype == inputs["W"].dtype == np.dtype("int8")
        assert inputs["B"].dtype == np.dtype("int32") and inputs["B"].nbytes == 8
        assert outputs["Y0"].dtype == np.dtype("int8") and outputs["Y0"].nbytes == 18
        assert outputs["Y0"].tolist() == expected["Y0"]
        # The physical harness consumes these exact bytes without an output cast or reshape adapter.
        path = Path(__file__).with_name("physical_panel.py")
        spec = importlib.util.spec_from_file_location("physical_fixture_consumer", path)
        physical = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(physical)
        assert physical._array_bytes(inputs, "B", tensors["B"]) == inputs["B"].tobytes()
        assert physical._array_bytes(outputs, "Y0", tensors["Y0"]) == outputs["Y0"].tobytes()
    bad = copy.deepcopy(tensors)
    bad["B"]["dtype"] = "i8"
    with pytest.raises(ValueError, match="differs from original contract"):
        api.write_fixtures(cap, directory, tensor_contract=bad)


def test_prepare_manifest_is_operator_only_and_drops_stale_admission(anchors, tmp_path, monkeypatch):
    api, a = module(), anchors
    # Construction metadata is checked for all 34 cases without materializing the large panel twice.
    def fixture_stub(cap, directory):
        return dict(schema=api.FIXTURES, testing="fixture allocation covered separately")
    monkeypatch.setattr(api, "write_fixtures", fixture_stub)
    dest = tmp_path / "panel"
    doc = api.prepare_panel(dest, gemm_capsule=a["gemm_dir"], qk_capsule=a["qk_dir"], facts=a["facts"], frozen_revision="a" * 40)
    assert doc["schema"] == api.SCHEMA and doc["counts"]["total"] == 34
    assert doc["operator_only"] and not doc["author_visible"] and not doc["allow_author_mount"]
    assert not doc["visibility"]["helper_imports_allowed"] and not doc["qualification_issued"]
    assert doc["admission"] == "source_ready_admission_pending"
    assert {row["split"] for row in doc["cases"]} == {"heldout"}
    for row in doc["cases"]:
        cap = yaml.safe_load(Path(row["capsule"]["path"]).read_text())
        assert "software_screen" not in cap
        assert Path(row["source"]["path"]).is_relative_to(dest)
        assert json.loads(Path(row["admission"]["path"]).read_text())["target_execution"] == "not_run"
    with pytest.raises(ValueError, match="new immutable"):
        api.prepare_panel(dest, gemm_capsule=a["gemm_dir"], qk_capsule=a["qk_dir"], facts=a["facts"])


@pytest.mark.skipif(not os.environ.get("MERLIN_TRANSFER_PANEL_CANONICAL"), reason="operator canonical inputs not selected")
def test_selected_real_anchor_sources_roundtrip():
    api = module()
    root = Path(os.environ["MERLIN_TRANSFER_PANEL_CANONICAL"])
    cap, text, _ = api._template(root / "TL_mm_o_m16k2048n2048_raw")
    api._gemm_template(cap, text)
    cap, text, _ = api._template(root / "TL_att_qk_q32kv4_m16k64n16_raw")
    api._qk_template(cap, text)


def test_real_qk_fixtures_keep_all_outputs_and_original_shared_key_inputs(anchors, tmp_path):
    api, a = module(), anchors
    row = dict(case="qk_small", family="qk", geometry=dict(query=3, key=5))
    cap, _, _ = api.make_case(row, a["gemm"], a["cb"], a["qk"], a["program"], a["sem"])
    dest = tmp_path / "qk-fixtures"
    dest.mkdir()
    api.write_fixtures(cap, dest)
    with np.load(dest / "inputs.npz", allow_pickle=False) as inputs, np.load(dest / "outputs.npz", allow_pickle=False) as outputs:
        assert len(inputs.files) == 36 and len(outputs.files) == 32
        assert {key for key in inputs.files if key.startswith("KT")} == {f"KT{i}" for i in range(4)}
        for head in range(32):
            q, kt = inputs[f"Q{head}"], inputs[f"KT{head // 8}"]
            expected = [[sum(int(q[i, k]) * int(kt[k, j]) for k in range(64)) for j in range(5)] for i in range(3)]
            assert outputs[f"S{head}"].tolist() == expected
            assert outputs[f"S{head}"].dtype == np.dtype("int32")


def test_unsafe_internal_mac_bound_is_refused_even_with_i32_storage(anchors):
    api, a = module(), anchors
    sem = copy.deepcopy(a["sem"])
    sem["internal_arithmetic"]["mac_result_bits"] = 12
    sem["internal_arithmetic"]["mac_result_overflow"] = "wrap_to_12_bits"
    row = dict(case="large_k", family="gemm", geometry=dict(M=16, K=4096, N=2048))
    with pytest.raises(ValueError, match="internal MAC bound"):
        api.make_case(row, a["gemm"], a["cb"], a["qk"], a["program"], sem)


def test_fresh_admission_refusal_is_not_saved_as_a_pass(anchors, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin_experiments.phase0 import evidence, program_admission
    api, a = module(), anchors
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    (evidence_root / "evidence-manifest.json").write_text("{}")
    monkeypatch.setattr(evidence, "load_exported_evidence", lambda path: SimpleNamespace())
    monkeypatch.setattr(program_admission, "screen_written", lambda *args, **kwargs: dict(status="unsupported", placements={}))
    monkeypatch.setattr(api, "write_fixtures", lambda *args: {})
    dest = tmp_path / "refused-panel"
    with pytest.raises(ValueError, match="admission refused"):
        api.prepare_panel(dest, gemm_capsule=a["gemm_dir"], qk_capsule=a["qk_dir"], facts=a["facts"], evidence_root=evidence_root)
    assert not (dest / "manifest.json").exists()
