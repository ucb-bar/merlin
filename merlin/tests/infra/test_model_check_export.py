"""``kernels-export`` writes what the package must lower, deterministically, and nothing of any tensor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.common.schemas import validate
from merlin.model_check import cli
from merlin.model_check import export as EX
from merlin.model_check import kernel_reference as KR
from merlin.model_check import report as R
from merlin.model_check.bundle import Bundle
from merlin.model_check.catalog import CATALOG_SCHEMA
from merlin.model_check.models import load

TARGET = "gemmini"  # the target whose interface binding states these entries
_COMMON = {
    "kind": "op",
    "operand_dtype": "int8",
    "scale_granularity": "tensor",
    "name": "group7",
    "cat": "layers",
    "source_role": "model_derived",
    "source_reference": "test",
}
_ENTRIES = {
    "k_conv": {
        **_COMMON,
        "op": "conv2d",
        "N": 8,
        "ci": 3,
        "Himg": 9,
        "Wimg": 9,
        "kh": 7,
        "kw": 7,
        "stride": [2, 2],
        "padding": [3, 3, 3, 3],
        "epilogue": [],
    },
    "k_mean": {**_COMMON, "op": "matmul", "M": 16, "K": 49, "N": 1, "epilogue": ["acc_scale"], "acc_scale": 0.03125},
    "k_act": {**_COMMON, "op": "matmul", "M": 8, "K": 8, "N": 4, "epilogue": [], "stationary_operand": "activation"},
}
_MODELS = """schema: merlin.model_check.models.v1
target: gemmini
models:
  toy:
    workload: toy
    chain: {kind: single}
    programs:
      - name: main
    judged: {kind: logits, outputs: [{program: main, output: 0}]}
"""


def _call_buffers(interface: str) -> list[dict]:
    from merlin.llvmlower.device_shim import logical_buffers

    buffers, _why = logical_buffers(interface)
    rank = {"input": 0, "weight": 1, "bias": 2, "out": 3}
    ordered = sorted(buffers, key=lambda b: (rank[b.role.rpartition("_")[0]], b.role))
    return [{"role": b.role, "shape": list(b.shape), "dtype": b.dtype} for b in ordered]


def _bundle(root: Path) -> Bundle:
    """A bundle with one program's catalog, sidecar and census, and no blobs or inputs at all."""
    program = root / "toy" / "programs" / "main"
    (program / "dev").mkdir(parents=True)
    interfaces = {
        s: KR.interface_for(e, target=TARGET, operand_dtype="int8", accum_dtype="int32") for s, e in _ENTRIES.items()
    }
    sidecar = {
        "entries": _ENTRIES,
        "call_buffers": {s: _call_buffers(i) for s, i in interfaces.items()},
        "arg_access": {s: ["read", "read", "write"] for s in interfaces},
        "programs": {"k_mean": {"notes": ["a mean over a trailing window: the stored operand is a constant one"]}},
        "routed": [
            {"symbol": "k_conv", "source_region_id": "matmul_0", "group": 1},
            {"symbol": "k_mean", "source_region_id": "reduce_0", "group": 9},
            {"symbol": "k_act", "source_region_id": "matmul_10", "group": 3},
            {"symbol": "k_act", "source_region_id": "matmul_2", "group": 2},
        ],
    }
    (program / "dev" / "device_signatures.json").write_text(json.dumps(sidecar))
    catalog = {
        "schema": CATALOG_SCHEMA,
        "target": TARGET,
        "stage": "/nonexistent",
        "ref_prepared": "/nonexistent",
        "dev_prepared": "/nonexistent",
        "routing": {"operand_dtype": "int8", "accum_dtype": "int32", "numeric_policy": None},
        "interfaces": interfaces,
    }
    (program / "catalog.json").write_text(json.dumps(catalog))
    census = {
        "denominator": 6,
        "device": 4,
        "host": {"host:not_quantized_by_recipe": 2},
        "device_mac_fraction": 0.9,
        "contractions": [
            {"op": "linalg.generic", "dtypes": ["i8", "i8"], "macs": 90, "placement": "device"},
            {"op": "linalg.generic", "dtypes": ["f32", "f32"], "macs": 6, "placement": "host:not_quantized_by_recipe"},
            {"op": "linalg.generic", "dtypes": ["f32", "f32"], "macs": 4, "placement": "host:not_quantized_by_recipe"},
        ],
    }
    (program / "census.json").write_text(json.dumps(census))
    (root / "MANIFEST.json").write_text(json.dumps({"schema": "merlin.model_check.bundle.v1", "models": {}}))
    return Bundle(root)


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_forms_name_the_kernel_shapes():
    calls = [{"role": "out_0", "dtype": "i32"}]
    assert EX.kernel_form(_ENTRIES["k_conv"], calls) == (
        "conv2d 7x7/s2 -> i32",
        "conv2d 7x7/s2 Cin=3 Cout=8 on 9x9 pad [3, 3, 3, 3] -> i32",
    )
    category, form = EX.kernel_form(_ENTRIES["k_mean"], [{"role": "out_0", "dtype": "i8"}])
    assert category == "matmul + acc_scale -> i8" and "acc_scale 0.03125" in form
    assert EX.kernel_form(_ENTRIES["k_act"], calls)[0] == "activation-stationary matmul -> i32"
    assert sorted(["matmul_10", "matmul_2", "matmul_1"], key=EX.natural_key) == ["matmul_1", "matmul_2", "matmul_10"]


def test_export_holds_interfaces_abi_and_entries_and_is_deterministic(tmp_path):
    bundle = _bundle(tmp_path / "bundle")
    model = load_models(tmp_path).model("toy")
    first, second = tmp_path / "a", tmp_path / "b"
    EX.export_model(bundle, model, first)
    EX.export_model(bundle, model, second)
    EX.export_model(bundle, model, second)  # a re-export replaces its own previous output
    assert _files(first) == _files(second)
    catalog = bundle.catalog("toy", "main")
    for symbol, interface in catalog.interfaces.items():
        assert (first / "toy" / "kernels" / "main" / symbol / "interface.mlir").read_text() == interface
    abi = json.loads((first / "toy" / "kernels" / "main" / "k_act" / "abi.json").read_text())
    assert abi["routed_groups"] == 2 and abi["source_region_ids"] == ["matmul_2", "matmul_10"]
    assert [b["role"] for b in abi["call_buffers"]] == ["input_0", "weight_0", "out_0"]
    assert {a["role"] for a in abi["kernel_arguments"]} == {"input_0", "weight_0", "out_0"}
    assert (
        "name" not in abi["entry"] and "cat" not in abi["entry"] and abi["entry"]["stationary_operand"] == "activation"
    )
    index = (first / "index.md").read_text()
    assert "conv2d 7x7/s2 -> i32" in index and "activation-stationary matmul -> i32" in index
    assert "2 x `linalg.generic` f32xf32 (host:not_quantized_by_recipe), 10.00%" in index
    assert "constant one" in (first / "toy" / "index.md").read_text()
    for text in _files(first).values():
        assert str(tmp_path).encode() not in text  # no paths: the export is relocatable


def test_export_refuses_to_replace_a_directory_it_did_not_write(tmp_path):
    bundle = _bundle(tmp_path / "bundle")
    (tmp_path / "out" / "toy").mkdir(parents=True)
    (tmp_path / "out" / "toy" / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        EX.export_model(bundle, load_models(tmp_path).model("toy"), tmp_path / "out")
    assert (tmp_path / "out" / "toy" / "notes.txt").read_text() == "mine"


def test_the_subcommand_writes_a_report_and_takes_only_the_visible_split(tmp_path):
    bundle = _bundle(tmp_path / "bundle")
    models = tmp_path / "models.yaml"
    models.write_text(_MODELS)
    out = tmp_path / "r.json"
    argv = ["kernels-export", "--models", str(models), "--bundle", str(bundle.root), "--model", "toy"]
    code = cli.main([*argv, "--out-dir", str(tmp_path / "x"), "--out", str(out)])
    doc = json.loads(out.read_text())
    assert code == R.EXIT_PASS and doc["status"] == "pass" and validate(doc, R.SCHEMA_NAME) == []
    assert [k["symbol"] for k in doc["programs"][0]["kernels"]] == ["k_act", "k_conv", "k_mean"]
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*argv, "--split", "sealed", "--out-dir", "x", "--out", "o.json"])


def load_models(tmp_path: Path):
    path = tmp_path / "models.yaml"
    if not path.is_file():
        path.write_text(_MODELS)
    return load(path)
