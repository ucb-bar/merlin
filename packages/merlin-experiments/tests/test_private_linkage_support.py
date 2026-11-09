"""Closed reviewed source requirements meet actual selected archive suppliers."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from merlin_experiments.phase1.feedback import private_compilation_inputs as compilation
from merlin_experiments.phase1.feedback import private_linalg_support as linalg
from merlin_experiments.phase1.feedback import private_linkage_support as linkage

from merlin.common import mlir_query as mq
from merlin.common.digest import sha256_file
from merlin.runtime.modelinputs import strict_tree_sha256
from merlin.llvmlower import toolchain
from merlin.llvmlower.compilation_recipe import FILENAME
from merlin.mining.registry import load_rvv_package
from merlin.runtime.backends import spike, spike_model
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities
from merlin.targetgen.host_linkage_contract import SCHEMA, SUPPLIER, validate_linkage_contract

_SOURCE = """module {
  func.func @forward(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %init = tensor.empty() : tensor<3xf32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<3xf32>) outs(%init : tensor<3xf32>) {
      ^bb0(%value: f32, %unused: f32):
        %sin = "math.sin"(%value) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32
        linalg.yield %sin : f32
    } -> tensor<3xf32>
    return %out : tensor<3xf32>
  }
}"""
_SOURCE_POW = _SOURCE.replace(
    '        %sin = "math.sin"(%value) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32',
    '        %base = "arith.constant"() <{value = 3.250000e+00 : f32}> : () -> f32\n'
    '        %sin = "math.powf"(%base, %value) '
    "<{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32",
)
_SOURCE_GELU = _SOURCE.replace(
    '        %sin = "math.sin"(%value) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32',
    "\n".join(
        (
            '        %half = "arith.constant"() <{value = 5.000000e-01 : f32}> : () -> f32',
            '        %one = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32',
            '        %cubic = "arith.constant"() <{value = 4.471500e-02 : f32}> : () -> f32',
            '        %scale = "arith.constant"() <{value = 7.9788456e-01 : f32}> : () -> f32',
            '        %square = "arith.mulf"(%value, %value) : (f32, f32) -> f32',
            '        %cube = "arith.mulf"(%square, %value) : (f32, f32) -> f32',
            '        %scaled = "arith.mulf"(%cubic, %cube) : (f32, f32) -> f32',
            '        %plus = "arith.addf"(%value, %scaled) : (f32, f32) -> f32',
            '        %inner = "arith.mulf"(%scale, %plus) : (f32, f32) -> f32',
            '        %tanh = "math.tanh"(%inner) : (f32) -> f32',
            '        %shifted = "arith.addf"(%one, %tanh) : (f32, f32) -> f32',
            '        %half_input = "arith.mulf"(%half, %value) : (f32, f32) -> f32',
            '        %sin = "arith.mulf"(%half_input, %shifted) : (f32, f32) -> f32',
        )
    ),
)


def _source(operation: str) -> str:
    return {
        "math.sin": _SOURCE,
        "literal_base_pow_f32": _SOURCE_POW,
        "tanh_gelu_f32": _SOURCE_GELU,
    }[operation]


def _declaration(archive_sha: str, *, operation: str = "math.sin") -> dict:
    composite = operation in {"literal_base_pow_f32", "tanh_gelu_f32"}
    return {
        "id": "reviewed_neutral_sine",
        "ops": ["linalg.generic"],
        "placement": "host",
        "signature": {"ordered_operand_dtypes": ["f32", "f32"], "ordered_result_dtypes": ["f32"], "ranks": [1]},
        "source_body": {
            "schema": "merlin.static_composite_math_body.v1" if composite else "merlin.static_f32_math_source_body.v1",
            "operation": operation,
        },
        "linkage_contract": {
            "schema": SCHEMA,
            "supplier": SUPPLIER,
            "archive_sha256": archive_sha,
            "symbols": ["powf" if operation == "literal_base_pow_f32" else "tanhf" if composite else "sinf"],
        },
    }


def _admission(
    archive_sha: str,
    *,
    source: bool = True,
    reviewed: bool = True,
    misplaced: bool = False,
    operation: str = "math.sin",
) -> tuple[dict, dict, tuple]:
    program = _source(operation)
    parsed = (next(mq.walk(mq.parse(program), "linalg.generic")),)
    row = {"mlir_operation": "linalg.generic", "count": 1, "ordinals": [0]}
    declaration = _declaration(archive_sha, operation=operation)
    if misplaced:
        declaration["numerical_contract"] = {"linkage_contract": declaration.pop("linkage_contract")}
    selected = {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed" if reviewed else "unreviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [declaration],
                "evidence": {"scope": "neutral test only"},
            },
        }
    }
    decision = admit_host_operation(
        selected,
        row,
        {
            "family": "elementwise_map",
            "ordered_operand_dtypes": ["f32", "f32"],
            "ordered_result_dtypes": ["f32"],
            "rank": 1,
        },
        source_operations=parsed if source else None,
    )
    return decision, row, parsed


def test_contract_is_closed_and_source_only_decision_stays_pending():
    contract = _declaration("c" * 64)["linkage_contract"]
    assert validate_linkage_contract(contract) == contract
    for mutation in (
        {**contract, "unknown": True},
        {**contract, "archive_sha256": "x" * 64},
        {**contract, "symbols": ["sinf", "sinf"]},
        {**contract, "symbols": ["-Wl,inject"]},
        {**contract, "symbols": []},
    ):
        with pytest.raises(ValueError):
            validate_linkage_contract(mutation)
    selected = {
        "schema": "merlin.host_capabilities.v1",
        "status": "reviewed",
        "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
        "operations": [_declaration("c" * 64)],
        "evidence": {},
    }
    validate_host_capabilities(selected)
    for draft in (
        {**selected, "status": "unreviewed"},
        {**selected, "operations": [{**_declaration("c" * 64), "status": "unreviewed"}]},
    ):
        validate_host_capabilities(draft)
    for mutation in (
        {**selected, "operations": [{**_declaration("c" * 64), "source_body": None}]},
        {
            **selected,
            "operations": [
                {
                    **_declaration("c" * 64),
                    "source_body": {"schema": "merlin.static_f32_math_source_body.v1", "operation": "pow"},
                }
            ],
        },
    ):
        with pytest.raises(ValueError):
            validate_host_capabilities(mutation)
    decision, _, _ = _admission("c" * 64)
    assert decision["status"] == "admitted"
    assert decision["linkage_requirement"]["contract"] == contract
    assert "linked" not in json.dumps(decision["linkage_requirement"])
    assert "linkage_requirement" not in _admission("c" * 64, source=False)[0]
    assert "linkage_requirement" not in _admission("c" * 64, misplaced=True)[0]
    assert "linkage_requirement" not in _admission("c" * 64, reviewed=False)[0]


def _bundle(path: Path, *, operation: str = "math.sin") -> Path:
    path.mkdir()
    (path / "model.mlir").write_text(_source(operation), encoding="utf-8")
    (path / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    np.savez(path / "inputs.npz", in0=np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    return path


@pytest.mark.parametrize("operation", ("math.sin", "literal_base_pow_f32", "tanh_gelu_f32"))
def test_real_math_link_supplier_joins_reviewed_source_and_rejects_override(tmp_path, operation):
    package_name = os.environ.get("MERLIN_TEST_HOST_PACKAGE")
    if not package_name or not spike.gcc_path().is_file() or not toolchain.m2m_python().is_file():
        pytest.skip("explicit neutral host package and cross toolchain not selected")
    package = Path(package_name).resolve(strict=True)
    pkg = load_rvv_package(package)
    if pkg.backend != "scalar":
        pytest.skip("neutral scalar host package not selected")
    package_sha = strict_tree_sha256(package)["sha256"]
    march = [flag.removeprefix("-march=") for flag in pkg.cflags if flag.startswith("-march=")]
    assert len(march) == 1
    dts = tmp_path / "neutral.dts"
    dts.write_text('/ { cpus { cpu@0 { riscv,isa = "' + march[0] + '"; }; }; };')
    selected = linkage._selected_archive(package, package_sha, None, dts, sha256_file(dts))
    admission, row, parsed = _admission(selected["archive_sha256"], operation=operation)
    raw = "d" * 64
    linalg_witness = linalg.begin(raw, raw, 1)
    linalg.record(linalg_witness, row, admission, parsed, {0: row})
    witness = linkage.begin(raw, raw, 1)
    linkage.record(witness, row, admission, {0: row}, linalg_witness)
    source = {
        "source_sha256": raw,
        "normalized_source_sha256": raw,
        "n_source_operations": 1,
        "linalg_host_support": linalg_witness,
        linkage.FIELD: witness,
    }
    symbol = "powf" if operation == "literal_base_pow_f32" else "tanhf" if operation == "tanh_gelu_f32" else "sinf"
    assert linkage.symbols(source) == [symbol]
    built = spike_model.build(
        _bundle(tmp_path / "bundle", operation=operation),
        tmp_path / "build",
        arena_mb=1,
        backend="scalar",
        cflags_override=list(pkg.cflags),
        math_archive_symbols=linkage.symbols(source),
    )
    elf = Path(built["elf"])
    recipe = elf.parent / FILENAME
    output = {"compilation_recipe": {"path": str(recipe), "sha256": sha256_file(recipe)}}
    binding = compilation.verify(output, elf, required=True)
    linked_build = {
        "capture_tree_sha256": "c" * 64,
        "candidate_tree_sha256": "e" * 64,
        "elf_sha256": sha256_file(elf),
    }
    entry = {
        "source_sha256": raw,
        "capture_tree_sha256": "c" * 64,
        "elf_sha256": sha256_file(elf),
        "compilation_recipe": binding,
    }
    assert not linkage.linked_complete(source, entry, "e" * 64)
    linkage.link(
        source,
        entry,
        host_package=package,
        host_package_tree_sha256=package_sha,
        vlen=None,
        dts=dts,
        dts_sha256=sha256_file(dts),
        host_isa=march[0],
        simulator_isa=march[0],
        linked_build=linked_build,
    )
    assert linkage.linked_complete(source, entry, "e" * 64)
    changed = deepcopy(source)
    changed[linkage.FIELD].update(count=0, occurrences=[], selection=None, recipe_sha256=None)
    assert not linkage.linked_complete(changed, entry, "e" * 64)
    changed = deepcopy(source)
    changed["linalg_host_support"]["occurrences"][0]["linkage_requirement"]["archive_sha256"] = "0" * 64
    assert not linkage.linked_complete(changed, entry, "e" * 64)
    changed = deepcopy(source)
    changed[linkage.FIELD]["occurrences"][0]["contract"]["archive_sha256"] = "0" * 64
    assert not linkage.linked_complete(changed, entry, "e" * 64)
    changed = deepcopy(source)
    changed[linkage.FIELD]["selection"]["gcc_cflags"].append("-march=another")
    assert not linkage.linked_complete(changed, entry, "e" * 64)
    changed = deepcopy(source)
    changed[linkage.FIELD]["linked_build"]["capture_tree_sha256"] = "0" * 64
    assert not linkage.linked_complete(changed, entry, "e" * 64)
    for field in ("count", "n_source_operations"):
        changed = deepcopy(source)
        changed[linkage.FIELD][field] = True
        assert not linkage.linked_complete(changed, entry, "e" * 64)
    changed = deepcopy(source)
    changed[linkage.FIELD]["selection"]["archive"] = str(tmp_path / "other.a")
    assert not linkage.linked_complete(changed, entry, "e" * 64)
    original_dts = dts.read_bytes()
    dts.write_bytes(original_dts + b"\n")
    assert not linkage.linked_complete(source, entry, "e" * 64)
    dts.write_bytes(original_dts)
    # Rehashing a forged observed supplier does not make it independently selected.
    document = json.loads(recipe.read_text(encoding="utf-8"))
    document["link_suppliers"]["symbols"][symbol]["definition"] = str(elf.parent / "model.o")
    recipe.write_text(json.dumps(document), encoding="utf-8")
    entry["compilation_recipe"]["recipe_sha256"] = sha256_file(recipe)
    assert not linkage.linked_complete(source, entry, "e" * 64)


def test_default_no_contract_needs_explicit_empty_roster_and_no_extra_build_arg():
    source = {
        "source_sha256": "a" * 64,
        "normalized_source_sha256": "b" * 64,
        "n_source_operations": 1,
        "linalg_host_support": {"occurrences": []},
        linkage.FIELD: linkage.begin("a" * 64, "b" * 64, 1),
    }
    assert linkage.symbols(source) is None
    entry = {"source_sha256": "a" * 64, "capture_tree_sha256": "c" * 64, "elf_sha256": "d" * 64}
    linkage.link(
        source,
        entry,
        host_package=Path("/unused"),
        host_package_tree_sha256="e" * 64,
        vlen=None,
        dts=Path("/unused"),
        dts_sha256="f" * 64,
        host_isa="unused",
        simulator_isa="unused",
        linked_build={"capture_tree_sha256": "c" * 64, "candidate_tree_sha256": "e" * 64, "elf_sha256": "d" * 64},
    )
    assert linkage.linked_complete(source, entry, "e" * 64)
    assert not linkage.linked_complete({**source, linkage.FIELD: None}, entry, "e" * 64)


def test_compiled_receipt_cannot_substitute_same_byte_host_package_path(tmp_path):
    package = tmp_path / "selected"
    package.mkdir()
    (package / "manifest.yaml").write_text("neutral: true\n")
    source = {linkage.FIELD: linkage.begin("a" * 64, "b" * 64, 1)}
    receipt = {"inputs": {"package": str(tmp_path / "other"), "package_tree": strict_tree_sha256(package)}}
    with pytest.raises(ValueError, match="different selected host package"):
        linkage.link_compiled(
            source,
            {},
            receipt,
            package,
            receipt["inputs"]["package_tree"]["sha256"],
            tmp_path / "unused-catalog",
            "unused",
            tmp_path / "unused-dts",
            {},
        )
