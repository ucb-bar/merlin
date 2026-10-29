"""Neutral exact-source and linked-byte checks for the ordered f32 scan."""

# ruff: noqa: E501 - emitted neutral MLIR retains the producer's operation form.

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_ordered_scan_support as scan

from merlin.frontends.capture_normalization import CaptureNormalizationError, normalize_capture_mlir

_SOURCE = """builtin.module {
  func.func @forward(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %flat = tensor.collapse_shape %arg0 [[0 : i64, 1 : i64]] {prov.region_id = "scan_0", prov.family = "scan", prov.aten = "aten.cumsum.default"} : tensor<2x3xf32> into tensor<6xf32>
    %zero_index = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 0 : index
    %one = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 1 : index
    %total = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 6 : index
    %stride = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 1 : index
    %extent = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 3 : index
    %zero = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 0.000000e+00 : f64
    %out_init = "tensor.empty"() {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} : () -> tensor<6xf32>
    %carry_init = "tensor.empty"() {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} : () -> tensor<6xf64>
    %out, %carry = "scf.for"(%zero_index, %total, %one, %out_init, %carry_init) ({
    ^bb0(%i: index, %out_arg: tensor<6xf32>, %carry_arg: tensor<6xf64>):
      %current = tensor.extract %flat[%i] : tensor<6xf32>
      %wide = arith.extf %current : f32 to f64
      %quotient = arith.divui %i, %stride : index
      %position = arith.remui %quotient, %extent : index
      %first = arith.cmpi eq, %position, %zero_index : index
      %prior = scf.if %first -> (f64) {
        scf.yield %zero : f64
      } else {
        %prior_index = arith.subi %i, %stride : index
        %prior_value = tensor.extract %carry_arg[%prior_index] : tensor<6xf64>
        scf.yield %prior_value : f64
      }
      %sum = arith.addf %prior, %wide : f64
      %rounded = arith.truncf %sum : f64 to f32
      %new_out = tensor.insert %rounded into %out_arg[%i] : tensor<6xf32>
      %new_carry = tensor.insert %sum into %carry_arg[%i] : tensor<6xf64>
      scf.yield %new_out, %new_carry : tensor<6xf32>, tensor<6xf64>
    }) {prov.region_id = "scan_0", prov.family = "scan", prov.aten = "aten.cumsum.default"} : (index, index, index, tensor<6xf32>, tensor<6xf64>) -> (tensor<6xf32>, tensor<6xf64>)
    %result = tensor.expand_shape %out [[0 : i64, 1 : i64]] output_shape [2, 3] {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} : tensor<6xf32> into tensor<2x3xf32>
    func.return %result : tensor<2x3xf32>
  }
}"""


def _selected(bits=64):
    return {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": f"e-p:{bits}:{bits}",
        "index_bits": bits,
        "scope": "neutral test premise",
    }


def _actual(selected):
    bits = selected["index_bits"]
    return {
        **selected,
        "effective_pipeline": ",".join(
            f"{name}{{index-bitwidth={bits}}}"
            for name in (
                "convert-index-to-llvm",
                "convert-arith-to-llvm",
                "finalize-memref-to-llvm",
                "convert-func-to-llvm",
                "convert-cf-to-llvm",
            )
        ),
    }


def _begin(tmp_path, text=_SOURCE, *, bits=64):
    path = tmp_path / "neutral.mlir"
    path.write_text(text)
    normalized, _ = normalize_capture_mlir(text)
    selected = _selected(bits)
    witness = scan.begin(
        path, sha256(text.encode()).hexdigest(), sha256(normalized.encode()).hexdigest(), "b" * 64, selected
    )
    source = {
        "source_sha256": witness["raw_source_sha256"],
        "normalized_source_sha256": witness["normalized_source_sha256"],
        "capture_receipt_sha256": witness["capture_receipt_sha256"],
        "n_source_operations": witness["n_source_operations"],
        "selected_index_observation": selected,
        scan.FIELD: witness,
    }
    return path, source


def _linked(source):
    witness = source[scan.FIELD]
    witness["admissions"] = [
        {
            "root_ordinal": item["root_ordinal"],
            "all_ordinals": item["all_ordinals"],
            "software_declaration": "reviewed_scan",
            "host_profile": "selected",
            "host_declaration": "reviewed_scan",
            "host_package_sha256": "a" * 64,
            "host_spec_sha256": "b" * 64,
            "host_dtype_strategy": "int8_w8a8",
            "software_record_sha256": "c" * 64,
            "capability_record_sha256": "d" * 64,
            "semantic_signature": scan._semantic_signature(item),  # noqa: PLC2701 -- exact fixture
            "hardware_refusals": [
                {"rank": rank, "family": "reduction", "refusal": "input_dtype"}
                for rank in sorted({1, len(item["input_shape"])})
            ],
        }
        for item in witness["occurrences"]
    ]
    witness["admissions_sha256"] = scan._digest(witness["admissions"])  # noqa: PLC2701 -- exact fixture
    witness["admission_verified"] = True
    triple = {"candidate_tree_sha256": "c" * 64, "capture_tree_sha256": "d" * 64, "elf_sha256": "e" * 64}
    actual = _actual(source["selected_index_observation"])
    scan.link(source, actual, triple)
    entry = {**triple, "source_sha256": source["source_sha256"], "index_lowering": actual}
    return entry


@pytest.mark.parametrize(
    "variant",
    [
        "reviewed",
        "unreviewed_software",
        "missing_software",
        "ambiguous_software",
        "unreviewed_host",
        "missing_host",
        "ambiguous_host",
        "eligible_hardware",
        "unknown_hardware",
        "wrong_semantic_policy",
        "mixed_row",
        "extra_scalar",
    ],
)
def test_reviewed_scan_accounts_for_its_closed_body_at_full_model_source_gate(tmp_path, monkeypatch, variant):
    """A reviewed algorithm may own its proved body, not unrelated arithmetic."""
    import json

    from merlin_experiments.phase1.feedback import private_bucketize_support as bucketize
    from merlin_experiments.phase1.feedback import private_data_movement as movement
    from merlin_experiments.phase1.feedback import private_full_models as gate

    from merlin.common import mlir_query as mq
    from merlin.targetgen import application_inventory as ai
    from merlin.xdsl_dialects.lowering import compute_groups as cg

    capture = tmp_path / "capture"
    capture.mkdir()
    # The complete stage has a separate, typed yield-only data movement. A
    # zero-linalg stage is reserved for the direct-return proof boundary.
    return_line = "func.return %result : tensor<2x3xf32>"
    copied = """%copy = "linalg.generic"(%result, %result) <{indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 1, 1>}> ({
        ^bb0(%a: f32, %old: f32):
          "linalg.yield"(%a) : (f32) -> ()
    }) {prov.family = "layout", prov.aten = "aten.clone.default"} :
      (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    func.return %copy : tensor<2x3xf32>"""
    text = _SOURCE.replace(return_line, copied)
    if variant == "extra_scalar":
        text = text.replace("func.return %copy", "%outside = arith.addf %zero, %zero : f64\n    func.return %copy")
    (capture / "model.mlir").write_text(text)
    (capture / "frontend-trace.json").write_text(json.dumps({"graphs": {"prepared": {"nodes": []}}}))

    def verified(_source):
        return {"status": "verified_materialized", "receipt_sha256": "b" * 64}

    monkeypatch.setattr(ai, "verify_capture_receipt", verified)
    monkeypatch.setattr(bucketize, "verify_capture_receipt", verified)
    monkeypatch.setattr(cg, "form_groups", lambda _module, _target: [])
    monkeypatch.setattr(cg, "plan", lambda _module, _target, *, groups: {})
    monkeypatch.setattr(cg, "require_explained", lambda _plan: None)

    signature = {
        "ordered_operand_dtypes": ["f32"],
        "ordered_result_dtypes": ["f32"],
        "compute_dtypes": ["f64"],
        "ranks": [2],
    }
    declaration = {
        "id": "reviewed_ordered_scan",
        "ops": ["aten.cumsum.default"],
        "placement": "host",
        "signature": signature,
    }
    software = {"status": "reviewed", "operations": [declaration]}
    capability = {
        "name": "fixture",
        "compute_units": [
            {
                "name": "unit",
                "kind": "vector",
                "dtypes": ["int8"],
                "ops": ["add"],
                "semantic_capabilities": [{"family": "reduction", "dtypes": ["int8"]}],
            }
        ],
    }
    host_document = {
        "schema": "merlin.host_capabilities.v1",
        "status": "reviewed",
        "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
        "operations": [declaration],
        "evidence": {"scope": "neutral synthetic review only"},
    }
    host = {
        "selected": {
            "status": "reviewed",
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "c" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": host_document,
        }
    }
    if variant == "unreviewed_software":
        software["status"] = "unreviewed"
    elif variant == "missing_software":
        software["operations"] = []
    elif variant == "ambiguous_software":
        software["operations"].append({**declaration, "id": "second_ordered_scan"})
    elif variant == "unreviewed_host":
        host_document["status"] = "unreviewed"
    elif variant == "missing_host":
        host_document["operations"] = []
    elif variant == "ambiguous_host":
        host_document["operations"].append({**declaration, "id": "second_ordered_scan"})
    elif variant == "eligible_hardware":
        capability["compute_units"][0]["semantic_capabilities"][0]["dtypes"] = ["f32"]
        capability["compute_units"][0]["dtypes"] = ["f32"]
    elif variant == "unknown_hardware":
        capability["compute_units"][0]["semantic_capabilities"] = []
        capability["semantic_capabilities_unknown"] = [{"family": "reduction"}]
    elif variant == "wrong_semantic_policy":
        signature["ranks"] = [1]
    elif variant == "mixed_row":
        original_join = movement.source_inventory_by_ordinal

        def mixed_join(parsed, inventory, raw_sha, normalized_sha):
            rows = original_join(parsed, inventory, raw_sha, normalized_sha)
            root = next(i for i, op in enumerate(parsed) if mq.op_name(op) == "scf.for")
            rows[root] = {**rows[root], "ordinals": [root, root + 1]}
            return rows

        monkeypatch.setattr(movement, "source_inventory_by_ordinal", mixed_join)
    if variant not in {"reviewed", "missing_software"}:
        with pytest.raises(ValueError):
            gate._source_obligations(capture, "fixture", software, capability, host, _selected())
        return
    proof = gate._source_obligations(capture, "fixture", software, capability, host, _selected())
    assert proof[scan.FIELD]["count"] == 1
    assert proof[scan.FIELD]["source_verified"] is True
    assert proof[scan.FIELD]["admission_verified"] is True
    if variant == "missing_software":
        assert proof[scan.FIELD]["admissions"][0]["software_declaration"] is None
    assert proof[scan.FIELD]["admissions"][0]["root_ordinal"] == proof[scan.FIELD]["occurrences"][0]["root_ordinal"]


def test_closed_ordered_source_and_exact_linked_tuple(tmp_path):
    _, source = _begin(tmp_path)
    witness = source[scan.FIELD]
    assert witness["count"] == 1
    assert len(witness["occurrences"][0]["all_ordinals"]) == 26
    assert witness["occurrences"][0]["input_shape"] == [2, 3]
    with pytest.raises(ValueError, match="admission"):
        scan.link(
            source,
            _actual(source["selected_index_observation"]),
            {"candidate_tree_sha256": "c" * 64, "capture_tree_sha256": "d" * 64, "elf_sha256": "e" * 64},
        )
    entry = _linked(source)
    assert scan.linked_complete(source, entry, "c" * 64)
    for key in ("capture_tree_sha256", "elf_sha256", "index_lowering"):
        damaged = deepcopy(entry)
        damaged.pop(key)
        assert not scan.linked_complete(source, damaged, "c" * 64)
    damaged = deepcopy(entry)
    damaged["index_lowering"]["index_bits"] = 32
    assert not scan.linked_complete(source, damaged, "c" * 64)
    damaged = deepcopy(source)
    damaged[scan.FIELD]["occurrences"][0]["axis_extent"] = 2
    damaged[scan.FIELD]["occurrences_sha256"] = scan._digest(damaged[scan.FIELD]["occurrences"])  # noqa: PLC2701
    assert not scan.linked_complete(damaged, entry, "c" * 64)
    for field in ("admissions", "admissions_sha256", "admission_verified"):
        damaged = deepcopy(source)
        damaged[scan.FIELD].pop(field)
        assert not scan.linked_complete(damaged, entry, "c" * 64)
    damaged = deepcopy(source)
    damaged[scan.FIELD]["admissions"] = []
    damaged[scan.FIELD]["admissions_sha256"] = scan._digest([])  # noqa: PLC2701
    assert not scan.linked_complete(damaged, entry, "c" * 64)
    damaged = deepcopy(source)
    damaged[scan.FIELD]["admissions"][0]["root_ordinal"] += 1
    damaged[scan.FIELD]["admissions_sha256"] = scan._digest(damaged[scan.FIELD]["admissions"])  # noqa: PLC2701
    assert not scan.linked_complete(damaged, entry, "c" * 64)
    damaged = deepcopy(source)
    damaged[scan.FIELD]["admissions"][0]["hardware_refusals"][0]["refusal"] = "rank"
    damaged[scan.FIELD]["admissions_sha256"] = scan._digest(damaged[scan.FIELD]["admissions"])  # noqa: PLC2701
    assert not scan.linked_complete(damaged, entry, "c" * 64)
    damaged = deepcopy(source)
    damaged[scan.FIELD]["admissions"][0]["semantic_signature"]["rank"] = 1
    damaged[scan.FIELD]["admissions_sha256"] = scan._digest(damaged[scan.FIELD]["admissions"])  # noqa: PLC2701
    assert not scan.linked_complete(damaged, entry, "c" * 64)
    damaged = deepcopy(source)
    damaged[scan.FIELD]["admissions"][0]["semantic_signature"]["rank"] = 2.0
    damaged[scan.FIELD]["admissions_sha256"] = scan._digest(damaged[scan.FIELD]["admissions"])  # noqa: PLC2701
    assert not scan.linked_complete(damaged, entry, "c" * 64)
    damaged = deepcopy(source)
    damaged[scan.FIELD]["admissions"][0]["root_ordinal"] = True
    damaged[scan.FIELD]["admissions_sha256"] = scan._digest(damaged[scan.FIELD]["admissions"])  # noqa: PLC2701
    assert not scan.linked_complete(damaged, entry, "c" * 64)
    for malformed in (None, {"input_shape": 4}):
        damaged = deepcopy(source)
        damaged[scan.FIELD]["occurrences"][0] = malformed
        assert not scan.linked_complete(damaged, entry, "c" * 64)


@pytest.mark.parametrize(
    "old,new",
    [
        ("arith.addf %prior, %wide", "arith.subf %prior, %wide"),
        ("arith.addf %prior, %wide : f64", "arith.addf %prior, %wide fastmath<reassoc> : f64"),
        ("arith.cmpi eq, %position", "arith.cmpi ne, %position"),
        ("%sum = arith.addf %prior, %wide", "%sum = arith.addf %wide, %prior"),
        ("%new_carry = tensor.insert %sum", "%new_carry = tensor.insert %prior"),
        (
            '%total = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 6',
            '%total = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 5',
        ),
        (
            '%stride = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 1',
            '%stride = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 2',
        ),
        ("0.000000e+00 : f64", "-0.000000e+00 : f64"),
        ("%prior_value = tensor.extract %carry_arg", "%prior_value = tensor.extract %out_arg"),
        ("scf.yield %new_out, %new_carry", "scf.yield %new_carry, %new_out"),
        (
            "scf.yield %new_out, %new_carry",
            "%leak = arith.negf %sum : f64\n      scf.yield %new_out, %new_carry",
        ),
        (
            ') {prov.region_id = "scan_0", prov.family = "scan", prov.aten = "aten.cumsum.default"}',
            ') {prov.region_id = "scan_0", prov.family = "scan", prov.other = "unowned"}',
        ),
        (
            "%result = tensor.expand_shape",
            '%extra = arith.constant {prov.region_id = "scan_0", prov.aten = "aten.cumsum.default"} 2 : index\n    %result = tensor.expand_shape',
        ),
        ("output_shape [2, 3]", "output_shape [3, 2]"),
        ("output_shape [2, 3]", "output_shape [%one, 3]"),
        (
            "scf.if %first -> (f64) {\n        scf.yield",
            "scf.if %first -> (f64) {\n      ^bb0(%extra: f64):\n        scf.yield",
        ),
    ],
)
def test_changed_algorithm_or_region_refuses(tmp_path, old, new):
    assert old in _SOURCE
    with pytest.raises((ValueError, CaptureNormalizationError)):
        _begin(tmp_path, _SOURCE.replace(old, new, 1))


def test_source_bytes_width_and_empty_roster_are_mandatory(tmp_path):
    path, source = _begin(tmp_path)
    path.write_text(path.read_text().replace("arith.addf", "arith.subf"))
    with pytest.raises(ValueError, match="raw source bytes changed"):
        scan.begin(path, source["source_sha256"], source["normalized_source_sha256"], "b" * 64, _selected())
    with pytest.raises(ValueError, match="selected signed index width"):
        _begin(tmp_path, bits=3)
    no_scan = (
        """builtin.module { func.func @id(%x: tensor<2xf32>) -> tensor<2xf32> { func.return %x : tensor<2xf32> } }"""
    )
    _, empty = _begin(tmp_path, no_scan)
    assert empty[scan.FIELD]["count"] == 0
    entry = _linked(empty)
    assert scan.linked_complete(empty, entry, "c" * 64)
    missing = deepcopy(empty)
    missing.pop(scan.FIELD)
    assert not scan.linked_complete(missing, entry, "c" * 64)
